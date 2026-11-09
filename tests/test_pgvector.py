# fitcheck: no-index (test data, not evidence)
"""PgVectorStore against a real PostgreSQL + pgvector. Skipped unless TEST_DATABASE_URL is set.

    docker run -d --name fitcheck-pgtest -e POSTGRES_PASSWORD=test -p 55432:5432 pgvector/pgvector:pg16
    TEST_DATABASE_URL=postgresql://postgres:test@localhost:55432/postgres pytest tests/test_pgvector.py
"""

import os
import random

import pytest

from app.models import Chunk

DSN = os.environ.get("TEST_DATABASE_URL")
pytestmark = pytest.mark.skipif(not DSN, reason="set TEST_DATABASE_URL to run against real pgvector")
DIM = 64


def unit(rng: random.Random) -> list[float]:
    v = [rng.gauss(0, 1) for _ in range(DIM)]
    n = sum(x * x for x in v) ** 0.5
    return [x / n for x in v]


@pytest.fixture
def store():
    import psycopg

    from app.rag.store import PgVectorStore

    # A test table is tiny, so Postgres would just scan it and never touch the HNSW index.
    # Production tables are big enough that it does use the index, which is where the
    # filtered-search bug lives. Turning seq scans off makes the test take that path.
    with psycopg.connect(DSN, autocommit=True) as admin:
        admin.execute("ALTER DATABASE postgres SET enable_seqscan = off")
    s = PgVectorStore(DSN)
    s.clear()
    yield s
    s.clear()
    s.close()
    with psycopg.connect(DSN, autocommit=True) as admin:
        admin.execute("ALTER DATABASE postgres RESET enable_seqscan")


def code_chunks(repo: str, n: int) -> list[Chunk]:
    return [Chunk(id=f"{repo}:{i}", repo=repo, path=f"f{i}.py", start_line=1, end_line=5, text=f"code {i}",
                  ref="sha1", url_base=f"https://github.com/me/{repo}") for i in range(n)]


def test_cv_search_finds_cv_sections_among_many_code_chunks(store):
    """The bug this guards: HNSW returned ~40 nearest rows and THEN filtered by source_type,
    so with CV sections at ~1% of rows the CV search came back empty."""
    rng = random.Random(7)
    code = code_chunks("big", 800)
    store.upsert(code, [unit(rng) for _ in code])
    cv = [Chunk(id=f"profile:{i}", repo="profile", path="profile.md", start_line=i, end_line=i + 1,
                text=f"cv {i}", source_type="self_reported", title=f"S{i}") for i in range(8)]
    store.upsert(cv, [unit(rng) for _ in cv])

    hits = store.search(unit(rng), 3, "self_reported")
    assert len(hits) == 3 and all(h.chunk.source_type == "self_reported" for h in hits)
    sims = [h.similarity for h in hits]
    assert sims == sorted(sims, reverse=True)

    code_hits = store.search(unit(rng), 5, "code")
    assert len(code_hits) == 5 and all(h.chunk.source_type == "code" for h in code_hits)


def test_replace_repo_swaps_in_one_transaction_and_summary_reports_it(store):
    rng = random.Random(1)
    store.upsert(code_chunks("a", 3) + code_chunks("b", 2), [unit(rng) for _ in range(5)])
    new = [c.model_copy(update={"ref": "sha2"}) for c in code_chunks("a", 1)]
    store.replace_repo("a", new, [unit(rng)])
    assert store.indexed_refs() == {"a": "sha2", "b": "sha1"}
    summary = {r["repo"]: r for r in store.repo_summary()}
    assert summary["a"]["chunks"] == 1 and summary["a"]["source"] == "github"
    assert summary["a"]["url"] == "https://github.com/me/a/tree/sha2"
    assert store.count() == 3


def test_pool_recovers_after_the_server_drops_connections(store):
    """Neon suspends idle databases and drops connections. The pool must hand out a fresh one."""
    rng = random.Random(2)
    store.upsert(code_chunks("a", 2), [unit(rng) for _ in range(2)])
    with store._pool.connection() as conn:
        conn.execute("SELECT pg_terminate_backend(pid) FROM pg_stat_activity "
                     "WHERE datname = current_database() AND pid <> pg_backend_pid()")
    assert store.count() == 2
    assert len(store.search(unit(rng), 2)) == 2


TEXTS = [
    ("app/graph.py", "from langgraph.graph import END, START, StateGraph\ngraph = StateGraph(FitState)"),
    ("README.md", "Stack: FastAPI, LangGraph, pgvector, NVIDIA"),
    ("app/main.py", "from fastapi import FastAPI\napp = FastAPI()"),
    ("consumer.py", "python kafka consumer"),
    ("src/config.py", "SKILLS = ['python', 'kafka', 'kubernetes', 'spark']"),
    *[(f"util{i}.py", "python helper python") for i in range(6)],
    *[(f"misc/f{i}.txt", f"notes about item {i}") for i in range(60)],
]


def text_chunks() -> list[Chunk]:
    return [Chunk(id=f"t:{i}", repo="t", path=p, start_line=1, end_line=2, text=t)
            for i, (p, t) in enumerate(TEXTS)]


def test_keyword_search_ranks_like_the_in_memory_store(store):
    """Same tokenizer, same BM25 formula: the SQL version must give the same order."""
    from app.rag.store import InMemoryStore

    rng = random.Random(3)
    chunks = text_chunks()
    vectors = [unit(rng) for _ in chunks]
    store.upsert(chunks, vectors)
    mem = InMemoryStore()
    mem.upsert(chunks, vectors)

    q = unit(rng)
    for text in ("AutoGen, CrewAI, or LangGraph", "Python and Kafka", "FastAPI", "Kubernetes", "Rust"):
        pg_hits = store.keyword_search(text, q, 4)
        mem_hits = mem.keyword_search(text, q, 4)
        assert [h.chunk.id for h in pg_hits] == [h.chunk.id for h in mem_hits], text
        for a, b in zip(pg_hits, mem_hits):
            assert abs(a.similarity - b.similarity) < 1e-2  # halfvec rounding
            assert a.via == "keyword"
    assert store.keyword_search("Rust", q, 4) == []


def test_old_tables_get_keyword_columns_filled_on_startup(store):
    """A table indexed before hybrid retrieval has no term counts; opening the store fills them."""
    from app.rag.store import PgVectorStore

    rng = random.Random(4)
    store.upsert(text_chunks(), [unit(rng) for _ in TEXTS])
    store._execute("UPDATE chunks SET terms = NULL, n_terms = NULL")
    assert store.keyword_search("LangGraph", unit(rng), 3) == []

    reopened = PgVectorStore(DSN)
    try:
        assert {h.chunk.path for h in reopened.keyword_search("LangGraph", unit(rng), 3)} == \
            {"app/graph.py", "README.md"}
    finally:
        reopened.close()


# ---------- limits and sync status shared by every Lambda instance (app/shared.py) ----------

@pytest.fixture
def shared(store):
    from app.shared import PgState

    for table in ("fitcheck_hits", "fitcheck_daily", "fitcheck_meta"):
        store.execute(f"DROP TABLE IF EXISTS {table}")
    return PgState(store, "test-secret")


def test_shared_rate_limit_counts_per_visitor_and_stores_no_ip(shared, store):
    from app.shared import PgRateLimiter

    limiter = PgRateLimiter(shared, max_requests=3)
    assert [limiter.allow("203.0.113.9")[0] for _ in range(4)] == [True, True, True, False]
    allowed, retry = limiter.allow("203.0.113.9")
    assert not allowed and 3500 < retry <= 3600
    assert limiter.allow("198.51.100.4") == (True, 0)
    keys = {row[0] for row in store.execute("SELECT key FROM fitcheck_hits")}
    assert len(keys) == 2 and not any("203.0.113" in k for k in keys)


def test_shared_daily_cap(shared):
    from datetime import UTC, datetime

    from app.shared import PgDailyCap

    cap = PgDailyCap(shared, limit=2)
    day = datetime(2026, 11, 9, 23, 0, tzinfo=UTC)
    assert [cap.allow(day)[0] for _ in range(3)] == [True, True, False]
    assert cap.allow(day)[1] == 3600  # until midnight UTC
    assert cap.allow(datetime(2026, 11, 10, 0, 1, tzinfo=UTC))[0]  # a new day


def test_shared_meta_round_trip(shared):
    assert shared.get_meta("github_sync") is None
    shared.set_meta("github_sync", {"last_checked": "2026-11-09T03:00:00+00:00", "last_changes": {}})
    shared.set_meta("github_sync", {"last_checked": "2026-11-09T03:30:00+00:00", "last_changes": {"a": 1}})
    assert shared.get_meta("github_sync")["last_changes"] == {"a": 1}
