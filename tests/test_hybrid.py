# fitcheck: no-index (test data, not evidence)
"""Hybrid retrieval: keyword (BM25) search, rank fusion, and the LangGraph miss it fixes."""

import math

from app.models import Chunk, Evidence
from app.rag.hybrid import fuse, retrieve
from app.rag.keywords import query_terms, term_counts
from app.rag.store import InMemoryStore, cosine


def chunk(cid: str, path: str, text: str, source_type: str = "code") -> Chunk:
    return Chunk(id=cid, repo="demo", path=path, start_line=1, end_line=5, text=text,
                 source_type=source_type)


def ev(cid: str, sim: float = 0.5) -> Evidence:
    return Evidence(chunk=chunk(cid, f"{cid}.py", cid), similarity=sim)


def angle(deg: float) -> list[float]:
    return [math.cos(math.radians(deg)), math.sin(math.radians(deg))]


def filler(n: int = 60) -> list[tuple[Chunk, list[float]]]:
    """Unrelated chunks, so the small test indexes have realistic term frequencies."""
    return [(chunk(f"f{i:03d}", f"misc/f{i}.txt", f"notes about item {i}"), angle(170)) for i in range(n)]


# --- tokenizer

def test_query_terms_drop_jd_filler_and_keep_names():
    assert query_terms("Hands-on experience with AutoGen, CrewAI, or LangGraph") == \
        ["autogen", "crewai", "langgraph"]


def test_query_terms_keep_dotted_names_whole_but_split_slashes_and_dashes():
    assert query_terms("Next.js") == ["next.js"]
    assert query_terms("CI/CD pipelines") == ["ci", "cd", "pipelines"]
    assert query_terms("Redis-compatible caching") == ["redis", "compatible", "caching"]


def test_term_counts_split_imports_and_camel_case():
    counts = term_counts("from langgraph.graph import StateGraph, END")
    for term in ("langgraph.graph", "langgraph", "graph", "stategraph", "state", "import", "end"):
        assert counts[term] >= 1, term


def test_a_dotted_query_does_not_match_its_parts():
    # "next.config.js" is not Next.js: the query term "next.js" must not match it.
    assert "next.js" not in term_counts("files = ['next.config.js', 'package.json']")


# --- fusion

def test_fuse_puts_chunks_found_by_both_searches_first():
    out = fuse([ev("a"), ev("b"), ev("c")], [ev("c"), ev("d")], extra=3)
    assert [e.chunk.id for e in out][0] == "c"
    assert {e.chunk.id: e.via for e in out} == {"c": "both", "a": "vector", "d": "keyword", "b": "vector"}


def test_fuse_only_adds_keyword_hits_and_never_drops_a_vector_hit():
    vector = [ev(f"v{i}") for i in range(5)]
    out = fuse(vector, [ev("k1"), ev("k2"), ev("k3"), ev("k4")], extra=3)
    ids = [e.chunk.id for e in out]
    assert set(ids) == {"v0", "v1", "v2", "v3", "v4", "k1", "k2", "k3"}
    assert ids.index("k1") < ids.index("v4")  # ordered by RRF, not appended


def test_fuse_does_not_spend_extra_slots_on_chunks_vector_already_found():
    out = fuse([ev("a"), ev("b")], [ev("a"), ev("b"), ev("x")], extra=1)
    assert {e.chunk.id for e in out} == {"a", "b", "x"}


def test_fuse_breaks_rank_ties_toward_the_vector_hit():
    out = fuse([ev("a"), ev("b")], [ev("x"), ev("y")], extra=2)
    assert [e.chunk.id for e in out] == ["a", "x", "b", "y"]


def test_fuse_keeps_the_cosine_similarity():
    out = fuse([ev("a", 0.31)], [ev("a", 0.99)], extra=1)
    assert out[0].similarity == 0.31  # the vector hit's real cosine, not a fused score


# --- in-memory keyword search

def store_with(rows: list[tuple[Chunk, list[float]]]) -> InMemoryStore:
    s = InMemoryStore()
    s.upsert([c for c, _ in rows], [v for _, v in rows])
    return s


def test_keyword_search_finds_names_and_reports_real_cosine():
    s = store_with([
        (chunk("g", "app/graph.py", "from langgraph.graph import StateGraph"), angle(80)),
        (chunk("m", "app/main.py", "from fastapi import FastAPI"), angle(5)),
        *filler(),
    ])
    hits = s.keyword_search("LangGraph", angle(0), 5)
    assert [h.chunk.id for h in hits] == ["g"]
    assert hits[0].via == "keyword"
    assert abs(hits[0].similarity - cosine(angle(0), angle(80))) < 1e-9


def test_keyword_search_ranks_rare_terms_above_common_ones():
    s = store_with([
        (chunk("common", "a.py", "python python python service"), angle(0)),
        (chunk("rare", "b.py", "python kafka consumer"), angle(0)),
        *[(chunk(f"p{i}", f"p{i}.py", "python helper"), angle(0)) for i in range(5)],
        *filler(150),
    ])
    assert s.keyword_search("Python and Kafka", angle(0), 1)[0].chunk.id == "rare"


def test_common_words_do_not_search():
    """'Container orchestration with Kubernetes' with no Kubernetes anywhere must not
    match a UI file on the word 'container' (in 41 of 711 real chunks)."""
    s = store_with([
        *[(chunk(f"ui{i}", f"ui/el{i}.js", "const container = document.body"), angle(0)) for i in range(10)],
        *filler(90),
    ])
    assert s.keyword_search("Container orchestration with Kubernetes", angle(0), 5) == []


def test_keyword_search_never_mixes_code_and_cv():
    s = store_with([
        (chunk("code", "a.py", "langgraph"), angle(0)),
        (chunk("cv", "profile.md", "LangGraph", "self_reported"), angle(0)),
        *filler(),
    ])
    assert [h.chunk.id for h in s.keyword_search("LangGraph", angle(0), 5)] == ["code"]
    assert [h.chunk.id for h in s.keyword_search("LangGraph", angle(0), 5, "self_reported")] == ["cv"]


def test_keyword_counts_follow_a_reindexed_chunk():
    s = store_with([(chunk("a", "a.py", "flask app"), angle(0)), *filler()])
    assert s.keyword_search("flask", angle(0), 5)
    s.replace_repo("demo", [chunk("a", "a.py", "django app")], [angle(0)])
    assert not s.keyword_search("flask", angle(0), 5)
    assert s.keyword_search("django", angle(0), 5)


# --- the held-out miss, reproduced

def test_hybrid_finds_the_code_that_vector_search_missed():
    """'AutoGen, CrewAI, or LangGraph' embeds close to nothing in particular: vector search
    returned five unrelated chunks and the README, never app/graph.py. Keyword search finds it."""
    query = angle(0)
    rows = [(chunk(f"x{i}", f"src/other{i}.py", f"def handler_{i}(): pass"), angle(10 + i)) for i in range(5)]
    rows.append((chunk("readme", "README.md", "Stack: FastAPI, LangGraph, pgvector"), angle(16)))
    rows.append((chunk("graph", "app/graph.py",
                       "from langgraph.graph import END, START, StateGraph\n"
                       "graph = StateGraph(FitState)"), angle(60)))
    s = store_with(rows + filler())
    text = "Experience with AutoGen, CrewAI, or LangGraph"

    vector_only = retrieve(s, text, query, 5, mode="vector")
    assert "graph" not in [e.chunk.id for e in vector_only]

    hybrid = retrieve(s, text, query, 5, mode="hybrid", extra=3)
    assert "graph" in [e.chunk.id for e in hybrid]
    # ...and everything vector search found is still there.
    assert {e.chunk.id for e in vector_only} <= {e.chunk.id for e in hybrid}
    assert len(hybrid) <= 5 + 3


def test_vector_mode_is_exactly_the_old_search():
    s = store_with([(chunk(f"c{i}", f"{i}.py", "x"), angle(i * 7)) for i in range(8)])
    assert retrieve(s, "anything", angle(0), 3, mode="vector") == s.search(angle(0), 3)
