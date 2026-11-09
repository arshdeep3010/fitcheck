"""Vector stores behind one interface: a JSON file for local dev, pgvector for deployment.

`search` takes a source_type so code (proof) and profile (claims) are never mixed:
code retrieval only ever sees code, the CV check only ever sees the profile.
"""

import json
import math
from collections import Counter
from pathlib import Path
from typing import Protocol

from app.models import Chunk, Evidence, SourceType
from app.rag.keywords import B, K1, MAX_DF, bm25, chunk_text, query_terms, term_counts


class VectorStore(Protocol):
    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> None: ...
    def search(self, vector: list[float], k: int, source_type: SourceType = "code") -> list[Evidence]: ...
    # BM25 over the chunk text; `vector` is only used to report each hit's cosine similarity.
    def keyword_search(self, text: str, vector: list[float], k: int,
                       source_type: SourceType = "code") -> list[Evidence]: ...
    def count(self) -> int: ...
    def clear(self) -> None: ...
    def indexed_refs(self) -> dict[str, str]: ...  # repo -> commit SHA it was indexed at
    def delete_repo(self, repo: str) -> None: ...
    # Swap all of one repo's chunks in a single step (searches never see a half-indexed repo).
    def replace_repo(self, repo: str, chunks: list[Chunk], vectors: list[list[float]]) -> None: ...
    def repo_summary(self) -> list[dict]: ...  # one row per indexed source, for the status page


def _summary_row(repo: str, source_type: str, ref: str, url_base: str, chunks: int) -> dict:
    if source_type == "self_reported":
        source = "cv"
    elif url_base:
        source = "github"
    else:
        source = "local"
    return {"repo": repo, "source": source, "chunks": chunks, "commit": ref or None,
            "url": f"{url_base}/tree/{ref}" if url_base and ref else (url_base or None)}


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


class InMemoryStore:
    """Brute-force cosine search. Fine for a few thousand chunks; persists to a JSON file.

    Copy-on-write: every change builds a new dict and swaps it in, so searches running in
    other threads during a live sync iterate a stable snapshot.
    """

    def __init__(self, path: str | None = None):
        self._path = Path(path) if path else None
        self._rows: dict[str, tuple[Chunk, list[float]]] = {}
        self._terms: dict[str, Counter] = {}  # chunk id + text hash -> term counts, built lazily
        if self._path and self._path.exists():
            for row in json.loads(self._path.read_text(encoding="utf-8")):
                self._rows[row["chunk"]["id"]] = (Chunk(**row["chunk"]), row["vector"])

    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        rows = dict(self._rows)
        for chunk, vector in zip(chunks, vectors, strict=True):
            rows[chunk.id] = (chunk, vector)
        self._rows = rows

    def search(self, vector: list[float], k: int, source_type: SourceType = "code") -> list[Evidence]:
        scored = [Evidence(chunk=c, similarity=cosine(vector, v))
                  for c, v in self._rows.values() if c.source_type == source_type]
        scored.sort(key=lambda e: e.similarity, reverse=True)
        return scored[:k]

    def keyword_search(self, text: str, vector: list[float], k: int,
                       source_type: SourceType = "code") -> list[Evidence]:
        # Sorted by id so equal scores come out in the same order as in PgVectorStore.
        rows = sorted(((c, v) for c, v in self._rows.values() if c.source_type == source_type),
                      key=lambda row: row[0].id.encode())
        docs = [self._counts(c) for c, _ in rows]
        return [Evidence(chunk=rows[i][0], similarity=cosine(vector, rows[i][1]), via="keyword")
                for i, _ in bm25(query_terms(text), docs, k)]

    def _counts(self, chunk: Chunk) -> Counter:
        key = f"{chunk.id}|{hash(chunk.text)}"  # a re-indexed chunk with new text gets new counts
        if key not in self._terms:
            self._terms[key] = term_counts(chunk_text(chunk.path, chunk.text))
        return self._terms[key]

    def count(self) -> int:
        return len(self._rows)

    def clear(self) -> None:
        self._rows = {}
        self._terms = {}

    def indexed_refs(self) -> dict[str, str]:
        return {c.repo: c.ref for c, _ in self._rows.values() if c.ref}

    def delete_repo(self, repo: str) -> None:
        self._rows = {k: v for k, v in self._rows.items() if v[0].repo != repo}

    def replace_repo(self, repo: str, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        rows = {k: v for k, v in self._rows.items() if v[0].repo != repo}
        for chunk, vector in zip(chunks, vectors, strict=True):
            rows[chunk.id] = (chunk, vector)
        self._rows = rows

    def repo_summary(self) -> list[dict]:
        groups: dict[str, list[Chunk]] = {}
        for chunk, _ in self._rows.values():
            groups.setdefault(chunk.repo, []).append(chunk)
        return [_summary_row(repo, cs[0].source_type, cs[0].ref, cs[0].url_base, len(cs))
                for repo, cs in groups.items()]

    def save(self) -> None:
        if not self._path:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        rows = [{"chunk": c.model_dump(), "vector": v} for c, v in self._rows.values()]
        # Write then rename, so a crash mid-save never leaves a broken index behind.
        tmp = self._path.with_suffix(self._path.suffix + ".tmp")
        tmp.write_text(json.dumps(rows), encoding="utf-8")
        tmp.replace(self._path)


def _vector_literal(vector: list[float]) -> str:
    return "[" + ",".join(f"{x:.7f}" for x in vector) + "]"


class PgVectorStore:
    """PostgreSQL + pgvector, cosine distance with an HNSW index.

    Vectors are stored as `halfvec` (16-bit floats): HNSW indexes `vector` only up to
    2000 dimensions, but `halfvec` up to 4000, and NVIDIA's embeddings are 2048-d.
    Half precision also halves storage, which matters on a free-tier database.

    Two things a hosted Postgres (e.g. Neon) needs:
    - A connection pool that checks each connection before use. Neon suspends an idle
      database and drops its connections; a single long-lived connection would then fail
      every search until the app restarts.
    - Filtered search that still returns rows. HNSW finds ~40 nearest rows and only then
      applies `WHERE source_type = ...`. CV sections are ~1% of rows, so a plain filtered
      HNSW query almost always finds none. CV search is therefore exact (it orders by an
      expression the index can't serve, over a handful of rows), and code search turns on
      pgvector's iterative scan (0.8+) so the filter can't starve it either.

    Keyword search: each row stores its term counts (`terms`, jsonb, GIN-indexed) and
    `n_terms`, computed in Python by the same tokenizer as the in-memory store. The SQL in
    `keyword_search` is the same BM25 formula, so both stores rank keyword hits alike.
    """

    _SELECT = ("SELECT id, repo, path, start_line, end_line, text, source_type, title, ref, url_base, "
               "1 - (embedding <=> %(q)s::halfvec) AS similarity FROM chunks WHERE source_type = %(t)s ")

    def __init__(self, dsn: str, min_size: int = 1, max_size: int = 4):
        # imported lazily so offline/dev runs don't need a database driver
        from psycopg_pool import ConnectionPool

        self._pool = ConnectionPool(
            dsn, min_size=min_size, max_size=max_size, open=True,
            kwargs={"autocommit": True}, configure=self._configure,
            check=ConnectionPool.check_connection, max_idle=300)
        with self._pool.connection() as conn:
            conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        if self._has_table():
            self._migrate()

    @staticmethod
    def _configure(conn) -> None:
        """Per connection: let filtered HNSW scans keep going until they have k rows."""
        import psycopg

        try:
            conn.execute("SET hnsw.iterative_scan = strict_order")
        except psycopg.Error:  # pgvector < 0.8: no iterative scans; code search is ~99% of rows anyway
            pass

    def close(self) -> None:
        self._pool.close()

    def execute(self, sql: str, params=None) -> list[tuple]:
        """For the other tables in the same database (app/shared.py)."""
        return self._execute(sql, params)

    def _execute(self, sql: str, params=None) -> list[tuple]:
        """Run one statement on a pooled (health-checked) connection; returns rows, if any."""
        with self._pool.connection() as conn:
            cur = conn.execute(sql, params)
            return cur.fetchall() if cur.description else []

    def _ensure_table(self, dim: int) -> None:
        self._execute(f"""
            CREATE TABLE IF NOT EXISTS chunks (
                id text PRIMARY KEY,
                repo text NOT NULL,
                path text NOT NULL,
                start_line int NOT NULL,
                end_line int NOT NULL,
                text text NOT NULL,
                source_type text NOT NULL DEFAULT 'code',
                title text NOT NULL DEFAULT '',
                ref text NOT NULL DEFAULT '',
                url_base text NOT NULL DEFAULT '',
                embedding halfvec({int(dim)}) NOT NULL
            )""")
        self._execute(
            "CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw "
            "ON chunks USING hnsw (embedding halfvec_cosine_ops)")
        self._migrate()

    def _migrate(self) -> None:
        """Add the keyword-search columns to a table made before hybrid retrieval, and fill them."""
        self._execute("ALTER TABLE chunks ADD COLUMN IF NOT EXISTS terms jsonb")
        self._execute("ALTER TABLE chunks ADD COLUMN IF NOT EXISTS n_terms int")
        self._execute("CREATE INDEX IF NOT EXISTS chunks_terms_gin ON chunks USING gin (terms)")
        missing = self._execute("SELECT id, path, text FROM chunks WHERE terms IS NULL")
        if missing:
            with self._pool.connection() as conn, conn.transaction(), conn.cursor() as cur:
                for cid, path, text in missing:
                    counts = term_counts(chunk_text(path, text))
                    cur.execute("UPDATE chunks SET terms = %s::jsonb, n_terms = %s WHERE id = %s",
                                (json.dumps(counts), sum(counts.values()), cid))

    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        if not chunks:
            return
        self._ensure_table(len(vectors[0]))
        with self._pool.connection() as conn, conn.cursor() as cur:
            self._insert(cur, chunks, vectors)

    @staticmethod
    def _insert(cur, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        for chunk, vector in zip(chunks, vectors, strict=True):
            counts = term_counts(chunk_text(chunk.path, chunk.text))
            cur.execute(
                """INSERT INTO chunks (id, repo, path, start_line, end_line, text,
                                       source_type, title, ref, url_base, embedding, terms, n_terms)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::halfvec, %s::jsonb, %s)
                   ON CONFLICT (id) DO UPDATE
                   SET text = EXCLUDED.text, title = EXCLUDED.title, ref = EXCLUDED.ref,
                       url_base = EXCLUDED.url_base, embedding = EXCLUDED.embedding,
                       terms = EXCLUDED.terms, n_terms = EXCLUDED.n_terms""",
                (chunk.id, chunk.repo, chunk.path, chunk.start_line, chunk.end_line,
                 chunk.text, chunk.source_type, chunk.title, chunk.ref, chunk.url_base,
                 _vector_literal(vector), json.dumps(counts), sum(counts.values())))

    def replace_repo(self, repo: str, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        """Delete + insert in one transaction: searches never see a half-indexed repo."""
        if chunks:
            self._ensure_table(len(vectors[0]))
        elif not self._has_table():
            return
        with self._pool.connection() as conn, conn.transaction(), conn.cursor() as cur:
            cur.execute("DELETE FROM chunks WHERE repo = %s", (repo,))
            self._insert(cur, chunks, vectors)

    def repo_summary(self) -> list[dict]:
        if not self._has_table():
            return []
        rows = self._execute(
            """SELECT repo, min(source_type), max(ref), max(url_base), count(*)
               FROM chunks GROUP BY repo""")
        return [_summary_row(*row) for row in rows]

    def search(self, vector: list[float], k: int, source_type: SourceType = "code") -> list[Evidence]:
        if not self._has_table():
            return []
        params = {"q": _vector_literal(vector), "t": source_type, "k": k}
        if source_type == "code":
            order = "ORDER BY embedding <=> %(q)s::halfvec LIMIT %(k)s"  # HNSW (+ iterative scan)
        else:
            order = "ORDER BY 1 - (embedding <=> %(q)s::halfvec) DESC LIMIT %(k)s"  # exact: index can't serve it
        return [self._evidence(r) for r in self._execute(self._SELECT + order, params)]

    _KEYWORD_SQL = """
        WITH q AS (SELECT unnest(%(terms)s::text[]) AS term),
             stats AS (SELECT count(*)::float AS n, coalesce(avg(n_terms), 1)::float AS avgdl
                       FROM chunks WHERE source_type = %(t)s),
             df AS (SELECT q.term, (SELECT count(*) FROM chunks c
                                    WHERE c.source_type = %(t)s AND c.terms ? q.term)::float AS df
                    FROM q),
             bm AS (SELECT c.id,
                           sum(ln(1 + (stats.n - df.df + 0.5) / (df.df + 0.5))
                               * (c.terms ->> df.term)::float * (%(k1)s + 1)
                               / ((c.terms ->> df.term)::float
                                  + %(k1)s * (1 - %(b)s + %(b)s * c.n_terms / stats.avgdl))) AS score
                    FROM chunks c CROSS JOIN stats JOIN df ON c.terms ? df.term
                    WHERE c.source_type = %(t)s AND c.terms ?| %(terms)s::text[]
                      AND df.df <= greatest(1, %(max_df)s * stats.n)  -- specific terms only (keywords.MAX_DF)
                    GROUP BY c.id)
        SELECT ch.id, ch.repo, ch.path, ch.start_line, ch.end_line, ch.text, ch.source_type,
               ch.title, ch.ref, ch.url_base, 1 - (ch.embedding <=> %(q)s::halfvec) AS similarity
        FROM bm JOIN chunks ch ON ch.id = bm.id
        ORDER BY bm.score DESC, ch.id COLLATE "C" LIMIT %(k)s"""

    def keyword_search(self, text: str, vector: list[float], k: int,
                       source_type: SourceType = "code") -> list[Evidence]:
        """BM25, the same formula as keywords.bm25. Only rows sharing a term are scored
        (`terms ?| ...` uses the GIN index); df and the average length cover every row of
        this source type, as in the in-memory store."""
        terms = query_terms(text)
        if not terms or not self._has_table():
            return []
        params = {"q": _vector_literal(vector), "t": source_type, "k": k,
                  "terms": terms, "k1": K1, "b": B, "max_df": MAX_DF}
        return [self._evidence(r, via="keyword") for r in self._execute(self._KEYWORD_SQL, params)]

    @staticmethod
    def _evidence(r: tuple, via: str = "vector") -> Evidence:
        return Evidence(chunk=Chunk(id=r[0], repo=r[1], path=r[2], start_line=r[3], end_line=r[4],
                                    text=r[5], source_type=r[6], title=r[7], ref=r[8], url_base=r[9]),
                        similarity=float(r[10]), via=via)

    def count(self) -> int:
        return self._execute("SELECT count(*) FROM chunks")[0][0] if self._has_table() else 0

    def clear(self) -> None:
        self._execute("DROP TABLE IF EXISTS chunks")

    def _has_table(self) -> bool:
        return self._execute("SELECT to_regclass('public.chunks')")[0][0] is not None

    def indexed_refs(self) -> dict[str, str]:
        if not self._has_table():
            return {}
        rows = self._execute("SELECT DISTINCT repo, ref FROM chunks WHERE ref <> ''")
        return {repo: ref for repo, ref in rows}

    def delete_repo(self, repo: str) -> None:
        if self._has_table():
            self._execute("DELETE FROM chunks WHERE repo = %s", (repo,))
