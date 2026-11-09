"""Keyword search (BM25) over chunks, the half of hybrid retrieval that matches exact names.

Why: a requirement like "AutoGen, CrewAI, or LangGraph" is a list of product names. Its
embedding is close to nothing in particular (~0.20 similarity to every chunk), so vector
search returned the README's stack line and missed `app/graph.py`. Names are exactly what
keyword search is good at.

Both stores use this module, so they score the same way: InMemoryStore runs `bm25` in
Python; PgVectorStore stores `term_counts` per row and runs the same formula in SQL.
"""

import math
import re
from collections import Counter

K1 = 1.2  # term-frequency saturation (standard BM25 values)
B = 0.75  # length normalisation
# Only specific terms search: ones in at most 3% of chunks. Keyword search is here to find
# names (langgraph: 2 of 711 chunks, kafka: 3); common words ("container": 41, "python": 67)
# are vector search's job, and matching on them only added noise like a UI file for
# "Container orchestration with Kubernetes". Measured on the development set only.
MAX_DF = 0.03

# Words that say nothing about a skill. Job descriptions are full of them.
STOPWORDS = frozenset("""
a an and are as at be by can for from has have in into is it its of on or that the this to
use used using with within via etc e.g i.e
experience experienced hands hands-on strong solid good deep proven working knowledge
understanding familiarity familiar exposure ability skills skill proficiency proficient
expertise building build developing develop development implementing implement designing
design years year plus nice preferred required must should will including like such
""".split())

_QUERY_TOKEN = re.compile(r"[a-z0-9][a-z0-9+#]*(?:\.[a-z0-9+#]+)*")
_DOC_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9+#]*(?:[._/-][A-Za-z0-9+#]+)*")
_CAMEL = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|[0-9]+")


def _keep(term: str) -> bool:
    return len(term) > 1 and term not in STOPWORDS


def query_terms(text: str) -> list[str]:
    """Terms of a requirement, deduplicated, in order. Dotted names stay whole: "Next.js"
    must match "next.js", not every chunk that says "next" (CI/CD and Redis-compatible split)."""
    seen: dict[str, None] = {}
    for tok in _QUERY_TOKEN.findall(text.lower()):
        if _keep(tok):
            seen.setdefault(tok)
    return list(seen)


def term_counts(text: str) -> Counter:
    """Terms of a chunk with their counts. Each compound token also yields its parts, so
    `from langgraph.graph import StateGraph` matches "langgraph", "stategraph", "graph"."""
    counts: Counter = Counter()
    for raw in _DOC_TOKEN.findall(text):
        whole = raw.lower()
        parts = {p.lower() for piece in re.split(r"[._/-]", raw) for p in [piece, *_CAMEL.findall(piece)]}
        for term in {whole} | parts:
            if _keep(term):
                counts[term] += 1
    return counts


def chunk_text(path: str, text: str) -> str:
    """What keyword search sees: the file path (Dockerfile, ci.yml) plus the content."""
    return f"{path}\n{text}"


def idf(n_docs: int, df: int) -> float:
    return math.log(1 + (n_docs - df + 0.5) / (df + 0.5))


def bm25(terms: list[str], docs: list[Counter], k: int, max_df: float = MAX_DF) -> list[tuple[int, float]]:
    """Top-k (doc index, score) for the query terms; only docs sharing a specific term
    (document frequency <= max_df of all docs) are returned."""
    if not docs or not terms:
        return []
    n = len(docs)
    lengths = [sum(d.values()) for d in docs]
    avgdl = (sum(lengths) / n) or 1.0
    df = {t: sum(1 for d in docs if t in d) for t in terms}
    terms = [t for t in terms if 0 < df[t] <= max(1.0, max_df * n)]  # floor: tiny indexes
    scored = []
    for i, (doc, length) in enumerate(zip(docs, lengths)):
        score = 0.0
        for t in terms:
            tf = doc.get(t, 0)
            if tf:
                score += idf(n, df[t]) * tf * (K1 + 1) / (tf + K1 * (1 - B + B * length / avgdl))
        if score > 0:
            scored.append((i, score))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:k]
