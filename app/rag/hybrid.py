"""Hybrid retrieval: vector search + keyword search, merged by reciprocal rank fusion.

Vector search finds meaning ("REST API" ~ a FastAPI route). Keyword search finds names
("LangGraph" -> `from langgraph.graph import StateGraph`), which embed poorly.

Keyword search can only ADD evidence: the scorer always sees the full vector top k, plus up
to `extra` chunks that only keyword search found. So vector-only evidence is a strict
subset of hybrid evidence, and a proving file can never be pushed out by a keyword hit.
(Plain RRF top-k would always drop vector hits #4 and #5 for keyword hits #1 and #2.)

The merged list is ordered by reciprocal rank fusion (RRF): each chunk gets
sum(1 / (60 + rank)) over the lists it appears in. RRF compares ranks, not scores, because
a cosine of 0.2 and a BM25 of 5.1 are on different scales. 60 is the constant from the
original RRF paper; it stops one list's #1 from dominating.

Evidence.similarity stays the real cosine similarity for every hit, keyword hits included.
"""

from app.models import Evidence, SourceType

RRF_K = 60


def fuse(vector_hits: list[Evidence], keyword_hits: list[Evidence], extra: int) -> list[Evidence]:
    """Every vector hit, plus up to `extra` keyword-only hits, in RRF order."""
    in_vector = {ev.chunk.id for ev in vector_hits}
    added = [ev for ev in keyword_hits if ev.chunk.id not in in_vector][:extra]
    keep = in_vector | {ev.chunk.id for ev in added}

    scores: dict[str, float] = {}
    found: dict[str, Evidence] = {}
    by: dict[str, set[str]] = {}
    for kind, hits in (("vector", vector_hits), ("keyword", keyword_hits)):
        for rank, ev in enumerate(hits, start=1):
            cid = ev.chunk.id
            if cid not in keep:
                continue
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (RRF_K + rank)
            found.setdefault(cid, ev)
            by.setdefault(cid, set()).add(kind)
    # Ties (same rank in one list each) go to the vector hit.
    order = sorted(scores, key=lambda cid: (-scores[cid], "vector" not in by[cid]))
    via = {frozenset({"vector"}): "vector", frozenset({"keyword"}): "keyword"}
    return [found[cid].model_copy(update={"via": via.get(frozenset(by[cid]), "both")})
            for cid in order]


def retrieve(store, text: str, vector: list[float], k: int, mode: str = "vector",
             extra: int = 3, source_type: SourceType = "code") -> list[Evidence]:
    """The one retrieval entry point (the graph and the evals both use it).

    vector mode: the vector top k. hybrid mode: that, plus up to `extra` keyword-only hits.
    """
    vector_hits = store.search(vector, k, source_type)
    if mode != "hybrid":
        return vector_hits
    # Ask for k + extra so chunks already in the vector list don't use up the extra slots.
    keyword_hits = store.keyword_search(text, vector, k + extra, source_type)
    return fuse(vector_hits, keyword_hits, extra)
