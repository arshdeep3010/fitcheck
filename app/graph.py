"""The FitCheck pipeline as a LangGraph state graph.

    START -> parse_jd --(no requirements)--> report -> END
                 |
                 v
              retrieve -> score -> ground -> check_cv -> report -> END

check_cv only looks at what code didn't prove, and can only add "Stated on CV" rows.

Every node reads the shared state and returns only the keys it changes.
Failures never produce claims: a broken retrieval or scorer ends in UNVERIFIED rows.
"""

import asyncio
import logging
import operator
import time
from dataclasses import dataclass, field
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph

from app.config import Settings
from app.grounding import CV_CHECKABLE_VERDICTS, github_link_builder, ground, ground_self_reported
from app.jd_parser import JdParser
from app.llm.credits import CreditPause
from app.llm.embeddings import Embedder
from app.llm.http import CreditsExhaustedError
from app.models import Evidence, FitReport, Requirement, RequirementResult, RequirementScore
from app.rag.hybrid import retrieve as retrieve_evidence
from app.rag.store import VectorStore
from app.report import build_report
from app.scoring import FitScorer

log = logging.getLogger(__name__)


def _merge_dicts(left: dict, right: dict) -> dict:
    return {**left, **right}


class FitState(TypedDict, total=False):
    jd_text: str
    requirements: list[Requirement]
    evidence: dict[str, list[Evidence]]
    retrieval_failed: bool
    scores: list[RequirementScore]
    results: list[RequirementResult]
    report: FitReport
    # Reducers: each node appends its own warnings / timings instead of overwriting.
    warnings: Annotated[list[str], operator.add]
    timings_ms: Annotated[dict[str, int], _merge_dicts]
    scorer_used: str  # set only when the fallback scorer stood in for the configured one


@dataclass
class Deps:
    parser: JdParser
    embedder: Embedder
    store: VectorStore
    scorer: FitScorer
    settings: Settings
    models: dict[str, str] = field(default_factory=dict)
    # Free scorer to use when the paid one is out of credits (None = fail closed instead).
    fallback_scorer: FitScorer | None = None
    credits: CreditPause = field(default_factory=CreditPause)


# Shown to every visitor, so it explains, without account details. The fix goes to the log.
FALLBACK_NOTICE = ("The main scorer ({main}) is out of credits right now, so this check was "
                   "scored by the backup model ({backup}).")
NO_CREDITS_NOTICE = "The fit scorer is out of credits right now; nothing is claimed."


def _ms(start: float) -> int:
    return int((time.perf_counter() - start) * 1000)


def build_graph(deps: Deps):
    settings = deps.settings
    link_for = github_link_builder(settings.github_base_url, settings.github_branch)

    async def parse_jd(state: FitState) -> dict:
        start = time.perf_counter()
        requirements, warnings = await deps.parser.parse(state["jd_text"])
        return {"requirements": requirements, "warnings": warnings,
                "timings_ms": {"parse_jd": _ms(start)}}

    async def run_scorer(requirements: list[Requirement], evidence: dict[str, list[Evidence]]
                         ) -> tuple[list[RequirementScore], dict]:
        """Score with the configured scorer, or with the free fallback while it's out of credits.

        Returns (scores, extra state). Raises like the scorer does when there's no fallback.
        """
        fallback = deps.fallback_scorer
        if fallback is None or not deps.credits.paused:
            try:
                return await deps.scorer.score(requirements, evidence), {}
            except CreditsExhaustedError as exc:
                deps.credits.trip(str(exc))
                log.warning("%s is out of credits (%s). Using %s for the next %d min. Top up "
                            "credits, or set SCORER=llm_json.", deps.scorer.name, exc,
                            fallback.name if fallback else "nothing", deps.credits.pause_s // 60)
                if fallback is None:
                    raise
        scores = await fallback.score(requirements, evidence)
        notice = FALLBACK_NOTICE.format(main=deps.scorer.name,
                                        backup=fallback.name.removeprefix("llm-json:"))
        return scores, {"warnings": [notice], "scorer_used": fallback.name}

    def checkable(state: FitState) -> list[Requirement]:
        return [r for r in state["requirements"] if r.checkable]

    async def retrieve(state: FitState) -> dict:
        start = time.perf_counter()
        requirements = checkable(state)
        if not requirements:
            return {"evidence": {}, "retrieval_failed": False, "timings_ms": {"retrieve": 0}}
        try:
            vectors = await deps.embedder.embed([r.text for r in requirements], kind="query")
            hits = await asyncio.gather(*(
                asyncio.to_thread(retrieve_evidence, deps.store, r.text, v,
                                  settings.retrieve_top_k, settings.retrieval, settings.keyword_extra)
                for r, v in zip(requirements, vectors, strict=True)))
            evidence = {r.id: h for r, h in zip(requirements, hits, strict=True)}
            return {"evidence": evidence, "retrieval_failed": False,
                    "timings_ms": {"retrieve": _ms(start)}}
        except Exception as exc:  # noqa: BLE001 - any failure here must fail closed
            log.exception("retrieval failed")
            return {"evidence": {}, "retrieval_failed": True,
                    "warnings": [f"Evidence search was unavailable ({type(exc).__name__}); nothing is claimed."],
                    "timings_ms": {"retrieve": _ms(start)}}

    async def score(state: FitState) -> dict:
        start = time.perf_counter()
        requirements = checkable(state)
        unknown = [RequirementScore(requirement_id=r.id, status="not_met", confidence=0.0)
                   for r in requirements]
        if not requirements:
            return {"scores": [], "timings_ms": {"score": 0}}
        if state.get("retrieval_failed"):
            return {"scores": unknown, "timings_ms": {"score": 0}}
        try:
            scores, extra = await run_scorer(requirements, state["evidence"])
            return {"scores": scores, **extra, "timings_ms": {"score": _ms(start)}}
        except CreditsExhaustedError:
            return {"scores": unknown, "warnings": [NO_CREDITS_NOTICE],
                    "timings_ms": {"score": _ms(start)}}
        except Exception as exc:  # noqa: BLE001 - scorer outage must not become a verdict
            log.exception("scoring failed")
            return {"scores": unknown,
                    "warnings": [f"Fit scoring was unavailable ({type(exc).__name__}); nothing is claimed."],
                    "timings_ms": {"score": _ms(start)}}

    def ground_node(state: FitState) -> dict:
        start = time.perf_counter()
        results = ground(state["requirements"], state.get("evidence", {}), state["scores"],
                         settings.min_confidence, link_for)
        return {"results": results, "timings_ms": {"ground": _ms(start)}}

    async def check_cv(state: FitState) -> dict:
        """Second pass, only for what code didn't prove: does my CV at least state it?

        Searches self-reported profile chunks only. A hit becomes SELF_REPORTED ("Stated
        on CV"), never MET. Any failure here leaves the code verdicts untouched.
        """
        start = time.perf_counter()
        results = state.get("results", [])
        pending = [r.requirement for r in results if r.verdict in CV_CHECKABLE_VERDICTS]
        if not pending or state.get("retrieval_failed"):
            return {"timings_ms": {"check_cv": 0}}
        try:
            vectors = await deps.embedder.embed([r.text for r in pending], kind="query")
            hits = await asyncio.gather(*(
                asyncio.to_thread(deps.store.search, v, settings.cv_top_k, "self_reported")
                for v in vectors))
            cv_evidence = {r.id: h for r, h in zip(pending, hits, strict=True) if h}
            if not cv_evidence:
                return {"timings_ms": {"check_cv": _ms(start)}}
            askable = [r.model_copy(update={"checkable": True}) for r in pending if r.id in cv_evidence]
            scores, extra = await run_scorer(askable, cv_evidence)
            claims = ground_self_reported(pending, cv_evidence, scores, settings.min_confidence,
                                          settings.cv_url or None)
            merged = [claims.get(r.requirement.id, r) for r in results]
            return {"results": merged, **extra, "timings_ms": {"check_cv": _ms(start)}}
        except CreditsExhaustedError:
            return {"warnings": ["CV check was unavailable (the fit scorer is out of credits)."],
                    "timings_ms": {"check_cv": _ms(start)}}
        except Exception as exc:  # noqa: BLE001 - CV check is optional; never break the report
            log.exception("CV check failed")
            return {"warnings": [f"CV check was unavailable ({type(exc).__name__})."],
                    "timings_ms": {"check_cv": _ms(start)}}

    def report_node(state: FitState) -> dict:
        models = deps.models
        if state.get("scorer_used"):  # the report names the model that actually judged
            models = {**models, "scorer": state["scorer_used"]}
        warnings = list(dict.fromkeys(state.get("warnings", [])))  # score + check_cv may repeat one
        report = build_report(settings.candidate_name, state.get("results", []),
                              warnings, state.get("timings_ms", {}), models)
        return {"report": report}

    def after_parse(state: FitState) -> str:
        return "retrieve" if state.get("requirements") else "report"

    graph = StateGraph(FitState)
    graph.add_node("parse_jd", parse_jd)
    graph.add_node("retrieve", retrieve)
    graph.add_node("score", score)
    graph.add_node("ground", ground_node)
    graph.add_node("check_cv", check_cv)
    graph.add_node("report", report_node)

    graph.add_edge(START, "parse_jd")
    graph.add_conditional_edges("parse_jd", after_parse, {"retrieve": "retrieve", "report": "report"})
    graph.add_edge("retrieve", "score")
    graph.add_edge("score", "ground")
    graph.add_edge("ground", "check_cv")
    graph.add_edge("check_cv", "report")
    graph.add_edge("report", END)
    return graph.compile()


async def run_fit_check(compiled_graph, jd_text: str) -> FitReport:
    final = await compiled_graph.ainvoke({"jd_text": jd_text, "warnings": [], "timings_ms": {}})
    return final["report"]


def step_detail(node: str, update: dict) -> str:
    """One human line per finished step, for the live progress UI."""
    if node == "parse_jd":
        reqs = update.get("requirements", [])
        not_checkable = sum(1 for r in reqs if not r.checkable)
        return f"{len(reqs)} requirements" + (f", {not_checkable} not code-checkable" if not_checkable else "")
    if node == "retrieve":
        if update.get("retrieval_failed"):
            return "evidence search unavailable"
        snippets = sum(len(v) for v in update.get("evidence", {}).values())
        return f"{snippets} code snippets compared"
    if node == "score":
        return f"{len(update.get('scores', []))} requirements judged"
    if node == "ground":
        proven = sum(1 for r in update.get("results", []) if r.verdict.value in ("met", "partial"))
        return f"{proven} backed by cited code"
    if node == "check_cv":
        if "results" not in update:
            return "nothing left for the CV"
        claims = sum(1 for r in update["results"] if r.verdict.value == "self_reported")
        return f"{claims} found on CV"
    return ""


async def stream_fit_check(compiled_graph, jd_text: str):
    """Yields ("step", node, ms, detail) as each node finishes, then ("report", FitReport)."""
    initial = {"jd_text": jd_text, "warnings": [], "timings_ms": {}}
    async for chunk in compiled_graph.astream(initial, stream_mode="updates"):
        for node, update in chunk.items():
            update = update or {}
            if node == "report":
                yield ("report", update["report"])
            else:
                ms = (update.get("timings_ms") or {}).get(node, 0)
                yield ("step", node, ms, step_detail(node, update))
