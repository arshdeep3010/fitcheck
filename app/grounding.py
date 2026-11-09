"""Grounding: turn the scorer's opinions into verdicts. This is where code owns the decision.

Rules, in order:
1. A positive claim (met / partial) needs a citation to a chunk that was actually
   retrieved for THAT requirement. No citation -> UNVERIFIED.
2. Confidence below the threshold -> UNVERIFIED, whatever the scorer said.
3. Otherwise the scorer's status stands.

UNVERIFIED is never shown as a strength. "Not sure" must not become "yes".
Requirements code can't prove at all (years, degrees) are NOT_CHECKABLE and never scored.
"""

from collections.abc import Callable

from app.models import (Chunk, Citation, Evidence, Requirement, RequirementResult,
                        RequirementScore, Verdict)

SNIPPET_LINES = 12


def ground(requirements: list[Requirement], evidence: dict[str, list[Evidence]],
           scores: list[RequirementScore], min_confidence: float,
           link_for: Callable[[Chunk], str | None] = lambda _c: None) -> list[RequirementResult]:
    by_req = {s.requirement_id: s for s in scores}
    results: list[RequirementResult] = []

    for req in requirements:
        if not req.checkable:
            results.append(RequirementResult(
                requirement=req, verdict=Verdict.NOT_CHECKABLE, confidence=1.0,
                note="Years, degrees and similar can't be proven from code. See my CV."))
            continue
        score = by_req.get(req.id)
        if score is None:
            results.append(RequirementResult(requirement=req, verdict=Verdict.UNVERIFIED,
                                             confidence=0.0, note="No assessment returned."))
            continue

        retrieved = {ev.chunk.id: ev.chunk for ev in evidence.get(req.id, [])}

        if score.status in ("met", "partial"):
            chunk = retrieved.get(score.evidence_id or "")
            if chunk is None:
                results.append(RequirementResult(
                    requirement=req, verdict=Verdict.UNVERIFIED, confidence=score.confidence,
                    note="Model leaned yes, but cited no retrieved evidence, so it isn't claimed."))
                continue
            if score.confidence < min_confidence:
                results.append(RequirementResult(
                    requirement=req, verdict=Verdict.UNVERIFIED, confidence=score.confidence,
                    citation=_citation(chunk, link_for),
                    note="Possible match, but confidence is below the threshold."))
                continue
            results.append(RequirementResult(
                requirement=req, verdict=Verdict(score.status), confidence=score.confidence,
                citation=_citation(chunk, link_for)))
            continue

        # not_met
        if score.confidence < min_confidence:
            results.append(RequirementResult(requirement=req, verdict=Verdict.UNVERIFIED,
                                             confidence=score.confidence,
                                             note="Unclear from the evidence."))
        else:
            results.append(RequirementResult(requirement=req, verdict=Verdict.NOT_MET,
                                             confidence=score.confidence,
                                             note="No evidence in my public repos."))
    return results


def _citation(chunk: Chunk, link_for: Callable[[Chunk], str | None]) -> Citation:
    from_github = bool(chunk.url_base)
    return Citation(source=chunk.source, url=link_for(chunk), snippet=_snippet(chunk.text),
                    kind="github" if from_github else "code", repo=chunk.repo,
                    commit=chunk.ref[:7] if from_github and chunk.ref else None)


def github_link_builder(base_url: str, branch: str) -> Callable[[Chunk], str | None]:
    """Chunks synced from GitHub carry the commit SHA they came from, so their links are
    permalinks: they keep pointing at the right lines even after the code changes."""
    base = base_url.rstrip("/")

    def link(c: Chunk) -> str | None:
        if c.source_type != "code":
            return None
        if c.url_base:
            return f"{c.url_base}/blob/{c.ref or branch}/{c.path}#L{c.start_line}-L{c.end_line}"
        if not base:
            return None
        return f"{base}/{c.repo}/blob/{c.ref or branch}/{c.path}#L{c.start_line}-L{c.end_line}"

    return link


# Rows the CV may still speak to: code didn't prove them, or code can't prove them at all.
CV_CHECKABLE_VERDICTS = {Verdict.NOT_MET, Verdict.UNVERIFIED, Verdict.NOT_CHECKABLE}


def ground_self_reported(requirements: list[Requirement], evidence: dict[str, list[Evidence]],
                         scores: list[RequirementScore], min_confidence: float,
                         cv_url: str | None = None) -> dict[str, RequirementResult]:
    """CV pass. Only a confident "met" that cites a retrieved CV section becomes SELF_REPORTED.

    Everything else is dropped, so the code-based verdict stays. A CV claim never becomes MET.
    """
    by_id = {r.id: r for r in requirements}
    claims: dict[str, RequirementResult] = {}
    for score in scores:
        req = by_id.get(score.requirement_id)
        retrieved = {ev.chunk.id: ev.chunk for ev in evidence.get(score.requirement_id, [])}
        chunk = retrieved.get(score.evidence_id or "")
        if req is None or chunk is None or chunk.source_type != "self_reported":
            continue
        if score.status != "met" or score.confidence < min_confidence:
            continue
        claims[req.id] = RequirementResult(
            requirement=req, verdict=Verdict.SELF_REPORTED, confidence=score.confidence,
            citation=Citation(source=chunk.source, url=cv_url, snippet=_snippet(chunk.text),
                              kind="cv", repo="CV"),
            note="Not proven in my public code. Stated on my CV.")
    return claims


def _snippet(text: str) -> str:
    return "\n".join(text.splitlines()[:SNIPPET_LINES])
