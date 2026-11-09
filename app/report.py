"""Deterministic report: the scores and summary are computed in code, not written by a model.

Two scores, so a reader never confuses a claim with proof:
  score          proven in code only ("Stated on CV" counts as 0)
  score_with_cv  also counts "Stated on CV" rows as met
Requirements code can't prove (years, degrees) are left out of the code score, and only
enter the CV score if the CV actually states them.
"""

from app.models import FitReport, RequirementResult, Verdict

CODE_CREDIT = {Verdict.MET: 1.0, Verdict.PARTIAL: 0.5, Verdict.NOT_MET: 0.0,
               Verdict.UNVERIFIED: 0.0, Verdict.SELF_REPORTED: 0.0}
CV_CREDIT = {**CODE_CREDIT, Verdict.SELF_REPORTED: 1.0}
MUST_HAVE_WEIGHT = 2.0
NICE_TO_HAVE_WEIGHT = 1.0


def _weighted(results: list[RequirementResult], credit: dict[Verdict, float], code_only: bool) -> int:
    total = earned = 0.0
    for r in results:
        if r.verdict == Verdict.NOT_CHECKABLE or (code_only and not r.requirement.checkable):
            continue
        weight = MUST_HAVE_WEIGHT if r.requirement.must_have else NICE_TO_HAVE_WEIGHT
        total += weight
        earned += weight * credit[r.verdict]
    return round(100 * earned / total) if total else 0


def fit_score(results: list[RequirementResult]) -> int:
    """Proven in code only."""
    return _weighted(results, CODE_CREDIT, code_only=True)


def fit_score_with_cv(results: list[RequirementResult]) -> int:
    """Also counting what my CV states but code doesn't prove."""
    return _weighted(results, CV_CREDIT, code_only=False)


def summarize(results: list[RequirementResult]) -> str:
    if not results:
        return "No checkable requirements were found in this job description."
    count = {v: sum(1 for r in results if r.verdict == v) for v in Verdict}
    parts = [f"{count[Verdict.MET]} met", f"{count[Verdict.PARTIAL]} partly met",
             f"{count[Verdict.NOT_MET]} not met"]
    if count[Verdict.SELF_REPORTED]:
        parts.append(f"{count[Verdict.SELF_REPORTED]} stated on CV only")
    if count[Verdict.UNVERIFIED]:
        parts.append(f"{count[Verdict.UNVERIFIED]} unverified")
    if count[Verdict.NOT_CHECKABLE]:
        parts.append(f"{count[Verdict.NOT_CHECKABLE]} not checkable")
    return f"{len(results)} requirements: " + ", ".join(parts) + ". Every 'met' links to the code that proves it."


def build_report(candidate: str, results: list[RequirementResult], warnings: list[str],
                 timings_ms: dict[str, int], models: dict[str, str]) -> FitReport:
    return FitReport(candidate=candidate, score=fit_score(results),
                     score_with_cv=fit_score_with_cv(results), summary=summarize(results),
                     results=results, warnings=warnings, timings_ms=timings_ms, models=models)
