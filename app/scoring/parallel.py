"""Fit scoring with Parallel Decisions (free on OpenRouter), the same questions as System One.

Parallel searches the web on every request and that can't be switched off, so this scorer
makes sure the web can't become evidence:
- Citations: the proof question can only pick e0..eN (chunks we retrieved) or none; the
  model's web sources are dropped by the client and never shown; grounding then re-checks
  that every cited chunk is one of this requirement's retrieved chunks.
- Identity: the candidate's name and GitHub user are replaced with "the candidate" before
  anything is sent, so a search can't find the candidate's own profiles or posts.
- Instructions: the state says to judge only from the evidence given. Parallel still
  searches, so whether web results sway the verdict is measured by the evals, not assumed.
"""

import re

from app.llm.decisions import DecisionsClient
from app.models import Evidence, Requirement, RequirementScore
from app.scoring.system_one import build_request, parse_answers

OFFLINE_RULE = ("Judge ONLY from the requirements and evidence in this state. Web pages are "
                "not evidence about this candidate: ignore them. A skill counts only if one of "
                "the listed evidence items shows it.")


def make_redactor(names: list[str]):
    """Replaces each name (case-insensitive, whole words) with "the candidate"."""
    names = sorted({n.strip() for n in names if n and len(n.strip()) >= 3}, key=len, reverse=True)
    if not names:
        return lambda text: text
    pattern = re.compile(r"\b(" + "|".join(re.escape(n) for n in names) + r")\b", re.IGNORECASE)
    return lambda text: pattern.sub("the candidate", text)


def identity_terms(candidate_name: str, github_user: str) -> list[str]:
    """The full name, each part of it, and the GitHub user."""
    parts = candidate_name.split()
    return [candidate_name, github_user, *parts]


class ParallelFitScorer:
    def __init__(self, client: DecisionsClient, redact_terms: list[str] | None = None):
        self._client = client
        self._redact = make_redactor(redact_terms or [])
        self.name = client.model

    async def score(self, requirements: list[Requirement],
                    evidence: dict[str, list[Evidence]]) -> list[RequirementScore]:
        state, questions, option_maps = build_request(requirements, evidence, redact=self._redact)
        state["rules"] = OFFLINE_RULE
        answers = await self._client.ask(state, questions) if questions else {}
        return parse_answers(requirements, answers, option_maps)
