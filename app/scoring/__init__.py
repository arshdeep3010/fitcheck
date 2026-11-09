"""Fit scorers: given requirements and retrieved evidence, judge each requirement.

Every scorer implements the same interface, so they can be swapped by config and
benchmarked against each other with evals/run_evals.py.
"""

from typing import Protocol

from app.models import Evidence, Requirement, RequirementScore


class FitScorer(Protocol):
    name: str

    async def score(self, requirements: list[Requirement],
                    evidence: dict[str, list[Evidence]]) -> list[RequirementScore]: ...
