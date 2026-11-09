"""Offline baseline scorer: word overlap between requirement and evidence.

Used in fake mode (no keys) and as the floor every model-based scorer must beat in evals.
"""

import re

from app.models import Evidence, Requirement, RequirementScore

_WORD = re.compile(r"[a-z0-9][a-z0-9+#.]*")
STOPWORDS = {
    "a", "an", "and", "the", "of", "in", "on", "for", "with", "to", "or", "at", "by", "is", "are",
    "experience", "knowledge", "strong", "good", "working", "using", "use", "years", "year",
    "ability", "understanding", "skills", "skill", "hands", "on", "proficiency", "familiarity",
}


def keywords(text: str) -> set[str]:
    return {w.strip(".") for w in _WORD.findall(text.lower())} - STOPWORDS


class KeywordFitScorer:
    name = "keyword-baseline"

    async def score(self, requirements: list[Requirement],
                    evidence: dict[str, list[Evidence]]) -> list[RequirementScore]:
        scores: list[RequirementScore] = []
        for req in requirements:
            wanted = keywords(req.text)
            best_id, best = None, 0.0
            for ev in evidence.get(req.id, []):
                have = keywords(f"{ev.chunk.path} {ev.chunk.text}")
                overlap = len(wanted & have) / len(wanted) if wanted else 0.0
                if overlap > best:
                    best_id, best = ev.chunk.id, overlap
            if best >= 0.6:
                scores.append(RequirementScore(requirement_id=req.id, status="met", confidence=0.8,
                                               evidence_id=best_id))
            elif best >= 0.3:
                scores.append(RequirementScore(requirement_id=req.id, status="partial", confidence=0.6,
                                               evidence_id=best_id))
            else:
                scores.append(RequirementScore(requirement_id=req.id, status="not_met", confidence=0.8))
        return scores
