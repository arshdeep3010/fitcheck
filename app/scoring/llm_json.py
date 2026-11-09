"""Comparison scorer: a regular chat LLM asked for JSON.

Exists for the evals: "decision model vs plain LLM" on accuracy, false claims and latency.
Note its confidence is self-reported text, not a calibrated probability.
"""

import json

from pydantic import BaseModel, Field, ValidationError

from app.llm.chat import ChatModel, extract_json
from app.models import Evidence, Requirement, RequirementScore, Status

SYSTEM_PROMPT = "You assess job-fit strictly from provided evidence. Reply with a single JSON object only."


class _Item(BaseModel):
    requirement_id: str
    status: Status
    confidence: float = Field(ge=0.0, le=1.0)
    evidence_id: str | None = None


class _Reply(BaseModel):
    scores: list[_Item]


class LlmJsonFitScorer:
    def __init__(self, chat: ChatModel):
        self._chat = chat
        self.name = f"llm-json:{chat.name}"

    async def score(self, requirements: list[Requirement],
                    evidence: dict[str, list[Evidence]]) -> list[RequirementScore]:
        payload = {
            req.id: {
                "requirement": req.text,
                "evidence": {ev.chunk.id: ev.chunk.text[:1200] for ev in evidence.get(req.id, [])},
            }
            for req in requirements
        }
        user = (
            "For each requirement, decide met / partial / not_met using ONLY its evidence.\n"
            "Be strict:\n"
            # Examples are deliberately NOT technologies from evals/dataset.jsonl.
            "- met: the evidence shows EVERY part of the requirement. If it names several things "
            "(e.g. 'Flask, Celery and RabbitMQ') and any one is missing, it is partial at best.\n"
            "- A related technology is not the same one, and is not_met: Flask is not Django, "
            "MySQL is not PostgreSQL, Jenkins is not GitHub Actions.\n"
            "- Text that only NAMES a technology is not experience with it: a comment, a doc, "
            "a README stack line, or a list of names in code (e.g. a set of skill or keyword "
            "strings). Only code that imports, calls, configures or runs it counts.\n"
            "- confidence: your honest probability that the status is right, from 0 to 1.\n"
            "evidence_id must be one of that requirement's evidence keys, or null.\n"
            'Return {"scores": [{"requirement_id": "...", "status": "...", "confidence": 0.0, '
            '"evidence_id": "..."}]}\n\n' + json.dumps(payload)
        )
        try:
            reply = _Reply.model_validate(extract_json(await self._chat.complete(SYSTEM_PROMPT, user)))
        except (ValueError, ValidationError):
            reply = _Reply(scores=[])

        by_id = {s.requirement_id: s for s in reply.scores}
        out: list[RequirementScore] = []
        for req in requirements:
            item = by_id.get(req.id)
            if item is None:
                out.append(RequirementScore(requirement_id=req.id, status="not_met", confidence=0.0))
            else:
                out.append(RequirementScore(requirement_id=req.id, status=item.status,
                                            confidence=item.confidence, evidence_id=item.evidence_id))
        return out
