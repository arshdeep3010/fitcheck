"""Fit scoring with a System One decision model (Mercury Decide / Jev).

All requirements go in ONE request: questions run in parallel on the model side,
and one call per JD keeps us well inside free-tier rate limits.

Per requirement we ask two Choice questions:
  <id>_status  met | partial | not_met        -> with a calibrated confidence
  <id>_proof   e0 | e1 | ... | none            -> which retrieved chunk proves it
The proof question can only pick from chunks we actually retrieved, so the model
cannot point at evidence that doesn't exist.
"""

from app.llm.system_one import SystemOneClient
from app.models import Evidence, Requirement, RequirementScore

# "partial" means SOME of the named parts are shown, never "something similar is shown".
# A related-but-different technology is not_met: it used to be listed under partial, which made
# the model lean yes on near misses (grounding then had to demote them to Unverified).
# Examples are deliberately technologies that are NOT in evals/dataset.jsonl, so the evals
# measure the rule, not memorised answers.
STATUS_CRITERIA = {
    "met": ("The evidence shows the candidate built or used EVERY part of this requirement, "
            "at the level asked."),
    "partial": ("The requirement names several things, and the evidence shows the candidate "
                "built or used some of them but not all."),
    "not_met": ("The evidence does not show this exact skill being built or used. A related but "
                "different technology does not count (Flask is not Django, MySQL is not "
                "PostgreSQL, Jenkins is not GitHub Actions), and neither does a comment or doc "
                "that only mentions it."),
}
SNIPPET_CHARS = 1200


def build_request(requirements: list[Requirement], evidence: dict[str, list[Evidence]]
                  ) -> tuple[dict, dict, dict[str, dict[str, str]]]:
    """Returns (state, questions, option_to_chunk_id per requirement)."""
    from_cv = any(ev.chunk.source_type == "self_reported" for items in evidence.values() for ev in items)
    state: dict = {
        "about": ("Evidence is the candidate's own CV text: self-reported claims."
                  if from_cv else "Evidence is code and docs from the candidate's own repositories."),
        "requirements": {},
        "evidence": {},
    }
    questions: dict = {}
    option_maps: dict[str, dict[str, str]] = {}

    for req in requirements:
        items = evidence.get(req.id, [])
        if not items:
            continue
        state["requirements"][req.id] = req.text
        state["evidence"][req.id] = {
            f"e{i}": {"source": ev.chunk.source, "text": ev.chunk.text[:SNIPPET_CHARS]}
            for i, ev in enumerate(items)
        }
        option_maps[req.id] = {f"e{i}": ev.chunk.id for i, ev in enumerate(items)}

        questions[f"{req.id}_status"] = {
            "type": "choice",
            "instructions": (f'Judging only from evidence.{req.id}, does the candidate meet '
                             f'requirement {req.id}: "{req.text}"?'),
            "criteria": STATUS_CRITERIA,
        }
        proof_options = {f"e{i}": ev.chunk.source for i, ev in enumerate(items)}
        proof_options["none"] = "No listed item proves this requirement."
        questions[f"{req.id}_proof"] = {
            "type": "choice",
            "instructions": (f'Which item in evidence.{req.id} best proves requirement '
                             f'{req.id} ("{req.text}")? Choose none if no item proves it.'),
            "criteria": proof_options,
        }
    return state, questions, option_maps


def parse_answers(requirements: list[Requirement], answers: dict,
                  option_maps: dict[str, dict[str, str]]) -> list[RequirementScore]:
    scores: list[RequirementScore] = []
    for req in requirements:
        if req.id not in option_maps:
            # Nothing retrieved at all: code decides, no model needed.
            scores.append(RequirementScore(requirement_id=req.id, status="not_met", confidence=1.0))
            continue
        status = answers.get(f"{req.id}_status") or {}
        proof = answers.get(f"{req.id}_proof") or {}
        choice = status.get("choice")
        if choice not in STATUS_CRITERIA:
            # Missing/odd answer: report zero confidence and let grounding mark it unverified.
            scores.append(RequirementScore(requirement_id=req.id, status="not_met", confidence=0.0))
            continue
        evidence_id = option_maps[req.id].get(proof.get("choice", "none"))
        scores.append(RequirementScore(
            requirement_id=req.id,
            status=choice,
            confidence=float(status.get("confidence") or 0.0),
            evidence_id=evidence_id,
        ))
    return scores


class SystemOneFitScorer:
    def __init__(self, client: SystemOneClient):
        self._client = client
        self.name = client.model

    async def score(self, requirements: list[Requirement],
                    evidence: dict[str, list[Evidence]]) -> list[RequirementScore]:
        state, questions, option_maps = build_request(requirements, evidence)
        answers = await self._client.ask(state, questions) if questions else {}
        return parse_answers(requirements, answers, option_maps)
