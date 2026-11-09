# fitcheck: no-index (sample requirements below are test data, not evidence)
from app.models import Chunk, Evidence, Requirement
from app.scoring.system_one import SystemOneFitScorer, build_request, parse_answers

REQS = [Requirement(id="r1", text="Spring Boot"), Requirement(id="r2", text="Kafka")]
CHUNKS = [Chunk(id=f"repo:F{i}.java:1-10", repo="repo", path=f"F{i}.java", start_line=1, end_line=10,
                text=f"code {i}") for i in range(2)]
EVIDENCE = {"r1": [Evidence(chunk=c, similarity=0.7) for c in CHUNKS], "r2": []}


def test_request_has_status_and_proof_questions_only_for_requirements_with_evidence():
    state, questions, option_maps = build_request(REQS, EVIDENCE)
    assert set(questions) == {"r1_status", "r1_proof"}
    assert questions["r1_status"]["type"] == "choice"
    assert set(questions["r1_status"]["criteria"]) == {"met", "partial", "not_met"}
    assert set(questions["r1_proof"]["criteria"]) == {"e0", "e1", "none"}
    assert state["evidence"]["r1"]["e1"]["source"] == "repo/F1.java:L1-L10"
    assert option_maps == {"r1": {"e0": CHUNKS[0].id, "e1": CHUNKS[1].id}}


def test_answers_map_back_to_chunk_ids():
    _, _, option_maps = build_request(REQS, EVIDENCE)
    answers = {
        "r1_status": {"type": "choice", "choice": "met", "confidence": 0.82},
        "r1_proof": {"type": "choice", "choice": "e1", "confidence": 0.7},
    }
    scores = {s.requirement_id: s for s in parse_answers(REQS, answers, option_maps)}
    assert scores["r1"].status == "met" and scores["r1"].evidence_id == CHUNKS[1].id
    assert scores["r1"].confidence == 0.82
    # no evidence at all: decided in code
    assert scores["r2"].status == "not_met" and scores["r2"].confidence == 1.0


def test_proof_none_means_no_evidence_id():
    _, _, option_maps = build_request(REQS, EVIDENCE)
    answers = {"r1_status": {"choice": "met", "confidence": 0.9}, "r1_proof": {"choice": "none"}}
    assert parse_answers(REQS, answers, option_maps)[0].evidence_id is None


def test_missing_answer_gets_zero_confidence():
    _, _, option_maps = build_request(REQS, EVIDENCE)
    assert parse_answers(REQS, {}, option_maps)[0].confidence == 0.0


class RecordingClient:
    model = "inception/mercury-decide:free"

    def __init__(self):
        self.calls = 0

    async def ask(self, state, questions):
        self.calls += 1
        return {"r1_status": {"choice": "partial", "confidence": 0.6}, "r1_proof": {"choice": "e0"}}


async def test_one_api_call_per_jd():
    client = RecordingClient()
    scores = await SystemOneFitScorer(client).score(REQS, EVIDENCE)
    assert client.calls == 1
    assert [s.status for s in scores] == ["partial", "not_met"]
