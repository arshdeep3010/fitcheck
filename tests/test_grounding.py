# fitcheck: no-index (sample requirements below are test data, not evidence)
from app.grounding import github_link_builder, ground
from app.models import Chunk, Evidence, Requirement, RequirementScore, Verdict

REQ = Requirement(id="r1", text="Spring Boot REST APIs")
CHUNK = Chunk(id="repo:A.java:1-20", repo="repo", path="A.java", start_line=1, end_line=20, text="@RestController")
OTHER = Chunk(id="repo:B.java:1-20", repo="repo", path="B.java", start_line=1, end_line=20, text="other")
EVIDENCE = {"r1": [Evidence(chunk=CHUNK, similarity=0.8)]}


def verdict(score: RequirementScore, evidence=EVIDENCE) -> Verdict:
    return ground([REQ], evidence, [score], min_confidence=0.5)[0].verdict


def test_met_with_retrieved_citation_and_confidence_is_met():
    results = ground([REQ], EVIDENCE, [RequirementScore(requirement_id="r1", status="met", confidence=0.9,
                                                        evidence_id=CHUNK.id)],
                     0.5, github_link_builder("https://github.com/me", "main"))
    assert results[0].verdict == Verdict.MET
    assert results[0].citation.url == "https://github.com/me/repo/blob/main/A.java#L1-L20"


def test_met_without_citation_is_not_claimed():
    assert verdict(RequirementScore(requirement_id="r1", status="met", confidence=0.95)) == Verdict.UNVERIFIED


def test_citation_must_be_retrieved_for_this_requirement():
    score = RequirementScore(requirement_id="r1", status="met", confidence=0.95, evidence_id=OTHER.id)
    assert verdict(score) == Verdict.UNVERIFIED


def test_low_confidence_yes_is_not_claimed():
    score = RequirementScore(requirement_id="r1", status="partial", confidence=0.3, evidence_id=CHUNK.id)
    assert verdict(score) == Verdict.UNVERIFIED


def test_confident_not_met_is_reported_as_gap():
    assert verdict(RequirementScore(requirement_id="r1", status="not_met", confidence=0.9)) == Verdict.NOT_MET


def test_missing_score_is_unverified():
    assert ground([REQ], EVIDENCE, [], 0.5)[0].verdict == Verdict.UNVERIFIED


def test_not_checkable_requirement_is_never_scored():
    from app.report import fit_score
    years = Requirement(id="r2", text="3+ years of experience", checkable=False)
    results = ground([REQ, years], EVIDENCE,
                     [RequirementScore(requirement_id="r1", status="met", confidence=0.9, evidence_id=CHUNK.id)], 0.5)
    assert results[1].verdict == Verdict.NOT_CHECKABLE
    assert fit_score(results) == 100  # the un-provable item neither helps nor hurts
