# fitcheck: no-index (sample requirements below are test data, not evidence)
"""The CV / profile source: indexed separately, used only for what code didn't prove,
never shown as Met, and scored separately."""

import asyncio
from pathlib import Path

import httpx

from app.config import Settings
from app.deps import build_deps
from app.graph import build_graph, run_fit_check
from app.grounding import ground_self_reported
from app.llm.embeddings import HashEmbedder
from app.models import Chunk, Evidence, Requirement, RequirementResult, RequirementScore, Verdict
from app.rag.chunker import chunk_profile, chunk_repo
from app.rag.ingest import ingest
from app.rag.store import InMemoryStore
from app.report import fit_score, fit_score_with_cv

PROFILE = """<!-- comment, not a section -->
# Me

## Current work
- Build Apache Kafka event pipelines for a healthcare client.

## Education
- B.E. in Computer Science, 2025.

## Empty section
"""


def write_profile(tmp_path: Path) -> Path:
    path = tmp_path / "profile.md"
    path.write_text(PROFILE, encoding="utf-8")
    return path


def test_profile_is_one_self_reported_chunk_per_section(tmp_path):
    chunks = chunk_profile(write_profile(tmp_path))
    assert [c.title for c in chunks] == ["Current work", "Education"]
    assert all(c.source_type == "self_reported" for c in chunks)
    assert chunks[0].source == "CV: Current work"


def test_profile_folder_is_never_indexed_as_code(sample_repo):
    (sample_repo / "profile").mkdir()
    (sample_repo / "profile" / "profile.md").write_text("## Skills\n- Kafka\n- Redis\n- Go\n", encoding="utf-8")
    assert all(not c.path.startswith("profile/") for c in chunk_repo("demo-repo", sample_repo))


async def test_code_search_never_returns_cv_and_vice_versa(tmp_path, sample_repo):
    store = InMemoryStore()
    await ingest({"demo-repo": sample_repo}, HashEmbedder(), store, profile=write_profile(tmp_path))
    vec = (await HashEmbedder().embed(["Apache Kafka"], "query"))[0]
    assert all(e.chunk.source_type == "code" for e in store.search(vec, 10))
    assert all(e.chunk.source_type == "self_reported" for e in store.search(vec, 10, "self_reported"))


async def test_unchanged_profile_is_not_re_embedded(tmp_path):
    store = InMemoryStore()
    profile = write_profile(tmp_path)
    first = await ingest({}, HashEmbedder(), store, profile=profile)
    second = await ingest({}, HashEmbedder(), store, profile=profile)
    assert first["profile (CV)"] == 2 and second["profile (CV)"] == "unchanged"


CV_CHUNK = Chunk(id="profile:p.md:4-6", repo="profile", path="p.md", start_line=4, end_line=6,
                 text="Kafka pipelines", source_type="self_reported", title="Current work")
CODE_CHUNK = Chunk(id="repo:A.java:1-9", repo="repo", path="A.java", start_line=1, end_line=9, text="kafka")
REQ = Requirement(id="r1", text="Apache Kafka")


def test_confident_cv_match_becomes_self_reported_never_met():
    claims = ground_self_reported([REQ], {"r1": [Evidence(chunk=CV_CHUNK, similarity=0.9)]},
                                  [RequirementScore(requirement_id="r1", status="met", confidence=0.9,
                                                    evidence_id=CV_CHUNK.id)], 0.5)
    assert claims["r1"].verdict == Verdict.SELF_REPORTED
    assert claims["r1"].citation.source == "CV: Current work"


def test_cv_pass_ignores_partial_low_confidence_and_code_chunks():
    ev = {"r1": [Evidence(chunk=CV_CHUNK, similarity=0.9), Evidence(chunk=CODE_CHUNK, similarity=0.9)]}
    for score in [RequirementScore(requirement_id="r1", status="partial", confidence=0.9, evidence_id=CV_CHUNK.id),
                  RequirementScore(requirement_id="r1", status="met", confidence=0.2, evidence_id=CV_CHUNK.id),
                  RequirementScore(requirement_id="r1", status="met", confidence=0.9, evidence_id=CODE_CHUNK.id)]:
        assert ground_self_reported([REQ], ev, [score], 0.5) == {}


def test_two_scores_keep_claims_and_proof_apart():
    met = RequirementResult(requirement=Requirement(id="r1", text="Spring Boot"), verdict=Verdict.MET, confidence=1)
    cv = RequirementResult(requirement=Requirement(id="r2", text="Kafka"), verdict=Verdict.SELF_REPORTED, confidence=1)
    degree = RequirementResult(requirement=Requirement(id="r3", text="B.E. degree", checkable=False),
                               verdict=Verdict.SELF_REPORTED, confidence=1)
    results = [met, cv, degree]
    assert fit_score(results) == 50          # Kafka unproven; degree not code-checkable at all
    assert fit_score_with_cv(results) == 100  # CV states both


JD = """We are hiring.
- Build REST APIs with Spring Boot
- Apache Kafka event pipelines
- B.E. in Computer Science or a related degree
"""


async def test_end_to_end_cv_rows_only_fill_gaps(tmp_path, sample_repo):
    index = tmp_path / "index.json"
    store = InMemoryStore(str(index))
    await ingest({"demo-repo": sample_repo}, HashEmbedder(), store, profile=write_profile(tmp_path))
    store.save()
    settings = Settings(_env_file=None, provider_mode="fake", vector_store="memory", index_path=str(index))
    async with httpx.AsyncClient() as http:
        report = await run_fit_check(build_graph(build_deps(settings, http)), JD)
    verdicts = {r.requirement.text: r.verdict for r in report.results}
    assert verdicts["Apache Kafka event pipelines"] == Verdict.SELF_REPORTED
    assert verdicts["Build REST APIs with Spring Boot"] in (Verdict.MET, Verdict.PARTIAL)
    assert report.score_with_cv >= report.score
    for r in report.results:
        if r.verdict in (Verdict.MET, Verdict.PARTIAL):
            assert not r.citation.source.startswith("CV")
