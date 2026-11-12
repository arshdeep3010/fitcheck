# fitcheck: no-index (test data, not evidence)
"""Parallel Decisions: free, but it searches the web, which can't be switched off. These tests
pin down what FitCheck guarantees anyway: web sources are never cited or passed on, the
candidate isn't named in what's sent, and grounding still decides."""

import httpx
import pytest

from app.config import Settings
from app.deps import build_fallback, build_scorer
from app.grounding import ground
from app.llm.decisions import DecisionsClient
from app.models import Chunk, Evidence, Requirement
from app.scoring.parallel import OFFLINE_RULE, ParallelFitScorer, identity_terms, make_redactor

URL = "https://openrouter.ai/api/alpha/decisions"
REQS = [Requirement(id="r1", text="Build REST APIs with FastAPI")]
CHUNK = Chunk(id="demo:app/main.py:1-12", repo="demo", path="app/main.py", start_line=1, end_line=12,
              text="# Written by Arshdeep Singh (github.com/arshdeep3010)\nfrom fastapi import FastAPI\n")
EVIDENCE = {"r1": [Evidence(chunk=CHUNK, similarity=0.8)]}

# What the live API returned on 12 Nov 2026 (trimmed): the answers plus web sources.
LIVE_LIKE = {
    "id": "dec-1", "model": "parallel/decisions-fast-20261106", "provider": "Parallel",
    "answers": {
        "r1_status": {"type": "choice", "choice": "met",
                      "probabilities": {"met": 0.74, "partial": 0.21, "not_met": 0.05}, "confidence": 0.61},
        "r1_proof": {"type": "choice", "choice": "e0",
                     "probabilities": {"e0": 0.985, "none": 0.015}, "confidence": 0.97},
    },
    "annotations": [{"type": "url_citation",
                     "url_citation": {"url": "https://example.com/fastapi-tutorial", "title": "A tutorial",
                                      "content": "FastAPI is a web framework"}}] * 3,
    "usage": {"input_tokens": 2971, "output_tokens": 0, "cost": 0},
}


def verdict(scores) -> str:
    return ground(REQS, EVIDENCE, scores, min_confidence=0.5)[0].verdict.value


def client_returning(payload: dict, seen: list) -> DecisionsClient:
    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=payload)
    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return DecisionsClient(http, URL, "test-key", "parallel/decisions-fast:free")


async def test_web_sources_are_dropped_and_only_retrieved_chunks_are_cited():
    seen: list = []
    client = client_returning(LIVE_LIKE, seen)
    scores = await ParallelFitScorer(client).score(REQS, EVIDENCE)
    assert client.web_sources_discarded == 3
    assert scores[0].status == "met" and scores[0].evidence_id == CHUNK.id
    assert "example.com" not in scores[0].model_dump_json()
    # grounding accepts it only because the cited chunk is one this requirement retrieved
    assert verdict(scores) == "met"


async def test_the_candidate_is_not_named_in_what_is_sent():
    seen: list = []
    client = client_returning(LIVE_LIKE, seen)
    scorer = ParallelFitScorer(client, redact_terms=identity_terms("Arshdeep Singh", "arshdeep3010"))
    await scorer.score(REQS, EVIDENCE)
    body = seen[0].content.decode().lower()
    assert seen[0].url == URL
    assert "arshdeep" not in body and "singh" not in body
    assert "the candidate" in body
    assert OFFLINE_RULE.split(".")[0].lower() in body


def test_redactor_matches_whole_words_only():
    redact = make_redactor(["Arshdeep Singh", "arshdeep3010", "Arshdeep", "Singh"])
    assert redact("By ARSHDEEP SINGH, see github.com/arshdeep3010") == "By the candidate, see github.com/the candidate"
    assert redact("Singhania Ltd") == "Singhania Ltd"
    assert make_redactor([""])("unchanged") == "unchanged"


async def test_a_refusal_or_odd_answer_is_unverified_not_met():
    payload = {"answers": {"r1_status": {"type": "refusal"}, "r1_proof": {"type": "refusal"}}}
    scores = await ParallelFitScorer(client_returning(payload, [])).score(REQS, EVIDENCE)
    assert scores[0].confidence == 0.0
    assert verdict(scores) == "unverified"


async def test_a_web_url_as_proof_cannot_be_cited():
    payload = {**LIVE_LIKE, "answers": {**LIVE_LIKE["answers"],
                                         "r1_proof": {"type": "choice", "choice": "https://example.com"}}}
    scores = await ParallelFitScorer(client_returning(payload, [])).score(REQS, EVIDENCE)
    assert scores[0].evidence_id is None
    assert verdict(scores) != "met"


def test_config_builds_parallel_with_a_free_fallback():
    settings = Settings(_env_file=None, scorer="parallel", provider_mode="live", openrouter_api_key="k",
                        nvidia_api_key="n", candidate_name="Arshdeep Singh", github_user="arshdeep3010")
    http = httpx.AsyncClient()
    scorer = build_scorer(settings, http, chat=None)
    assert isinstance(scorer, ParallelFitScorer) and scorer.name == "parallel/decisions-fast:free"
    chat = type("Chat", (), {"name": "nvidia/nemotron"})()
    assert build_fallback(settings, http, chat).name == "llm-json:nvidia/nemotron"  # when the free quota runs out
    with pytest.raises(ValueError):
        build_fallback(settings.model_copy(update={"scorer_fallback": "system_one"}), http, chat)
