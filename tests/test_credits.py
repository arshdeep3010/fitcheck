# fitcheck: no-index (sample requirements below are test data, not evidence)
"""OpenRouter out of credits: no retries, the free fallback scores, and the report says so."""

import httpx
import pytest

from app.deps import build_chat, build_deps, build_fallback
from app.graph import FALLBACK_NOTICE, NO_CREDITS_NOTICE, build_graph, run_fit_check
from app.llm.credits import CreditPause
from app.llm.http import CreditsExhaustedError, ProviderError, post_json
from app.scoring.keyword import KeywordFitScorer
from app.scoring.llm_json import LlmJsonFitScorer

JD = """We are hiring a backend engineer.
- Build REST APIs with Spring Boot
- RAG retrieval over a pgvector vector database
"""
URL = "https://openrouter.ai/api/v1/systemone"


def replying(status: int, body: dict, calls: list):
    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(status, json=body)
    return httpx.MockTransport(handler)


async def test_402_is_out_of_credits_and_not_retried():
    calls: list = []
    body = {"error": {"code": 402, "message": "Insufficient credits. Add more using https://openrouter.ai/credits"}}
    async with httpx.AsyncClient(transport=replying(402, body, calls)) as http:
        with pytest.raises(CreditsExhaustedError, match="Insufficient credits"):
            await post_json(http, URL, "key", {}, backoff_s=0)
    assert len(calls) == 1


async def test_free_daily_limit_is_out_of_credits_and_not_retried():
    calls: list = []
    body = {"error": {"code": 429, "message": "Rate limit exceeded: free-models-per-day. "
                                              "Add 10 credits to unlock 1000 free model requests per day"}}
    async with httpx.AsyncClient(transport=replying(429, body, calls)) as http:
        with pytest.raises(CreditsExhaustedError):
            await post_json(http, URL, "key", {}, backoff_s=0)
    assert len(calls) == 1


async def test_plain_rate_limit_is_still_retried():
    calls: list = []
    body = {"error": {"code": 429, "message": "Rate limit exceeded: limit_rpm/inception/mercury-decide"}}
    async with httpx.AsyncClient(transport=replying(429, body, calls)) as http:
        with pytest.raises(ProviderError) as info:
            await post_json(http, URL, "key", {}, attempts=3, backoff_s=0)
    assert not isinstance(info.value, CreditsExhaustedError)
    assert len(calls) == 3


def test_pause_expires():
    now = [100.0]
    pause = CreditPause(pause_s=60, clock=lambda: now[0])
    assert not pause.paused
    pause.trip("402")
    assert pause.paused and pause.seconds_left() == 60
    now[0] += 61
    assert not pause.paused


class BrokeScorer:
    """Stands in for Mercury Decide on an account with no credits left."""
    name = "inception/mercury-decide"

    def __init__(self):
        self.calls = 0

    async def score(self, requirements, evidence):
        self.calls += 1
        raise CreditsExhaustedError(f"{URL} returned 402: Insufficient credits")


async def run(settings, **deps_overrides):
    http = httpx.AsyncClient()
    deps = build_deps(settings, http)
    for key, value in deps_overrides.items():
        setattr(deps, key, value)
    graph = build_graph(deps)
    report = await run_fit_check(graph, JD)
    second = await run_fit_check(graph, JD)
    await http.aclose()
    return report, second


async def test_out_of_credits_falls_back_and_says_so(fake_settings):
    broke, backup = BrokeScorer(), KeywordFitScorer()
    report, second = await run(fake_settings, scorer=broke, fallback_scorer=backup)

    notice = FALLBACK_NOTICE.format(main=broke.name, backup=backup.name.removeprefix("llm-json:"))
    assert report.warnings.count(notice) == 1  # score and check_cv don't repeat it
    assert report.models["scorer"] == backup.name  # the report names who actually judged
    assert any(r.verdict.value in ("met", "partial") for r in report.results)  # backup really judged
    # Paused: the second check doesn't ask OpenRouter again, and still says why.
    assert broke.calls == 1
    assert notice in second.warnings


async def test_out_of_credits_without_fallback_claims_nothing(fake_settings):
    report, _ = await run(fake_settings, scorer=BrokeScorer(), fallback_scorer=None)
    assert all(r.verdict.value == "unverified" for r in report.results)
    assert NO_CREDITS_NOTICE in report.warnings
    assert report.score == 0


def test_fallback_only_for_the_paid_scorer(fake_settings):
    http = httpx.AsyncClient()
    chat = object()  # never called while building
    live = fake_settings.model_copy(update={"provider_mode": "live"})
    assert build_fallback(live.model_copy(update={"scorer": "llm_json"}), http, chat) is None
    assert build_fallback(live.model_copy(update={"scorer": "system_one", "scorer_fallback": ""}),
                          http, chat) is None
    real_chat = build_chat(live, http)
    backup = build_fallback(live.model_copy(update={"scorer": "system_one"}), http, real_chat)
    assert isinstance(backup, LlmJsonFitScorer)  # the default fallback is the free Nemotron judge
    with pytest.raises(ValueError):
        build_fallback(live.model_copy(update={"scorer": "system_one", "scorer_fallback": "system_one"}),
                       http, chat)
