# fitcheck: no-index (sample requirements below are test data, not evidence)
import httpx

from app.deps import build_deps
from app.graph import build_graph, run_fit_check
from app.models import Verdict

JD = """We are hiring a backend engineer.
- Build REST APIs with Spring Boot
- RAG retrieval over a pgvector vector database
- Experience with Kubernetes operators
"""


async def make_graph(settings, **overrides):
    http = httpx.AsyncClient()
    deps = build_deps(settings, http)
    for key, value in overrides.items():
        setattr(deps, key, value)
    return build_graph(deps), http


async def test_end_to_end_offline(fake_settings):
    graph, http = await make_graph(fake_settings)
    report = await run_fit_check(graph, JD)
    await http.aclose()

    verdicts = {r.requirement.text: r.verdict for r in report.results}
    assert verdicts["Build REST APIs with Spring Boot"] in (Verdict.MET, Verdict.PARTIAL)
    assert verdicts["Experience with Kubernetes operators"] in (Verdict.NOT_MET, Verdict.UNVERIFIED)
    for r in report.results:
        if r.verdict in (Verdict.MET, Verdict.PARTIAL):
            assert r.citation is not None and r.citation.url.startswith("https://github.com/someone/demo-repo/")
    assert 0 <= report.score <= 100
    assert set(report.timings_ms) == {"parse_jd", "retrieve", "score", "ground", "check_cv"}


async def test_no_requirements_skips_to_report(fake_settings):
    graph, http = await make_graph(fake_settings)
    report = await run_fit_check(graph, "Great culture. Free snacks. Hybrid office in a nice city with views.")
    await http.aclose()
    assert report.results == [] and report.score == 0


class BrokenScorer:
    name = "broken"

    async def score(self, requirements, evidence):
        raise RuntimeError("provider down")


class BrokenEmbedder:
    name = "broken"

    async def embed(self, texts, kind):
        raise RuntimeError("embeddings down")


async def test_scorer_outage_claims_nothing(fake_settings):
    graph, http = await make_graph(fake_settings, scorer=BrokenScorer())
    report = await run_fit_check(graph, JD)
    await http.aclose()
    assert all(r.verdict == Verdict.UNVERIFIED for r in report.results)
    assert any("scoring was unavailable" in w for w in report.warnings)
    assert report.score == 0


async def test_retrieval_outage_claims_nothing(fake_settings):
    graph, http = await make_graph(fake_settings, embedder=BrokenEmbedder())
    report = await run_fit_check(graph, JD)
    await http.aclose()
    assert all(r.verdict == Verdict.UNVERIFIED for r in report.results)
    assert any("Evidence search was unavailable" in w for w in report.warnings)
