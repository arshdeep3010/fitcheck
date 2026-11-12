# fitcheck: no-index (test data, not evidence)
"""Abuse guards, and the code-vs-CV distinction carried on every citation."""

from datetime import UTC, datetime

from fastapi.testclient import TestClient

from app.grounding import github_link_builder, ground, ground_self_reported
from app.main import create_app
from app.models import Chunk, Evidence, Requirement, RequirementScore, Verdict
from app.ratelimit import DailyCap, RateLimiter, client_ip

JD = """Backend engineer
- Build REST APIs with Spring Boot
- RAG retrieval over a vector database
"""


# ---------- client IP behind a proxy ----------

def test_forged_x_forwarded_for_is_ignored():
    # The visitor sends a fake left-hand entry; our one proxy appends the real address.
    assert client_ip("6.6.6.6, 203.0.113.9", "10.0.0.1", trust_proxy=True, proxy_hops=1) == "203.0.113.9"


def test_two_proxies_take_the_entry_before_them():
    assert client_ip("6.6.6.6, 203.0.113.9, 10.1.1.1", "10.0.0.1", True, proxy_hops=2) == "203.0.113.9"


def test_header_is_ignored_unless_trusted():
    assert client_ip("6.6.6.6", "198.51.100.4", trust_proxy=False, proxy_hops=1) == "198.51.100.4"
    assert client_ip("", "198.51.100.4", trust_proxy=True, proxy_hops=1) == "198.51.100.4"


def test_rotating_forged_ips_share_one_rate_limit(fake_settings):
    fake_settings.trust_proxy_headers = True
    with TestClient(create_app(fake_settings)) as client:
        codes = [client.post("/api/fit", json={"jd": JD},
                             headers={"X-Forwarded-For": f"1.2.3.{n}, 203.0.113.9"}).status_code
                 for n in range(4)]
    assert codes == [200, 200, 200, 429]


def test_rate_limiter_forgets_idle_visitors():
    limiter = RateLimiter(max_requests=5, window_s=10)
    for n in range(50):
        limiter.allow(f"ip{n}", now=0)
    limiter.allow("late", now=25)
    assert limiter.tracked() == 1


def test_daily_cap_is_shared_and_resets_at_midnight_utc():
    cap = DailyCap(2)
    day = datetime(2026, 11, 7, 23, 0, tzinfo=UTC)
    assert cap.allow(day)[0] and cap.allow(day)[0]
    allowed, retry = cap.allow(day)
    assert not allowed and retry == 3600
    assert cap.allow(datetime(2026, 11, 8, 0, 1, tzinfo=UTC))[0]
    assert DailyCap(0).allow(day) == (True, 0)


def test_daily_cap_returns_429(fake_settings):
    fake_settings.rate_limit_per_hour = 100
    fake_settings.daily_limit = 2
    with TestClient(create_app(fake_settings)) as client:
        codes = [client.post("/api/fit", json={"jd": JD}).status_code for _ in range(3)]
        assert codes == [200, 200, 429]
        assert "today" in client.post("/api/fit", json={"jd": JD}).json()["detail"]


# ---------- code is proof, the CV is a claim ----------

def _chunk(**extra) -> Chunk:
    values = {"id": "c1", "repo": "proj", "path": "app/main.py", "start_line": 1, "end_line": 9, "text": "x = 1"}
    return Chunk(**(values | extra))


def _ground(chunk: Chunk):
    req = Requirement(id="r1", text="FastAPI")
    score = RequirementScore(requirement_id="r1", status="met", confidence=0.9, evidence_id=chunk.id)
    return ground([req], {"r1": [Evidence(chunk=chunk, similarity=0.8)]}, [score], 0.5,
                  github_link_builder("", "main"))[0]


def test_github_code_citation_is_marked_github_with_its_commit():
    result = _ground(_chunk(ref="abcdef123456", url_base="https://github.com/me/proj"))
    assert result.verdict == Verdict.MET
    assert result.citation.kind == "github" and result.citation.commit == "abcdef1"
    assert result.citation.url.startswith("https://github.com/me/proj/blob/abcdef123456/")


def test_local_code_citation_is_marked_code_without_a_commit():
    result = _ground(_chunk())
    assert result.citation.kind == "code" and result.citation.commit is None


def test_cv_citation_is_marked_cv_and_never_met():
    cv = _chunk(id="cv1", repo="profile", path="profile.md", source_type="self_reported", title="Skills")
    req = Requirement(id="r1", text="Kubernetes")
    score = RequirementScore(requirement_id="r1", status="met", confidence=0.95, evidence_id="cv1")
    claim = ground_self_reported([req], {"r1": [Evidence(chunk=cv, similarity=0.7)]}, [score], 0.5)["r1"]
    assert claim.verdict == Verdict.SELF_REPORTED
    assert claim.citation.kind == "cv" and claim.citation.source == "CV: Skills"


def test_api_report_carries_the_citation_kind(fake_settings):
    with TestClient(create_app(fake_settings)) as client:
        results = client.post("/api/fit", json={"jd": JD}).json()["results"]
    kinds = {r["citation"]["kind"] for r in results if r["citation"]}
    assert kinds <= {"github", "code", "cv"} and kinds


def test_page_shows_the_candidate_without_waiting_for_health(fake_settings):
    from fastapi.testclient import TestClient

    from app.main import create_app

    fake_settings.candidate_name = "Ada <Lovelace>"
    fake_settings.github_base_url = "https://github.com/ada"
    with TestClient(create_app(fake_settings)) as client:
        page = client.get("/")
    assert page.status_code == 200
    assert "{{" not in page.text                      # every placeholder filled
    assert "Ada &lt;Lovelace&gt;" in page.text         # escaped, never raw HTML
    assert ">AL<" in page.text                          # initials
    assert 'href="https://github.com/ada"' in page.text
    assert "s-maxage=300" in page.headers["cache-control"]
