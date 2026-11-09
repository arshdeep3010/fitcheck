# fitcheck: no-index (sample requirements below are test data, not evidence)
from fastapi.testclient import TestClient

from app.main import create_app
from app.ratelimit import RateLimiter

JD = """Backend engineer
- Build REST APIs with Spring Boot
- RAG retrieval over a vector database
"""


def test_fit_endpoint_returns_report(fake_settings):
    with TestClient(create_app(fake_settings)) as client:
        res = client.post("/api/fit", json={"jd": JD})
        assert res.status_code == 200
        body = res.json()
        assert body["candidate"] == "Arshdeep Singh"
        assert len(body["results"]) == 2


def test_health_reports_index_size(fake_settings):
    with TestClient(create_app(fake_settings)) as client:
        body = client.get("/health").json()
        assert body["indexed_chunks"] > 0
        assert body["models"]["scorer"] == "keyword-baseline"


def test_index_page_is_served(fake_settings):
    with TestClient(create_app(fake_settings)) as client:
        assert "FitCheck" in client.get("/").text


def test_validation_and_size_limits(fake_settings):
    fake_settings.max_jd_chars = 200
    with TestClient(create_app(fake_settings)) as client:
        assert client.post("/api/fit", json={"jd": "too short"}).status_code == 422
        assert client.post("/api/fit", json={"jd": "x" * 201}).status_code == 413


def test_rate_limit_returns_429_with_retry_after(fake_settings):
    with TestClient(create_app(fake_settings)) as client:
        codes = [client.post("/api/fit", json={"jd": JD}).status_code for _ in range(4)]
        assert codes == [200, 200, 200, 429]
        res = client.post("/api/fit", json={"jd": JD})
        assert int(res.headers["retry-after"]) > 0


def test_rate_limiter_window_expires():
    limiter = RateLimiter(max_requests=1, window_s=10)
    assert limiter.allow("ip", now=0)[0]
    assert not limiter.allow("ip", now=5)[0]
    assert limiter.allow("ip", now=10.1)[0]


def test_stream_sends_each_step_then_the_report(fake_settings):
    import json
    with TestClient(create_app(fake_settings)) as client:
        with client.stream("POST", "/api/fit/stream", json={"jd": JD}) as res:
            assert res.status_code == 200
            assert res.headers["content-type"].startswith("text/event-stream")
            body = "".join(res.iter_text())
    events = [block.split("\n") for block in body.strip().split("\n\n")]
    names = [lines[0].removeprefix("event: ") for lines in events]
    payloads = [json.loads(lines[1].removeprefix("data: ")) for lines in events]
    assert names == ["step"] * 5 + ["report"]
    assert [p["node"] for p in payloads[:5]] == ["parse_jd", "retrieve", "score", "ground", "check_cv"]
    assert payloads[0]["detail"].startswith("2 requirements")
    assert payloads[-1]["candidate"] == "Arshdeep Singh"


def test_stream_is_rate_limited_before_it_opens(fake_settings):
    with TestClient(create_app(fake_settings)) as client:
        codes = [client.post("/api/fit/stream", json={"jd": JD}).status_code for _ in range(4)]
        assert codes[-1] == 429
