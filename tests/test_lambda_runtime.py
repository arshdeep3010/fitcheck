# fitcheck: no-index (test data, not evidence)
"""Running on AWS Lambda: scheduled sync events, the visitor's IP, secrets from SSM."""

import asyncio
import json
import os

from fastapi.testclient import TestClient

from app.config import load_ssm
from app.main import create_app
from app.rag.sync import GitHubSyncer
from app.ratelimit import lambda_source_ip

JD = """Backend engineer
- Build REST APIs with Spring Boot
- RAG retrieval over a vector database
"""


def context(ip: str) -> dict:
    return {"x-amzn-request-context": json.dumps({"http": {"sourceIp": ip, "method": "POST"}})}


class FakeSyncer:
    enabled = True
    running = False

    def __init__(self):
        self.runs = 0

    async def run_once(self):
        self.runs += 1
        return {"agentops": "unchanged"}


def lambda_client(fake_settings, token="s3cret"):
    fake_settings.runtime = "lambda"
    fake_settings.sync_token = token
    client = TestClient(create_app(fake_settings))
    return client


# ---------- the scheduled sync (EventBridge -> Lambda Web Adapter -> POST /events) ----------

def test_sync_event_runs_the_sync_and_waits_for_it(fake_settings):
    with lambda_client(fake_settings) as client:
        client.app.state.syncer = syncer = FakeSyncer()
        r = client.post("/events", json={"action": "sync", "token": "s3cret"})
    assert r.status_code == 200 and r.json() == {"synced": {"agentops": "unchanged"}}
    assert syncer.runs == 1


def test_sync_event_needs_the_token(fake_settings):
    # /events is reachable through the public URL too.
    with lambda_client(fake_settings) as client:
        client.app.state.syncer = syncer = FakeSyncer()
        assert client.post("/events", json={"action": "sync", "token": "guess"}).status_code == 401
        assert client.post("/events", json={"action": "sync"}).status_code == 401
        assert client.post("/events", json={"action": "delete"}).status_code == 400
        assert client.post("/events", content=b"not json").status_code == 400
    assert syncer.runs == 0


def test_sync_event_is_off_without_a_token_or_off_lambda(fake_settings):
    with lambda_client(fake_settings, token="") as client:
        client.app.state.syncer = FakeSyncer()
        assert client.post("/events", json={"action": "sync", "token": ""}).status_code == 404
    fake_settings.runtime = "server"
    with TestClient(create_app(fake_settings)) as client:
        assert client.post("/events", json={"action": "sync", "token": "s3cret"}).status_code == 404


def test_manual_sync_on_lambda_finishes_before_answering(fake_settings):
    with lambda_client(fake_settings) as client:
        client.app.state.syncer = syncer = FakeSyncer()
        r = client.post("/api/sync", headers={"Authorization": "Bearer s3cret"})
    assert r.status_code == 202 and r.json()["result"] == {"agentops": "unchanged"}
    assert syncer.runs == 1


# ---------- the visitor's IP ----------

def test_lambda_source_ip_reads_the_adapters_request_context():
    assert lambda_source_ip(json.dumps({"http": {"sourceIp": "203.0.113.9"}})) == "203.0.113.9"
    assert lambda_source_ip("") == ""
    assert lambda_source_ip("not json") == ""
    assert lambda_source_ip(json.dumps(["odd"])) == ""


def test_rate_limit_on_lambda_is_per_source_ip(fake_settings):
    # fake_settings allows 3 checks per hour per visitor. The peer address is the same for
    # everyone on Lambda (the adapter), so the source IP must be what's counted.
    with lambda_client(fake_settings) as client:
        first = [client.post("/api/fit", json={"jd": JD}, headers=context("203.0.113.9")).status_code
                 for _ in range(4)]
        other = client.post("/api/fit", json={"jd": JD}, headers=context("198.51.100.4")).status_code
    assert first == [200, 200, 200, 429]
    assert other == 200


# ---------- secrets from SSM ----------

class FakeSsm:
    def __init__(self, params):
        self._params = params

    def get_paginator(self, name):
        assert name == "get_parameters_by_path"
        params = self._params

        class Pages:
            def paginate(self, Path, WithDecryption):  # noqa: N803 - boto3's argument names
                assert WithDecryption
                return [{"Parameters": [p for p in params if p["Name"].startswith(Path + "/")]}]
        return Pages()


def test_load_ssm_fills_the_environment(monkeypatch):
    for name in ("NVIDIA_API_KEY", "DATABASE_URL", "OPENROUTER_API_KEY", "GITHUB_TOKEN"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("GITHUB_TOKEN", "from-env")
    load_ssm("/fitcheck", FakeSsm([
        {"Name": "/fitcheck/nvidia_api_key", "Value": "nv-test"},
        {"Name": "/fitcheck/database_url", "Value": "postgresql://test"},
        {"Name": "/fitcheck/openrouter_api_key", "Value": "set-me"},  # placeholder: skipped
        {"Name": "/fitcheck/github_token", "Value": "from-ssm"},  # the environment wins
        {"Name": "/other/nvidia_api_key", "Value": "wrong-prefix"},
    ]))
    assert os.environ["NVIDIA_API_KEY"] == "nv-test"
    assert os.environ["DATABASE_URL"] == "postgresql://test"
    assert "OPENROUTER_API_KEY" not in os.environ
    assert os.environ["GITHUB_TOKEN"] == "from-env"


# ---------- every instance shows the same "last checked" ----------

class FakeShared:
    def __init__(self):
        self.meta = {}

    def set_meta(self, key, value):
        self.meta[key] = json.loads(json.dumps(value))

    def get_meta(self, key):
        return self.meta.get(key)


def test_sync_status_is_shared_between_instances(fake_settings, monkeypatch):
    shared = FakeShared()
    store = type("S", (), {"repo_summary": lambda self: []})()
    one = GitHubSyncer(None, fake_settings, None, store, shared)
    two = GitHubSyncer(None, fake_settings, None, store, shared)

    async def fake_sync():
        return {"agentops": 187}
    monkeypatch.setattr(one, "_sync", fake_sync)
    asyncio.run(one.run_once())

    fake_settings.github_user, fake_settings.provider_mode = "someone", "live"  # sync enabled
    status = two.status()  # an instance that never synced itself
    assert status["last_checked"] is not None
    assert status["last_changes"] == {"agentops": 187}
    assert status["next_check"] is not None
    assert one.status()["next_check"] == status["next_check"]  # the one that synced, too
