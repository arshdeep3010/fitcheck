# fitcheck: no-index (test data, not evidence)
"""The in-app GitHub sync and its status endpoint, against a fake GitHub: no network."""

import httpx
from fastapi.testclient import TestClient

from app.config import Settings
from app.llm.embeddings import HashEmbedder
from app.main import create_app
from app.models import Chunk
from app.rag.store import InMemoryStore
from app.rag.sync import GitHubSyncer
from tests.test_github_sync import FakeGitHub


def live_settings(**overrides) -> Settings:
    values = {"provider_mode": "live", "github_user": "me", "github_sync_interval_min": 30,
              "profile_path": ""}
    return Settings(_env_file=None, **(values | overrides))


def make_syncer(fake: FakeGitHub, store: InMemoryStore, http: httpx.AsyncClient, **overrides) -> GitHubSyncer:
    return GitHubSyncer(http, live_settings(**overrides), HashEmbedder(), store)


async def test_sync_indexes_repos_and_reports_their_commits():
    fake, store = FakeGitHub({"new-project": "abc1234567"}), InMemoryStore()
    async with httpx.AsyncClient(transport=httpx.MockTransport(fake.handler)) as http:
        syncer = make_syncer(fake, store, http)
        assert syncer.enabled
        await syncer.run_once()
    status = syncer.status()
    assert status["last_error"] is None and status["last_checked"]
    assert status["last_changes"] == {"new-project": 1}
    assert status["repos"] == [{"repo": "new-project", "source": "github", "chunks": 1, "commit": "abc1234567",
                                "url": "https://github.com/me/new-project/tree/abc1234567"}]


async def test_a_push_shows_up_on_the_next_check_and_quiet_checks_download_nothing():
    fake, store = FakeGitHub({"proj": "sha1"}), InMemoryStore()
    async with httpx.AsyncClient(transport=httpx.MockTransport(fake.handler)) as http:
        syncer = make_syncer(fake, store, http)
        await syncer.run_once()
        await syncer.run_once()
        assert fake.downloads == ["proj"] and syncer.last_changes == {}
        fake.repos["proj"] = "sha2"
        fake.repos["brand-new"] = "sha3"
        await syncer.run_once()
    assert store.indexed_refs() == {"proj": "sha2", "brand-new": "sha3"}
    assert syncer.last_changes == {"proj": 1, "brand-new": 1}


async def test_failed_check_keeps_the_old_index_and_reports_why():
    fake, store = FakeGitHub({"proj": "sha1"}), InMemoryStore()
    async with httpx.AsyncClient(transport=httpx.MockTransport(fake.handler)) as http:
        syncer = make_syncer(fake, store, http)
        await syncer.run_once()

    def down(request):
        return httpx.Response(503, text="GitHub is down")
    async with httpx.AsyncClient(transport=httpx.MockTransport(down)) as http:
        syncer._http = http
        await syncer.run_once()
    assert "503" in syncer.status()["last_error"]
    assert store.indexed_refs() == {"proj": "sha1"} and store.count() == 1


async def test_sync_saves_the_json_index_only_when_something_changed(tmp_path):
    path = tmp_path / "index.json"
    fake, store = FakeGitHub({"proj": "sha1"}), InMemoryStore(str(path))
    async with httpx.AsyncClient(transport=httpx.MockTransport(fake.handler)) as http:
        syncer = make_syncer(fake, store, http)
        await syncer.run_once()
        saved = path.stat().st_mtime_ns
        await syncer.run_once()
    assert path.stat().st_mtime_ns == saved
    assert InMemoryStore(str(path)).indexed_refs() == {"proj": "sha1"}


def test_sync_is_off_without_a_user_or_in_fake_mode():
    http = httpx.AsyncClient()
    store = InMemoryStore()
    assert not GitHubSyncer(http, live_settings(github_user=""), HashEmbedder(), store).enabled
    assert not GitHubSyncer(http, live_settings(provider_mode="fake"), HashEmbedder(), store).enabled
    assert not GitHubSyncer(http, live_settings(github_sync_interval_min=0), HashEmbedder(), store).enabled


def chunk(repo: str, n: int, **extra) -> Chunk:
    return Chunk(id=f"{repo}:{n}", repo=repo, path="a.py", start_line=1, end_line=3, text="x", **extra)


def test_replace_repo_swaps_one_repo_and_leaves_the_rest():
    store = InMemoryStore()
    store.upsert([chunk("a", 1), chunk("a", 2), chunk("b", 1)], [[1.0]] * 3)
    snapshot = store._rows
    store.replace_repo("a", [chunk("a", 9)], [[1.0]])
    assert sorted(store._rows) == ["a:9", "b:1"]
    assert sorted(snapshot) == ["a:1", "a:2", "b:1"]  # a search mid-swap keeps its own snapshot


def test_repo_summary_labels_github_local_and_cv_sources():
    store = InMemoryStore()
    store.upsert([chunk("gh", 1, ref="abc", url_base="https://github.com/me/gh"), chunk("local", 1),
                  chunk("profile", 1, source_type="self_reported", ref="d1")], [[1.0]] * 3)
    by_repo = {r["repo"]: r for r in store.repo_summary()}
    assert by_repo["gh"]["source"] == "github" and by_repo["gh"]["url"] == "https://github.com/me/gh/tree/abc"
    assert by_repo["local"]["source"] == "local" and by_repo["local"]["url"] is None
    assert by_repo["profile"]["source"] == "cv"


def test_status_endpoint_and_how_it_works_page(fake_settings):
    with TestClient(create_app(fake_settings)) as client:
        status = client.get("/api/sync/status").json()
        assert status["enabled"] is False
        assert [r["repo"] for r in status["repos"]] == ["demo-repo"]
        assert status["repos"][0]["source"] == "local"
        page = client.get("/how-it-works")
        assert page.status_code == 200 and "Synced with GitHub" in page.text
        assert client.get("/health").json()["settings"]["min_confidence"] == 0.5


class StubSyncer:
    enabled, running = True, False

    def __init__(self):
        self.runs = 0

    async def run_once(self):
        self.runs += 1


def test_manual_sync_needs_the_token(fake_settings):
    with TestClient(create_app(fake_settings)) as client:
        assert client.post("/api/sync").status_code == 404  # no SYNC_TOKEN configured
    fake_settings.sync_token = "s3cret-token"
    with TestClient(create_app(fake_settings)) as client:
        stub = StubSyncer()
        client.app.state.syncer = stub
        assert client.post("/api/sync").status_code == 401
        assert client.post("/api/sync", headers={"Authorization": "Bearer wrong"}).status_code == 401
        res = client.post("/api/sync", headers={"Authorization": "Bearer s3cret-token"})
        assert res.status_code == 202
    assert stub.runs == 1
