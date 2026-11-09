# fitcheck: no-index (test data, not evidence)
"""GitHub sync against a fake GitHub (httpx.MockTransport): no network."""

import io
import json
import stat
import zipfile

import httpx
import pytest

from app.grounding import github_link_builder
from app.llm.embeddings import HashEmbedder
from app.rag.github import safe_extract
from app.rag.ingest import sync_github
from app.rag.store import InMemoryStore

CODE = "from fastapi import FastAPI\n\napp = FastAPI()\n\n@app.get('/health')\ndef health():\n    return {}\n"


def make_zip(files: dict[str, str], symlink: str | None = None) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for name, content in files.items():
            z.writestr(name, content)
        if symlink:
            info = zipfile.ZipInfo(symlink)
            info.external_attr = (stat.S_IFLNK | 0o777) << 16
            z.writestr(info, "/etc/passwd")
    return buf.getvalue()


class FakeGitHub:
    """Serves a user's repo list, head SHAs and zipballs; counts archive downloads."""

    def __init__(self, repos: dict[str, str], private: tuple[str, ...] = (), forks: tuple[str, ...] = ()):
        self.repos = repos  # name -> sha
        self.private, self.forks = private, forks
        self.downloads: list[str] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/users/me/repos":
            listing = [{"name": n, "default_branch": "main", "html_url": f"https://github.com/me/{n}",
                        "private": n in self.private, "fork": n in self.forks, "archived": False, "size": 10}
                       for n in self.repos]
            return httpx.Response(200, json=listing)
        parts = path.strip("/").split("/")  # repos/me/<name>/commits/main | repos/me/<name>/zipball/<sha>
        name = parts[2]
        if parts[3] == "commits":
            return httpx.Response(200, text=self.repos[name])
        if parts[3] == "zipball":
            self.downloads.append(name)
            return httpx.Response(200, content=make_zip({f"me-{name}-{parts[4]}/app/main.py": CODE}))
        return httpx.Response(404)


async def run_sync(fake: FakeGitHub, store: InMemoryStore):
    async with httpx.AsyncClient(transport=httpx.MockTransport(fake.handler)) as http:
        return await sync_github(http, "me", HashEmbedder(), store)


async def test_new_repo_is_indexed_with_permalink_citations():
    fake, store = FakeGitHub({"new-project": "abc123"}), InMemoryStore()
    stats = await run_sync(fake, store)
    assert stats["new-project"] == 1
    chunk = store.search((await HashEmbedder().embed(["FastAPI health"], "query"))[0], 1)[0].chunk
    assert chunk.path == "app/main.py" and chunk.ref == "abc123"
    link = github_link_builder("", "main")(chunk)
    assert link == "https://github.com/me/new-project/blob/abc123/app/main.py#L1-L7"


async def test_unchanged_repo_is_not_downloaded_again():
    fake, store = FakeGitHub({"proj": "sha1"}), InMemoryStore()
    await run_sync(fake, store)
    stats = await run_sync(fake, store)
    assert stats["proj"] == "unchanged" and fake.downloads == ["proj"]


async def test_new_commit_replaces_old_chunks():
    fake, store = FakeGitHub({"proj": "sha1"}), InMemoryStore()
    await run_sync(fake, store)
    fake.repos["proj"] = "sha2"
    await run_sync(fake, store)
    assert store.indexed_refs() == {"proj": "sha2"} and store.count() == 1


async def test_private_forked_and_deleted_repos_are_not_kept():
    fake = FakeGitHub({"keep": "s1", "secret": "s2", "someone-elses": "s3"},
                      private=("secret",), forks=("someone-elses",))
    store = InMemoryStore()
    await run_sync(fake, store)
    assert set(store.indexed_refs()) == {"keep"}
    del fake.repos["keep"]
    stats = await run_sync(fake, store)
    assert stats["keep"].startswith("removed") and store.count() == 0


def test_zip_cannot_write_outside_destination(tmp_path):
    data = make_zip({"top/ok.py": "x = 1\n", "../../evil.py": "boom\n"}, symlink="top/link")
    root = safe_extract(data, tmp_path / "out")
    assert (root / "ok.py").exists()
    assert not (tmp_path / "evil.py").exists() and not (tmp_path.parent / "evil.py").exists()
    assert not (root / "link").exists()


async def test_rate_limit_error_is_explained():
    def handler(request):
        return httpx.Response(403, headers={"x-ratelimit-remaining": "0"}, text="limit")
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        with pytest.raises(RuntimeError, match="GITHUB_TOKEN"):
            await sync_github(http, "me", HashEmbedder(), InMemoryStore())
