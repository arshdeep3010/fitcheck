"""Pull my public GitHub repositories so new projects show up as evidence automatically.

Only public, non-fork, non-archived repos are used: their snippets end up on a public page.
Downloads are commit-pinned zipballs, extracted defensively (no path escapes, no symlinks,
size-capped), and every chunk remembers the commit SHA so citations are permalinks.
"""

import io
import logging
import stat
import zipfile
from dataclasses import dataclass
from pathlib import Path

import httpx

log = logging.getLogger(__name__)

GITHUB_API = "https://api.github.com"
MAX_ARCHIVE_BYTES = 50_000_000
MAX_MEMBER_BYTES = 5_000_000


class GitHubError(RuntimeError):
    pass


@dataclass(frozen=True)
class RepoInfo:
    name: str
    default_branch: str
    html_url: str


def _headers(token: str, accept: str = "application/vnd.github+json") -> dict[str, str]:
    headers = {"Accept": accept, "User-Agent": "fitcheck-indexer", "X-GitHub-Api-Version": "2022-11-28"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _check(response: httpx.Response, what: str) -> None:
    if response.status_code == 403 and response.headers.get("x-ratelimit-remaining") == "0":
        raise GitHubError(f"GitHub rate limit hit while fetching {what}; set GITHUB_TOKEN to raise it")
    if response.status_code >= 400:
        raise GitHubError(f"GitHub returned {response.status_code} for {what}: {response.text[:200]}")


async def list_public_repos(http: httpx.AsyncClient, user: str, token: str = "") -> list[RepoInfo]:
    repos: list[RepoInfo] = []
    page = 1
    while True:
        response = await http.get(f"{GITHUB_API}/users/{user}/repos", headers=_headers(token),
                                  params={"per_page": 100, "page": page, "type": "owner", "sort": "pushed"})
        _check(response, f"repos of {user}")
        batch = response.json()
        for repo in batch:
            if repo.get("private") or repo.get("fork") or repo.get("archived") or not repo.get("size"):
                continue
            repos.append(RepoInfo(name=repo["name"], default_branch=repo["default_branch"],
                                  html_url=repo["html_url"]))
        if len(batch) < 100:
            return repos
        page += 1


async def head_sha(http: httpx.AsyncClient, user: str, repo: RepoInfo, token: str = "") -> str:
    response = await http.get(f"{GITHUB_API}/repos/{user}/{repo.name}/commits/{repo.default_branch}",
                              headers=_headers(token, accept="application/vnd.github.sha"))
    _check(response, f"latest commit of {repo.name}")
    return response.text.strip()


async def download(http: httpx.AsyncClient, user: str, repo: RepoInfo, sha: str, dest: Path,
                   token: str = "") -> Path:
    """Download the repo at `sha` and extract it under `dest`. Returns the repo root folder."""
    response = await http.get(f"{GITHUB_API}/repos/{user}/{repo.name}/zipball/{sha}",
                              headers=_headers(token), follow_redirects=True)
    _check(response, f"archive of {repo.name}")
    if len(response.content) > MAX_ARCHIVE_BYTES:
        raise GitHubError(f"{repo.name} archive is larger than {MAX_ARCHIVE_BYTES // 1_000_000} MB; skipped")
    return safe_extract(response.content, dest)


def safe_extract(data: bytes, dest: Path) -> Path:
    """Extract a zip without trusting it: every member must land inside `dest`."""
    dest = dest.resolve()
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        for member in archive.infolist():
            if member.is_dir():
                continue
            if stat.S_ISLNK(member.external_attr >> 16):  # symlinks could point anywhere
                continue
            if member.file_size > MAX_MEMBER_BYTES:
                continue
            target = (dest / member.filename).resolve()
            if not target.is_relative_to(dest):
                log.warning("skipping unsafe zip member %r", member.filename)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archive.read(member))
    # GitHub zipballs contain one top folder: <owner>-<repo>-<sha>/
    folders = [p for p in dest.iterdir() if p.is_dir()]
    return folders[0] if len(folders) == 1 else dest
