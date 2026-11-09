"""Build and refresh the index: chunk -> embed -> store.

Three sources:
  local folders   ingest()        repo name = folder name or the name you give
  GitHub          sync_github()   every public repo of a user, re-indexed only when its commit changed
  profile (CV)    in both         self-reported claims, re-indexed only when the file changed

Indexing a repo always replaces that repo's old chunks, so re-running is safe.
"""

import asyncio
import hashlib
import logging
import tempfile
from pathlib import Path

import httpx

from app.llm.embeddings import Embedder
from app.models import Chunk
from app.rag import github
from app.rag.chunker import chunk_profile, chunk_repo, embedding_text
from app.rag.store import VectorStore

log = logging.getLogger(__name__)
PROFILE_REPO = "profile"


async def _replace(name: str, chunks: list[Chunk], embedder: Embedder, store: VectorStore) -> int:
    # Embed first: if the embeddings API fails, the repo's old chunks stay in place.
    vectors = await embedder.embed([embedding_text(c) for c in chunks], kind="passage") if chunks else []
    # Store calls can be slow (a remote Postgres), so they run off the event loop: the app
    # keeps answering requests while a live sync writes.
    await asyncio.to_thread(store.replace_repo, name, chunks, vectors)
    log.info("indexed %s: %d chunks", name, len(chunks))
    return len(chunks)


async def ingest_profile(profile: Path, embedder: Embedder, store: VectorStore) -> int | str:
    digest = hashlib.sha256(profile.read_bytes()).hexdigest()[:16]
    if (await asyncio.to_thread(store.indexed_refs)).get(PROFILE_REPO) == digest:
        return "unchanged"
    chunks = [c.model_copy(update={"ref": digest}) for c in chunk_profile(profile)]
    return await _replace(PROFILE_REPO, chunks, embedder, store)


async def ingest(repos: dict[str, Path], embedder: Embedder, store: VectorStore,
                 clear: bool = False, profile: Path | None = None) -> dict[str, int | str]:
    """Index local folders (and the profile). Returns chunk counts per source."""
    if clear:
        store.clear()
    stats: dict[str, int | str] = {}
    for name, root in repos.items():
        stats[name] = await _replace(name, chunk_repo(name, root), embedder, store)
    if profile is not None:
        stats["profile (CV)"] = await ingest_profile(profile, embedder, store)
    return stats


async def sync_github(http: httpx.AsyncClient, user: str, embedder: Embedder, store: VectorStore,
                      token: str = "", exclude: set[str] | None = None) -> dict[str, int | str]:
    """Bring the index in line with the user's public GitHub repos.

    unchanged repo -> skipped (no download, no embedding calls)
    new/updated    -> downloaded at its latest commit and re-indexed
    gone/private   -> its chunks are removed
    """
    exclude = exclude or set()
    repos = [r for r in await github.list_public_repos(http, user, token) if r.name not in exclude]
    indexed = await asyncio.to_thread(store.indexed_refs)
    stats: dict[str, int | str] = {}

    for repo in repos:
        sha = await github.head_sha(http, user, repo, token)
        if indexed.get(repo.name) == sha:
            stats[repo.name] = "unchanged"
            continue
        with tempfile.TemporaryDirectory() as tmp:
            root = await github.download(http, user, repo, sha, Path(tmp), token)
            chunks = [c.model_copy(update={"ref": sha, "url_base": repo.html_url})
                      for c in chunk_repo(repo.name, root)]
            stats[repo.name] = await _replace(repo.name, chunks, embedder, store)

    current = {r.name for r in repos}
    for name in indexed:
        if name != PROFILE_REPO and name not in current:
            await asyncio.to_thread(store.delete_repo, name)
            stats[name] = "removed (no longer public)"
    return stats
