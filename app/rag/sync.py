"""Keep the index in step with GitHub while the app runs.

Every GITHUB_SYNC_INTERVAL_MIN minutes (and once at startup) the server asks GitHub for
the latest commit of each public repo. Only repos whose commit changed are downloaded
and re-embedded, so a quiet check costs 1 + one-per-repo small API calls and no embeddings.
A new or updated repo is citable a few minutes after it is pushed, with no redeploy.

On AWS Lambda there's no loop (a frozen function can't wake itself): an EventBridge schedule
calls run_once() through POST /events, and the result is kept in Postgres (`shared`), so
every instance shows the same "last checked".

Requests keep being served during a sync: each repo's chunks are swapped in one step,
so a check never sees a half-indexed repo.
"""

import asyncio
import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx

from app.config import Settings
from app.llm.embeddings import Embedder
from app.rag.ingest import ingest_profile, sync_github
from app.rag.store import InMemoryStore, VectorStore

log = logging.getLogger(__name__)


def _now() -> datetime:
    return datetime.now(UTC)


def _iso(value: datetime | None) -> str | None:
    return value.isoformat(timespec="seconds") if value else None


class GitHubSyncer:
    def __init__(self, http: httpx.AsyncClient, settings: Settings, embedder: Embedder,
                 store: VectorStore, shared=None):
        self._http = http
        self._settings = settings
        self._embedder = embedder
        self._store = store
        self._lock = asyncio.Lock()
        self.last_started: datetime | None = None
        self.last_finished: datetime | None = None
        self.last_error: str | None = None
        self.last_changes: dict[str, int | str] = {}
        self.next_run: datetime | None = None
        self._shared = shared  # app.shared.PgState, or None for in-memory status

    @property
    def enabled(self) -> bool:
        s = self._settings
        return bool(s.github_user) and s.github_sync_interval_min > 0 and s.provider_mode == "live"

    @property
    def running(self) -> bool:
        return self._lock.locked()

    async def run_once(self) -> dict[str, int | str]:
        """One sync. If one is already running, waits for it instead of starting another."""
        if self._lock.locked():
            async with self._lock:
                return self.last_changes
        async with self._lock:
            self.last_started = _now()
            try:
                stats = await self._sync()
                self.last_changes = {k: v for k, v in stats.items() if v != "unchanged"}
                self.last_error = None
                if self.last_changes:
                    log.info("GitHub sync changed: %s", self.last_changes)
                return stats
            except Exception as exc:  # noqa: BLE001 - a failed sync keeps the old index serving
                log.exception("GitHub sync failed")
                self.last_error = f"{type(exc).__name__}: {str(exc)[:200]}"
                return {}
            finally:
                self.last_finished = _now()
                if self._shared is not None:
                    await asyncio.to_thread(self._save_status)

    async def _sync(self) -> dict[str, int | str]:
        s = self._settings
        exclude = {name.strip() for name in s.github_exclude.split(",") if name.strip()}
        stats = await sync_github(self._http, s.github_user, self._embedder, self._store,
                                  token=s.github_token, exclude=exclude)
        profile = Path(s.profile_path) if s.profile_path else None
        if profile and profile.is_file():
            stats["profile (CV)"] = await ingest_profile(profile, self._embedder, self._store)
        if isinstance(self._store, InMemoryStore) and any(v != "unchanged" for v in stats.values()):
            await asyncio.to_thread(self._store.save)
        return stats

    async def loop(self) -> None:
        """Runs until cancelled: sync now, then every interval."""
        interval = timedelta(minutes=self._settings.github_sync_interval_min)
        while True:
            await self.run_once()
            self.next_run = _now() + interval
            await asyncio.sleep(interval.total_seconds())

    def _save_status(self) -> None:
        try:
            self._shared.set_meta("github_sync", {
                "last_checked": _iso(self.last_finished), "last_error": self.last_error,
                "last_changes": self.last_changes})
        except Exception:  # noqa: BLE001 - status is cosmetic; never fail a sync over it
            log.exception("could not save the sync status")

    def _load_status(self) -> None:
        """Another instance may have synced since: take its result."""
        try:
            saved = self._shared.get_meta("github_sync") or {}
        except Exception:  # noqa: BLE001
            log.exception("could not read the sync status")
            return
        if saved.get("last_checked"):
            finished = datetime.fromisoformat(saved["last_checked"])
            if self.last_finished is None or finished > self.last_finished:
                self.last_finished = finished
                self.last_error = saved.get("last_error")
                self.last_changes = saved.get("last_changes") or {}
        if self.last_finished is not None:  # the schedule runs every interval
            self.next_run = self.last_finished + timedelta(minutes=self._settings.github_sync_interval_min)

    def status(self) -> dict:
        """Public view: when GitHub was last checked and what is indexed at which commit."""
        if self._shared is not None:
            self._load_status()
        order = {"github": 0, "local": 1, "cv": 2}
        repos = sorted(self._store.repo_summary(), key=lambda r: (order[r["source"]], r["repo"].lower()))
        return {
            "enabled": self.enabled,
            "github_user": self._settings.github_user or None,
            "interval_min": self._settings.github_sync_interval_min,
            "running": self.running,
            "last_checked": _iso(self.last_finished),
            "next_check": _iso(self.next_run) if self.enabled else None,
            "last_error": self.last_error,
            "last_changes": self.last_changes,
            "repos": repos,
        }
