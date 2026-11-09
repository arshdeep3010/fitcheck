"""State every app instance must share, kept in Postgres: rate limits and the last GitHub sync.

One server keeps these in memory (app/ratelimit.py, app/rag/sync.py). On AWS Lambda each
concurrent request can run in its own instance, and an instance is frozen between requests,
so per-instance memory would let a visitor get N x the hourly limit and would forget when
GitHub was last checked. RATE_LIMIT_STORE=postgres puts them in the same database as the index.

Privacy: a visitor's IP is never stored. The rate-limit key is an HMAC of it with a secret
(SYNC_TOKEN), and rows are deleted once they're older than the window (1 hour).
"""

import hashlib
import hmac
import json
import random
from datetime import UTC, datetime
from typing import Protocol

_TABLES = (
    """CREATE TABLE IF NOT EXISTS fitcheck_hits (
        key text NOT NULL,
        at timestamptz NOT NULL DEFAULT now())""",
    "CREATE INDEX IF NOT EXISTS fitcheck_hits_key_at ON fitcheck_hits (key, at)",
    """CREATE TABLE IF NOT EXISTS fitcheck_daily (
        day date PRIMARY KEY,
        used int NOT NULL)""",
    """CREATE TABLE IF NOT EXISTS fitcheck_meta (
        key text PRIMARY KEY,
        value jsonb NOT NULL,
        updated timestamptz NOT NULL DEFAULT now())""",
)

# One statement, so two requests can't both read "19 of 20" and both get in. It counts the
# window, inserts a hit only if there's room, and says when the oldest hit expires.
_HIT_SQL = """
WITH recent AS (
    SELECT count(*) AS n, min(at) AS oldest FROM fitcheck_hits
    WHERE key = %(k)s AND at > now() - make_interval(secs => %(w)s)
), ins AS (
    INSERT INTO fitcheck_hits (key) SELECT %(k)s FROM recent WHERE n < %(max)s RETURNING 1
)
SELECT n, ceil(extract(epoch FROM oldest + make_interval(secs => %(w)s) - now()))::int FROM recent
"""

# Counts today's check only if the cap isn't reached yet (no row back = cap reached).
_DAILY_SQL = """
INSERT INTO fitcheck_daily (day, used) VALUES (%(day)s, 1)
ON CONFLICT (day) DO UPDATE SET used = fitcheck_daily.used + 1
WHERE fitcheck_daily.used < %(limit)s
RETURNING used
"""


class Executor(Protocol):
    def execute(self, sql: str, params=None) -> list[tuple]: ...


class PgState:
    def __init__(self, db: Executor, secret: str):
        self._db = db
        self._secret = (secret or "fitcheck").encode()
        for sql in _TABLES:
            db.execute(sql)

    def ip_key(self, ip: str) -> str:
        return hmac.new(self._secret, ip.encode(), hashlib.sha256).hexdigest()[:32]

    def hit(self, key: str, max_requests: int, window_s: float) -> tuple[bool, int]:
        n, retry = self._db.execute(_HIT_SQL, {"k": key, "w": window_s, "max": max_requests})[0]
        if random.random() < 0.05:  # now and then, forget hits older than the window
            self._db.execute("DELETE FROM fitcheck_hits WHERE at < now() - make_interval(secs => %s)",
                             (window_s,))
        if n < max_requests:
            return True, 0
        return False, max(1, retry or 1)

    def count_today(self, limit: int, now: datetime | None = None) -> bool:
        day = (now or datetime.now(UTC)).date()
        return bool(self._db.execute(_DAILY_SQL, {"day": day, "limit": limit}))

    def get_meta(self, key: str) -> dict | None:
        rows = self._db.execute("SELECT value FROM fitcheck_meta WHERE key = %s", (key,))
        return rows[0][0] if rows else None

    def set_meta(self, key: str, value: dict) -> None:
        self._db.execute(
            "INSERT INTO fitcheck_meta (key, value) VALUES (%s, %s::jsonb) "
            "ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value, updated = now()",
            (key, json.dumps(value)))


class PgRateLimiter:
    """Same interface as ratelimit.RateLimiter, shared by every instance."""

    def __init__(self, state: PgState, max_requests: int, window_s: float = 3600.0):
        self._state, self._max, self._window = state, max_requests, window_s

    def allow(self, ip: str) -> tuple[bool, int]:
        return self._state.hit(self._state.ip_key(ip), self._max, self._window)


class PgDailyCap:
    """Same interface as ratelimit.DailyCap, shared by every instance. 0 disables it."""

    def __init__(self, state: PgState, limit: int):
        self._state, self._limit = state, limit

    def allow(self, now: datetime | None = None) -> tuple[bool, int]:
        if self._limit <= 0:
            return True, 0
        now = now or datetime.now(UTC)
        if self._state.count_today(self._limit, now):
            return True, 0
        midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)
        return False, 86400 - int((now - midnight).total_seconds())
