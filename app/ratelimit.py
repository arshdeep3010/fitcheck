"""Abuse guards for the public demo, in memory.

RateLimiter  sliding window per client IP (e.g. 20 checks/hour)
DailyCap     one shared budget for everyone (e.g. 200 checks/day), so rotating or
             spoofed IPs still can't burn through the free model quotas

In memory: good for one server. On Lambda (several instances) app/shared.py keeps the
same limits in Postgres.
"""

import ipaddress
import json
import time
from collections import deque
from datetime import UTC, datetime
from threading import Lock


class RateLimiter:
    def __init__(self, max_requests: int, window_s: float = 3600.0):
        self._max = max_requests
        self._window = window_s
        self._hits: dict[str, deque[float]] = {}
        self._lock = Lock()
        self._last_prune = 0.0

    def allow(self, key: str, now: float | None = None) -> tuple[bool, int]:
        """Returns (allowed, seconds until the next request is allowed)."""
        now = time.monotonic() if now is None else now
        with self._lock:
            self._prune(now)
            hits = self._hits.setdefault(key, deque())
            while hits and now - hits[0] >= self._window:
                hits.popleft()
            if len(hits) >= self._max:
                return False, int(self._window - (now - hits[0])) + 1
            hits.append(now)
            return True, 0

    def _prune(self, now: float) -> None:
        """Forget IPs whose hits have all expired, so memory doesn't grow with every visitor."""
        if now - self._last_prune < self._window:
            return
        self._last_prune = now
        self._hits = {k: h for k, h in self._hits.items() if h and now - h[-1] < self._window}

    def tracked(self) -> int:
        return len(self._hits)


class DailyCap:
    """At most `limit` checks per UTC day across all visitors. 0 disables it."""

    def __init__(self, limit: int):
        self._limit = limit
        self._day = ""
        self._used = 0
        self._lock = Lock()

    def allow(self, now: datetime | None = None) -> tuple[bool, int]:
        """Returns (allowed, seconds until midnight UTC)."""
        if self._limit <= 0:
            return True, 0
        now = now or datetime.now(UTC)
        day = now.strftime("%Y-%m-%d")
        with self._lock:
            if day != self._day:
                self._day, self._used = day, 0
            if self._used >= self._limit:
                midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)
                return False, 86400 - int((now - midnight).total_seconds())
            self._used += 1
            return True, 0


def lambda_source_ip(request_context: str) -> str:
    """The visitor's IP on AWS Lambda, from the request context the Lambda Web Adapter adds.

    The adapter sets the x-amzn-request-context header itself from the Lambda event (a header
    the visitor sends with that name is replaced), and the function URL fills sourceIp from
    the TCP connection, so it can't be forged the way X-Forwarded-For can.
    """
    try:
        context = json.loads(request_context) if request_context else {}
        return str(context.get("http", {}).get("sourceIp") or "")
    except (ValueError, AttributeError):
        return ""


def cloudfront_viewer_ip(viewer_address: str) -> str:
    """The visitor's IP from CloudFront's CloudFront-Viewer-Address header ("ip:port", e.g.
    198.51.100.10:46532; for IPv6 the port follows the last colon too).

    Only trustworthy when CloudFront is the only way in: the function URL then requires
    CloudFront's own signature (AWS_IAM + origin access control), so nobody can reach the app
    with a header they wrote themselves. Returns "" for anything that isn't an IP address.
    """
    host = viewer_address.strip().rsplit(":", 1)[0].strip("[]") if ":" in viewer_address else ""
    try:
        return str(ipaddress.ip_address(host))
    except ValueError:
        return ""


def client_ip(forwarded_for: str, peer: str, trust_proxy: bool, proxy_hops: int) -> str:
    """The visitor's IP.

    X-Forwarded-For is "client, proxy1, proxy2...": each proxy APPENDS the address it saw.
    The left part is whatever the client sent, so it can be forged. Only the last
    `proxy_hops` entries were written by our own proxies, so the visitor is the entry
    just before them: ips[-proxy_hops].
    """
    if trust_proxy and forwarded_for and proxy_hops > 0:
        ips = [ip.strip() for ip in forwarded_for.split(",") if ip.strip()]
        if ips:
            return ips[-proxy_hops] if len(ips) >= proxy_hops else ips[0]
    return peer
