"""HTTP API + the one-page UI.

POST /api/fit          {"jd": "..."}  -> FitReport (JSON)
POST /api/fit/stream   {"jd": "..."}  -> Server-Sent Events: one "step" per pipeline node, then "report"
GET  /api/sync/status                 -> when GitHub was last checked + every indexed repo and commit
POST /api/sync                        -> check GitHub now (needs SYNC_TOKEN as a Bearer token)
GET  /health                          -> index size + configured models
GET  /                                -> paste-a-JD page
GET  /how-it-works                    -> how the pipeline, grounding and GitHub sync work
"""

import asyncio
import contextlib
import hmac
import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from app.config import Settings, get_settings
from app.deps import build_deps
from app.graph import build_graph, run_fit_check, stream_fit_check
from app.models import FitReport
from app.rag.sync import GitHubSyncer
from app.ratelimit import DailyCap, RateLimiter, client_ip, lambda_source_ip
from app.shared import PgDailyCap, PgRateLimiter, PgState

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger(__name__)
STATIC_DIR = Path(__file__).parent / "static"


class FitRequest(BaseModel):
    jd: str = Field(min_length=50, description="The job description, pasted as plain text.")


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        async with httpx.AsyncClient(timeout=settings.http_timeout_s) as http:
            deps = build_deps(settings, http)
            app.state.deps = deps
            app.state.graph = build_graph(deps)
            shared = None
            if settings.rate_limit_store == "postgres":
                if settings.vector_store != "pgvector":
                    raise ValueError("RATE_LIMIT_STORE=postgres needs VECTOR_STORE=pgvector")
                shared = PgState(deps.store, settings.sync_token)
                app.state.limiter = PgRateLimiter(shared, settings.rate_limit_per_hour)
                app.state.daily_cap = PgDailyCap(shared, settings.daily_limit)
            else:
                app.state.limiter = RateLimiter(settings.rate_limit_per_hour)
                app.state.daily_cap = DailyCap(settings.daily_limit)
            local = [r["repo"] for r in deps.store.repo_summary() if r["source"] == "local"]
            if local:  # added from a folder: no public link, and the sync never removes them
                log.warning("The index has local-folder sources with no public link: %s. Remove "
                            "them; use --eval to index local folders for evals only.", local)
            syncer = GitHubSyncer(http, settings, deps.embedder, deps.store, shared)
            app.state.syncer = syncer
            # On Lambda the schedule calls POST /events instead: a frozen function can't loop.
            looping = syncer.enabled and settings.runtime != "lambda"
            task = asyncio.create_task(syncer.loop()) if looping else None
            try:
                yield
            finally:
                if task:
                    task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await task
                if hasattr(deps.store, "close"):
                    deps.store.close()

    app = FastAPI(title="FitCheck", version="0.2.0", lifespan=lifespan)

    def visitor_ip(request: Request) -> str:
        if settings.runtime == "lambda":
            ip = lambda_source_ip(request.headers.get("x-amzn-request-context", ""))
            if ip:
                return ip
        return client_ip(request.headers.get("x-forwarded-for", ""),
                         request.client.host if request.client else "unknown",
                         settings.trust_proxy_headers, settings.proxy_hops)

    def guard(body: FitRequest, request: Request) -> None:
        """Size + rate limits, checked before any model call (and before a stream opens)."""
        if len(body.jd) > settings.max_jd_chars:
            raise HTTPException(413, f"Job description is too long (max {settings.max_jd_chars} characters).")
        allowed, retry_after = request.app.state.limiter.allow(visitor_ip(request))
        if not allowed:
            raise HTTPException(429, "Too many checks from this address. Try again later.",
                                headers={"Retry-After": str(retry_after)})
        allowed, retry_after = request.app.state.daily_cap.allow()
        if not allowed:
            raise HTTPException(429, "The demo has reached today's limit. Please try again tomorrow.",
                                headers={"Retry-After": str(retry_after)})

    @app.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/how-it-works", include_in_schema=False)
    async def how_it_works() -> FileResponse:
        return FileResponse(STATIC_DIR / "how-it-works.html")

    @app.get("/api/sync/status")
    async def sync_status(request: Request) -> dict:
        return await asyncio.to_thread(request.app.state.syncer.status)

    def check_sync_token(given: str) -> GitHubSyncer:
        syncer: GitHubSyncer = app.state.syncer
        if not settings.sync_token or not syncer.enabled:
            raise HTTPException(404, "Manual sync is not enabled.")
        if not hmac.compare_digest(given.encode(), settings.sync_token.encode()):
            raise HTTPException(401, "Wrong or missing sync token.")
        return syncer

    @app.post("/api/sync", status_code=202)
    async def sync_now(request: Request) -> dict:
        """Force a GitHub check now, e.g. from a deploy hook. Off unless SYNC_TOKEN is set."""
        syncer = check_sync_token(request.headers.get("authorization", "").removeprefix("Bearer ").strip())
        if settings.runtime == "lambda":
            # Lambda freezes the function once it answers, so finish the sync first.
            return {"started": True, "result": await syncer.run_once()}
        if not syncer.running:
            # Keep a reference so the task isn't garbage-collected before it finishes.
            request.app.state.manual_sync = asyncio.create_task(syncer.run_once())
        return {"started": True}

    @app.post("/events", include_in_schema=False)
    async def lambda_event(request: Request) -> dict:
        """Non-HTTP Lambda invocations (the EventBridge sync schedule, the deploy workflow).

        The Lambda Web Adapter posts the event's JSON here. This path is also reachable
        through the public URL, so the event must carry the sync token.
        """
        if settings.runtime != "lambda":
            raise HTTPException(404, "Not found.")
        try:
            event = await request.json()
        except ValueError:
            event = None
        if not isinstance(event, dict) or event.get("action") != "sync":
            raise HTTPException(400, "Unknown event.")
        syncer = check_sync_token(str(event.get("token", "")))
        return {"synced": await syncer.run_once()}

    @app.get("/health")
    async def health(request: Request) -> dict:
        deps = request.app.state.deps
        return {"status": "ok", "indexed_chunks": deps.store.count(), "models": deps.models,
                "candidate": settings.candidate_name,
                "settings": {"min_confidence": settings.min_confidence, "top_k": settings.retrieve_top_k,
                             "retrieval": settings.retrieval,
                             "keyword_extra": settings.keyword_extra if settings.retrieval == "hybrid" else 0},
                # out_of_credits = OpenRouter said the credits ran out; the fallback is judging.
                "scorer_status": {"out_of_credits": deps.credits.paused,
                                  "fallback": deps.fallback_scorer.name if deps.fallback_scorer else None,
                                  "retry_in_s": deps.credits.seconds_left()},
                # What actually runs each step, so the UI can show the real stack.
                "stack": {"framework": "LangGraph", "api": "FastAPI", "store": settings.vector_store}}

    # Nothing about a request is stored: no JD, no IP beyond the in-memory rate window.

    @app.post("/api/fit", response_model=FitReport)
    async def fit(body: FitRequest, request: Request) -> FitReport:
        guard(body, request)
        return await run_fit_check(request.app.state.graph, body.jd)

    @app.post("/api/fit/stream")
    async def fit_stream(body: FitRequest, request: Request) -> StreamingResponse:
        guard(body, request)
        graph = request.app.state.graph

        async def events():
            try:
                async for item in stream_fit_check(graph, body.jd):
                    if item[0] == "step":
                        _, node, ms, detail = item
                        yield _sse("step", {"node": node, "ms": ms, "detail": detail})
                    else:
                        yield _sse("report", item[1].model_dump(mode="json"))
            except Exception:  # noqa: BLE001 - the client must always get a terminal event
                log.exception("stream failed")
                yield _sse("error", {"message": "Something went wrong while checking. Please try again."})

        return StreamingResponse(events(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    return app


app = create_app()
