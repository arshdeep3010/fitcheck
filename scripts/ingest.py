"""Build or refresh the FitCheck index.

Usage (from the fitcheck folder):
    # every public repo on my GitHub (only changed repos are re-indexed) + my CV profile
    python -m scripts.ingest --github arshdeep3010 --profile profile/profile.md

    # local folders go ONLY into the eval index (data/eval-index.json), never the app's:
    # they have no public link, and may not be public at all
    python -m scripts.ingest --eval ../Healthdecissionagent ../vibe-audit

    # start from scratch
    python -m scripts.ingest --rebuild --github arshdeep3010 --profile profile/profile.md

Re-running is safe: each repo's old chunks are replaced, unchanged sources are skipped.
Only index PUBLIC code and a profile you're happy to show: snippets appear on a public page.
"""

import argparse
import asyncio
import logging
from pathlib import Path

import httpx

from app.config import get_settings
from app.deps import build_embedder, build_store
from app.rag.ingest import ingest, sync_github
from app.rag.store import InMemoryStore


def parse_repos(args: list[str]) -> dict[str, Path]:
    repos: dict[str, Path] = {}
    for arg in args:
        name, _, path = arg.partition("=") if "=" in arg else ("", "", arg)
        root = Path(path).resolve()
        if not root.is_dir():
            raise SystemExit(f"Not a directory: {root}")
        repos[name or root.name] = root
    return repos


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("repos", nargs="*", help="local repo folders, optionally as name=path")
    parser.add_argument("--github", metavar="USER", help="sync every public repo of this GitHub user")
    parser.add_argument("--exclude", nargs="*", default=None, help="GitHub repo names to skip")
    parser.add_argument("--profile", type=Path, help="Markdown profile/CV to index as self-reported claims")
    parser.add_argument("--rebuild", action="store_true", help="wipe the whole index first")
    parser.add_argument("--eval", action="store_true",
                        help="write to the eval index (EVAL_INDEX_PATH) instead of the app's")
    args = parser.parse_args()
    if not (args.repos or args.github or args.profile):
        parser.error("give local repos, --github USER, and/or --profile FILE")
    if args.repos and not args.eval:
        # The app shows snippets publicly and must link every claim to public code.
        parser.error("local folders can only go into the eval index (add --eval). For the app, "
                     "make the repo public and use --github: the sync then indexes it with links")

    settings = get_settings()
    if args.eval:
        if settings.vector_store != "memory":
            parser.error("--eval needs VECTOR_STORE=memory (the eval index is a local file)")
        settings = settings.model_copy(update={"index_path": settings.eval_index_path})
    exclude = set(args.exclude if args.exclude is not None
                  else filter(None, (s.strip() for s in settings.github_exclude.split(","))))
    async with httpx.AsyncClient(timeout=settings.http_timeout_s) as http:
        embedder = build_embedder(settings, http)
        store = build_store(settings)
        stats = await ingest(parse_repos(args.repos), embedder, store,
                             clear=args.rebuild, profile=args.profile)
        if args.github:
            stats.update(await sync_github(http, args.github, embedder, store,
                                           token=settings.github_token, exclude=exclude))
        if isinstance(store, InMemoryStore):
            store.save()

    for name, result in stats.items():
        print(f"{name:32s} {result if isinstance(result, str) else f'{result} chunks'}")
    print(f"{'total in index':32s} {store.count()} chunks  (embedder: {embedder.name}, store: {settings.vector_store})")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    asyncio.run(main())
