"""Measure fit scorers on labelled requirements, using the real index.

Usage (from the fitcheck folder, after building the index):
    python -m evals.run_evals                         # all scorers available in this mode
    python -m evals.run_evals --scorers keyword system_one
    python -m evals.run_evals --scorers parallel        # Parallel Decisions (free)
    python -m evals.run_evals --dataset heldout        # the held-out set: run it ONCE per prompt version

Requirements are given directly (the JD parser is skipped), so this isolates
retrieval + scoring + grounding, the parts that decide what gets claimed.

Metrics
  accuracy          final verdict == expected label (unverified counts as wrong)
  false_claim_rate  expected not_met but reported met/partial  <- must stay ~0
  coverage          share of requirements NOT left unverified
  overclaims        verdict stronger than the label (not_met -> met/partial, partial -> met)
  avg_latency_ms    scoring time per JD

Every results file also records the dataset's SHA-256 and what was indexed (repo -> commit,
chunk count), because labels are only true for a given index.
"""

import argparse
import asyncio
import hashlib
import json
import time
from datetime import datetime
from pathlib import Path

import httpx

from app.config import get_settings
from app.deps import build_chat, build_embedder, build_scorer, build_store
from app.grounding import ground
from app.llm.http import CreditsExhaustedError
from app.llm.system_one import SystemOneClient
from app.models import Requirement, Verdict
from app.rag.hybrid import retrieve
from app.scoring.keyword import KeywordFitScorer
from app.scoring.llm_json import LlmJsonFitScorer
from app.scoring.system_one import SystemOneFitScorer

DATASETS = {
    "main": Path(__file__).parent / "dataset.jsonl",  # used while developing prompts
    "heldout": Path(__file__).parent / "heldout.jsonl",  # fresh JDs, labelled before any run
}
RESULTS_DIR = Path(__file__).parent / "results"
STRENGTH = {"not_met": 0, "unverified": 0, "not_checkable": 0, "partial": 1, "met": 2}


def load_dataset(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


async def evaluate(scorer, cases: list[dict], embedder, store, settings) -> dict:
    total = correct = false_claims = negatives = covered = overclaims = 0
    latencies: list[float] = []
    rows: list[dict] = []

    for case in cases:
        reqs = [Requirement(id=f"r{i + 1}", text=r["text"], must_have=r["must_have"])
                for i, r in enumerate(case["requirements"])]
        expected = {f"r{i + 1}": r["expected"] for i, r in enumerate(case["requirements"])}

        vectors = await embedder.embed([r.text for r in reqs], kind="query")
        evidence = {r.id: retrieve(store, r.text, v, settings.retrieve_top_k, settings.retrieval,
                                   settings.keyword_extra)
                    for r, v in zip(reqs, vectors)}

        start = time.perf_counter()
        scores = await scorer.score(reqs, evidence)
        latencies.append((time.perf_counter() - start) * 1000)

        raw = {s.requirement_id: s for s in scores}
        for res in ground(reqs, evidence, scores, settings.min_confidence):
            want = expected[res.requirement.id]
            opinion = raw.get(res.requirement.id)
            got = res.verdict.value
            total += 1
            correct += got == want
            covered += res.verdict != Verdict.UNVERIFIED
            if want == "not_met":
                negatives += 1
                false_claims += got in ("met", "partial")
            overclaims += STRENGTH.get(got, 0) > STRENGTH.get(want, 0)
            rows.append({"case": case["id"], "requirement": res.requirement.text,
                         "expected": want, "got": got, "confidence": round(res.confidence, 3),
                         "citation": res.citation.source if res.citation else None,
                         # What the model said before grounding, so a demoted answer is visible:
                         # e.g. model_status "partial" + got "unverified" = leaned yes, cited nothing.
                         "model_status": opinion.status if opinion else None,
                         "note": res.note,
                         # What the scorer was shown, and which search found it.
                         "retrieved": [f"{e.chunk.source} [{e.via}]"
                                       for e in evidence[res.requirement.id]]})

    return {
        "scorer": scorer.name,
        "accuracy": round(correct / total, 3) if total else 0.0,
        "false_claim_rate": round(false_claims / negatives, 3) if negatives else 0.0,
        "coverage": round(covered / total, 3) if total else 0.0,
        "avg_latency_ms": round(sum(latencies) / len(latencies)) if latencies else 0,
        "overclaims": overclaims,
        "n_requirements": total,
        "rows": rows,
    }


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scorers", nargs="*", default=None,
                        help="keyword | system_one | llm_json | parallel (default: all available)")
    parser.add_argument("--dataset", choices=sorted(DATASETS), default="main")
    parser.add_argument("--retrieval", choices=["hybrid", "vector"], default=None,
                        help="override RETRIEVAL from the environment (default: vector)")
    args = parser.parse_args()

    settings = get_settings()
    if args.retrieval:
        settings = settings.model_copy(update={"retrieval": args.retrieval})
    if settings.vector_store == "memory":
        # The labels assume projects that the public app index doesn't have: use the eval index.
        if not Path(settings.eval_index_path).exists():
            raise SystemExit(f"No eval index at {settings.eval_index_path}. Build it with "
                             "`python -m scripts.ingest --eval <folders> --github <user> --profile <cv>`.")
        settings = settings.model_copy(update={"index_path": settings.eval_index_path})
    dataset_path = DATASETS[args.dataset]
    cases = load_dataset(dataset_path)
    async with httpx.AsyncClient(timeout=settings.http_timeout_s) as http:
        embedder = build_embedder(settings, http)
        store = build_store(settings)
        if store.count() == 0:
            raise SystemExit("Index is empty. Run `python -m scripts.ingest --eval <repos>` first.")
        indexed = {r["repo"] for r in store.repo_summary()}
        missing = sorted({repo for case in cases for repo in case.get("assumes_indexed", [])} - indexed)
        if missing:  # a label is only true if what it assumes is in the index
            raise SystemExit(f"The labels assume {missing}, which aren't in {settings.index_path}.")

        available = {"keyword": KeywordFitScorer()}
        if settings.provider_mode != "fake":
            available["system_one"] = SystemOneFitScorer(SystemOneClient(
                http, settings.openrouter_base_url, settings.openrouter_api_key, settings.decide_model))
            available["llm_json"] = LlmJsonFitScorer(build_chat(settings, http))
            # Same construction as the app (deps.build_scorer): name redacted, web sources dropped.
            available["parallel"] = build_scorer(settings.model_copy(update={"scorer": "parallel"}),
                                                 http, chat=None)
        chosen = args.scorers or list(available)

        try:
            reports = [await evaluate(available[name], cases, embedder, store, settings)
                       for name in chosen]
        except CreditsExhaustedError as exc:
            # No fallback here: an eval must measure the scorer it names, or nothing.
            raise SystemExit(f"Out of credits, no results written: {exc}\n"
                             "Top up the OpenRouter credits, or run --scorers keyword llm_json.") from exc
        snapshot = sorted(store.repo_summary(), key=lambda r: r["repo"].lower())

    digest = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
    print(f"\ndataset {args.dataset} ({dataset_path.name}, sha256 {digest[:12]}), "
          f"{sum(r['chunks'] for r in snapshot)} chunks from {len(snapshot)} sources")
    print(f"{'scorer':45s} {'accuracy':>9s} {'false claims':>13s} {'overclaims':>11s} "
          f"{'coverage':>9s} {'latency':>9s}")
    for r in reports:
        print(f"{r['scorer']:45s} {r['accuracy']:9.0%} {r['false_claim_rate']:13.0%} "
              f"{r['overclaims']:11d} {r['coverage']:9.0%} {r['avg_latency_ms']:7d}ms")

    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / f"eval-{args.dataset}-{datetime.now():%Y%m%d-%H%M%S}.json"
    meta = {"dataset": args.dataset, "dataset_file": dataset_path.name, "dataset_sha256": digest,
            "run_at": datetime.now().isoformat(timespec="seconds"),
            "settings": {"retrieval": settings.retrieval, "keyword_extra": settings.keyword_extra,
                         "retrieve_top_k": settings.retrieve_top_k,
                         "min_confidence": settings.min_confidence,
                         "embedding_model": settings.embedding_model},
            "index_snapshot": snapshot}
    out.write_text(json.dumps({"meta": meta, "reports": reports}, indent=2), encoding="utf-8")
    print(f"\nDetails: {out}")


if __name__ == "__main__":
    asyncio.run(main())
