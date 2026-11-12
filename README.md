# FitCheck

**Paste a job description. Get an honest fit report, where every "met" links to the code that proves it.**

FitCheck reads a JD, searches my real repositories, and rates every requirement as **Met**, **Partly met**, **Not met**, or **Unverified**. It can't claim a skill it can't cite.

`Python` · `LangGraph` · `FastAPI` · `PostgreSQL + pgvector` · `NVIDIA Nemotron` · `Mercury Decide` · `Docker`

**Live demo:** https://d3fkoasdzz9loq.cloudfront.net/ (AWS Lambda behind CloudFront; the first check after a quiet spell takes a few seconds longer while it starts). How it's deployed: [`infra/`](infra/).

## How it works

```
parse_jd (Nemotron) → retrieve (vector search on pgvector; keyword search opt-in) → score (1 call: Nemotron, Mercury Decide or Parallel Decisions) → ground (code rules) → report (code)
```

- The scorer picks a status and its proof **only from chunks that were actually retrieved**. It's a switch: the free Nemotron scorer by default, or a decision model whose confidence varies with the evidence instead of sitting at ~0.95: Mercury Decide (`SCORER=system_one`, paid) or Parallel Decisions (`SCORER=parallel`, free). Parallel always searches the web and that can't be turned off, so FitCheck drops its web sources, lets it cite only retrieved code, and replaces my name before sending anything. It's live-checked, not yet measured on the held-out set.
- Technology lists are split ("Spring Security, OAuth2 and JWT" → 3 rows), so each one needs its own evidence.
- **Code decides what's claimed:** no citation or low confidence → Unverified. Outages → nothing claimed.
- **Proof and claims never look alike:** GitHub code gets a solid badge and a commit permalink; CV lines get a dashed "CV claim" badge, sit in their own box marked "Not proven by code", and count only in the second score.
- **Evidence sources:** my public GitHub repos (the running app checks GitHub every 30 minutes and re-indexes only repos with a new commit; permalink citations) and my CV, which is shown only as **Stated on CV** and never as Met.
- Two scores, **proven in code** and **including CV claims**, both computed deterministically.

## Results

25 hand-labelled requirements, including near misses (Docker ≠ Kubernetes, LangGraph ≠ LangChain, JWT ≠ OAuth2, React ≠ Angular). Live NVIDIA embeddings over 719 chunks (12 repos + CV), measured 8 Nov 2026. The scorer prompts use no examples from this set.

| Scorer | Accuracy | False claims | Latency per JD |
|---|---|---|---|
| Keyword baseline | 52% | 54% | 6 ms |
| Plain LLM (Nemotron 3 Ultra, JSON) — default, free | 100% | 0% | 3.5–14 s |
| Mercury Decide (paid, ~₹0.03/check) | 100% | 0% | 2.3 s |

**Held-out set** (the honest number): 31 requirements from 4 real job postings, labelled by reading the code *before* any model ran, frozen by hash, run once.

| Scorer | Accuracy | False claims | Overclaims | Latency per JD |
|---|---|---|---|---|
| Keyword baseline | 55% | 54% | 7 | 2 ms |
| Plain LLM (Nemotron) | 90% | 0% | 1 | 6.3 s |
| Mercury Decide | 97% | 0% | 0 | 2.9 s |

Nemotron's overclaim: "Spring Boot **and microservices**" marked met by citing an architecture doc of a single service. Mercury's miss: retrieval didn't surface the LangGraph code for "AutoGen, CrewAI, or LangGraph". **Hybrid retrieval** (`RETRIEVAL=hybrid`, opt-in) fixes that: keyword search (BM25) adds up to 3 chunks that vector search missed, and `app/graph.py` is retrieved. It's **off by default**: with the Nemotron scorer it raised held-out false claims from 0% to 8–15% over 3 runs, e.g. claiming PyTorch from a list of skill names. Recall isn't worth a false claim here.

Mercury first scored 84%: on 4 near misses (Kubernetes, Spring Cloud, OAuth2, microservices) it leaned yes but couldn't cite proof, so grounding marked them **Unverified** instead of claiming them. The cause was our own definition of "partial", which included "a related technology". After fixing it, Mercury scored 100% in two runs. That fix was found on this same set, so a held-out set of fresh JDs is next.

25 requirements is a small set; one label had to be corrected after the GitHub sync added a React repo (details in [PROJECT.md §9](PROJECT.md#9-evals)).

## Run it

```bash
pip install -r requirements-dev.txt
PROVIDER_MODE=fake pytest -q                         # offline, no keys (111 pass, 8 pgvector tests skip)
cp .env.example .env                                 # add NVIDIA + OpenRouter keys
python -m scripts.ingest --github <your-user> --profile profile/profile.md
uvicorn app.main:app --reload
```

**Deploy:** `infra/` has the Terraform for AWS Lambda (container image, CloudFront in front of a streaming function URL that only CloudFront can call, an EventBridge schedule for the GitHub sync, secrets in SSM, a GitHub OIDC deploy role, an optional budget alert), and `.github/workflows/deploy.yml` ships every green push to `main`. Expected cost: ₹0 to a few rupees a month. Steps: [infra/README.md](infra/README.md).

A **How it works** page (`/how-it-works`) explains the pipeline and grounding rules, and shows live which repos are indexed at which commit (`GET /api/sync/status`).

Full design, diagrams, deployment and evals: **[PROJECT.md](PROJECT.md)**.
