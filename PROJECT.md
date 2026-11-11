<!-- fitcheck: no-index (this doc quotes sample requirements; it is not evidence of skills) -->
# FitCheck — everything about the project

> **One line:** paste a job description, get an honest fit report for me, where every "met" links to the exact lines of code that prove it, and gaps are shown as gaps.
>
> **Status (8 Nov 2026):** recruiter mode built · **113 tests** (105 offline + 8 against a real pgvector 0.8.7, also run in CI) · live runs done with NVIDIA chat + embeddings · **the app syncs with GitHub on its own** (startup + every 30 min, §3.6) · a **How it works** page at `/how-it-works` · **final check done 8 Nov**: all tests, evals, a real JD end to end in the browser, the Docker image (§9).
> **Scoring today:** the free Nemotron scorer (`SCORER=llm_json`, now also the default in code). Mercury Decide costs ~₹0.03 per check and needs OpenRouter credits (§7). **On 8 Nov OpenRouter lists Mercury Decide with 0 provider endpoints** (`404 No endpoints found`, with a new key too), so `SCORER=system_one` currently gives Unverified rows with a warning. Nothing in the app can fix that; switch back when it's served again.
> **Fixed on 7 Nov:** CV search on pgvector returned almost nothing (measured, §6.3); a forged `X-Forwarded-For` could dodge the rate limit; no reconnect after Neon drops idle connections; the default scorer in code didn't match the docs.
> **Mercury Decide run live:** 84% on 7 Nov (it leaned yes on 4 near misses; grounding demoted all 4). After fixing its status criteria on 8 Nov: 100% on the development set.
> **Held-out set (8 Nov, 4 real postings, 31 requirements, labelled before any run, run once):** Mercury **97%**, 0 false claims, 0 overclaims, 2.9 s · Nemotron **90%**, 0 false claims, 1 overclaim, 6.3 s · keyword 55% (§9).
> **Hybrid retrieval (8 Nov), built but off by default:** keyword search (BM25) next to vector search, which can only add evidence. It fixes the held-out LangGraph miss, but with Nemotron it raised held-out false claims from 0% to 8–15% (3 runs each). So the default is `RETRIEVAL=vector`, and hybrid is opt-in until it's re-tested with Mercury (§9).
> **Final numbers (8 Nov, current defaults, 664-chunk eval index):** development set, Nemotron 100%, 0 false claims, 4.1 s.

> **8 Nov: the app's index is public GitHub + CV only.** Four projects had been indexed from folders on my laptop (Healthdecissionagent, vibe-audit, fitcheck, linkedin-bot: 492 of 664 chunks). None of them is public on GitHub, so the app could mark a skill *met* from code a recruiter can't open. They're removed: the app index now holds only what's public on GitHub, plus the CV. `scripts.ingest` refuses local folders unless `--eval` is given, and the app logs a warning at startup if a local-folder source is ever in its index. To bring a project back, make it public: the sync indexes it within 30 minutes, with permalinks.
>
> **Evals use their own index** (`data/eval-index.json`, `EVAL_INDEX_PATH`): the 664-chunk index the labels were written for. Their labels assume those projects (`assumes_indexed` in each case), and `run_evals` refuses to run if one is missing. This index stays on my laptop (`data/` is in `.gitignore` and `.dockerignore`) and is never served. Re-run on it, the development set still gives Nemotron 100% / 0 false claims (`eval-main-20261108-131502.json`).
> **Live since 9 Nov 2026:** https://d3fkoasdzz9loq.cloudfront.net/ (CloudFront in front of AWS Lambda in ap-south-1, since the same evening) + Neon (Singapore), code at github.com/arshdeep3010/fitcheck. First live check through the public URL: the report streamed step by step in 13.8 s, 6 of 7 requirements met with AgentOps permalinks, Kubernetes not met. The schedule's first sync indexed agentops, portfolio-website and the CV (215 chunks). Through CloudFront: the Lambda URL answers 403 to anyone else, a POST without the body hash is refused, `/health` reports `client_ip_from: cloudfront`, and a real check streamed step by step over ~16 s.

---

## Contents

1. [Why this project exists](#1-why-this-project-exists)
2. [What a visitor sees](#2-what-a-visitor-sees)
3. [How it works](#3-how-it-works)
4. [Design decisions](#4-design-decisions)
5. [Code map](#5-code-map)
6. [Low-level design](#6-low-level-design)
7. [Models and cost](#7-models-and-cost)
8. [Run it locally](#8-run-it-locally)
9. [Evals](#9-evals)
10. [Deploy it publicly](#10-deploy-it-publicly)
11. [Security and privacy](#11-security-and-privacy)
12. [Known limitations](#12-known-limitations)
13. [Next steps, in order](#13-next-steps-in-order)
14. [Roadmap: private mode](#14-roadmap-private-mode)

---

## 1. Why this project exists

- **It closes the gap on my CV.** I list LangGraph, RAG and agentic AI. This project uses all of them in real code: a LangGraph state graph, retrieval over embeddings in pgvector, a decision model, and evals.
- **It is the same philosophy as my claim agent:** *AI-assisted, code-decided.* The model gives opinions; code decides what is claimed.
- **It is a demo, portfolio and pitch in one.** A recruiter who uses it learns about me while using it.
- **It is honest by construction.** It cannot claim a skill without pointing at code that exists. That matches my "no code, no claim" rule.

---

## 2. What a visitor sees

1. A single page with a text box: "Paste a job description."
2. They paste a JD and click **Check fit**.
3. They get:
   - an overall **fit score** (0–100%) and a one-line summary
   - **two scores**: *proven in code* and *including CV claims*
   - one row per requirement with a badge: **Met**, **Partly met**, **Not met**, **Unverified**, **Stated on CV** (not proven in code, but my CV says it), or **Not checkable** (e.g. years of experience the CV doesn't state)
   - for Met / Partly met: a **link to the exact file and lines on GitHub**, plus a code snippet
   - a confidence percentage for every row
   - a **source badge** on every cited row, so proof and claims never look alike: solid **GitHub** (code in a public repo, pinned to a commit), outlined **Code** (a local folder with no link: no longer used, since the app index is public GitHub only), dashed **CV claim** (the CV's own words). The "Stated on CV" rows sit in their own dashed, hatched box under "Not proven by code", and a legend under the scores explains the badges.
4. Nothing they paste is stored.
5. A **How it works** page (`/how-it-works`) explains the six steps, the grounding rules, *code is proof vs the CV is a claim*, the verdicts and scores, and shows a **live sync panel**: when GitHub was last checked and which commit each repo is indexed at.

---

## 3. How it works

### 3.1 Big picture

```mermaid
flowchart LR
    subgraph OFFLINE["At startup + every 30 min: GitHub sync (only new commits)"]
        R["My public GitHub repos"] --> C["Chunker<br/>60-line windows, skips secrets/.env/build"]
        C --> E1["NVIDIA embeddings<br/>(passage)"]
        E1 --> DB[("pgvector<br/>or JSON file")]
    end
    subgraph ONLINE["Per request"]
        JD["Recruiter pastes JD"] --> API["FastAPI /api/fit<br/>size limit + rate limit"]
        API --> G["LangGraph pipeline"]
        G --> REP["Fit report<br/>with GitHub links"]
    end
    DB -. search .-> G
```

### 3.2 The LangGraph pipeline

```mermaid
stateDiagram-v2
    [*] --> parse_jd
    parse_jd --> report: no requirements found
    parse_jd --> retrieve: requirements found
    retrieve --> score
    score --> ground
    ground --> check_cv
    check_cv --> report
    report --> [*]
```

| Node | What it does | Who decides |
|---|---|---|
| `parse_jd` | Nemotron (NVIDIA) turns the JD into ≤ 12 short requirements, each marked must-have or nice-to-have, and **checkable or not** (code double-checks: "1-3 years of experience" or a degree is never checkable). Falls back to simple rules if the model fails. | LLM, with code checks and a fallback |
| `retrieve` | Embeds each requirement (as a *query*) and pulls the top 5 most similar chunks of my code. With `RETRIEVAL=hybrid` (opt-in) it adds up to 3 more that only keyword search (BM25) found, e.g. `from langgraph.graph import StateGraph` for "LangGraph" (§9). | Vector search (+ keyword, opt-in) |
| `score` | **One** scorer call for the whole JD (Mercury Decide, or the free Nemotron scorer via `SCORER=llm_json`). Per requirement it asks: *met / partial / not_met?* and *which retrieved chunk proves it?* | Decision model (an opinion) |
| `ground` | Applies hard rules: no citation → unverified; low confidence → unverified. | **Code** |
| `check_cv` | Only for rows code didn't prove (not met, unverified, not checkable): searches my **profile** and asks the scorer again. A confident match becomes **Stated on CV**, never Met. | Scorer + **code** |
| `report` | Computes the score and summary deterministically. | **Code** |

### 3.3 One request, step by step

```mermaid
sequenceDiagram
    autonumber
    participant V as Visitor
    participant API as FastAPI
    participant RL as RateLimiter
    participant P as parse_jd (Nemotron)
    participant RT as retrieve (embeddings + store)
    participant S as score (Mercury Decide)
    participant GR as ground (code rules)
    participant RP as report (code)

    V->>API: POST /api/fit {jd}
    API->>API: 50 ≤ length ≤ MAX_JD_CHARS else 422/413
    API->>RL: allow(ip)?
    RL-->>API: no → 429 + Retry-After
    API->>P: jd text
    P-->>API: [r1..rN] (or heuristic fallback + warning)
    API->>RT: embed requirements (query) → top-k chunks each
    API->>S: ONE System One call: r*_status + r*_proof questions
    S-->>API: choice + confidence per question
    API->>GR: scores + retrieved evidence
    GR-->>API: verdicts (met / partial / not_met / unverified)
    API->>RP: verdicts
    RP-->>V: score, summary, rows, citations, timings
```

### 3.4 What the decision model is actually asked

For each requirement that has evidence, two **Choice** questions go into the same request:

```json
{
  "model": "inception/mercury-decide",
  "state": {
    "about": "Evidence is code and docs from the candidate's own repositories.",
    "requirements": { "r1": "REST APIs with Java and Spring Boot" },
    "evidence": {
      "r1": {
        "e0": { "source": "Healthdecissionagent/src/main/java/.../ClaimController.java:L1-L60", "text": "..." },
        "e1": { "source": "...", "text": "..." }
      }
    }
  },
  "questions": {
    "r1_status": {
      "type": "choice",
      "instructions": "Judging only from evidence.r1, does the candidate meet requirement r1: \"REST APIs with Java and Spring Boot\"?",
      "criteria": {
        "met": "The evidence clearly shows the candidate built or used this, at the level asked.",
        "partial": "The evidence shows related or smaller-scale work, but not fully what is asked.",
        "not_met": "None of the evidence shows this skill or experience."
      }
    },
    "r1_proof": {
      "type": "choice",
      "instructions": "Which item in evidence.r1 best proves requirement r1 (...)? Choose none if no item proves it.",
      "criteria": { "e0": "...ClaimController.java:L1-L60", "e1": "...", "none": "No listed item proves this requirement." }
    }
  }
}
```

The proof question can **only** choose from chunks that were really retrieved. The model cannot invent a file.

### 3.5 Grounding rules (the heart of the project)

```mermaid
flowchart TD
    S["Scorer says: status + confidence + evidence_id"] --> Q1{"status is met / partial?"}
    Q1 -->|yes| Q2{"evidence_id is a chunk retrieved<br/>for THIS requirement?"}
    Q2 -->|no| U1["UNVERIFIED<br/>'leaned yes, but cited nothing'"]
    Q2 -->|yes| Q3{"confidence ≥ MIN_CONFIDENCE?"}
    Q3 -->|no| U2["UNVERIFIED<br/>'possible match, low confidence'"]
    Q3 -->|yes| M["MET / PARTIAL<br/>+ GitHub link + snippet"]
    Q1 -->|no, not_met| Q4{"confidence ≥ MIN_CONFIDENCE?"}
    Q4 -->|yes| N["NOT MET"]
    Q4 -->|no| U3["UNVERIFIED<br/>'unclear'"]
```

**Unverified never counts as a strength.** "Not sure" must not become "yes".

Before any of this, requirements marked **not checkable** skip retrieval and scoring entirely and become **Not checkable** (the CV pass may still turn them into **Stated on CV**). This was added after the first live run, where a plain LLM marked "1-3 years of software engineering experience" as *Met* by citing a Python file.

### 3.6 Where the evidence comes from

```mermaid
flowchart LR
    GH["My public GitHub repos<br/>(auto: every non-fork, non-archived repo)"] -->|"in-app sync every 30 min<br/>only changed commits"| IDX[("Index")]
    CV["profile/profile.md<br/>(my CV, one claim per ## section)"] -->|"--profile"| IDX
    IDX -->|"source_type = code"| PROOF["Proof → Met / Partly met"]
    IDX -->|"source_type = self_reported"| CLAIM["Claim → Stated on CV"]
```

| Source | How it gets in | Shown as | Kept fresh by |
|---|---|---|---|
| **GitHub** | The running app (or `--github arshdeep3010` by hand) lists public repos, downloads each at its latest commit | Met / Partly met, solid **GitHub** badge, **permalink** to that commit | **The app itself**: at startup and every `GITHUB_SYNC_INTERVAL_MIN` (30) minutes; unchanged repos are skipped. The daily Action is an optional backup |
| ~~Local folders~~ | Not in the app any more (8 Nov): `scripts.ingest` takes folders only with `--eval`, into the local eval index | n/a | Make the repo public; the sync indexes it with links |
| **CV profile** | `--profile profile/profile.md` | **Stated on CV** only, dashed **CV claim** badge | The app re-indexes it when the file's hash changes; the Action on every push to `profile/` |

Code search and CV search never mix: every chunk has a `source_type`, and each search asks for one type. Private repos and forks are never indexed, because snippets are shown publicly.

**How the live sync works** ([`app/rag/sync.py`](app/rag/sync.py)). `GitHubSyncer.loop()` runs as a background task started in the FastAPI lifespan, when `PROVIDER_MODE=live`, `GITHUB_USER` is set and the interval is above 0.

1. List the user's public repos (1 call) and get each one's latest commit SHA (1 small call per repo).
2. Compare with the SHA stored on that repo's chunks. Unchanged → skip (no download, no embeddings).
3. Changed or new → download the zipball at that commit, chunk it, embed it, then **swap all of that repo's chunks in one step** (`VectorStore.replace_repo`: a copy-on-write dict swap in memory, one transaction in Postgres). Embedding happens first, so if NVIDIA fails, the old chunks stay.
4. A repo that is gone or private → its chunks are removed.
5. The CV file is re-indexed if its hash changed. The JSON index is saved (write + rename) only if something changed.

Measured on 7 Nov against the real GitHub account: 8 repos checked in **3.4 s**, nothing changed, so no embeddings. A failed check keeps the old index and shows the reason on the status panel. `GET /api/sync/status` is public; `POST /api/sync` forces a check now, only with `SYNC_TOKEN` as a Bearer token (constant-time compare).

**Keeping proof and claims apart, end to end.** Stored apart (`source_type`), searched apart (code first; the CV only for what code didn't prove), scored apart (CV rows never become Met and count only in the CV score), and shown apart: every `Citation` carries a `kind` (`github` / `code` / `cv`), and the UI draws each kind differently (§2).

### 3.7 Scoring formula

| Verdict | Credit |
|---|---|
| Met | 1.0 |
| Partly met | 0.5 |
| Not met / Unverified | 0 |

Must-have requirements weigh **2**, nice-to-have weigh **1**. Two scores:

- **Proven in code:** Stated on CV counts 0; requirements code can't prove (years, degrees) are left out.
- **Including CV claims:** Stated on CV counts like Met; only rows still *Not checkable* are left out.

 Score = earned ÷ total × 100. All in [`app/report.py`](app/report.py), no model involved.

---

## 4. Design decisions

| Decision | Why |
|---|---|
| **Code owns the verdict** | The model's answer is an opinion. Grounding rules decide what's claimed. Same pattern as the claim agent. |
| **A decision model for scoring, not a chat LLM** | Mercury Decide returns a *choice* with a *calibrated probability*, instead of free text I'd have to parse and trust. Confidence lets code route: claim, or mark unverified. |
| **Proof must be chosen from retrieved chunks** | Citations can't be hallucinated: the options *are* the retrieved chunk ids. Grounding double-checks the id belongs to that requirement. |
| **One System One call per JD** | Questions run in parallel on the model side; one call keeps latency low and stays inside free-tier rate limits. |
| **Fail closed** | If retrieval or scoring is down, every row becomes Unverified with a warning. An outage never becomes "not met" or "met". |
| **JD is untrusted input** | A visitor can paste prompt-injection text. It can only affect the *list* of requirements, and a "met" still has to cite a real retrieved snippet, shown next to the row. Honest limit: retrieval has no similarity floor, so a crafted requirement could get a confident "met" citing a real but unrelated snippet. The visible snippet makes that easy to spot; a similarity floor and an injection eval case would close it (§13). |
| **Sync inside the app, not only a cron job** | A project pushed today should be citable today, with no redeploy and no extra service. Checking a SHA is cheap, so polling every 30 min stays inside GitHub's 60/hour no-token limit. |
| **Swap a repo in one step** | Requests keep running during a sync. Replacing a repo's chunks atomically means a check never sees half a repo. |
| **Hybrid retrieval, and keyword search can only add** | Product names ("AutoGen, CrewAI, or LangGraph") embed close to nothing in particular, so vector search missed `app/graph.py`. Keyword search finds names. But plain rank fusion would push out vector hits #4–5, and common words brought noise, so keyword hits are *extra* (≤ 3), and only rare terms (≤ 3% of chunks) search. **Off by default:** with the Nemotron scorer it raised held-out false claims from 0% to 8–15%, and false claims matter more than recall here. |
| **Exact search for the CV, HNSW for code** | HNSW filters *after* finding its nearest ~40 rows. CV sections are ~1% of rows, so a filtered HNSW search for them came back nearly empty (measured, §6.3). The CV has a handful of rows, so exact search is cheap; code search keeps HNSW plus pgvector's iterative scan. |
| **Proof and claims look different** | A reader skimming the report must never mistake a CV line for proven code. Different badges, a separate dashed box, and two separate scores. |
| **No-index marker + secret filter** | Files that *quote* requirements (sample JDs, prompts, tests) are not evidence. Chunks that look like credentials are never indexed, because snippets are shown publicly. |
| **Strategy pattern everywhere** | Chat, embeddings, scorer and store are interfaces. Swap Mercury Decide ↔ Jev ↔ LLM, or JSON ↔ pgvector, by config. That's also what makes the evals possible. |
| **Offline "fake" mode** | Tests and CI need no keys and no network. Anyone can run the project in 1 minute. |
| **Deterministic report** | The score and summary are arithmetic, so they're reproducible and explainable. |

---

## 5. Code map

```
fitcheck/
├── app/
│   ├── main.py            FastAPI app: /, /how-it-works, /health, /api/fit(+/stream), /api/sync(/status); guards
│   ├── config.py          All settings from env vars (.env)
│   ├── deps.py            Builds providers for "live" or "fake" mode (the only place that knows)
│   ├── graph.py           LangGraph state + nodes + edges; fail-closed handling; timings
│   ├── models.py          Pydantic domain models (Requirement, Chunk, Evidence, Verdict, FitReport…)
│   ├── jd_parser.py       JD → requirements (LLM, heuristic fallback)
│   ├── grounding.py       Opinions → verdicts (the hard rules) + GitHub link builder
│   ├── report.py          Score + summary, deterministic
│   ├── ratelimit.py       Per-IP sliding window, daily cap (in memory), X-Forwarded-For + Lambda source IP
│   ├── shared.py          The same limits + the sync status in Postgres, shared by every Lambda instance
│   ├── llm/
│   │   ├── http.py        POST with auth, retries + exponential backoff on 429/5xx/529
│   │   ├── chat.py        OpenAI-compatible chat (NVIDIA Nemotron), JSON extraction
│   │   ├── embeddings.py  NVIDIA embeddings (query/passage) + offline HashEmbedder
│   │   └── system_one.py  System One client (Mercury Decide / Jev)
│   ├── rag/
│   │   ├── chunker.py     Repo → line-ranged chunks; skips secrets, env, build dirs, no-index files
│   │   ├── store.py       VectorStore: InMemoryStore (JSON, copy-on-write) + PgVectorStore (pool, HNSW, exact CV search, BM25 in SQL)
│   │   ├── keywords.py    Tokenizer + BM25 for keyword search (shared by both stores)
│   │   ├── hybrid.py      retrieve(): vector top k + keyword-only extras, RRF order
│   │   ├── github.py      list public repos, head SHA, safe zipball download
│   │   ├── ingest.py      local folders, GitHub sync, profile → embed → replace_repo (incremental)
│   │   └── sync.py        GitHubSyncer: background sync loop + public status
│   ├── scoring/
│   │   ├── __init__.py    FitScorer interface
│   │   ├── system_one.py  Mercury Decide scorer (build_request / parse_answers)
│   │   ├── llm_json.py    Plain-LLM scorer, for comparison in evals
│   │   └── keyword.py     Offline baseline scorer
│   └── static/
│       ├── index.html         The paste-a-JD page (vanilla JS, no HTML injection)
│       └── how-it-works.html  How it works + live sync panel
├── scripts/ingest.py      CLI: --github USER, --profile FILE, --rebuild; local folders only with --eval
├── evals/
│   ├── dataset.jsonl      Development set: 25 hand-labelled requirements in 4 cases
│   ├── heldout.jsonl      Held-out set: 31 requirements from 4 real postings, labelled with evidence, frozen by hash
│   └── run_evals.py       Accuracy, false-claim rate, coverage, latency per scorer
├── profile/profile.md     My CV as self-reported claims (public)
├── tests/                 113 tests: 105 offline + 8 against real pgvector (skipped without TEST_DATABASE_URL)
├── Dockerfile             Non-root, $PORT-aware (any container host)
├── lambda.Dockerfile      Same app on the Lambda base image + Lambda Web Adapter
├── docker-compose.yml     Postgres + pgvector + app
├── infra/                 Terraform: ECR, Lambda + streaming URL, EventBridge sync, SSM, OIDC deploy role, budget
└── .github/workflows/
    ├── ci.yml             every test, with a real pgvector service
    └── deploy.yml         after ci on main: build, push to ECR, update Lambda, sync, check /health
```

---

## 6. Low-level design

### 6.1 Class diagram

```mermaid
classDiagram
    direction LR
    class Deps {
        +JdParser parser
        +Embedder embedder
        +VectorStore store
        +FitScorer scorer
        +Settings settings
    }
    class JdParser {
        +parse(jd) (list~Requirement~, warnings)
    }
    class ChatModel {
        <<interface>>
        +complete(system, user) str
    }
    class OpenAICompatibleChat
    class Embedder {
        <<interface>>
        +embed(texts, kind) list~vector~
    }
    class OpenAICompatibleEmbedder
    class HashEmbedder
    class VectorStore {
        <<interface>>
        +upsert(chunks, vectors)
        +search(vector, k, source_type) list~Evidence~
        +replace_repo(repo, chunks, vectors)
        +repo_summary() list
        +indexed_refs() dict
        +count() int
        +clear()
    }
    class GitHubSyncer {
        +run_once() stats
        +loop()
        +status() dict
    }
    class InMemoryStore
    class PgVectorStore
    class FitScorer {
        <<interface>>
        +score(requirements, evidence) list~RequirementScore~
    }
    class SystemOneFitScorer
    class LlmJsonFitScorer
    class KeywordFitScorer
    class SystemOneClient {
        +ask(state, questions) answers
    }

    Deps --> JdParser
    Deps --> Embedder
    Deps --> VectorStore
    Deps --> FitScorer
    JdParser --> ChatModel
    ChatModel <|.. OpenAICompatibleChat
    Embedder <|.. OpenAICompatibleEmbedder
    Embedder <|.. HashEmbedder
    VectorStore <|.. InMemoryStore
    VectorStore <|.. PgVectorStore
    FitScorer <|.. SystemOneFitScorer
    FitScorer <|.. LlmJsonFitScorer
    FitScorer <|.. KeywordFitScorer
    SystemOneFitScorer --> SystemOneClient
    LlmJsonFitScorer --> ChatModel
    GitHubSyncer --> VectorStore
    GitHubSyncer --> Embedder
```

### 6.2 Data model

```mermaid
classDiagram
    class Requirement {
        id
        text
        must_have
    }
    class Chunk {
        id
        repo
        path
        start_line
        end_line
        text
    }
    class Evidence {
        chunk
        similarity
    }
    class RequirementScore {
        requirement_id
        status
        confidence
        evidence_id
    }
    class RequirementResult {
        requirement
        verdict
        confidence
        citation
        note
    }
    class Citation {
        source
        url
        snippet
        kind: github | code | cv
        repo
        commit
    }
    class FitReport {
        candidate
        score
        score_with_cv
        summary
        results
        warnings
        timings_ms
        models
    }
    Evidence --> Chunk
    RequirementResult --> Requirement
    RequirementResult --> Citation
    FitReport --> RequirementResult
```

`RequirementScore` is the model's **opinion**. `RequirementResult` is the **verdict** after grounding. Keeping them as separate types makes it impossible to accidentally show an un-grounded opinion.

### 6.3 Database (pgvector mode)

```sql
CREATE TABLE chunks (
  id text PRIMARY KEY,            -- "repo:path:start-end"
  repo text, path text, start_line int, end_line int,
  text text,
  source_type text,               -- 'code' (proof) | 'self_reported' (CV claim)
  title text,                     -- CV section heading
  ref text,                       -- commit SHA (GitHub) or file hash (CV)
  url_base text,                  -- https://github.com/<user>/<repo> for synced repos
  embedding halfvec(<dim>),       -- dim from the embedding model at ingest (2048 for Nemotron)
  terms jsonb,                    -- {"langgraph": 1, "stategraph": 2, ...}: term counts for keyword search
  n_terms int                     -- chunk length in terms (BM25 length normalisation)
);
CREATE INDEX chunks_embedding_hnsw ON chunks USING hnsw (embedding halfvec_cosine_ops);
CREATE INDEX chunks_terms_gin ON chunks USING gin (terms);
-- keyword search: rows WHERE terms ?| $terms (GIN), scored by the same BM25 formula as keywords.bm25
-- code search: WHERE source_type='code' ORDER BY embedding <=> $q LIMIT k   (HNSW + iterative scan)
-- CV search:   WHERE source_type='self_reported' ORDER BY 1 - (embedding <=> $q) DESC LIMIT k   (exact)
```

**Why `halfvec`:** pgvector's HNSW index supports `vector` only up to 2000 dimensions, and Nemotron embeddings have 2048. `halfvec` (16-bit floats) indexes up to 4000 dimensions and halves storage, which matters on a free-tier database. The precision loss is negligible for cosine ranking.

**The filtered-search bug (found and fixed 7 Nov).** HNSW returns its ~40 nearest rows (`hnsw.ef_search`) and *then* applies `WHERE source_type = ...`. CV sections are 8 of 719 rows (~1%), so the CV search usually found nothing. Measured on pgvector 0.8.7 with 800 code + 8 CV rows, forcing the index plan a big table would get: the old query returned **0–2 of the 3 rows asked for, and 0 in 9 of 20 searches**. Fix: CV search orders by an expression the index can't serve (`1 - distance DESC`), so it's an exact scan over a handful of rows; code search turns on `hnsw.iterative_scan = strict_order` (pgvector 0.8+) so the filter can't starve it. Both return 3 of 3 every time. `tests/test_pgvector.py` forces the index plan (`enable_seqscan = off`); with the old query it fails (1 of 3), with the fix it passes. A tiny test table never showed the bug, because Postgres just scans it without the index.

**Keyword columns.** Term counts are computed in Python, by the same tokenizer the in-memory store uses, and stored per row. The SQL runs the same BM25 formula, and a test checks that both stores return the same ranking. A table indexed before hybrid retrieval gets the columns added and filled when the app starts, so there's no manual migration.

**Connections.** A `psycopg_pool.ConnectionPool` (1–4 connections) checks each connection before handing it out and recycles idle ones after 5 min. Neon suspends an idle database and drops its connections; with the old single connection every search would have failed until a restart. A test kills all server connections and checks the next query still works.

### 6.4 Patterns used

| Pattern | Where |
|---|---|
| Strategy | `ChatModel`, `Embedder`, `VectorStore`, `FitScorer` |
| Dependency injection | `Deps` built once in `deps.py`, passed into `build_graph` |
| Pipeline / state machine | LangGraph `StateGraph` with reducers for warnings and timings |
| Fail-closed / graceful degradation | `retrieve` and `score` nodes; heuristic JD fallback |
| Retry with exponential backoff | `llm/http.py` (429, 5xx, 529 only) |
| Factory | `build_store`, `build_embedder`, `build_chat`, `build_deps` |

---

## 7. Models and cost

| Step | Model | Provider | Cost |
|---|---|---|---|
| JD → requirements | `nvidia/nemotron-3-ultra-550b-a55b` | NVIDIA API (build.nvidia.com) | Free tier |
| Embeddings | `nvidia/nemotron-3-embed-1b` (2048-d) | NVIDIA API | Free tier |
| Fit scoring (default today) | `nvidia/nemotron-3-ultra-550b-a55b` as a JSON scorer (`SCORER=llm_json`) | NVIDIA API | Free tier |
| Fit scoring (calibrated) | `inception/mercury-decide` (`SCORER=system_one`) | OpenRouter System One API | $0.02 / M input tokens, output free ≈ **₹0.03 per check**; needs OpenRouter credits |
| Database | Postgres + pgvector | Neon / Supabase free tier | ₹0 |
| Hosting | AWS Lambda (container image) behind CloudFront, Terraform in `infra/` | Lambda always-free allowance | ₹0 to a few rupees a month (§10) |

Every model is one env var, so if a free model disappears, change one line. That already happened once: `inception/mercury-decide:free` was withdrawn in Nov 2026, so the scorer became a switch (`SCORER`), and the app runs free on Nemotron until there are OpenRouter credits.

**When OpenRouter credits run out (`SCORER=system_one`):** OpenRouter answers `402` (no credits) or `429 free-models-per-day` (free daily quota used up). `app/llm/http.py` raises `CreditsExhaustedError` for both and doesn't retry them, because retrying can't help (a normal `429` rate limit is still retried). The graph then:
- scores that check with the free fallback (`SCORER_FALLBACK=llm_json`, Nemotron), and the report says so: *"The main scorer (inception/mercury-decide) is out of credits right now, so this check was scored by the backup model (nvidia/nemotron-...)."* `models.scorer` in the report names the model that actually judged.
- stops asking OpenRouter for `CREDIT_PAUSE_MINUTES` (15), then tries once more, because credits may have been topped up and the daily quota resets at midnight UTC. `/health` shows `scorer_status.out_of_credits` and `retry_in_s`.
- logs the fix for the owner (top up, or `SCORER=llm_json`). The visitor-facing notice has no account details.
- with `SCORER_FALLBACK=` (empty), fails closed: every row Unverified, "The fit scorer is out of credits right now; nothing is claimed."

The eval runner never falls back: an eval must measure the scorer it names, so it stops with "Out of credits, no results written". Tested in `tests/test_credits.py`, and once live: OpenRouter faked to answer 402, real Nemotron took over and the report said so.

**Stricter scoring rules (added after live runs):** the parser splits technology lists ("Spring Security, OAuth2 and JWT" → three rows, "Flyway or Liquibase" stays one), and both scorers are told a requirement is *met* only if every part is evidenced, that a related technology isn't the same one (Docker ≠ Kubernetes, one service ≠ microservices) and that a mention isn't experience. Before this, Nemotron marked OAuth2 as met because JWT was present; after it, OAuth2 is correctly *not met*.

**8 Nov: examples taken out of the prompts.** Both scorers' instructions used to give examples that are *in the eval set* (Docker/Kubernetes, microservices, LangGraph/LangChain, "Spring Security, OAuth2 and JWT"), which flatters the evals. They now use technologies that appear nowhere in `evals/dataset.jsonl` (Flask/Django, MySQL/PostgreSQL, Jenkins/GitHub Actions, "Flask, Celery and RabbitMQ"). Mercury's `partial` criterion was also fixed (§9).

**Why not `nemotron-3.5-embed-1b`?** On the free tier it returns `500 service inference failed` whenever one request carries several long code chunks (single short texts work). `nemotron-3-embed-1b` batches fine. The embedder also splits a failing batch in half and retries, so either model works, just slower. `CHAT_THINKING=true` turns on Nemotron's reasoning mode: better on messy JDs, much slower.

Free tiers are rate limited. A check makes **3 model calls** (1 chat, 1 embeddings batch, 1 scorer call), plus **2 more** (embeddings + scorer) when some rows go to the CV pass. That's why the public demo allows 20 checks per IP per hour and **200 per day in total** (`DAILY_LIMIT`), so rotating IPs can't burn the free quotas.

---

## 8. Run it locally

### 8.1 Offline in 1 minute (no keys)

```bash
cd fitcheck
python -m venv .venv
.venv\Scripts\activate            # Windows (macOS/Linux: source .venv/bin/activate)
pip install -r requirements-dev.txt
set PROVIDER_MODE=fake             # PowerShell: $env:PROVIDER_MODE="fake"
python -m pytest -q                # 70 passed, 3 skipped (the pgvector tests, §8.3)
python -m scripts.ingest --github arshdeep3010 --profile profile/profile.md
uvicorn app.main:app --reload      # open http://localhost:8000
```

Offline mode uses a hashing embedder and a keyword scorer. It proves the plumbing, **not** the quality.

### 8.2 Live (real models)

1. Copy `.env.example` to `.env`.
2. Put your **NVIDIA** key in `NVIDIA_API_KEY` and your **OpenRouter** key in `OPENROUTER_API_KEY`.
3. Set `PROVIDER_MODE=live`.
4. Rebuild the index with real embeddings (an index built in fake mode can't be reused: different vectors):
   ```bash
   # everything public on GitHub + my CV (re-runs only re-index what changed)
   python -m scripts.ingest --github arshdeep3010 --profile profile/profile.md
   # the eval index (evals only, never served): the labels also assume projects that aren't public
   python -m scripts.ingest --eval --github arshdeep3010 --profile profile/profile.md ../Healthdecissionagent ../vibe-audit ../linkedin-bot .
   ```
   The app index holds only public GitHub repos and the CV: every claim must link to code a recruiter can open. Local folders go only into the eval index (`--eval`); without it, `scripts.ingest` refuses them.
5. `uvicorn app.main:app --reload`.

### 8.3 With Postgres (like production)

```bash
docker compose up -d db
# in .env: VECTOR_STORE=pgvector, DATABASE_URL=postgresql://fitcheck:fitcheck@localhost:5432/fitcheck
python -m scripts.ingest --github arshdeep3010 --profile profile/profile.md
docker compose up --build app
```

Test the pgvector store against a real database (what the 3 skipped tests need):

```bash
docker run -d --name fitcheck-pgtest -e POSTGRES_PASSWORD=test -p 55432:5432 pgvector/pgvector:pg16
TEST_DATABASE_URL=postgresql://postgres:test@localhost:55432/postgres python -m pytest -q tests/test_pgvector.py
docker rm -f fitcheck-pgtest
```

---

## 9. Evals

**Goal:** prove with numbers that the system doesn't over-claim.

| Metric | Meaning | Target |
|---|---|---|
| **false_claim_rate** | Expected *not met*, but reported met/partial | **≈ 0%** (the most important number) |
| accuracy | Final verdict equals the label (unverified counts as wrong) | as high as possible |
| coverage | Share of requirements not left unverified | high, but never at the cost of false claims |
| avg_latency_ms | Scoring time per JD | for the README |

```bash
python -m evals.run_evals                       # all scorers available in this mode
python -m evals.run_evals --scorers keyword system_one llm_json
```

**Dataset format** (`evals/dataset.jsonl`, one JD per line):

```json
{"id": "java-backend", "assumes_indexed": ["Healthdecissionagent"],
 "requirements": [{"text": "REST APIs with Java and Spring Boot", "must_have": true, "expected": "met"}]}
```

The dataset has 4 cases / 25 requirements (12 expected *met*, 13 expected *not met*), including a **near-misses** case: each "not met" there has something nearby in the code that a careless scorer mistakes for the real thing (Docker ≠ Kubernetes, LangGraph ≠ LangChain, React ≠ Angular, a comment saying "would need Redis" ≠ Redis experience). **Grow it to 30–50 JDs**: copy real JDs you apply to and label each requirement honestly against your code.

**Measured 8 Nov 2026** (live NVIDIA embeddings, 719 chunks: 8 repos synced from GitHub + 4 local folders + the CV; 25 requirements; current prompts):

```
scorer                                         accuracy  false claims  coverage   latency
keyword-baseline                                  52%         54%        100%        2ms
inception/mercury-decide                         100%          0%        100%     2296ms   (2 runs, both 100%)
llm-json:nvidia/nemotron-3-ultra-550b-a55b       100%          0%        100%     3468ms
```

Results files: `eval-20261108-090936.json` (all three) and `eval-20261108-090954.json` (Mercury again), local and gitignored. Mercury's latency in the first file is wrong (7.7 million ms average): that run was started at night and one request stayed open while the laptop slept, so its latency is discarded; 2.3 s is from the second run. Accuracy was identical in both runs.

**What changed between 7 and 8 Nov: one sentence in Mercury's criteria.** On 7 Nov the `partial` criterion said "only some of the parts listed, **or a related but different technology (e.g. Docker for Kubernetes, one service for microservices)**". So a Docker snippet for a Kubernetes requirement was, by *our own definition*, partial: Mercury did what it was told, and the definition disagreed with the labels. Now `partial` means only "some of the named parts are shown", and a related-but-different technology is explicitly `not_met`. With `model_status` now logged, the four near misses show the change directly:

| Requirement | 7 Nov (old criteria) | 8 Nov (new criteria) |
|---|---|---|
| Container orchestration with Kubernetes | leaned yes, no proof → Unverified | `not_met`, 0.99 |
| Microservices with Spring Cloud | leaned yes, no proof → Unverified | `not_met`, 0.95 |
| Authentication with OAuth2 | leaned yes, no proof → Unverified | `not_met`, 0.99 |
| Microservices architecture | leaned yes, no proof → Unverified | `not_met`, 0.94 |

On 8 Nov grounding demoted **nothing**, for either model: every final verdict is the model's own answer. Confidence still behaves differently: Mercury used 14–16 distinct values (0.82–1.00), Nemotron 2 (0.90 and 0.95).

**Be honest about this result:** the fix was found by looking at these same 25 rows, so 100% on them is optimistic. The examples in the prompts are not from the eval set, but the *idea* of the fix came from it. The real test is a fresh, held-out set of JDs (§13).

**Held-out set, 8 Nov — the number to quote.** `evals/heldout.jsonl`: 4 real postings read on 8 Nov from hirist.tech (SOFTPATH *GenAI Engineer*, LeadSquared *Lead AI/ML Backend Engineer*, Stellaspire *Lead Backend Java Developer*, Boost Tech *Full Stack Developer*), **31 requirements: 16 met, 2 partly met, 13 not met**. None of the text overlaps `dataset.jsonl`.

How it was made, so the result means something:
1. Requirements taken from the postings in their own words, one skill per item.
2. **Labelled by reading the indexed code, before any scorer ran.** Every label stores its evidence (the file that proves a *met*, or why a *not met* has nothing, e.g. "only a skill keyword list in redrob-ranker/src/config.py mentions it"). 13 items were left out and listed with the reason: 9 were genuinely arguable ("LangChain, LlamaIndex, **or similar**", "Hugging Face" when only sentence-transformers is used, "guardrails"), 4 were duplicates.
3. Frozen at SHA-256 `6968cb0dc71ce5ded2d5e73d422e7bc736260bd2f4af2af6d47a8a793a4fa8ea`, then each scorer run **once**. The results file stores that hash and the index snapshot (repos, commits, chunk counts).

```
dataset heldout (sha256 6968cb0dc71c), 719 chunks from 13 sources
scorer                                         accuracy  false claims  overclaims  coverage   latency
keyword-baseline                                  55%          54%           7       100%        2ms
inception/mercury-decide                          97%           0%           0       100%     2943ms
llm-json:nvidia/nemotron-3-ultra-550b-a55b        90%           0%           1        94%     6279ms
```

(`overclaims` = a verdict stronger than the label: not met → met/partial, or partial → met.)

**What went wrong, row by row:**

| Scorer | Requirement | Label | Model said | Final | Why |
|---|---|---|---|---|---|
| Mercury | Experience with AutoGen, CrewAI, or LangGraph | met | not met, 0.99 | Not met | **Retrieval miss.** `fitcheck/app/graph.py` wasn't in the top 5 (all 5 hits ≈ 0.20 similarity). The only LangGraph evidence it saw was the README's stack line, and "a doc that only mentions it does not count". The scorer followed the rules; search didn't find the code. |
| Nemotron | Strong programming skills in Python | met | met, 0.95 | Unverified | Leaned yes but gave no valid evidence id, so grounding didn't claim it. |
| Nemotron | OCR technologies and document processing | partly met | met, 0.90 | Unverified | Would have been an overclaim (there's no OCR); no valid citation, so grounding caught it. |
| Nemotron | Spring Boot and microservices architecture | partly met | met, 0.95 | **Met** | **The one overclaim.** It cited `Healthdecissionagent/ARCHITECTURE.md` L51–110, which describes one Spring Boot service's modules and never mentions microservices. Grounding can't catch this: the citation was real and the confidence high. |

What this changes:
- **On fresh data, Mercury did better than Nemotron** (97% vs 90%, 0 vs 1 overclaim, 2.9 s vs 6.3 s). On the development set they were tied at 100%, so the development set was hiding the difference.
- **Both had zero false claims on not-met rows.** The weak spot is *partly met*: Nemotron rounded both partial rows up to met; Mercury got both right.
- **Grounding stopped 2 of Nemotron's 3 mistakes** (no valid citation → Unverified). The third shows its limit: a real but weak citation passes.
- **Confidence is not proof of calibration:** Mercury's only miss came with 0.99 confidence. It's a retrieval miss rather than a judgement error, but a threshold wouldn't have caught it.
- **The next fix is retrieval, not the prompt.** A requirement that lists framework names embeds poorly (~0.20 similarity to everything). Adding keyword search next to vector search (hybrid retrieval) should find `graph.py`. That has to be developed on the main set, and then tested on a *new* held-out set: this one has now been seen.

**Hybrid retrieval, 8 Nov (developed on the development set only).** The held-out miss above was a search failure, so the fix is in search, not in a prompt.

What it does (`app/rag/hybrid.py`, `app/rag/keywords.py`):
- **Vector search** as before (top 5), **plus keyword search** (BM25 over the chunk text and file path).
- **Keyword search can only add evidence.** The scorer sees the full vector top 5 plus up to 3 chunks that only keyword search found (`KEYWORD_EXTRA=3`). Vector-only evidence is a strict subset of hybrid evidence, so a proving file can't be pushed out.
- **Only specific terms search**: a term found in more than 3% of chunks is skipped. "langgraph" is in 2 of 711 code chunks and "kafka" in 3, while "container" is in 41 and "python" in 67. Common words are vector search's job.
- The merged list is ordered by reciprocal rank fusion (RRF, k = 60), which compares ranks, not scores: a cosine of 0.2 and a BM25 of 5.1 aren't on the same scale. `Evidence.similarity` stays the real cosine for every hit, and `Evidence.via` records which search found it.
- `RETRIEVAL=hybrid` switches it on; the default is `vector` (see "Final check" below for why). The CV search stays vector-only.

**The first version was worse, so I changed it** (still on the development set): plain RRF, keeping the fused top 5. It changed the evidence for 24 of 25 requirements. 42 of the 125 evidence slots went to chunks only keyword search found, and many were noise. "Container orchestration with Kubernetes" pulled in a front-end `elements.js` through the word "container". Plain RRF top 5 also *always* drops vector hits #4 and #5 when keyword search finds two other chunks, so a proving file at vector rank 4 would be lost. Nemotron still scored 100% with it, but I didn't want correctness to rest on the scorer ignoring noise.

```
dataset main (sha256 472cd80d07cf), 719 chunks from 13 sources, 8 Nov
retrieval                    scorer               accuracy  false claims  overclaims  latency
vector (before)              keyword-baseline        52%         54%           7         1ms
                             Nemotron (llm_json)    100%          0%           0      5799ms
hybrid v1 (RRF top 5)        keyword-baseline        48%         92%          12         2ms
                             Nemotron (llm_json)    100%          0%           0      4342ms
hybrid (final)               keyword-baseline        40%         92%          12         2ms
                             Nemotron (llm_json)    100%          0%           0      8888ms
```

Results files: `eval-main-20261108-104308.json` (vector), `-104340` (v1), `-104709` (final). **Mercury Decide could not be measured:** from 10:43 on 8 Nov, OpenRouter answered `404 No endpoints found for inception/mercury-decide`, and the model was no longer in its public model list (461 models).

How to read it:
- **The development set is saturated for Nemotron** (100% before and after), so it can only show that nothing broke. It does show that: no false claims, no overclaims, in two hybrid runs.
- **The keyword baseline got worse** (54% → 92% false claims). Hybrid search shows the scorer more chunks that *mention* a skill, such as a skill keyword list. A scorer that counts mentions then claims more. That is the risk hybrid search adds, and why the scorer's rule "a mention is not experience" matters more now.
- **Latency rose** to 8.9 s, from 4.3–5.8 s in the two runs before it. The scorer now reads up to 8 chunks instead of 5, but free-tier latency also varies a lot (3.5–14 s across earlier runs), so this is not a clean measurement.

**Retrieval check on the held-out set (a diagnostic, not a score).** For the 12 held-out "met" rows whose label names a file, I checked whether that file was retrieved. No scorer ran, and nothing was changed after looking. **Vector: 6 of 12. Hybrid: 7 of 12.** The row that changed is "AutoGen, CrewAI, or LangGraph": `fitcheck/app/graph.py` is now retrieved. The other 5 misses are generic requirements ("Strong programming skills in Python", "React"), where the label names one example file but other valid proof was retrieved; Mercury got all 5 right on the held-out run.

**Still open:** the real test is a **new** held-out set, scored once, with Mercury once it's back on OpenRouter. This held-out set was used to find the miss, so it can't judge the fix.

**Final check, 8 Nov: hybrid made Nemotron over-claim, so it is off by default.** Before finalising I ran Nemotron on the held-out set with both retrieval modes, three runs each (the first hybrid run was before, and the other runs after, the prompt change below). **This set was already seen, so these are not clean held-out numbers**; the frozen run above stays the number to quote. They are only a fair *comparison of the two modes*: the same set, model, prompt and index, run back to back.

```
held-out set, Nemotron (llm_json), 719 chunks, 8 Nov 11:17-11:21
retrieval   runs   accuracy        false claims (of 13 not-met)   overclaims
vector       3     90%, 87%, 87%   0, 0, 0                        1, 1, 2
hybrid       3     94%, 90%, 87%   1, 2, 2   (8%, 15%, 15%)       2, 3, 3
```

What went wrong with hybrid:
- **PyTorch → met**, citing `redrob-ranker/src/config.py`: a Python set of skill names (`"PyTorch", "TensorFlow", ...`), with no PyTorch code anywhere in the index. Only keyword search had retrieved it. This is the risk named above: keyword search finds chunks that only *name* a skill.
- **LLM tool and function calling → met**, citing `vibe-audit/backend/app/llm/client.py`, a plain chat client. No chunk in the index uses `tools` or `tool_calls`. That chunk was a *vector* hit, which vector mode also showed, yet Nemotron didn't claim it there. With 8 chunks instead of 5 it was more willing to say yes.
- **OpenTelemetry or APM tracing → partly met**, citing a correlation-id filter (once).

I tried one prompt fix first. Nemotron's rule "a comment or doc that only mentions a technology is not experience" became "text that only **names** a technology (a comment, a doc, a README stack line, or a list of names in code) is not experience; only code that imports, calls, configures or runs it counts". It's a general rule, but I wrote it after seeing a held-out row. It didn't fix hybrid (2 false claims in both runs after it). With it, the development set stays at 100%, and vector mode makes 0 false claims, so it stays in. Mercury's criteria are unchanged, because Mercury couldn't be run to test a change.

**Decision: `RETRIEVAL=vector` is the default. Hybrid stays in the code as an opt-in** (`RETRIEVAL=hybrid`). False claims are the number this project must keep at about 0. Hybrid trades them for recall with the default scorer: it fixes the LangGraph miss but adds 1–2 false claims per run. It should be re-tested with Mercury, which made 0 overclaims on near misses, on a new held-out set.

**Index on 8 Nov 11:21:** the live sync removed 4 repos that GitHub no longer serves (404: `ai-landing-3d-demo`, `optiextract-file-uploader`, `unicall`, `Meet-the-strangers`), as designed, because snippets are shown publicly. That left 664 chunks from 9 sources. On that index, with the final defaults (vector retrieval, current prompt), the development set gives keyword 52% / 54% false claims and **Nemotron 100% / 0% / 0 overclaims, 4.1 s** (`eval-main-20261108-112411.json`). The held-out labels still hold on it, because none of their evidence was in the removed repos.

**Before the fix, 7 Nov** (all three scorers in one run, `eval-20261107-231944.json`): keyword 52% / 54% / 3 ms; Mercury Decide **84% accuracy, 0% false claims, 84% coverage**, 1.7 s; Nemotron 100% / 0% / 4.6 s (its prompt still named some eval items then).

**What Mercury Decide did on 7 Nov.** It got 21 of 25 right and made **no false claims**. The 4 it missed are all near misses that should be *not met*: Kubernetes, Spring Cloud microservices, OAuth2, and microservices architecture. Each came back **Unverified with confidence ≥ 0.5 and no citation**. Under the grounding rules (§3.5) that combination has one cause: the model leaned *met* or *partly met* but answered "none" to "which snippet proves it?". So **Mercury leaned yes on four skills I don't have, and the citation rule stopped all four from becoming claims.** That's the grounding layer doing its job on a real model, not only in unit tests. (This is worked out from the rules, not logged: `run_evals.py` now also records each row's `model_status` before grounding, so the next run shows it directly.)

**Confidence on 7 Nov, Mercury vs Nemotron** (25 rows, so indicative only):

| | Mercury Decide | Nemotron (JSON) |
|---|---|---|
| Distinct confidence values | 21 | 2 (0.90 and 0.95) |
| Range | 0.52 – 1.00 | 0.90 – 0.95 |
| Mean when right | 0.97 | 0.94 |
| Mean when it missed | 0.79 | — (no misses) |

Mercury's confidence moves: it was lower on the rows it got wrong. Nemotron's is close to a constant, so a threshold on it can't separate good answers from bad ones. 25 rows can't prove calibration; it needs a bigger set and a reliability curve.

**Trade-off now (8 Nov):** both scorers are 100% / 0% on this set. Mercury is faster (2.3 s vs 3.5 s per JD in this run) and its confidence varies with the evidence; Nemotron is free. Default stays `llm_json`; `system_one` costs about ₹0.03 per check. On 7 Nov, before the criteria fix, both made zero false claims too, because grounding decides what is claimed.

**What happened on the way.** The first 7 Nov run, on the old 24 labels, gave Nemotron **96% accuracy and 8% false claims**: it marked "Frontend development with React" as *met*. That label was *not met* when only 4 local repos were indexed. The GitHub sync had since added `frontdesk-ai-supervisor`, whose `frontend/src/index.js` is a real React entry point, so the model was right and the label was stale. I relabelled React as *met* (the reason is stored in the dataset as `label_note`), added **Angular** as the new near miss (React code nearby, no Angular), and re-ran: the table above. Lesson: **labels depend on the index**, so every eval run must record what was indexed.

**Earlier run, 4 Nov** (492 chunks from 4 local repos, 24 requirements): keyword 50% / 54% false claims / 4 ms; Nemotron 100% / 0% / 5.7 s.

How to read it honestly:
- The keyword baseline is the floor: it claims **more than half of the skills I don't have**. That's why grounding and a real scorer matter.
- Nemotron was perfect on 25 requirements, but **25 is small**, and in a full-JD run it once marked "Familiarity with Kubernetes" as *partly met* because of a Docker sandbox. Its confidence is ~0.95 on nearly everything, so it isn't calibrated.
- Mercury's 7 Nov 84% was a coverage cost, not a wrong claim: every miss became Unverified. A threshold couldn't fix it (the model leaned yes); fixing the status definition did.
- Nemotron's latency varies a lot on the free tier: 3.5 s to 14.0 s per JD across four runs in two days.

---

## 10. Deploy it publicly

**Target: AWS Lambda, with Terraform and GitHub Actions** (`infra/`, step by step in [infra/README.md](infra/README.md)). It's designed for the public: no login, nothing stored, rate limited.

```
visitor ──HTTPS──> CloudFront (edge; caches the 2 pages) ──signed (OAC)──> function URL (AWS_IAM, streaming)
                                                              └──> Lambda: Lambda Web Adapter -> uvicorn -> FastAPI
EventBridge, every 30 min ──{"action":"sync"}──> same function -> POST /events -> GitHub sync
Neon Postgres + pgvector: index, shared rate limits, sync status · SSM: keys, database URL, sync token
```

- **One function, the same app.** `lambda.Dockerfile` puts the FastAPI app on the Lambda base image with the **Lambda Web Adapter**, which turns Lambda requests into HTTP for uvicorn and streams the response back, so the step-by-step report (SSE) works through a `RESPONSE_STREAM` function URL. Locally the adapter does nothing.
- **CloudFront in front** (`infra/cloudfront.tf`). Visitors connect to an edge location near them (price class 200 includes India). The function URL is `AWS_IAM`: it answers only requests CloudFront signed with **origin access control**, so nobody can go around CloudFront. What the app does for it:
  - **POST bodies are signed too,** and Lambda doesn't accept unsigned payloads, so the page sends the body's SHA-256 in `x-amz-content-sha256` (`bodyHash` in `index.html`; ignored without CloudFront).
  - **The visitor's IP:** behind CloudFront the connection's `sourceIp` is always a CloudFront server, so with `BEHIND_CLOUDFRONT=true` the IP comes from `CloudFront-Viewer-Address`, which CloudFront adds. It's trusted only then, because only CloudFront can reach the function (tested). `/health` reports which source was used (`client_ip_from`, never the IP), and the deploy warns if it isn't `cloudfront`.
  - **Caching:** only the two pages, which send `s-maxage=300` (5 minutes at the edge; browsers always revalidate). API responses send no `Cache-Control`, and the cache policy's default TTL is 0, so they're never cached. `/api/*` isn't compressed, so the streamed events aren't held back. Each deploy clears the cache (`/*`, one free invalidation path).
  - **Timeout:** CloudFront waits up to 60 s between bytes; the longest quiet gap in a check is one model call. Syncs never go through CloudFront: the schedule and the deploy invoke the function directly.
- **What changes on Lambda (`RUNTIME=lambda`)**, because a function is frozen between requests and there can be several at once:
  - **GitHub sync:** no background loop. An EventBridge rule invokes the function every 30 minutes; the adapter posts the event to `POST /events`, which runs the sync and answers only when it's done. `/events` is reachable through CloudFront too, so the event carries a generated token. The deploy workflow syncs once right after each deploy.
  - **Rate limits:** `RATE_LIMIT_STORE=postgres` (`app/shared.py`): the per-visitor hourly window and the daily cap live in the same Neon database, so every instance counts the same visitors. Each check is one atomic SQL statement, so two requests can't both take the last slot. The key is an HMAC of the IP, never the IP (tested).
  - **Visitor IP:** from `CloudFront-Viewer-Address` behind CloudFront (above), else from the request context the adapter adds (`sourceIp` of the TCP connection). Never from `X-Forwarded-For`, which a visitor can forge.
  - **Sync status:** "last checked" is saved in Postgres, so every instance shows the same one.
  - **Secrets:** read from SSM Parameter Store (`SSM_PREFIX=/fitcheck`) at startup; nothing secret in environment variables or GitHub.
- **No VPC on purpose:** the function only calls public APIs (NVIDIA, GitHub, OpenRouter, Neon). A VPC would need a NAT gateway for that, about $30+ a month even when idle.
- **CI/CD:** `ci.yml` runs every test, including the pgvector ones against a real `pgvector/pgvector:pg16` service. When it passes on `main`, `deploy.yml` signs in to AWS with GitHub OIDC (no AWS keys stored), builds and pushes the image to ECR, updates the function, syncs, clears CloudFront's copy of the pages, and checks `/health` through CloudFront.
- **Cost:** expected ₹0 to a few rupees a month. Lambda's always-free allowance covers ~20,000 checks a month at 1 GB and the daily limit stops at ~6,000; CloudFront's always-free tier (1 TB and 10M requests a month), function URLs, EventBridge, SSM standard parameters and this much logging are free; ECR is a few cents; Neon's free plan holds the index. An optional AWS Budget emails at 80% of `budget_usd`. These are from AWS pricing, not a measured bill.

**First deploy, in short** (details in `infra/README.md`): `terraform apply` with `create_function = false` → put the NVIDIA key and the Neon **direct** URL into SSM → set `AWS_ROLE_ARN` and `AWS_REGION` as repo variables → push to `main` (pushes the image) → `create_function = true`, `terraform apply` → run the deploy workflow once to fill the index. Terraform apply is run by hand.

**Without AWS:** the plain `Dockerfile` runs anywhere that runs containers (Render, Fly). Set `PROVIDER_MODE=live`, the NVIDIA key, `VECTOR_STORE=pgvector` + `DATABASE_URL`, `GITHUB_USER`, `SYNC_TOKEN`, and `TRUST_PROXY_HEADERS=true` with the right `PROXY_HOPS`. On one long-running server the app syncs by itself every 30 minutes and keeps the limits in memory.

Citations to GitHub repos are commit permalinks, so they never point at the wrong lines, even before the next sync.

---

## 11. Security and privacy

- [x] Keys only from env vars; `.env` is gitignored and dockerignored
- [x] Error messages never echo request headers (which carry keys)
- [x] Chunks that look like credentials are never indexed; `.env*` files are skipped
- [x] GitHub sync indexes only public, non-fork, non-archived repos; a repo that goes private is dropped at the next sync
- [x] Repo archives are extracted defensively: no path escapes, no symlinks, size caps
- [x] `profile/profile.md` is public by design: no phone, email, salary or job title
- [x] Build output, dependencies, uploads and generated CVs are skipped
- [x] JD length capped, per-IP rate limit with `Retry-After`
- [x] Behind a proxy, the visitor IP is read from the **right** of X-Forwarded-For (`PROXY_HOPS`), so a forged left-hand entry can't dodge the limit (tested)
- [x] A shared **daily cap** (`DAILY_LIMIT`) for all visitors, so rotating IPs can't burn the free model quotas; idle IPs are forgotten so the limiter's memory doesn't grow
- [x] `POST /api/sync` is off unless `SYNC_TOKEN` is set; the token is compared in constant time
- [x] The public sync status shows only public repo names, commits and chunk counts
- [x] No JD or IP is stored. On one server, IPs live only in the in-memory rate window. On Lambda the shared window stores an HMAC of the IP (keyed with a generated secret), deleted after the hour (tested: the table never contains the IP)
- [x] On Lambda: secrets only in SSM; the function's role can read only its own `/fitcheck/*` parameters; the deploy role can push one image and update one function (OIDC, no AWS keys in GitHub); the public `/events` path needs the generated sync token
- [x] UI renders API data with `textContent` (no HTML injection)
- [x] Container runs as a non-root user
- [ ] **You:** rotate the NVIDIA key you pasted into a chat on 4 Nov 2026
- [ ] **You:** only index repos that are public

---

## 12. Known limitations

| Limitation | Impact | Fix later |
|---|---|---|
| Prompts were fixed by looking at the development set | The 8 Nov 100% on `dataset.jsonl` is optimistic; the held-out 97% / 90% is the honest number | Keep developing on `dataset.jsonl`; build a new held-out set for each prompt or retrieval change |
| Retrieval misses requirements that are lists of names | "AutoGen, CrewAI, or LangGraph" retrieved the README, not `app/graph.py` (≈0.20 similarity everywhere) | Hybrid search retrieves `graph.py` (8 Nov), but it's **off by default**: with Nemotron it added 1–2 false claims per held-out run. Re-test it with Mercury on a new held-out set |
| Hybrid search shows more mention-only chunks | A scorer that counts mentions claims more: the keyword baseline went from 54% to 92% false claims on the development set, and Nemotron claimed PyTorch from a list of skill names | Keep hybrid opt-in; a stronger scorer (Mercury); near misses in every eval set |
| Mercury Decide not served (8 Nov) | OpenRouter lists it with 0 endpoints, so `SCORER=system_one` returns Unverified rows | Wait for it to come back, or switch to another decision model (one env var) |
| Grounding can't catch a weak but real citation | Nemotron cited an architecture doc as proof of "microservices" | A per-requirement check that the cited snippet contains the key term, or a second judge pass on the cited snippet only |
| pgvector tested in Docker, not on Neon | 3 tests pass on pgvector 0.8.7; Neon's suspend/resume is simulated by killing connections | One real ingest + check on Neon before going public |
| No similarity floor on retrieval | Every requirement gets 5 snippets, even unrelated ones; a crafted requirement could get a "met" citing an unrelated (but real, visible) snippet | A similarity floor + an injection case in the evals |
| Plain-LLM confidence is uncalibrated | Nemotron says 0.95 for almost everything | That's the case for Mercury Decide; measure it |
| Rate limiter, daily cap and sync run per instance | Reset on restart; several instances would each count and each sync | Redis / Postgres limiter; one syncing instance |
| Line-window chunking | A function split across two windows is cited as two halves | AST-aware chunking for Python/Java |
| Evidence = my repos only | Work done at my job (private) can't be cited | Add a "work experience" section from the CV, clearly labelled as self-reported |
| Small eval set | 25 labelled requirements; labels depend on what's indexed (React flipped once the GitHub sync added a React repo) | Grow to 30–50 JDs; store the index snapshot (repos + commits) with every run |
| Only 4 public repos to cite (8 Nov) | The app index is 172 chunks. Skills shown only in private projects (e.g. LangGraph, Spring Boot) come out *not met* or *Stated on CV* | Make those projects public; the sync adds them with links |
| Evals run on a private index | Their labels assume projects that aren't public, so the numbers describe the scorer, not what the public app can cite | Relabel on the public index once more projects are public |
| CV claims are only as honest as the profile | "Stated on CV" means exactly that | Keep `profile/profile.md` to what you can defend |
| Old public repos are indexed too | Weak or unrelated projects can be cited | Add names to `GITHUB_EXCLUDE` |

---

## 13. Next steps, in order

1. ~~NVIDIA key~~ ✅ · ~~OpenRouter key~~ ✅ · ~~Live run on the free scorer~~ ✅ · ~~pgvector tested on a real Postgres~~ ✅ · ~~Live GitHub sync~~ ✅ · ~~How it works page~~ ✅
2. ~~Mercury Decide evals~~ ✅ · ~~Fix its status criteria~~ ✅ · ~~Held-out set, run once~~ ✅ (Mercury 97%, Nemotron 90%, §9). ~~Hybrid retrieval~~ ✅ built, **off by default** (it fixes the LangGraph miss but made Nemotron over-claim). ~~Final check~~ ✅. Next: a **new** held-out set from fresh postings, scored once with Mercury (when OpenRouter serves it again) and Nemotron, in both retrieval modes.
3. **Evals:** grow the dataset to 30+ JDs (more near misses, one prompt-injection case), save the indexed repos + commits with each run, put the table in the README.
4. **Similarity floor** on retrieval, measured with the evals before and after.
5. **GitHub:** push `fitcheck`, `Healthdecissionagent`, `vibe-audit` (public). The sync picks them up within 30 minutes and their rows get links.
6. ~~**Deploy**~~ ✅ live on AWS Lambda + Neon since 9 Nov (§10).
7. **Record a 60-second demo:** paste a real JD, click a citation, show an Unverified row and explain why.

---

## 14. Roadmap: private mode

The same graph, extended for my own job search. It runs locally only and never acts without me.

```mermaid
stateDiagram-v2
    [*] --> parse_jd
    parse_jd --> retrieve
    retrieve --> score
    score --> ground
    ground --> decide_apply
    decide_apply --> skip: low fit
    decide_apply --> tailor_cv: good fit
    tailor_cv --> grounding_check
    grounding_check --> tailor_cv: invented line found
    grounding_check --> outreach_draft: every line cited
    outreach_draft --> approval: interrupt() — waits for me
    approval --> track: I approve (I apply manually)
    approval --> skip: I reject
    track --> [*]
    skip --> [*]
```

New pieces: a **Postgres checkpointer** so a run can pause at `interrupt()` and resume after I approve; the CV Tailor's LaTeX generator as the `tailor_cv` node; Langfuse tracing; a tracker table.
