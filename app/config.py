from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All configuration comes from environment variables (or a local .env file)."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Where it runs: "server" (uvicorn: locally, Docker, Render) or "lambda" (AWS Lambda behind
    # the Lambda Web Adapter). Lambda freezes the app between requests, so there's no
    # background loop: an EventBridge schedule triggers the GitHub sync (POST /events).
    runtime: str = "server"
    # On AWS, secrets live in SSM Parameter Store under this prefix (read once at startup).
    ssm_prefix: str = ""

    # "live" calls the real APIs; "fake" runs fully offline (no keys, no network).
    provider_mode: str = "live"
    http_timeout_s: float = 60.0

    # NVIDIA API (OpenAI-compatible): chat (JD parsing) + embeddings. Free tier.
    nvidia_api_key: str = ""
    nvidia_base_url: str = "https://integrate.api.nvidia.com/v1"
    chat_model: str = "nvidia/nemotron-3-ultra-550b-a55b"
    chat_thinking: bool = False  # reasoning mode: better on messy JDs, much slower
    embedding_model: str = "nvidia/nemotron-3-embed-1b"  # 3.5 rejects batched code chunks on the free tier
    embedding_input_type: bool = True  # NVIDIA retrieval models expect "query" vs "passage"

    # Fit scorer: "system_one" (decision model on OpenRouter, calibrated confidence, paid but
    # ~$0.02 per million input tokens) or "llm_json" (the NVIDIA chat model, free, uncalibrated).
    scorer: str = "llm_json"

    # OpenRouter: only for the System One decision model (fit scoring).
    openrouter_api_key: str = ""
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    decide_model: str = "inception/mercury-decide"  # the ":free" variant was withdrawn in Nov 2026
    # When OpenRouter says the credits (or the free daily quota) are used up, score with this
    # free scorer instead and say so in the report ("" = no fallback: nothing is claimed).
    scorer_fallback: str = "llm_json"
    # After a credits error, skip OpenRouter for this long, then try it once more.
    credit_pause_minutes: int = 15

    # Vector store: "memory" (JSON file) or "pgvector".
    vector_store: str = "memory"
    index_path: str = "data/index.json"
    # Evals only, never served: the labels in evals/*.jsonl assume projects that aren't public
    # (yet), so evals read their own local index. data/ is in .gitignore and .dockerignore.
    eval_index_path: str = "data/eval-index.json"
    database_url: str = ""
    db_pool_min: int = 1  # 0 on Lambda: a frozen function holds no idle connections

    # Behaviour
    retrieve_top_k: int = 5
    # "vector" = vector top k (the default). "hybrid" = that plus up to keyword_extra chunks
    # only keyword (BM25) search found: it finds exact names ("LangGraph") that embed poorly,
    # but with the Nemotron scorer it raised held-out false claims from 0% to 8-15% (3 runs
    # each, PROJECT.md §9), so it stays opt-in until it is re-tested with Mercury Decide.
    retrieval: str = "vector"
    keyword_extra: int = 3
    min_confidence: float = 0.5
    max_jd_chars: int = 12000
    max_requirements: int = 15  # room for split technology lists

    # Public demo guards
    rate_limit_per_hour: int = 20  # per visitor IP
    daily_limit: int = 200  # all visitors together, per UTC day (0 = off): protects the free quotas
    trust_proxy_headers: bool = False  # only behind a proxy that sets X-Forwarded-For
    # On Lambda behind CloudFront: the function URL accepts only CloudFront's signed requests
    # (AWS_IAM + origin access control), so every request's sourceIp is a CloudFront server and
    # the visitor's IP comes from the CloudFront-Viewer-Address header CloudFront adds.
    behind_cloudfront: bool = False
    # "memory" = per process (one server). "postgres" = shared by every instance, for Lambda
    # (needs VECTOR_STORE=pgvector; stores an HMAC of the IP, never the IP).
    rate_limit_store: str = "memory"
    proxy_hops: int = 1  # proxies in front of the app that append to X-Forwarded-For

    # GitHub sync: index every public repo of this user (scripts/ingest.py --github)
    github_user: str = ""
    github_token: str = ""  # optional: raises the API limit from 60 to 5000 requests/hour
    github_exclude: str = ""  # comma-separated repo names to skip
    # Live sync while the app runs: check GitHub at startup and every N minutes (0 = off).
    # A quiet check is 1 + one-per-repo tiny API calls, well inside the 60/hour no-token limit.
    github_sync_interval_min: int = 30
    # POST /api/sync with "Authorization: Bearer <token>" forces a check now. Empty = disabled.
    sync_token: str = ""

    # Self-reported profile (CV): searched only for what code didn't prove
    profile_path: str = "profile/profile.md"  # re-indexed by the live sync when it changes
    cv_top_k: int = 3
    cv_url: str = ""  # optional link shown on "Stated on CV" rows, e.g. a PDF of my CV

    # Report
    candidate_name: str = "Arshdeep Singh"
    github_base_url: str = ""
    github_branch: str = "main"


def load_ssm(prefix: str, client=None) -> None:
    """Copy the SSM parameters under `prefix` into the environment, names upper-cased
    (/fitcheck/nvidia_api_key -> NVIDIA_API_KEY). Values already in the environment win,
    and the "set-me" placeholder Terraform creates is skipped."""
    import os

    if client is None:
        import boto3  # in the Lambda base image; only needed on AWS

        client = boto3.client("ssm")
    for page in client.get_paginator("get_parameters_by_path").paginate(Path=prefix, WithDecryption=True):
        for param in page["Parameters"]:
            name = param["Name"].rsplit("/", 1)[-1].upper()
            if param["Value"] and param["Value"] != "set-me":
                os.environ.setdefault(name, param["Value"])


@lru_cache
def get_settings() -> Settings:
    import os

    if os.environ.get("SSM_PREFIX"):
        load_ssm(os.environ["SSM_PREFIX"])
    return Settings()
