"""Wire concrete providers from settings. The only place that knows about "live" vs "fake"."""

import httpx

from app.config import Settings
from app.graph import Deps
from app.jd_parser import JdParser
from app.llm.chat import OpenAICompatibleChat
from app.llm.credits import CreditPause
from app.llm.embeddings import Embedder, HashEmbedder, OpenAICompatibleEmbedder
from app.llm.decisions import DecisionsClient
from app.llm.system_one import SystemOneClient
from app.rag.store import InMemoryStore, PgVectorStore, VectorStore
from app.scoring import FitScorer
from app.scoring.keyword import KeywordFitScorer
from app.scoring.llm_json import LlmJsonFitScorer
from app.scoring.parallel import ParallelFitScorer, identity_terms
from app.scoring.system_one import SystemOneFitScorer


OPENROUTER_SCORERS = ("system_one", "parallel")


def build_store(settings: Settings) -> VectorStore:
    if settings.vector_store == "pgvector":
        return PgVectorStore(settings.database_url, min_size=settings.db_pool_min)
    return InMemoryStore(settings.index_path)


def build_embedder(settings: Settings, http: httpx.AsyncClient) -> Embedder:
    if settings.provider_mode == "fake":
        return HashEmbedder()
    return OpenAICompatibleEmbedder(http, settings.nvidia_base_url, settings.nvidia_api_key,
                                    settings.embedding_model, settings.embedding_input_type)


def build_chat(settings: Settings, http: httpx.AsyncClient) -> OpenAICompatibleChat:
    return OpenAICompatibleChat(http, settings.nvidia_base_url, settings.nvidia_api_key,
                                settings.chat_model, thinking=settings.chat_thinking)


def build_scorer(settings: Settings, http: httpx.AsyncClient, chat: OpenAICompatibleChat) -> FitScorer:
    if settings.scorer == "llm_json":
        return LlmJsonFitScorer(chat)
    if settings.scorer == "system_one":
        return SystemOneFitScorer(SystemOneClient(
            http, settings.openrouter_base_url, settings.openrouter_api_key, settings.decide_model))
    if settings.scorer == "parallel":
        return ParallelFitScorer(
            DecisionsClient(http, settings.openrouter_decisions_url, settings.openrouter_api_key,
                            settings.parallel_model),
            redact_terms=identity_terms(settings.candidate_name, settings.github_user))
    raise ValueError(f"Unknown SCORER {settings.scorer!r}: use llm_json, system_one or parallel")


def build_deps(settings: Settings, http: httpx.AsyncClient) -> Deps:
    store = build_store(settings)
    embedder = build_embedder(settings, http)
    if settings.provider_mode == "fake":
        parser = JdParser(chat=None, max_requirements=settings.max_requirements)
        scorer = KeywordFitScorer()
        models = {"parser": "heuristic", "embeddings": embedder.name, "scorer": scorer.name}
    else:
        chat = build_chat(settings, http)
        parser = JdParser(chat=chat, max_requirements=settings.max_requirements)
        scorer = build_scorer(settings, http, chat)
        models = {"parser": chat.name, "embeddings": embedder.name, "scorer": scorer.name}
    chat_or_none = None if settings.provider_mode == "fake" else chat
    return Deps(parser=parser, embedder=embedder, store=store, scorer=scorer,
                settings=settings, models=models,
                fallback_scorer=build_fallback(settings, http, chat_or_none),
                credits=CreditPause(settings.credit_pause_minutes * 60))


def build_fallback(settings: Settings, http: httpx.AsyncClient,
                   chat: OpenAICompatibleChat | None) -> FitScorer | None:
    """A scorer for when OpenRouter's credits or free daily quota run out. Only the OpenRouter
    scorers (system_one, parallel) need one."""
    if chat is None or settings.scorer not in OPENROUTER_SCORERS or not settings.scorer_fallback:
        return None
    if settings.scorer_fallback in OPENROUTER_SCORERS:
        raise ValueError("SCORER_FALLBACK must be a scorer that doesn't use OpenRouter (llm_json)")
    return build_scorer(settings.model_copy(update={"scorer": settings.scorer_fallback}), http, chat)
