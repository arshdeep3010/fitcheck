"""Shared fixtures: a tiny fake repo, an offline index built from it, offline settings.

fitcheck: no-index
"""

from pathlib import Path

import pytest

from app.config import Settings
from app.llm.embeddings import HashEmbedder
from app.rag.ingest import ingest
from app.rag.store import InMemoryStore

SPRING_CONTROLLER = """package com.demo.claims;

import org.springframework.web.bind.annotation.*;

// REST API for claims, built with Spring Boot
@RestController
@RequestMapping("/claims")
public class ClaimController {
    @PostMapping("/{id}/analyze")
    public Decision analyze(@PathVariable String id) {
        return orchestrator.analyze(id);
    }
}
"""

RAG_SERVICE = """# Retrieval service: RAG over policy documents with a pgvector vector database
def retrieve(query, top_k=5):
    vector = embed(query)
    rows = db.execute("SELECT text FROM chunks ORDER BY embedding <=> %s LIMIT %s", (vector, top_k))
    return [r[0] for r in rows]
"""


@pytest.fixture
def sample_repo(tmp_path: Path) -> Path:
    root = tmp_path / "demo-repo"
    (root / "src").mkdir(parents=True)
    (root / "src" / "ClaimController.java").write_text(SPRING_CONTROLLER, encoding="utf-8")
    (root / "src" / "rag.py").write_text(RAG_SERVICE, encoding="utf-8")
    (root / ".env").write_text("OPENAI_API_KEY=sk-should-never-be-indexed-000000000000\n", encoding="utf-8")
    (root / "node_modules" / "lib").mkdir(parents=True)
    (root / "node_modules" / "lib" / "index.js").write_text("module.exports = 1;\n// a\n// b\n", encoding="utf-8")
    return root


@pytest.fixture
def index_path(tmp_path: Path, sample_repo: Path) -> Path:
    import asyncio

    path = tmp_path / "index.json"
    store = InMemoryStore(str(path))
    asyncio.run(ingest({"demo-repo": sample_repo}, HashEmbedder(), store))
    store.save()
    return path


@pytest.fixture
def fake_settings(index_path: Path) -> Settings:
    return Settings(
        _env_file=None,
        provider_mode="fake",
        vector_store="memory",
        index_path=str(index_path),
        github_base_url="https://github.com/someone",
        rate_limit_per_hour=3,
        min_confidence=0.5,
    )
