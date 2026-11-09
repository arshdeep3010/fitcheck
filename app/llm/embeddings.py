"""Text embeddings: NVIDIA for real runs, a deterministic hashing embedder for offline mode."""

import hashlib
import math
import re
from typing import Literal, Protocol

import httpx

from app.llm.http import ProviderError, post_json

Kind = Literal["query", "passage"]


class Embedder(Protocol):
    name: str

    async def embed(self, texts: list[str], kind: Kind) -> list[list[float]]: ...


class OpenAICompatibleEmbedder:
    """POST /embeddings. NVIDIA retrieval models want input_type=query|passage."""

    def __init__(self, http: httpx.AsyncClient, base_url: str, api_key: str, model: str,
                 send_input_type: bool = True, batch_size: int = 16):
        self._http = http
        self._url = base_url.rstrip("/") + "/embeddings"
        self._api_key = api_key
        self.name = model
        self._send_input_type = send_input_type
        self._batch_size = batch_size

    async def embed(self, texts: list[str], kind: Kind) -> list[list[float]]:
        vectors: list[list[float]] = []
        for i in range(0, len(texts), self._batch_size):
            vectors.extend(await self._embed_batch(texts[i:i + self._batch_size], kind))
        return vectors

    async def _embed_batch(self, batch: list[str], kind: Kind) -> list[list[float]]:
        """Some free-tier endpoints reject large requests with a 500. Split and retry
        instead of failing the whole ingest; a single text that still fails is a real error."""
        payload: dict = {"model": self.name, "input": batch, "encoding_format": "float"}
        if self._send_input_type:
            payload["input_type"] = kind
            payload["truncate"] = "END"
        try:
            data = await post_json(self._http, self._url, self._api_key, payload,
                                   attempts=3 if len(batch) == 1 else 2)
        except ProviderError:
            if len(batch) == 1:
                raise
            mid = len(batch) // 2
            return await self._embed_batch(batch[:mid], kind) + await self._embed_batch(batch[mid:], kind)
        try:
            rows = sorted(data["data"], key=lambda r: r["index"])
            return [row["embedding"] for row in rows]
        except (KeyError, TypeError) as exc:
            raise ProviderError(f"Unexpected embeddings response: {str(data)[:200]}") from exc


_TOKEN = re.compile(r"[a-z0-9][a-z0-9+#.]*")


class HashEmbedder:
    """Offline stand-in: hashed bag of words, L2-normalised. Similar words -> similar vectors.

    Good enough for tests and a no-key demo; not a substitute for a real model.
    """

    name = "hash-embedder"

    def __init__(self, dim: int = 256):
        self.dim = dim

    async def embed(self, texts: list[str], kind: Kind) -> list[list[float]]:
        return [self._one(t) for t in texts]

    def _one(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for token in _TOKEN.findall(text.lower()):
            digest = hashlib.md5(token.encode()).digest()
            index = int.from_bytes(digest[:4], "little") % self.dim
            vec[index] += 1.0 if digest[4] % 2 == 0 else -1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]
