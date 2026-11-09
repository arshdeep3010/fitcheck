"""Chat completions on any OpenAI-compatible endpoint (NVIDIA by default)."""

import json
import re
from typing import Protocol

import httpx

from app.llm.http import ProviderError, post_json

_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL)


class ChatModel(Protocol):
    name: str

    async def complete(self, system: str, user: str) -> str: ...


class OpenAICompatibleChat:
    def __init__(self, http: httpx.AsyncClient, base_url: str, api_key: str, model: str,
                 thinking: bool = False, max_tokens: int = 4096):
        self._http = http
        self._url = base_url.rstrip("/") + "/chat/completions"
        self._api_key = api_key
        self.name = model
        self._thinking = thinking
        self._max_tokens = max_tokens

    async def complete(self, system: str, user: str) -> str:
        payload = {
            "model": self.name,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0.1,
            "max_tokens": self._max_tokens,
            "stream": False,
            # NVIDIA Nemotron: reasoning on/off. Ignored by endpoints that don't know it.
            "chat_template_kwargs": {"enable_thinking": self._thinking},
        }
        data = await post_json(self._http, self._url, self._api_key, payload)
        try:
            return data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError(f"Unexpected chat response shape: {str(data)[:200]}") from exc


def extract_json(text: str) -> dict:
    """Pull the first JSON object out of a model reply (tolerates <think> blocks and ``` fences)."""
    cleaned = _THINK_BLOCK.sub("", text)
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("no JSON object in model output")
    return json.loads(cleaned[start:end + 1])
