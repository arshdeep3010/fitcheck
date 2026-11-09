"""Client for the System One API (Mercury Decide, Jev): state + typed questions -> typed answers."""

import httpx

from app.llm.http import ProviderError, post_json


class SystemOneClient:
    def __init__(self, http: httpx.AsyncClient, base_url: str, api_key: str, model: str):
        self._http = http
        self._url = base_url.rstrip("/") + "/systemone"
        self._api_key = api_key
        self.model = model

    async def ask(self, state: dict | str, questions: dict) -> dict:
        """Returns the `answers` object, keyed by the same ids as `questions`."""
        payload = {"model": self.model, "state": state, "questions": questions}
        data = await post_json(self._http, self._url, self._api_key, payload)
        answers = data.get("answers")
        if not isinstance(answers, dict):
            raise ProviderError(f"Unexpected System One response: {str(data)[:200]}")
        return answers
