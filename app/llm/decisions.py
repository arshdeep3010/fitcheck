"""Client for OpenRouter's Decisions API (Parallel Decisions): state + typed questions ->
typed answers with a probability per option. Same request shape as System One.

Parallel's model researches the web for every request; neither OpenRouter nor Parallel has a
setting to turn that off (checked Nov 2026). The web sources come back as `annotations`.
FitCheck never uses them: this client returns only the answers, and the scorer can only cite
chunks it retrieved itself. The count of discarded sources is kept for logs and evals.
"""

import logging

import httpx

from app.llm.http import ProviderError, post_json

log = logging.getLogger("fitcheck.decisions")


class DecisionsClient:
    def __init__(self, http: httpx.AsyncClient, url: str, api_key: str, model: str):
        self._http = http
        self._url = url
        self._api_key = api_key
        self.model = model
        self.web_sources_discarded = 0  # from the last call

    async def ask(self, state: dict | str, questions: dict) -> dict:
        """Returns the `answers` object, keyed by the same ids as `questions`. Web sources
        (`annotations`, or `citations` on Parallel's own API) are dropped here."""
        payload = {"model": self.model, "state": state, "questions": questions}
        data = await post_json(self._http, self._url, self._api_key, payload)
        answers = data.get("answers")
        if not isinstance(answers, dict):
            raise ProviderError(f"Unexpected Decisions response: {str(data)[:200]}")
        self.web_sources_discarded = sum(len(data.get(k) or []) for k in ("annotations", "citations"))
        if self.web_sources_discarded:
            log.info("decision model returned %d web sources; discarded (only retrieved code is cited)",
                     self.web_sources_discarded)
        return answers
