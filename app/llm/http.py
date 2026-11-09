"""One place for HTTP calls to model APIs: auth, retries with backoff, clear errors."""

import asyncio
import logging

import httpx

log = logging.getLogger(__name__)

# Worth retrying: rate limited, overloaded, or a server-side hiccup.
RETRYABLE_STATUS = {429, 500, 502, 503, 504, 529}
# Words in a 429 body that mean "out of credits / free daily quota", not "slow down".
# OpenRouter's daily cap reads "Rate limit exceeded: free-models-per-day. Add 10 credits ...".
CREDIT_HINTS = ("credit", "free-models-per-day", "insufficient")


class ProviderError(RuntimeError):
    """A model API call failed for good (after retries, or with a non-retryable status)."""


class CreditsExhaustedError(ProviderError):
    """The account is out of credits, or used up its free daily quota. Retrying won't help."""


def _error_message(response: httpx.Response) -> str:
    """The provider's own error text (OpenRouter: {"error": {"message": ...}}), kept short."""
    try:
        error = response.json().get("error")
        if isinstance(error, dict) and error.get("message"):
            return str(error["message"])[:300]
    except (ValueError, AttributeError):
        pass
    return response.text[:300]


def _out_of_credits(status: int, message: str) -> bool:
    if status == 402:  # Payment Required: OpenRouter's "insufficient credits"
        return True
    return status == 429 and any(hint in message.lower() for hint in CREDIT_HINTS)


async def post_json(
    client: httpx.AsyncClient,
    url: str,
    api_key: str,
    payload: dict,
    attempts: int = 3,
    backoff_s: float = 1.0,
) -> dict:
    if not api_key:
        raise ProviderError(f"No API key configured for {url}")
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    for attempt in range(1, attempts + 1):
        try:
            response = await client.post(url, headers=headers, json=payload)
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            if attempt == attempts:
                raise ProviderError(f"{url} unreachable after {attempts} attempts: {exc}") from exc
            log.warning("transport error on %s (attempt %d): %s", url, attempt, exc)
        else:
            status = response.status_code
            if status < 400:
                return response.json()
            # Never echo request headers: they carry the key.
            message = _error_message(response)
            if _out_of_credits(status, message):
                raise CreditsExhaustedError(f"{url} returned {status}: {message}")
            if status not in RETRYABLE_STATUS or attempt == attempts:
                raise ProviderError(f"{url} returned {status}: {message}")
            log.warning("%s returned %d (attempt %d), retrying", url, status, attempt)
        await asyncio.sleep(backoff_s * 2 ** (attempt - 1))

    raise ProviderError(f"{url} failed")  # unreachable, keeps type checkers happy
