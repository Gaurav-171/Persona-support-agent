"""
llm_utils.py — Rate-limit-aware helpers for LLM API calls.

Provides:
    • ``llm_invoke_with_retry``  — Wrapper around ``llm.invoke()`` that retries
      on 429 / RESOURCE_EXHAUSTED errors with exponential back-off.
    • ``rate_limit_delay``       — Adds a configurable pause between successive
      API calls to stay within free-tier quotas.
"""

from __future__ import annotations

import time
from typing import Any

from src.config import (
    API_CALL_DELAY,
    API_MAX_RETRIES,
    API_RETRY_BASE_WAIT,
    get_logger,
)

logger = get_logger(__name__)

# Timestamp of the last API call — used to enforce minimum spacing.
_last_call_time: float = 0.0


def rate_limit_delay() -> None:
    """Sleep if needed so that consecutive API calls are spaced at least
    ``API_CALL_DELAY`` seconds apart."""
    global _last_call_time
    if _last_call_time > 0:
        elapsed = time.time() - _last_call_time
        if elapsed < API_CALL_DELAY:
            wait = API_CALL_DELAY - elapsed
            logger.debug("Rate-limit delay: sleeping %.1fs", wait)
            time.sleep(wait)


def llm_invoke_with_retry(llm: Any, messages: list, *, label: str = "LLM") -> Any:
    """
    Call ``llm.invoke(messages)`` with automatic retry on rate-limit errors.

    Parameters
    ----------
    llm : ChatGoogleGenerativeAI (or compatible)
        The LangChain LLM instance.
    messages : list
        List of ``SystemMessage`` / ``HumanMessage`` objects.
    label : str
        Human-readable label for log messages (e.g. ``"PersonaClassifier"``).

    Returns
    -------
    The response object from ``llm.invoke()``.

    Raises
    ------
    Exception
        Re-raises the last exception if all retries are exhausted.
    """
    global _last_call_time

    last_exc: Exception | None = None

    for attempt in range(1, API_MAX_RETRIES + 1):
        # Enforce spacing between calls
        rate_limit_delay()

        try:
            response = llm.invoke(messages)
            _last_call_time = time.time()
            return response

        except Exception as exc:
            last_exc = exc
            exc_str = str(exc).lower()

            is_rate_limit = any(
                kw in exc_str
                for kw in ("429", "resource_exhausted", "rate limit", "quota")
            )

            if is_rate_limit and attempt < API_MAX_RETRIES:
                wait = API_RETRY_BASE_WAIT * attempt  # linear back-off
                logger.warning(
                    "[%s] Rate-limited (attempt %d/%d) — retrying in %.0fs …",
                    label,
                    attempt,
                    API_MAX_RETRIES,
                    wait,
                )
                time.sleep(wait)
                _last_call_time = time.time()
            else:
                # Not a rate-limit error, or final attempt — re-raise
                logger.error(
                    "[%s] API call failed (attempt %d/%d): %s",
                    label,
                    attempt,
                    API_MAX_RETRIES,
                    exc,
                )
                raise

    # Should not reach here, but just in case
    raise last_exc  # type: ignore[misc]
