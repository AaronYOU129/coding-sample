"""
Purpose:    Thin wrapper around the Anthropic SDK that (1) builds a client from
            the ANTHROPIC_API_KEY environment variable and (2) makes every API
            call survive transient failures over a 10-day run via explicit
            exponential backoff. The SDK's own retries are disabled
            (max_retries=0) so the backoff is visible and logged in one place
            rather than split between the SDK and this code.
Inputs:     ANTHROPIC_API_KEY environment variable.
Outputs:    None (returns response text + usage to callers).
Key Steps:  Build client -> call Messages API behind a backoff loop -> return the
            joined text and the usage object as a plain dict for JSON storage.
How to Run: Imported by collect.py and extract.py; not run directly.
"""

import logging
import os
import random
import time

import anthropic

logger = logging.getLogger(__name__)

# Transient failures worth retrying. 4xx errors other than 429 (bad request,
# auth, not-found) are caller mistakes and must surface immediately, not retry.
_RETRYABLE_ERRORS = (
    anthropic.RateLimitError,
    anthropic.APIConnectionError,
    anthropic.APITimeoutError,
    anthropic.InternalServerError,
)


def build_client():
    """Return an Anthropic client, or fail loudly if the key is missing."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Export it before running "
            "(the key is read from the environment, never hardcoded)."
        )
    # max_retries=0: this module owns retry/backoff so it is logged in one place.
    return anthropic.Anthropic(max_retries=0)


def generate(client, model, prompt, max_tokens, system=None):
    """Send one user prompt and return (response_text, usage_dict).

    response_text joins all text blocks; usage_dict is the API usage object as a
    plain dict so it can be written straight into the output JSON.
    """
    response = _call_with_backoff(
        client.messages.create,
        model=model,
        max_tokens=max_tokens,
        system=system or anthropic.NOT_GIVEN,
        messages=[{"role": "user", "content": prompt}],
    )
    text = "".join(block.text for block in response.content if block.type == "text")
    return text, response.usage.model_dump()


def _call_with_backoff(api_call, *, max_retries=6, base_delay=2.0, max_delay=90.0, **kwargs):
    """Call api_call(**kwargs), retrying transient errors with exponential
    backoff and jitter. Non-retryable errors propagate immediately."""
    last_error = None
    for attempt in range(max_retries):
        try:
            return api_call(**kwargs)
        except _RETRYABLE_ERRORS as error:
            last_error = error
        except anthropic.APIStatusError as error:
            if error.status_code < 500:
                raise  # client-side error — retrying will not help
            last_error = error

        delay = min(base_delay * (2 ** attempt) + random.uniform(0, 1), max_delay)
        logger.warning(
            "API call failed (%s); retry %d/%d in %.1fs",
            type(last_error).__name__, attempt + 1, max_retries, delay,
        )
        time.sleep(delay)

    raise last_error
