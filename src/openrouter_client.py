"""LLM API client supporting OpenRouter and MiniMax providers."""

import time
import logging
import sys
from typing import Optional
from openai import OpenAI

from .config import OPENROUTER_BASE_URL, MINIMAX_BASE_URL, load_api_key

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler(sys.stdout))

# Module-level provider setting (set by main.py based on CLI args)
_active_provider: str = "openrouter"


def set_provider(provider: str) -> None:
    """Set the active LLM provider for candidate model calls."""
    global _active_provider
    _active_provider = provider


def get_provider() -> str:
    """Get the active LLM provider."""
    return _active_provider


def get_client(provider: str = "openrouter") -> OpenAI:
    """Create and return an OpenAI-compatible client for the given provider.

    Args:
        provider: LLM provider name ('openrouter' or 'minimax')

    Returns:
        OpenAI client configured for the specified provider
    """
    if provider == "minimax":
        return OpenAI(
            base_url=MINIMAX_BASE_URL,
            api_key=load_api_key("minimax"),
        )
    return OpenAI(
        base_url=OPENROUTER_BASE_URL,
        api_key=load_api_key("openrouter"),
    )


def call_llm(
    model: str,
    messages: list[dict],
    temperature: float = 0.7,
    max_tokens: int = 4096,
    timeout: int = 60,
    provider: Optional[str] = None,
) -> tuple[str, float]:
    """
    Call an LLM and return (response_text, latency_seconds).

    Args:
        model: Model ID (e.g. 'google/gemini-3.1-pro-preview' or 'MiniMax-M2.5')
        messages: List of message dicts with 'role' and 'content'
        temperature: Sampling temperature
        max_tokens: Maximum tokens in response
        timeout: Request timeout in seconds
        provider: LLM provider override (defaults to active provider)

    Returns:
        Tuple of (response_text, latency_in_seconds)
    """
    if provider is None:
        provider = _active_provider

    # MiniMax requires temperature in (0.0, 1.0] — clamp if needed
    if provider == "minimax":
        if temperature <= 0.0:
            temperature = 0.01
        if temperature > 1.0:
            temperature = 1.0

    client = get_client(provider)
    start = time.time()
    try:
        kwargs = dict(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )
        # Add OpenRouter-specific headers
        if provider == "openrouter":
            kwargs["extra_headers"] = {
                "HTTP-Referer": "https://llm-fitness-tool",
                "X-Title": "LLM Fitness Tool",
            }
        response = client.chat.completions.create(**kwargs)
        latency = time.time() - start
        # Guard against None choices or content
        if not response.choices or response.choices[0].message is None:
            return "", latency
        content = response.choices[0].message.content or ""
        return content, latency
    except Exception as e:
        latency = time.time() - start
        logger.error(f"Error calling {model}: {e}")
        return f"ERROR: {str(e)}", latency


def call_judge(messages: list[dict], temperature: float = 0.3, max_tokens: int = 8192) -> str:
    """
    Call the Judge LLM (Gemini 3.1 Pro via OpenRouter) and return response text.

    The Judge always uses OpenRouter regardless of the active provider,
    since the Judge model (Gemini) is only available through OpenRouter.

    Args:
        messages: List of message dicts
        temperature: Low temperature for consistent judging
        max_tokens: Maximum tokens

    Returns:
        Response text from the judge
    """
    from .config import JUDGE_MODEL_ID
    response, _ = call_llm(
        model=JUDGE_MODEL_ID,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=120,
        provider="openrouter",
    )
    return response
