"""Configuration module for the LLM Fitness Tool."""

import os
import json
import logging
from pathlib import Path

# OpenRouter configuration
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
JUDGE_MODEL_ID = "google/gemini-3.1-pro-preview"

# Credential path
OPENROUTER_CONFIG_PATH = Path.home() / ".config" / "openrouter" / "config"


def load_api_key() -> str:
    """Load OpenRouter API key from config file or environment variable."""
    # Try environment variable first
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if api_key:
        return api_key

    # Try config file
    if OPENROUTER_CONFIG_PATH.exists():
        try:
            with open(OPENROUTER_CONFIG_PATH) as f:
                config = json.load(f)
                api_key = config.get("api_key", "")
                if api_key:
                    return api_key
        except (json.JSONDecodeError, KeyError) as e:
            logging.warning(f"Failed to parse OpenRouter config: {e}")

    raise ValueError(
        "OpenRouter API key not found. Set OPENROUTER_API_KEY env var or "
        f"create {OPENROUTER_CONFIG_PATH} with {{\"api_key\": \"your-key\"}}"
    )


# Candidate model categories mapped to known top performers on OpenRouter
MODEL_CATEGORIES = {
    "coding": [
        "openai/gpt-5.3-codex",
        "z-ai/glm-5",
        "anthropic/claude-sonnet-4-6",
        "moonshotai/kimi-k2.5",
        "qwen/qwen3.5-397b-a17b",
        "google/gemini-3-flash-preview",
        "minimax/minimax-m2.5",
    ],
    "math": [
        "google/gemini-2.5-pro",
        "openai/gpt-4.1",
        "deepseek/deepseek-r1-0528",
        "anthropic/claude-sonnet-4-5",
        "qwen/qwen3-235b-a22b",
        "meta-llama/llama-4-maverick",
        "minimax/minimax-m2.5",
    ],
    "reasoning": [
        "google/gemini-2.5-pro",
        "openai/gpt-4.1",
        "deepseek/deepseek-r1-0528",
        "anthropic/claude-sonnet-4-5",
        "meta-llama/llama-4-maverick",
        "mistralai/mistral-large",
        "minimax/minimax-m2.5",
    ],
    "conversation": [
        "anthropic/claude-sonnet-4-5",
        "openai/gpt-4.1",
        "google/gemini-2.5-flash",
        "meta-llama/llama-4-maverick",
        "mistralai/mistral-large",
        "google/gemini-2.5-pro",
        "minimax/minimax-m2.5",
    ],
    "writing": [
        "anthropic/claude-sonnet-4-5",
        "openai/gpt-4.1",
        "google/gemini-2.5-pro",
        "meta-llama/llama-4-maverick",
        "mistralai/mistral-large",
        "google/gemini-2.5-flash",
        "minimax/minimax-m2.5",
    ],
    "general": [
        "openai/gpt-4.1",
        "google/gemini-2.5-pro",
        "anthropic/claude-sonnet-4-5",
        "deepseek/deepseek-r1-0528",
        "meta-llama/llama-4-maverick",
        "mistralai/mistral-large",
        "minimax/minimax-m2.5",
    ],
}

# Evaluation dimensions
EVAL_DIMENSIONS = ["accuracy", "hallucination", "grounding", "tool_calling", "clarity"]

# Benchmarking settings
MAX_CANDIDATES = 7
MAX_TEST_CASES = 5
REQUEST_TIMEOUT = 60  # seconds
