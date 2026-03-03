"""Prompt optimization using the Judge LLM."""

import logging
import sys
import re

from .openrouter_client import call_judge

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler(sys.stdout))


PROMPT_OPTIMIZATION_TEMPLATE = """You are an expert prompt engineer. Your task is to create an optimized system prompt for a specific AI task.

## Task Description
{task_description}

## Top Recommended Model
{top_model_id} ({top_model_name})

## Model Strengths
{strengths}

## Benchmark Performance
- Overall Score: {overall_score}/10
- Accuracy: {accuracy}/10
- Hallucination Resistance: {hallucination}/10
- Grounding: {grounding}/10
- Clarity: {clarity}/10

## Instructions
Create a structured, task-specific system prompt that:
1. Clearly defines the AI role and expertise for this task
2. Sets behavioral guidelines to maximize accuracy and minimize hallucination
3. Specifies output format requirements
4. Includes task-specific constraints or best practices
5. Is optimized for the model's known strengths

The system prompt should be production-ready and immediately usable.

IMPORTANT: Return ONLY the system prompt text itself. Do not include:
- Any explanation of what you are doing
- Markdown headers like "## System Prompt" or "Here is the prompt:"
- Preamble such as "Certainly!" or "Sure, here is..."
- Any text before or after the actual system prompt content

Start directly with the role definition (e.g. "You are an expert...") and end with the last instruction."""


def _clean_system_prompt(raw: str) -> str:
    """
    Clean a raw LLM response to extract only the system prompt text.

    Removes common LLM preamble patterns, markdown headers, and thinking tags
    that pollute the system prompt output.

    Args:
        raw: Raw text response from the Judge LLM

    Returns:
        Clean system prompt text ready for use
    """
    if not raw or not raw.strip():
        return _default_system_prompt()

    # Remove <think>...</think> blocks
    cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()

    # Remove common preamble patterns (case-insensitive)
    preamble_patterns = [
        r"^(certainly|sure|of course|absolutely|here is|here's|below is|the following)[^:]*:\s*",
        r"^(system prompt|optimized system prompt|task-specific system prompt)[:\s]*",
        r"^#+\s*(system prompt|optimized prompt|prompt)[^\n]*\n",
        r"^---+\n",
    ]
    for pattern in preamble_patterns:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE | re.MULTILINE).strip()

    # Remove trailing markdown fences
    cleaned = re.sub(r"```\s*$", "", cleaned).strip()
    # Remove leading markdown fences
    cleaned = re.sub(r"^```[a-z]*\s*", "", cleaned).strip()

    # If after cleaning we have nothing meaningful, return default
    if len(cleaned) < 50:
        logger.warning("System prompt too short after cleaning, using default.")
        return _default_system_prompt()

    return cleaned


def _default_system_prompt() -> str:
    """Return a safe default system prompt when generation fails."""
    return (
        "You are an expert AI assistant. Provide accurate, well-reasoned, and clearly structured responses. "
        "Always ground your answers in verifiable facts. Avoid speculation or fabrication. "
        "When uncertain, explicitly state your uncertainty rather than guessing. "
        "Format your responses for clarity and readability."
    )


def generate_optimized_prompt(
    task_description: str,
    top_model: dict,
    evaluation_scores: dict,
) -> str:
    """
    Generate an optimized system prompt for the top-ranked model.

    Uses the Judge LLM to craft a task-specific system prompt, then cleans
    the response to ensure only the prompt text is returned (no preamble,
    no markdown headers, no thinking traces).

    Args:
        task_description: The original task description
        top_model: Dict with model id, name, strengths, recommendation
        evaluation_scores: Aggregated evaluation scores for the model

    Returns:
        Clean, production-ready system prompt string
    """
    model_id = top_model.get("model_id", "unknown")
    logger.info(f"Generating optimized prompt for {model_id}...")

    strengths = top_model.get("strengths", [])
    if not strengths:
        strengths = ["Strong overall benchmark performance"]
    strengths_text = "\n".join(f"- {s}" for s in strengths)

    messages = [
        {
            "role": "user",
            "content": PROMPT_OPTIMIZATION_TEMPLATE.format(
                task_description=task_description,
                top_model_id=model_id,
                top_model_name=model_id.split("/")[-1],
                strengths=strengths_text,
                overall_score=round(evaluation_scores.get("overall", 0), 1),
                accuracy=round(evaluation_scores.get("accuracy", 0), 1),
                hallucination=round(evaluation_scores.get("hallucination", 0), 1),
                grounding=round(evaluation_scores.get("grounding", 0), 1),
                clarity=round(evaluation_scores.get("clarity", 0), 1),
            ),
        }
    ]

    raw = call_judge(messages, temperature=0.5, max_tokens=2048)
    prompt = _clean_system_prompt(raw)
    logger.info(f"Generated system prompt ({len(prompt)} chars) for {model_id}")
    return prompt
