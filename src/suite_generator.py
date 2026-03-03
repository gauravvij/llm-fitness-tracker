"""Test suite generation using the Judge LLM."""

import json
import logging
import sys
import re

from pydantic import ValidationError

from .openrouter_client import call_judge
from .schemas import TestCase

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler(sys.stdout))


TEST_SUITE_PROMPT = """You are an expert AI evaluator designing a rigorous benchmark suite.

The user wants to evaluate LLMs for the following task:
<task_description>
{task_description}
</task_description>

Generate exactly {num_tests} diverse, comprehensive test cases that thoroughly evaluate LLM performance on this task.

Each test case must cover different aspects: basic competency, edge cases, complex reasoning, accuracy under ambiguity, and (if applicable) tool/function calling.

You MUST respond with ONLY a valid JSON array. No markdown fences, no preamble, no explanation outside the JSON.

The JSON array must have exactly this structure:
[{{"id": 1, "category": "basic", "prompt": "The exact prompt to send to the candidate LLM", "evaluation_criteria": "What a correct/good response must contain or demonstrate", "expected_elements": ["key element 1", "key element 2"], "difficulty": "medium"}}, ...]

Rules:
- category must be one of: basic, reasoning, edge_case, accuracy, tool_calling
- difficulty must be one of: easy, medium, hard
- Make prompts realistic and representative of real user queries for this task
- Vary difficulty levels across the test cases
- Include at least one edge case or adversarial prompt
- For coding tasks: include at least one debugging and one implementation prompt
- For math tasks: include at least one multi-step problem
- For conversational tasks: include context-dependent follow-ups
- expected_elements must be a JSON array of strings, not a comma-separated string"""


def _extract_json_array(text: str) -> list:
    """
    Robustly extract a JSON array from text that may contain markdown fences,
    thinking tags, or extra prose.

    Args:
        text: Raw text from the Judge LLM

    Returns:
        Parsed list of dicts

    Raises:
        ValueError: If no valid JSON array can be extracted
    """
    if not text or not text.strip():
        raise ValueError("Cannot extract JSON array from empty response")

    # Remove <think>...</think> blocks
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    # Strip markdown fences
    cleaned = re.sub(r"```(?:json)?\s*", "", cleaned).strip().rstrip("`").strip()

    # Try direct parse
    try:
        result = json.loads(cleaned)
        if isinstance(result, list):
            return result
        if isinstance(result, dict) and "tests" in result:
            return result["tests"]
        if isinstance(result, dict) and "test_cases" in result:
            return result["test_cases"]
    except json.JSONDecodeError:
        pass

    # Find the outermost [ ... ] array
    bracket_depth = 0
    start_idx = None
    in_string = False
    escape_next = False
    for i, ch in enumerate(cleaned):
        if escape_next:
            escape_next = False
            continue
        if ch == "\\" and in_string:
            escape_next = True
            continue
        if ch == '"' and not escape_next:
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "[":
            if bracket_depth == 0:
                start_idx = i
            bracket_depth += 1
        elif ch == "]":
            bracket_depth -= 1
            if bracket_depth == 0 and start_idx is not None:
                candidate = cleaned[start_idx : i + 1]
                try:
                    result = json.loads(candidate)
                    if isinstance(result, list):
                        return result
                except json.JSONDecodeError:
                    # Try fixing trailing commas
                    fixed = re.sub(r",\s*([}\]])", r"\1", candidate)
                    try:
                        result = json.loads(fixed)
                        if isinstance(result, list):
                            return result
                    except json.JSONDecodeError:
                        pass
                start_idx = None
                bracket_depth = 0

    raise ValueError(f"Cannot extract JSON array from: {text[:300]}")


def _validate_test_cases(raw_cases: list, num_tests: int) -> list[dict]:
    """
    Validate and normalize test cases using Pydantic schemas.

    Args:
        raw_cases: Raw list of dicts from JSON parsing
        num_tests: Expected number of test cases

    Returns:
        List of validated test case dicts
    """
    validated = []
    for i, raw in enumerate(raw_cases):
        if not isinstance(raw, dict):
            logger.warning(f"Skipping non-dict test case at index {i}: {raw}")
            continue
        # Ensure id is set correctly
        raw.setdefault("id", i + 1)
        try:
            tc = TestCase(**raw)
            validated.append(tc.model_dump())
        except ValidationError as e:
            logger.warning(f"Test case {i+1} failed Pydantic validation, using defaults: {e}")
            # Force-create with defaults for invalid fields
            tc = TestCase(
                id=raw.get("id", i + 1),
                category=raw.get("category", "general"),
                prompt=str(raw.get("prompt", f"Task {i+1}: Demonstrate capability for the given task.")),
                evaluation_criteria=str(raw.get("evaluation_criteria", "Response should be accurate and relevant.")),
                expected_elements=raw.get("expected_elements", ["relevance", "accuracy"]),
                difficulty=raw.get("difficulty", "medium"),
            )
            validated.append(tc.model_dump())

    return validated


def generate_test_suite(task_description: str, num_tests: int = 5) -> list[dict]:
    """
    Generate a comprehensive test suite for the given task using the Judge LLM.

    Uses Pydantic validation to ensure all test cases have the correct schema
    with no missing or malformed fields.

    Args:
        task_description: Natural language description of the task
        num_tests: Number of test cases to generate

    Returns:
        List of validated test case dicts with id, category, prompt,
        evaluation_criteria, expected_elements, and difficulty.
    """
    logger.info(f"Generating {num_tests} test cases for task: {task_description[:80]}...")

    messages = [
        {
            "role": "user",
            "content": TEST_SUITE_PROMPT.format(
                task_description=task_description,
                num_tests=num_tests,
            ),
        }
    ]

    response = call_judge(messages, temperature=0.4)

    try:
        raw_cases = _extract_json_array(response)
        validated = _validate_test_cases(raw_cases, num_tests)
        if not validated:
            raise ValueError("No valid test cases after Pydantic validation")
        logger.info(f"Successfully generated {len(validated)} test cases")
        return validated
    except (ValueError, Exception) as e:
        logger.error(f"Failed to parse test suite JSON: {e}\nRaw response:\n{response[:500]}")
        return _fallback_test_suite(task_description, num_tests)


def _fallback_test_suite(task_description: str, num_tests: int) -> list[dict]:
    """
    Generate a simple fallback test suite if JSON parsing fails.

    Args:
        task_description: Natural language description of the task
        num_tests: Number of test cases to generate

    Returns:
        List of basic test case dicts
    """
    logger.warning("Using fallback test suite generation")
    categories = ["basic", "reasoning", "edge_case", "accuracy", "tool_calling"]
    difficulties = ["easy", "medium", "hard", "medium", "hard"]
    return [
        TestCase(
            id=i + 1,
            category=categories[i % len(categories)],
            prompt=f"Demonstrate your capability for the following task: {task_description}. Provide a detailed, practical example.",
            evaluation_criteria="Response should be accurate, relevant, well-structured, and demonstrate clear understanding of the task.",
            expected_elements=["relevance", "accuracy", "clarity", "practical example"],
            difficulty=difficulties[i % len(difficulties)],
        ).model_dump()
        for i in range(num_tests)
    ]
