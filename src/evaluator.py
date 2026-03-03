"""Multi-faceted evaluation of LLM responses using the Judge LLM."""

import json
import logging
import sys
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FuturesTimeoutError
from typing import Optional

from pydantic import ValidationError

from .openrouter_client import call_judge
from .schemas import EvaluationScore, RankingResult, RankingEntry

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler(sys.stdout))


# ---------------------------------------------------------------------------
# JSON extraction helpers
# ---------------------------------------------------------------------------

def _extract_json_object(text: str) -> dict:
    """
    Robustly extract a JSON object from text that may contain markdown fences,
    thinking tags, trailing commas, or truncated content.
    """
    if not text or not text.strip():
        raise ValueError("Cannot extract JSON from: empty response")

    # Remove <think>...</think> blocks (Gemini/DeepSeek reasoning traces)
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    # Strip markdown fences
    cleaned = re.sub(r"```(?:json)?\s*", "", cleaned).strip().rstrip("`").strip()

    # Try direct parse first
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Find the outermost { ... } block tracking full brace depth.
    # We track bracket depth too so nested arrays don't confuse the brace counter.
    brace_depth = 0
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
            bracket_depth += 1
        elif ch == "]":
            bracket_depth -= 1
        elif ch == "{":
            if brace_depth == 0:
                start_idx = i
            brace_depth += 1
        elif ch == "}":
            brace_depth -= 1
            # Only attempt parse when ALL braces AND brackets are closed
            if brace_depth == 0 and bracket_depth == 0 and start_idx is not None:
                candidate = cleaned[start_idx : i + 1]
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    fixed = re.sub(r",\s*([}\]])", r"\1", candidate)
                    try:
                        return json.loads(fixed)
                    except json.JSONDecodeError:
                        pass
                    # Don't reset — keep looking for a later valid block
                    start_idx = None
                    brace_depth = 0
                    bracket_depth = 0

    # Fix trailing commas in the whole text
    fixed = re.sub(r",\s*([}\]])", r"\1", cleaned)
    try:
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass

    # Last resort: extract individual numeric fields via regex
    scores: dict = {}
    for key in ["accuracy", "hallucination", "grounding", "reasoning", "clarity", "overall"]:
        m = re.search(rf'"{key}"\s*:\s*(\d+(?:\.\d+)?)', cleaned)
        if m:
            scores[key] = float(m.group(1))
    if len(scores) >= 3:
        for key in ["accuracy", "hallucination", "grounding", "reasoning", "clarity", "overall"]:
            scores.setdefault(key, 5.0)
        rt = re.search(r'"reasoning_text"\s*:\s*"([^"]*)"', cleaned)
        scores["reasoning_text"] = rt.group(1) if rt else "Extracted via regex fallback."
        return scores

    raise ValueError(f"Cannot extract JSON from: {text[:200]}")


def _parse_evaluation_score(raw: str) -> dict:
    """
    Parse and validate an evaluation score response using Pydantic.

    Falls back gracefully through multiple strategies before returning defaults.

    Args:
        raw: Raw text response from the Judge LLM

    Returns:
        Validated score dict with all required fields populated
    """
    # Strategy 1: Extract JSON then validate with Pydantic
    try:
        data = _extract_json_object(raw)
        score = EvaluationScore(**data)
        return score.to_dict()
    except (ValueError, ValidationError) as e:
        logger.debug(f"Pydantic validation failed on extracted JSON: {e}")

    # Strategy 2: Regex extraction of numeric fields
    scores: dict = {}
    for key in ["accuracy", "hallucination", "grounding", "reasoning", "clarity", "overall"]:
        m = re.search(rf'"{key}"\s*:\s*(\d+(?:\.\d+)?)', raw)
        if m:
            scores[key] = float(m.group(1))
    if scores:
        rt = re.search(r'"reasoning_text"\s*:\s*"([^"]*)"', raw)
        scores["reasoning_text"] = rt.group(1) if rt else "Extracted via regex fallback."
        try:
            score = EvaluationScore(**scores)
            return score.to_dict()
        except ValidationError:
            pass

    # Strategy 3: Return safe defaults via Pydantic
    logger.warning("All JSON extraction strategies failed; returning default scores.")
    return EvaluationScore().to_dict()


def _parse_ranking_result(raw: str, fallback_models: list[tuple[str, dict]]) -> dict:
    """
    Parse and validate a ranking response using Pydantic.

    Falls back to score-based ranking if the Judge LLM response cannot be parsed.

    Args:
        raw: Raw text response from the Judge LLM
        fallback_models: Sorted list of (model_id, scores) for fallback ranking

    Returns:
        Validated ranking dict
    """
    # Strategy 1: Extract JSON then validate with Pydantic
    try:
        data = _extract_json_object(raw)
        result = RankingResult(**data)
        return result.to_dict()
    except (ValueError, ValidationError) as e:
        logger.warning(f"Failed to parse ranking JSON via Pydantic: {e}")

    # Strategy 2: Try to extract ranking array directly
    try:
        arr_match = re.search(r'"ranking"\s*:\s*(\[.*?\])', raw, re.DOTALL)
        if arr_match:
            ranking_list = json.loads(arr_match.group(1))
            result = RankingResult(ranking=ranking_list)
            return result.to_dict()
    except (json.JSONDecodeError, ValidationError):
        pass

    # Strategy 3: Build fallback ranking from aggregated scores
    logger.warning("Ranking JSON parse failed — building fallback ranking from scores.")
    entries = []
    for i, (mid, scores) in enumerate(fallback_models[:3]):
        overall = scores.get("overall", 0.0)
        # Derive meaningful strengths/weaknesses from scores
        strengths = []
        weaknesses = []
        if scores.get("hallucination", 0) >= 8:
            strengths.append("High factual accuracy with minimal hallucination")
        if scores.get("accuracy", 0) >= 8:
            strengths.append("Strong response accuracy across test cases")
        if scores.get("grounding", 0) >= 8:
            strengths.append("Well-grounded responses tied to prompt requirements")
        if scores.get("avg_latency", 999) < 15:
            strengths.append(f"Fast response time ({scores['avg_latency']:.1f}s avg)")
        if scores.get("accuracy", 10) < 7:
            weaknesses.append("Accuracy below benchmark threshold on some test cases")
        if scores.get("grounding", 10) < 7:
            weaknesses.append("Responses occasionally drift from prompt constraints")
        if scores.get("avg_latency", 0) > 60:
            weaknesses.append(f"High latency ({scores['avg_latency']:.1f}s avg) may impact UX")
        if not strengths:
            strengths = ["Competitive overall benchmark performance"]
        if not weaknesses:
            weaknesses = ["No critical weaknesses identified in benchmark"]

        entries.append(
            RankingEntry(
                rank=i + 1,
                model_id=mid,
                overall_score=overall,
                strengths=strengths,
                weaknesses=weaknesses,
                recommendation=f"Ranked #{i+1} by aggregated benchmark score ({overall:.1f}/10).",
            )
        )

    result = RankingResult(
        ranking=entries,
        summary=(
            "Models ranked by aggregated evaluation scores across all test cases. "
            "The ranking reflects performance on accuracy, hallucination resistance, "
            "grounding, reasoning depth, and response clarity."
        ),
    )
    return result.to_dict()


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

EVALUATION_PROMPT = """You are a rigorous, expert AI evaluator with deep knowledge of LLM capabilities. Your task is to critically evaluate an LLM's response to a benchmark test. You must be STRICT and DISCRIMINATING — do not inflate scores. A score of 8+ should only be given for truly exceptional responses.

## Task Context
{task_description}

## Test Case
- **Category**: {test_category}
- **Difficulty**: {test_difficulty}
- **Prompt**: {prompt}
- **Evaluation Criteria**: {evaluation_criteria}
- **Expected Elements**: {expected_elements}

## LLM Response to Evaluate
{response}

## Critical Evaluation Instructions

You MUST evaluate with the following strict criteria:

### 1. accuracy (0-10)
- Does the response FULLY and CORRECTLY address ALL parts of the prompt?
- Does it include ALL expected elements listed above?
- Deduct 2 points for each missing expected element.
- Deduct 3 points for any factually incorrect code or logic.
- Deduct 2 points for incomplete responses (e.g., response cuts off mid-explanation).
- Score 9-10 ONLY if the response is complete, correct, and addresses every nuance.

### 2. hallucination (0-10)
- 10 = zero fabricated facts, APIs, or code that doesn't work.
- Deduct 3 points for any invented API calls or non-existent functions.
- Deduct 2 points for incorrect claims presented as facts.
- Deduct 1 point for each unverifiable assertion.
- Code that appears syntactically correct but has logical bugs counts as partial hallucination (deduct 1-2 points).

### 3. grounding (0-10)
- Is the response directly grounded in the specific requirements of the prompt?
- Deduct 2 points if the response gives a generic answer that ignores specific constraints.
- Deduct 2 points if the response misses key constraints (e.g., word boundaries, case-insensitivity, recursion).
- Deduct 3 points if the response solves a DIFFERENT problem than what was asked.
- Deduct 2 points for responses that are truncated or cut off before completing the solution.

### 4. reasoning (0-10)
- Does the response demonstrate DEEP reasoning and understanding of the problem?
- For coding tasks: Does it explain WHY certain approaches are used?
- For debugging tasks: Does it identify ALL bugs, not just the obvious ones?
- Deduct 2 points for shallow explanations that just state what the code does without explaining why.
- Deduct 3 points for missing critical edge cases that were explicitly mentioned.
- Score 9-10 ONLY for responses that show expert-level insight and anticipate edge cases.

### 5. clarity (0-10)
- Is the response well-structured, readable, and professional?
- Deduct 1 point for poor formatting or disorganized structure.
- Deduct 2 points for responses that are confusing or hard to follow.
- Deduct 2 points for overly verbose responses that bury the key information.

### Overall Score Calculation
The overall score MUST be a weighted average:
- accuracy: 35% weight
- hallucination: 20% weight
- grounding: 20% weight
- reasoning: 15% weight
- clarity: 10% weight

Formula: overall = (accuracy*0.35 + hallucination*0.20 + grounding*0.20 + reasoning*0.15 + clarity*0.10)

### IMPORTANT CALIBRATION NOTES:
- A response that is TRUNCATED or INCOMPLETE must score no higher than 5 on accuracy and grounding.
- A response that misses multiple expected elements should score 4-6 on accuracy.
- A response that correctly implements ALL expected elements with proper edge case handling deserves 8-9.
- Reserve scores of 9-10 for truly exceptional, production-ready responses.
- Do NOT give the same score to responses of clearly different quality.
- Be honest about weaknesses — every model has them. Do not leave weaknesses empty.

You MUST respond with ONLY a valid JSON object. No markdown fences, no preamble, no explanation outside the JSON.
The JSON must have exactly these keys: accuracy, hallucination, grounding, reasoning, clarity, overall, reasoning_text.

Example of the EXACT format required:
{{"accuracy": 7.5, "hallucination": 9.0, "grounding": 6.5, "reasoning": 7.0, "clarity": 8.0, "overall": 7.65, "reasoning_text": "The response correctly addresses the main requirements but misses two expected elements. The code logic is sound with no hallucinated APIs. Grounding is partial as the response ignores the case-insensitivity constraint. Reasoning is adequate but lacks depth on edge cases."}}"""


RANKING_PROMPT = """You are an expert AI systems evaluator with deep practical knowledge of LLM capabilities. Based on the benchmark results below, provide a final ranking and analysis.

## Task Description
{task_description}

## Benchmark Results Summary
{results_summary}

## Ranking Instructions
- Rank models based on their PRACTICAL suitability for the task, not just raw scores.
- Consider the CONSISTENCY of performance across all test cases (high variance is a weakness).
- A model with slightly lower scores but more consistent performance may be preferable.
- Consider latency as a tiebreaker — faster is better when scores are close (within 0.5 points).
- Be honest about weaknesses — do NOT leave weaknesses empty. Every model has areas for improvement.
- Strengths and weaknesses MUST each contain at least 1 specific, concrete item.

Provide a final ranking of the top 3 models.

You MUST respond with ONLY a valid JSON object. No markdown fences, no preamble, no explanation outside the JSON.
The JSON must have exactly this structure:

{{"ranking": [{{"rank": 1, "model_id": "<model_id>", "overall_score": <0-10>, "strengths": ["<specific strength 1>", "<specific strength 2>"], "weaknesses": ["<specific weakness 1>"], "recommendation": "<one sentence why this model is best for the task>"}}, {{"rank": 2, "model_id": "<model_id>", "overall_score": <0-10>, "strengths": ["<specific strength>"], "weaknesses": ["<specific weakness>"], "recommendation": "<one sentence>"}}, {{"rank": 3, "model_id": "<model_id>", "overall_score": <0-10>, "strengths": ["<specific strength>"], "weaknesses": ["<specific weakness>"], "recommendation": "<one sentence>"}}], "summary": "<3-4 sentence overall analysis that honestly compares the models and explains the ranking decisions>"}}"""


# ---------------------------------------------------------------------------
# Core evaluation functions
# ---------------------------------------------------------------------------

def evaluate_response(
    task_description: str,
    result: dict,
    retry_on_rate_limit: bool = True,
) -> dict:
    """
    Evaluate a single LLM response using the Judge LLM.

    Args:
        task_description: The original task description
        result: A benchmark result dict from benchmarker.run_single_test
        retry_on_rate_limit: Whether to retry on rate limit errors

    Returns:
        Validated evaluation dict with scores for each dimension
    """
    if result.get("error"):
        return EvaluationScore(
            accuracy=0, hallucination=0, grounding=0,
            reasoning=0, clarity=0,
            reasoning_text="Response contained an API error.",
        ).to_dict()

    messages = [
        {
            "role": "system",
            "content": (
                "You are a JSON-only responder. You MUST output a single valid JSON object "
                "and absolutely nothing else — no markdown fences, no preamble, no explanation, "
                "no thinking tags. Start your response with '{' and end with '}'."
            ),
        },
        {
            "role": "user",
            "content": EVALUATION_PROMPT.format(
                task_description=task_description,
                test_category=result.get("test_category", "general"),
                test_difficulty=result.get("test_difficulty", "medium"),
                prompt=result["prompt"],
                evaluation_criteria=result.get("evaluation_criteria", "N/A"),
                expected_elements=", ".join(result.get("expected_elements", [])),
                response=result["response"][:4000],
            ),
        },
    ]

    max_retries = 3 if retry_on_rate_limit else 1
    for attempt in range(max_retries):
        raw = call_judge(messages, temperature=0.1, max_tokens=1500)

        # Handle rate limit errors with exponential backoff
        if raw.startswith("ERROR:") and ("rate" in raw.lower() or "429" in raw):
            wait_time = (2 ** attempt) * 5  # 5s, 10s, 20s
            logger.warning(
                f"Rate limit hit for evaluation. Waiting {wait_time}s "
                f"before retry {attempt + 1}/{max_retries}..."
            )
            time.sleep(wait_time)
            continue

        scores = _parse_evaluation_score(raw)
        if scores.get("overall", 0) > 0 or attempt == max_retries - 1:
            return scores

    return EvaluationScore().to_dict()


def evaluate_all_results(
    task_description: str,
    benchmark_results: dict[str, list[dict]],
    max_parallel_evaluations: int = 4,
) -> dict[str, dict]:
    """
    Evaluate all benchmark results for all models using parallel execution.

    Runs judge evaluations concurrently across all (model, test) pairs to
    significantly reduce total evaluation time. Rate limits are handled
    gracefully with exponential backoff.

    Args:
        task_description: The original task description
        benchmark_results: Dict mapping model_id -> list of result dicts
        max_parallel_evaluations: Max concurrent judge API calls (default 4)

    Returns:
        Dict mapping model_id -> aggregated evaluation scores
    """
    all_tasks: list[tuple[str, dict]] = []
    for model_id, results in benchmark_results.items():
        for result in results:
            all_tasks.append((model_id, result))

    total_tasks = len(all_tasks)
    logger.info(
        f"Starting parallel evaluation: {total_tasks} judge calls "
        f"({max_parallel_evaluations} concurrent workers)"
    )

    raw_scores: dict[tuple[str, int], dict] = {}
    completed = 0

    def _eval_task(model_id: str, result: dict) -> tuple[str, int, dict]:
        """Evaluate a single (model, test) pair and return (model_id, test_id, scores)."""
        scores = evaluate_response(task_description, result, retry_on_rate_limit=True)
        scores["test_id"] = result["test_id"]
        scores["latency"] = result["latency"]
        return model_id, result["test_id"], scores

    with ThreadPoolExecutor(max_workers=max_parallel_evaluations) as executor:
        future_to_task = {
            executor.submit(_eval_task, model_id, result): (model_id, result["test_id"])
            for model_id, result in all_tasks
        }

        for future in as_completed(future_to_task):
            model_id_key, test_id_key = future_to_task[future]
            try:
                mid, tid, scores = future.result(timeout=180)
                raw_scores[(mid, tid)] = scores
                completed += 1
                logger.info(
                    f"  [{completed}/{total_tasks}] Evaluated {mid} | Test {tid} | "
                    f"Overall: {scores.get('overall', 0):.1f}"
                )
            except FuturesTimeoutError:
                logger.error(f"Evaluation timed out for {model_id_key} test {test_id_key}")
                raw_scores[(model_id_key, test_id_key)] = EvaluationScore(
                    reasoning_text="Evaluation timed out.",
                ).to_dict()
                raw_scores[(model_id_key, test_id_key)].update(
                    {"test_id": test_id_key, "latency": 0}
                )
            except Exception as e:
                logger.error(f"Evaluation failed for {model_id_key} test {test_id_key}: {e}")
                raw_scores[(model_id_key, test_id_key)] = EvaluationScore(
                    reasoning_text=f"Evaluation error: {str(e)}",
                ).to_dict()
                raw_scores[(model_id_key, test_id_key)].update(
                    {"test_id": test_id_key, "latency": 0}
                )

    # Aggregate scores per model
    model_evaluations: dict[str, dict] = {}
    dims = ["accuracy", "hallucination", "grounding", "reasoning", "tool_calling", "clarity", "overall"]

    for model_id, results in benchmark_results.items():
        per_test_scores = [
            raw_scores.get((model_id, r["test_id"]), {})
            for r in results
        ]
        per_test_scores = [s for s in per_test_scores if s]

        if per_test_scores:
            aggregated: dict = {}
            for dim in dims:
                vals = [s[dim] for s in per_test_scores if isinstance(s.get(dim), (int, float))]
                aggregated[dim] = round(sum(vals) / len(vals), 2) if vals else 0.0

            latencies = [s["latency"] for s in per_test_scores if isinstance(s.get("latency"), (int, float))]
            aggregated["avg_latency"] = round(sum(latencies) / len(latencies), 3) if latencies else 0.0
            aggregated["per_test"] = per_test_scores
        else:
            aggregated = {dim: 0.0 for dim in dims}
            aggregated["avg_latency"] = 0.0
            aggregated["per_test"] = []

        model_evaluations[model_id] = aggregated
        logger.info(
            f"  {model_id}: overall={aggregated['overall']:.1f}, "
            f"accuracy={aggregated['accuracy']:.1f}, "
            f"grounding={aggregated['grounding']:.1f}, "
            f"avg_latency={aggregated['avg_latency']:.2f}s"
        )

    return model_evaluations


def rank_models(
    task_description: str,
    model_evaluations: dict[str, dict],
    candidates: list[dict],
) -> dict:
    """
    Use the Judge LLM to produce a final ranked list of top 3 models.

    Args:
        task_description: The original task description
        model_evaluations: Dict mapping model_id -> aggregated scores
        candidates: List of candidate model dicts with metadata

    Returns:
        Validated ranking dict with top 3 models and analysis
    """
    candidate_map = {c["id"]: c for c in candidates}
    summary_lines = []

    for model_id, scores in model_evaluations.items():
        name = candidate_map.get(model_id, {}).get("name", model_id)
        per_test = scores.get("per_test", [])
        overall_scores = [
            t.get("overall", 0) for t in per_test
            if isinstance(t.get("overall"), (int, float))
        ]
        if len(overall_scores) > 1:
            mean = sum(overall_scores) / len(overall_scores)
            variance = round(
                sum((x - mean) ** 2 for x in overall_scores) / len(overall_scores), 2
            )
            consistency_note = f"Score variance: {variance:.2f} (lower = more consistent)"
        else:
            consistency_note = "Single test result"

        summary_lines.append(
            f"Model: {model_id} ({name})\n"
            f"  Overall: {scores['overall']:.1f}/10 | "
            f"Accuracy: {scores['accuracy']:.1f} | "
            f"Hallucination: {scores['hallucination']:.1f} | "
            f"Grounding: {scores['grounding']:.1f} | "
            f"Reasoning: {scores.get('reasoning', scores.get('tool_calling', 0)):.1f} | "
            f"Clarity: {scores['clarity']:.1f} | "
            f"Avg Latency: {scores['avg_latency']:.2f}s | "
            f"{consistency_note}"
        )

    results_summary = "\n\n".join(summary_lines)

    messages = [
        {
            "role": "system",
            "content": (
                "You are a JSON-only responder. You MUST output a single valid JSON object "
                "and absolutely nothing else — no markdown fences, no preamble, no explanation, "
                "no thinking tags. Start your response with '{' and end with '}'."
            ),
        },
        {
            "role": "user",
            "content": RANKING_PROMPT.format(
                task_description=task_description,
                results_summary=results_summary,
            ),
        },
    ]

    raw = call_judge(messages, temperature=0.2, max_tokens=4096)

    # Build sorted fallback list for use if parsing fails
    sorted_models = sorted(
        model_evaluations.items(),
        key=lambda x: x[1].get("overall", 0),
        reverse=True,
    )

    if not raw or raw.startswith("ERROR:"):
        logger.error(f"Judge LLM returned error or empty response for ranking: {raw[:200]}")
        return _parse_ranking_result("", sorted_models)

    return _parse_ranking_result(raw, sorted_models)
