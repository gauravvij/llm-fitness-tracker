"""
LLM Evaluator Tool - CLI entry point.

Automates the selection and evaluation of LLMs for any user-defined task.
Uses Gemini 3.1 Pro (via OpenRouter) as the Judge LLM.
"""

import sys
import logging
import argparse
from pathlib import Path

from rich.console import Console
from rich.prompt import Prompt
from rich.rule import Rule

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.config import load_api_key, MAX_TEST_CASES, MAX_CANDIDATES
from src.suite_generator import generate_test_suite
from src.model_discovery import discover_candidate_models
from src.benchmarker import run_benchmark, compute_latency_stats
from src.evaluator import evaluate_all_results, rank_models
from src.prompt_optimizer import generate_optimized_prompt
from src.reporter import (
    display_header,
    display_test_suite,
    display_candidates,
    display_evaluation_results,
    display_ranking,
    display_optimized_prompt,
    save_report,
    console,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def run_evaluation(
    task_description: str,
    num_tests: int = MAX_TEST_CASES,
    max_candidates: int = MAX_CANDIDATES,
    save_json: bool = True,
    output_dir: str = "./analysis",
) -> dict:
    """
    Run the full LLM evaluation pipeline for a given task.

    Args:
        task_description: Natural language description of the task
        num_tests: Number of test cases to generate
        max_candidates: Maximum number of candidate models to evaluate
        save_json: Whether to save a JSON report
        output_dir: Directory to save the report

    Returns:
        Dict with full results including ranking and optimized prompt
    """
    display_header(task_description)

    # ── Step 1: Generate test suite ──────────────────────────────────────────
    console.print(Rule("[bold green]Step 1/5 — Generating Test Suite[/bold green]"))
    console.print(f"[dim]Using Judge LLM: google/gemini-3.1-pro-preview[/dim]\n")
    test_cases = generate_test_suite(task_description, num_tests=num_tests)
    display_test_suite(test_cases)

    # ── Step 2: Discover candidate models ────────────────────────────────────
    console.print(Rule("[bold blue]Step 2/5 — Discovering Candidate Models[/bold blue]"))
    candidates = discover_candidate_models(task_description, max_candidates=max_candidates)
    display_candidates(candidates)

    # ── Step 3: Run benchmark ─────────────────────────────────────────────────
    console.print(Rule("[bold yellow]Step 3/5 — Running Benchmark[/bold yellow]"))
    console.print(f"[dim]Running {len(test_cases)} tests × {len(candidates)} models...[/dim]\n")
    benchmark_results = run_benchmark(candidates, test_cases, max_workers=3)

    # ── Step 4: Evaluate responses ────────────────────────────────────────────
    console.print(Rule("[bold magenta]Step 4/5 — Evaluating Responses[/bold magenta]"))
    console.print("[dim]Judge LLM scoring each response on accuracy, hallucination, grounding, tool-calling, clarity...[/dim]\n")
    model_evaluations = evaluate_all_results(task_description, benchmark_results)
    display_evaluation_results(model_evaluations, candidates)

    # ── Step 5: Rank models & generate optimized prompt ───────────────────────
    console.print(Rule("[bold cyan]Step 5/5 — Ranking & Prompt Optimization[/bold cyan]"))
    ranking = rank_models(task_description, model_evaluations, candidates)
    display_ranking(ranking, model_evaluations)

    # Generate optimized prompt for top model
    top_entries = ranking.get("ranking", [])
    optimized_prompt = ""
    if top_entries:
        top_entry = top_entries[0]
        top_model_id = top_entry.get("model_id", "")
        top_scores = model_evaluations.get(top_model_id, {})
        optimized_prompt = generate_optimized_prompt(task_description, top_entry, top_scores)
        display_optimized_prompt(optimized_prompt, top_model_id)

    # ── Save report ───────────────────────────────────────────────────────────
    report_path = ""
    if save_json:
        report_path = save_report(
            task_description=task_description,
            test_cases=test_cases,
            candidates=candidates,
            benchmark_results=benchmark_results,
            model_evaluations=model_evaluations,
            ranking=ranking,
            optimized_prompt=optimized_prompt,
            output_dir=output_dir,
        )

    console.print("\n[bold green]✅ Evaluation complete![/bold green]\n")

    return {
        "task_description": task_description,
        "test_cases": test_cases,
        "candidates": candidates,
        "benchmark_results": benchmark_results,
        "model_evaluations": model_evaluations,
        "ranking": ranking,
        "optimized_prompt": optimized_prompt,
        "report_path": report_path,
    }


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        prog="llm-evaluator",
        description="LLM Evaluator Tool — Automated LLM Selection & Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --task "Python software engineering assistant"
  python main.py --task "Math tutoring for high school students" --num-tests 3
  python main.py --task "Customer support chatbot" --max-candidates 4 --no-save
  python main.py  # Interactive mode (prompts for task)
        """,
    )
    parser.add_argument(
        "--task", "-t",
        type=str,
        default=None,
        help="Natural language description of the task to evaluate LLMs for",
    )
    parser.add_argument(
        "--num-tests", "-n",
        type=int,
        default=MAX_TEST_CASES,
        help=f"Number of test cases to generate (default: {MAX_TEST_CASES})",
    )
    parser.add_argument(
        "--max-candidates", "-c",
        type=int,
        default=MAX_CANDIDATES,
        help=f"Maximum number of candidate models (default: {MAX_CANDIDATES})",
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="./analysis",
        help="Directory to save JSON report (default: ./analysis)",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not save JSON report to disk",
    )
    return parser.parse_args()


def main() -> None:
    """Main entry point for the LLM Evaluator Tool CLI."""
    # Validate API key early
    try:
        load_api_key()
    except ValueError as e:
        console.print(f"[bold red]❌ Configuration Error:[/bold red] {e}")
        sys.exit(1)

    args = parse_args()

    # Get task description — from arg or interactive prompt
    task_description = args.task
    if not task_description:
        console.print(Panel(
            "[bold cyan]Welcome to the LLM Fitness Tool![/bold cyan]\n\n"
            "This tool automatically selects and evaluates the best LLMs for your task.\n"
            "It uses [bold]Gemini 3.1 Pro[/bold] as the Judge LLM via OpenRouter.",
            border_style="cyan",
        ))
        task_description = Prompt.ask(
            "\n[bold yellow]Enter your task description[/bold yellow]",
            default="Python software engineering assistant",
        )

    # Validate task description
    task_description = task_description.strip()
    if len(task_description) < 10:
        console.print("[bold red]❌ Task description too short. Please provide at least 10 characters.[/bold red]")
        sys.exit(1)
    if len(task_description) > 2000:
        console.print("[bold red]❌ Task description too long. Please keep it under 2000 characters.[/bold red]")
        sys.exit(1)

    # Run evaluation
    try:
        run_evaluation(
            task_description=task_description,
            num_tests=args.num_tests,
            max_candidates=args.max_candidates,
            save_json=not args.no_save,
            output_dir=args.output_dir,
        )
    except KeyboardInterrupt:
        console.print("\n[yellow]⚠ Evaluation interrupted by user.[/yellow]")
        sys.exit(0)
    except Exception as e:
        console.print(f"\n[bold red]❌ Evaluation failed:[/bold red] {e}")
        logger.exception("Unexpected error during evaluation")
        sys.exit(1)


if __name__ == "__main__":
    # Import Panel here to avoid circular import issues at module level
    from rich.panel import Panel
    main()
