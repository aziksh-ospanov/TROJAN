#!/usr/bin/env python3
"""
Generate synthetic error-injected solutions for math competition problems
using an agentic iterative self-refinement pipeline.

Pipeline per task:

    ┌──────────┐
    │ GENERATE │  Produce an initial flawed student solution
    └────┬─────┘
         │
         ▼
    ┌──────────┐
    │ CRITIQUE │  Grade it against the target score profile
    └────┬─────┘
         │
    aligned? ──yes──▶ Done ✓
         │
        no
         │
         ▼
    ┌──────────┐
    │  REFINE  │  Fix misalignments using the critique
    └────┬─────┘
         │
         └─────▶ back to CRITIQUE  (up to --max-iterations)

Usage:
    python generate_error_injected.py \\
        original_datasets/MathArena__imo_2025__train.json \\
        --n 10 --model deepseek-v3.2-no-thinking \\
        --max-iterations 3 --num-threads 4

Writes two files to --output-dir:
    <model>_<stem>.json        full trace (critique history, initial solution, …)
    <model>_<stem>_bench.json  benchmark format (see bench_format.py)
"""

import argparse
import json, sys
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import product
from pathlib import Path

from bench_format import dataset_tag_from_path, report_validation, save_bench, to_bench_records
from models.template_generator import TemplatedGenerator


# ═══════════════════════════════════════════════════════════════════════════
#  File I/O
# ═══════════════════════════════════════════════════════════════════════════


def load_json(path: str):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_text(path: str) -> str:
    with open(path, encoding="utf-8") as f:
        return f.read()


# ═══════════════════════════════════════════════════════════════════════════
#  Score-combination sampling
# ═══════════════════════════════════════════════════════════════════════════


def enumerate_score_combinations(grading_scheme: list[dict]) -> list[tuple[int, ...]]:
    """Return every possible score tuple for a grading scheme.

    Each criterion with max m points yields scores in {0, 1, …, m}.
    """
    per_criterion = [range(c["points"] + 1) for c in grading_scheme]
    return list(product(*per_criterion))


def is_perfect_score(scores: tuple[int, ...], grading_scheme: list[dict]) -> bool:
    return all(s == c["points"] for s, c in zip(scores, grading_scheme))


def sample_score_combinations(
    grading_scheme: list[dict],
    n: int,
    rng: random.Random,
) -> list[tuple[int, ...]]:
    """Sample up to *n* non-perfect score combinations uniformly at random."""
    pool = [
        combo
        for combo in enumerate_score_combinations(grading_scheme)
        if not is_perfect_score(combo, grading_scheme)
    ]
    return rng.sample(pool, min(n, len(pool)))


# ═══════════════════════════════════════════════════════════════════════════
#  Prompt-building helpers
# ═══════════════════════════════════════════════════════════════════════════


def build_score_profile_text(
    grading_scheme: list[dict],
    scores: tuple[int, ...],
) -> str:
    """Human-readable score profile used as LLM prompt context."""
    lines = []
    for i, (criterion, awarded) in enumerate(zip(grading_scheme, scores), 1):
        title   = criterion.get("title", f"Criterion {i}")
        desc    = criterion.get("desc") or criterion.get("description", "")
        max_pts = criterion["points"]
        lost    = max_pts - awarded
        tag     = "FULL MARKS" if lost == 0 else f"LOSE {lost}/{max_pts} points"

        lines.append(
            f"  {i}. \"{title}\" ({max_pts} pts) → Awarded {awarded}/{max_pts}  [{tag}]\n"
            f"     {desc}"
        )

    total   = sum(scores)
    maximum = sum(c["points"] for c in grading_scheme)
    lines.append(f"\n  Total score: {total}/{maximum}")
    return "\n".join(lines)


def build_solution_grading_scheme(
    grading_scheme: list[dict],
    scores: tuple[int, ...],
) -> list[dict]:
    """Annotated grading scheme stored alongside the generated solution."""
    return [
        {
            "title":          c.get("title", ""),
            "description":    c.get("desc") or c.get("description", ""),
            "max_points":     c["points"],
            "awarded_points": s,
        }
        for c, s in zip(grading_scheme, scores)
    ]


# ═══════════════════════════════════════════════════════════════════════════
#  Self-Refinement Pipeline
# ═══════════════════════════════════════════════════════════════════════════


class RefinementPipeline:
    """
    Agentic iterative self-refinement pipeline for error injection.

    Three LLM roles collaborate:
      • Generator  – writes an initial flawed student solution
      • Critic     – grades it against the target score profile (JSON output)
      • Refiner    – revises misaligned parts based on the critique

    The Critique → Refine loop runs up to *max_iterations* times.
    Set max_iterations=0 for single-shot generation (no refinement).
    """

    def __init__(
        self,
        generator: TemplatedGenerator,
        critic:    TemplatedGenerator,
        refiner:   TemplatedGenerator,
        max_tokens:          int,
        max_iterations:      int = 3,
        convergence_threshold: int = 1,
    ):
        self.generator      = generator
        self.critic         = critic
        self.refiner        = refiner
        self.max_tokens     = max_tokens
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold

    # ── Token-usage accumulator ──────────────────────────────────────

    @staticmethod
    def _empty_usage() -> dict:
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    @staticmethod
    def _add_usage(accumulator: dict, result: dict) -> None:
        """Add token counts from an LLM result dict into the accumulator."""
        usage = result.get("usage", {})
        for key in accumulator:
            accumulator[key] += usage.get(key, 0)

    # ── Step 1: Generate initial solution ─────────────────────────────

    def _generate(self, prob: dict, profile_text: str) -> dict:
        """Returns the full LLM response dict (content, usage, model, …)."""
        return self.generator.generate_from_template(
            variables={
                "problem":          prob["problem"],
                "grading_scheme":   prob["grading_scheme"],
                "correct_solution": prob["sample_solution"],
                "score_profile":    profile_text,
            },
            max_tokens=self.max_tokens,
        )

    # ── Step 2: Critique the candidate ────────────────────────────────

    def _critique(self, prob: dict, profile_text: str, candidate: str) -> tuple[dict, dict]:
        """Returns (parsed_critique, raw_llm_result)."""
        result = self.critic.generate_from_template(
            variables={
                "problem":            prob["problem"],
                "grading_scheme":     prob["grading_scheme"],
                "correct_solution":   prob["sample_solution"],
                "score_profile":      profile_text,
                "candidate_solution": candidate,
            },
            max_tokens=self.max_tokens,
            temperature=0.3,
        )
        critique = self._parse_critique(result["content"])
        return critique, result

    @staticmethod
    def _parse_critique(raw_text: str) -> dict:
        """Extract structured JSON from the critic's response."""
        # Try markdown-fenced JSON first
        fenced = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", raw_text)
        text = fenced.group(1).strip() if fenced else None

        # Fall back: extract outermost { … }
        if not text:
            start, end = raw_text.find("{"), raw_text.rfind("}")
            if start != -1 and end > start:
                text = raw_text[start : end + 1]

        if text:
            try:
                parsed = json.loads(text)
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                pass

        return {
            "criteria": [],
            "overall_aligned": False,
            "revision_guidance": f"[Could not parse critique JSON]\n{raw_text}",
            "_parse_error": True,
        }

    # ── Step 3: Refine using critique feedback ────────────────────────

    def _refine(
        self,
        prob: dict,
        profile_text: str,
        candidate: str,
        critique: dict,
    ) -> dict:
        """Returns the full LLM response dict (content, usage, model, …)."""
        return self.refiner.generate_from_template(
            variables={
                "problem":            prob["problem"],
                "grading_scheme":     prob["grading_scheme"],
                "correct_solution":   prob["sample_solution"],
                "score_profile":      profile_text,
                "candidate_solution": candidate,
                "critique":           critique,
            },
            max_tokens=self.max_tokens,
        )

    # ── Full pipeline: Generate → [Critique → Refine]* ───────────────

    def run(self, task: dict) -> dict:
        """Execute the full pipeline for a single (problem, score) pair."""
        prob   = task["problem_data"]
        scores = task["scores"]
        scheme = prob["grading_scheme"]
        pid    = prob.get("problem_idx", "?")

        total_max    = sum(c["points"] for c in scheme)
        total        = sum(scores)
        profile_text = build_score_profile_text(scheme, scores)
        token_usage  = self._empty_usage()

        tag = f"[P{pid} | {total}/{total_max}]"

        # ── Step 1: Generate ──────────────────────────────────────
        print(f"  {tag}  Generating initial solution...")
        gen_result = self._generate(prob, profile_text)
        self._add_usage(token_usage, gen_result)
        candidate = gen_result["content"]
        initial_solution = candidate

        # ── Steps 2–3: Critique → Refine loop ─────────────────────
        #
        # The solution must be judged "aligned" by the critic
        # `convergence_threshold` times in a row before we accept it.
        # This guards against a single lucky/noisy positive critique.

        critique_history = []
        converged = False
        consecutive_aligned = 0

        for iteration in range(1, self.max_iterations + 1):
            print(f"  {tag}  Critique round {iteration}/{self.max_iterations}...")
            critique, crit_result = self._critique(prob, profile_text, candidate)
            self._add_usage(token_usage, crit_result)
            aligned = critique.get("overall_aligned", False)

            critique_history.append({
                "iteration":          iteration,
                "candidate_solution": candidate,
                "overall_aligned":    aligned,
                "criteria":           critique.get("criteria", []),
                "revision_guidance":  critique.get("revision_guidance", ""),
            })

            if aligned:
                consecutive_aligned += 1
                print(f"  {tag}  ✓ Aligned ({consecutive_aligned}/{self.convergence_threshold})")
                if consecutive_aligned >= self.convergence_threshold:
                    converged = True
                    print(f"  {tag}  ✓ Converged after {iteration} round(s)")
                    break
            else:
                consecutive_aligned = 0
                print(f"  {tag}  Refining (round {iteration})...")
                ref_result = self._refine(prob, profile_text, candidate, critique)
                self._add_usage(token_usage, ref_result)
                candidate = ref_result["content"]

        if not converged and self.max_iterations > 0:
            print(f"  {tag}  ⚠ Max iterations reached without convergence")

        print(f"  {tag}  Tokens: {token_usage['total_tokens']} total "
              f"({token_usage['prompt_tokens']} prompt + "
              f"{token_usage['completion_tokens']} completion)")

        return {
            "problem_idx":              pid,
            "problem":                  prob["problem"],
            "max_points":               prob.get("points", total_max),
            "awarded_points":           total,
            "solution_grading_scheme":  build_solution_grading_scheme(scheme, scores),
            "solution":                 candidate,
            "initial_solution":         initial_solution,
            "original_sample_solution": prob["sample_solution"],
            "refinement_iterations":    len(critique_history),
            "converged":                converged,
            "critique_history":         critique_history,
            "token_usage":              token_usage,
        }


# ═══════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════


def main():
    ap = argparse.ArgumentParser(
        description="Generate error-injected math solutions with iterative self-refinement.",
    )
    ap.add_argument(
        "input",
        help="Path to problem JSON "
             "(e.g. original_datasets/MathArena__imo_2025__train.json)",
    )
    ap.add_argument(
        "--n", type=int, default=10,
        help="Total samples to generate, distributed across problems (default: 10)",
    )
    ap.add_argument(
        "--model", default="deepseek-v3.2-no-thinking",
        help="Model key from llm_config.json",
    )
    ap.add_argument(
        "--output-dir", default="error_injected_datasets",
        help="Directory for output files",
    )
    ap.add_argument(
        "--template", default="templates/error_injection_template_one_shot.txt",
        help="Generation prompt template",
    )
    ap.add_argument(
        "--critique-template", default="templates/critique_template.txt",
        help="Critique prompt template",
    )
    ap.add_argument(
        "--refine-template", default="templates/refinement_template.txt",
        help="Refinement prompt template",
    )
    ap.add_argument(
        "--config", default="llm_config.json",
        help="Path to LLM config JSON",
    )
    ap.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for reproducibility",
    )
    ap.add_argument(
        "--num-threads", type=int, default=1,
        help="Parallel threads for LLM calls (default: 1)",
    )
    ap.add_argument(
        "--max-iterations", type=int, default=3,
        help="Max critique→refine rounds per sample; 0 = single-shot (default: 3)",
    )
    ap.add_argument(
        "--convergence-threshold", type=int, default=1,
        help="Consecutive aligned critiques required before accepting (default: 1)",
    )

    ap.add_argument(
        "--name", type=str, default=None,
        help="Name of the dataset to save the results as",
    )
    ap.add_argument(
        "--dataset-tag", type=str, default=None,
        help="problem_idx prefix in the benchmark-format output "
             "(default: derived from input, e.g. imo_2025)",
    )
    args = ap.parse_args()

    # ── Load resources ────────────────────────────────────────────────
    problems = load_json(args.input)
    llm_cfg  = load_json(args.config)
    model    = llm_cfg[args.model]

    def make_generator(template_path: str) -> TemplatedGenerator:
        return TemplatedGenerator(
            template=load_text(template_path),
            api_key=model["api"],
            base_url=model["url"],
            model_name=model["name"],
        )

    pipeline = RefinementPipeline(
        generator=make_generator(args.template),
        critic=make_generator(args.critique_template),
        refiner=make_generator(args.refine_template),
        max_tokens=model.get("max_tokens", 8192),
        max_iterations=args.max_iterations,
        convergence_threshold=args.convergence_threshold,
    )

    rng = random.Random(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Identify eligible problems ────────────────────────────────────
    eligible = [p for p in problems if "sample_solution" in p]
    skipped  = len(problems) - len(eligible)
    if skipped:
        print(f"Skipped {skipped} problem(s) without sample_solution")
    if not eligible:
        print("No eligible problems. Exiting.")
        return

    # ── Distribute n samples evenly across problems ───────────────────
    base, extra = divmod(args.n, len(eligible))
    per_problem = [base + (1 if i < extra else 0) for i in range(len(eligible))]

    tasks = []
    for prob, n_samples in zip(eligible, per_problem):
        pid       = prob.get("problem_idx", "?")
        scheme    = prob["grading_scheme"]
        total_max = sum(c["points"] for c in scheme)
        combos    = sample_score_combinations(scheme, n=n_samples, rng=rng)

        print(f"Problem {pid}: {len(combos)} sample(s)  "
              f"({len(scheme)} criteria, {total_max} pts)")

        for scores in combos:
            tasks.append({"problem_data": prob, "scores": scores})

    print(f"\nTotal tasks: {len(tasks)}  |  Threads: {args.num_threads}  "
          f"|  Max rounds: {args.max_iterations}  "
          f"|  Convergence threshold: {args.convergence_threshold}\n")

    # ── Execute in parallel ───────────────────────────────────────────
    TASK_RETRIES = 3

    def run_with_retries(task: dict) -> dict:
        """Run the pipeline with task-level retries for transient failures."""
        pid = task["problem_data"].get("problem_idx", "?")
        last_exc = None
        for attempt in range(1, TASK_RETRIES + 1):
            try:
                return pipeline.run(task)
            except (ConnectionError, OSError) as exc:
                last_exc = exc
                if attempt < TASK_RETRIES:
                    wait = min(2 ** attempt, 30) + random.random()
                    print(f"  [P{pid}] Task retry {attempt}/{TASK_RETRIES} "
                          f"after {wait:.1f}s ({type(exc).__name__})")
                    time.sleep(wait)
        raise last_exc

    results = [None] * len(tasks)

    with ThreadPoolExecutor(max_workers=args.num_threads) as pool:
        futures = {
            pool.submit(run_with_retries, task): i
            for i, task in enumerate(tasks)
        }
        for future in as_completed(futures):
            idx = futures[future]
            try:
                results[idx] = future.result()
            except Exception as exc:
                pid = tasks[idx]["problem_data"].get("problem_idx", "?")
                print(f"  [P{pid}] ERROR: {exc}")
                results[idx] = {
                    "problem_idx": pid,
                    "error": str(exc),
                }

    # ── Save output ───────────────────────────────────────────────────
    stem     = Path(args.input).stem
    if args.name:
        stem = args.name
    else:
        stem = f"{stem}_conv_threshold_{args.convergence_threshold}_max_iterations_{args.max_iterations}_num_threads_{args.num_threads}_seed_{args.seed}"
    out_path = output_dir / f"{args.model}_{stem}.json"

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    dataset_tag = args.dataset_tag or dataset_tag_from_path(args.input)
    bench_path  = out_path.with_name(f"{out_path.stem}_bench.json")
    bench       = to_bench_records(results, dataset_tag)
    save_bench(bench, bench_path)

    successful   = [r for r in results if "error" not in r]
    n_converged  = sum(1 for r in successful if r.get("converged"))
    total_tokens = sum(r.get("token_usage", {}).get("total_tokens", 0) for r in successful)

    print(f"\nSaved {len(results)} sample(s) → {out_path}")
    print(f"Saved {len(bench)} benchmark-format sample(s) → {bench_path}")
    report_validation(bench, bench_path.name)
    print(f"  Converged: {n_converged}/{len(successful)}  "
          f"|  Errors: {len(results) - len(successful)}  "
          f"|  Total tokens: {total_tokens:,}")


if __name__ == "__main__":
    main()
