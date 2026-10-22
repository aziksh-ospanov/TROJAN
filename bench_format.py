#!/usr/bin/env python3
"""
Convert raw pipeline outputs into the Trojan benchmark format
(the format of ``trojan_bench_s.json``) and validate files against it.

Raw record (written by ``generate_error_injected.py``)::

    problem_idx, problem, max_points, awarded_points,
    solution_grading_scheme[{title, description, max_points, awarded_points}],
    solution, initial_solution, original_sample_solution,
    refinement_iterations, converged, critique_history, token_usage

Benchmark record::

    problem_idx      "<dataset_tag>_problem_<idx>_id_<i>"
    problem          problem statement
    max_points       maximum total score
    awarded_points   target (ground-truth) total score
    solution         error-injected candidate proof (Grader's Notes stripped)
    sample_solution  reference solution from the source dataset
    token_usage      {prompt_tokens, completion_tokens, total_tokens}
    justification    the generator's "Grader's Notes" section
    grading_scheme   [{title, max_points, awarded_points, desc}]

Usage:
    python bench_format.py convert error_injected_datasets/<raw>.json \\
        --dataset-tag imo_2025 -o error_injected_datasets/imo_2025_bench.json
    python bench_format.py check ../trojan_bench_s.json
"""

import argparse
import json
import sys
from pathlib import Path


BENCH_KEYS = (
    "problem_idx",
    "problem",
    "max_points",
    "awarded_points",
    "solution",
    "sample_solution",
    "token_usage",
    "justification",
    "grading_scheme",
)

BENCH_CRITERION_KEYS = ("title", "max_points", "awarded_points", "desc")

TOKEN_USAGE_KEYS = ("prompt_tokens", "completion_tokens", "total_tokens")

SOLUTION_HEADER = "# Candidate Solution"
NOTES_HEADER    = "# Grader's Notes"


# ═══════════════════════════════════════════════════════════════════════════
#  Conversion
# ═══════════════════════════════════════════════════════════════════════════


def dataset_tag_from_path(path: str) -> str:
    """``MathArena__imo_2025__train.json`` → ``imo_2025``; otherwise the file stem."""
    parts = Path(path).stem.split("__")
    return parts[1] if len(parts) == 3 else Path(path).stem


def extract_solution_and_justification(raw_solution: str) -> tuple[str, str]:
    """Split the generator output into the proof and the Grader's Notes."""
    solution = raw_solution or ""
    justification = ""

    if NOTES_HEADER in solution:
        solution, notes = solution.split(NOTES_HEADER, 1)
        justification = notes.strip()

    if SOLUTION_HEADER in solution:
        solution = solution.split(SOLUTION_HEADER, 1)[1]

    return solution.strip(), justification


def is_successful(record: dict) -> bool:
    return "error" not in record


def _problem_sort_key(record: dict):
    idx = str(record["problem_idx"])
    return (0, int(idx), "") if idx.isdigit() else (1, 0, idx)


def to_bench_record(raw: dict, problem_idx: str) -> dict:
    """Convert one successful raw pipeline record into a benchmark record."""
    solution, justification = extract_solution_and_justification(raw["solution"])
    return {
        "problem_idx":     problem_idx,
        "problem":         raw["problem"],
        "max_points":      raw["max_points"],
        "awarded_points":  raw["awarded_points"],
        "solution":        solution,
        "sample_solution": raw["original_sample_solution"],
        "token_usage":     raw["token_usage"],
        "justification":   justification,
        "grading_scheme": [
            {
                "title":          c["title"],
                "max_points":     c["max_points"],
                "awarded_points": c["awarded_points"],
                "desc":           c["description"],
            }
            for c in raw["solution_grading_scheme"]
        ],
    }


def to_bench_records(raw_records: list[dict], dataset_tag: str) -> list[dict]:
    """Drop failed records, order by problem, and assign benchmark ids.

    Ids are ``<dataset_tag>_problem_<idx>_id_<i>`` where ``i`` is the
    0-based position of the record within this dataset.
    """
    successful = sorted(filter(is_successful, raw_records), key=_problem_sort_key)
    return [
        to_bench_record(raw, f"{dataset_tag}_problem_{raw['problem_idx']}_id_{i}")
        for i, raw in enumerate(successful)
    ]


def save_bench(records: list[dict], path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=4, ensure_ascii=False)


# ═══════════════════════════════════════════════════════════════════════════
#  Validation
# ═══════════════════════════════════════════════════════════════════════════


def validate_bench_records(records) -> tuple[list[str], list[str]]:
    """Return ``(errors, warnings)`` for a list of benchmark records."""
    errors, warnings = [], []

    if not isinstance(records, list):
        return ["top-level JSON value must be a list"], warnings

    seen_ids = set()
    for n, rec in enumerate(records):
        where = f"record {n} ({rec.get('problem_idx', '?') if isinstance(rec, dict) else '?'})"
        if not isinstance(rec, dict):
            errors.append(f"{where}: not an object")
            continue

        if tuple(rec.keys()) != BENCH_KEYS:
            errors.append(f"{where}: keys {list(rec.keys())} != {list(BENCH_KEYS)}")
            continue

        for key in ("problem_idx", "problem", "solution", "sample_solution", "justification"):
            if not isinstance(rec[key], str):
                errors.append(f"{where}: '{key}' must be a string")
        for key in ("max_points", "awarded_points"):
            if not isinstance(rec[key], int):
                errors.append(f"{where}: '{key}' must be an integer")

        if rec["problem_idx"] in seen_ids:
            errors.append(f"{where}: duplicate problem_idx")
        seen_ids.add(rec["problem_idx"])

        usage = rec["token_usage"]
        if not isinstance(usage, dict) or set(usage) != set(TOKEN_USAGE_KEYS):
            errors.append(f"{where}: token_usage must have keys {list(TOKEN_USAGE_KEYS)}")

        scheme = rec["grading_scheme"]
        if not isinstance(scheme, list) or not scheme:
            errors.append(f"{where}: grading_scheme must be a non-empty list")
            continue
        for c in scheme:
            if not isinstance(c, dict) or tuple(c.keys()) != BENCH_CRITERION_KEYS:
                errors.append(f"{where}: criterion keys must be {list(BENCH_CRITERION_KEYS)}")
                break
            if not 0 <= c["awarded_points"] <= c["max_points"]:
                errors.append(f"{where}: criterion '{c['title']}' awarded outside [0, max]")
        else:
            if sum(c["max_points"] for c in scheme) != rec["max_points"]:
                errors.append(f"{where}: criterion max_points do not sum to max_points")
            if sum(c["awarded_points"] for c in scheme) != rec["awarded_points"]:
                errors.append(f"{where}: criterion awarded_points do not sum to awarded_points")

        if not rec["solution"]:
            warnings.append(f"{where}: empty solution")
        if not rec["justification"]:
            warnings.append(f"{where}: empty justification (no Grader's Notes section)")

    return errors, warnings


def report_validation(records, label: str) -> bool:
    """Print a validation summary; return True if there are no errors."""
    errors, warnings = validate_bench_records(records)
    for w in warnings:
        print(f"  warning: {w}")
    for e in errors:
        print(f"  error:   {e}")
    status = "OK" if not errors else "INVALID"
    print(f"{label}: {status}  ({len(records)} records, "
          f"{len(errors)} errors, {len(warnings)} warnings)")
    return not errors


# ═══════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="command", required=True)

    conv = sub.add_parser("convert", help="Convert raw pipeline output(s) to benchmark format")
    conv.add_argument("inputs", nargs="+", help="Raw output JSON file(s) of one dataset")
    conv.add_argument("--dataset-tag", required=True,
                      help="Prefix for problem_idx, e.g. imo_2025")
    conv.add_argument("-o", "--output", required=True, help="Output JSON path")

    chk = sub.add_parser("check", help="Validate file(s) against the benchmark format")
    chk.add_argument("files", nargs="+")

    args = ap.parse_args()

    if args.command == "convert":
        raw = []
        for path in args.inputs:
            with open(path, encoding="utf-8") as f:
                raw.extend(json.load(f))
        records = to_bench_records(raw, args.dataset_tag)
        save_bench(records, args.output)
        print(f"Converted {len(records)}/{len(raw)} record(s) → {args.output}")
        ok = report_validation(records, args.output)
    else:
        ok = True
        for path in args.files:
            with open(path, encoding="utf-8") as f:
                ok &= report_validation(json.load(f), path)

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
