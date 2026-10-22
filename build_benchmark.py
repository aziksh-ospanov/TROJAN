#!/usr/bin/env python3
"""
Build a Trojan benchmark file (``trojan_bench_s.json`` format) by merging raw
pipeline outputs per dataset, dropping failed samples, and drawing a seeded
random subset from each dataset.

Run with no ``--dataset`` arguments to rebuild ``trojan_bench_s.json`` from the
merged files in ``error_injected_datasets/`` (50 per dataset, seed 10).

Usage:
    python build_benchmark.py --output ../trojan_bench_s.json

    python build_benchmark.py \\
        --dataset imo_2025=error_injected_datasets/runA.json,error_injected_datasets/runB.json \\
        --dataset usamo_2025=error_injected_datasets/usamo_2025.json \\
        --per-dataset 30 --seed 0 --output my_bench.json
"""

import argparse
import json
import random
import sys

from bench_format import is_successful, report_validation, save_bench, to_bench_records


DEFAULT_DATASETS = [
    "imc_2025=error_injected_datasets/imc_2025.json",
    "imo_2025=error_injected_datasets/imo_2025.json",
    "usamo_2025=error_injected_datasets/usamo_2025.json",
    "usamo_2026=error_injected_datasets/usamo_2026.json",
]


def parse_dataset_arg(spec: str) -> tuple[str, list[str]]:
    tag, sep, paths = spec.partition("=")
    if not sep or not tag or not paths:
        raise argparse.ArgumentTypeError(
            f"expected TAG=PATH[,PATH...], got '{spec}'"
        )
    return tag, paths.split(",")


def main():
    ap = argparse.ArgumentParser(
        description="Merge, sample, and convert raw outputs into a benchmark file.",
    )
    ap.add_argument(
        "--dataset", action="append", type=parse_dataset_arg, default=None,
        metavar="TAG=PATH[,PATH...]",
        help="Dataset tag and its raw output file(s); repeatable. "
             "Order matters for seeded sampling. Default: the four merged files "
             "in error_injected_datasets/",
    )
    ap.add_argument(
        "--per-dataset", type=int, default=50,
        help="Samples drawn per dataset; -1 keeps every successful sample (default: 50)",
    )
    ap.add_argument("--seed", type=int, default=10, help="Sampling seed (default: 10)")
    ap.add_argument("--output", required=True, help="Output JSON path")
    args = ap.parse_args()

    datasets = args.dataset or [parse_dataset_arg(s) for s in DEFAULT_DATASETS]
    rng = random.Random(args.seed)

    bench = []
    for tag, paths in datasets:
        raw = []
        for path in paths:
            with open(path, encoding="utf-8") as f:
                raw.extend(json.load(f))
        successful = [r for r in raw if is_successful(r)]

        if args.per_dataset < 0:
            selected = successful
        else:
            selected = rng.sample(successful, k=min(args.per_dataset, len(successful)))

        records = to_bench_records(selected, tag)
        bench.extend(records)
        print(f"{tag}: {len(records)} selected from {len(successful)} successful "
              f"({len(raw) - len(successful)} failed) across {len(paths)} file(s)")

    save_bench(bench, args.output)
    print(f"\nSaved {len(bench)} record(s) → {args.output}")
    sys.exit(0 if report_validation(bench, args.output) else 1)


if __name__ == "__main__":
    main()
