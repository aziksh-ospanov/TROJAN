"""Download a Hugging Face dataset and save it as JSON to original_datasets/."""

import argparse
import json
from pathlib import Path

from datasets import load_dataset


def main():
    parser = argparse.ArgumentParser(
        description="Download a HuggingFace dataset and save as JSON."
    )
    parser.add_argument("dataset", help="HuggingFace dataset ID (e.g. 'squad', 'user/repo')")
    parser.add_argument("--subset", default=None, help="Dataset subset/config name")
    parser.add_argument("--split", default=None, help="Specific split to download (default: all splits)")
    parser.add_argument("--output-dir", default="original_datasets", help="Output directory")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ds = load_dataset(args.dataset, args.subset, split=args.split)

    safe_name = args.dataset.replace("/", "__")
    if args.subset:
        safe_name += f"__{args.subset}"

    if args.split:
        path = output_dir / f"{safe_name}__{args.split}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(ds.to_list(), f, indent=2, ensure_ascii=False)
        print(f"Saved {len(ds)} rows → {path}")
    else:
        for split_name, split_ds in ds.items():
            path = output_dir / f"{safe_name}__{split_name}.json"
            with open(path, "w", encoding="utf-8") as f:
                json.dump(split_ds.to_list(), f, indent=2, ensure_ascii=False)
            print(f"Saved {len(split_ds)} rows → {path}")


if __name__ == "__main__":
    main()
