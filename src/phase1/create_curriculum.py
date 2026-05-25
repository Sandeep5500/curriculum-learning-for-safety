"""
Create curriculum (easy→hard) and baseline (shuffled) orderings from the scored dataset.

Supports train/test split:
- curriculum_dataset_train.jsonl: Sorted by easiness (80% of data, easy→hard)
- curriculum_dataset_test.jsonl: Constant test set (20% of data, unsorted)
- baseline_dataset_train.jsonl: Shuffled training set (80% of data)
- baseline_dataset_test.jsonl: Constant test set (20% of data, same as curriculum test)

Sorting strategies:
- similarity: sorted by similarity_score descending (highest = easy first).
- ratio: sorted by ratio_score descending (sim_chosen / (sim_rejected + eps), highest = easy first).
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def load_jsonl(path: Path) -> list[dict]:
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def save_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def run(
    input_train_path: Path,
    input_test_path: Path,
    curriculum_train_path: Path,
    curriculum_test_path: Path,
    baseline_train_path: Path,
    baseline_test_path: Path,
    seed: int,
    sort_key: str = "similarity",
    reverse: bool = False,
) -> None:
    # Load train and test splits
    train_rows = load_jsonl(input_train_path)
    test_rows = load_jsonl(input_test_path)

    if not train_rows or not test_rows:
        raise SystemExit(f"No rows in {input_train_path} or {input_test_path}")

    print(f"Loaded {len(train_rows)} train rows and {len(test_rows)} test rows")

    # Determine sort key
    if sort_key == "ratio":
        if "ratio_score" not in train_rows[0]:
            raise SystemExit("Input must contain 'ratio_score' (run Phase 1 scoring first).")
        key_field = "ratio_score"
        label = "ratio (sim_chosen/sim_rejected)"
        sort_fn = lambda r: r[key_field]
    elif sort_key == "margin":
        if "similarity_score" not in train_rows[0] or "similarity_rejected" not in train_rows[0]:
            raise SystemExit("Input must contain 'similarity_score' and 'similarity_rejected' (run Phase 1 scoring first).")
        key_field = None
        label = "margin (sim_chosen - sim_rejected)"
        sort_fn = lambda r: r["similarity_score"] - r["similarity_rejected"]
    elif sort_key == "logprob":
        if "logprob_margin" not in train_rows[0]:
            raise SystemExit("Input must contain 'logprob_margin' (run score_logprob.py first).")
        key_field = "logprob_margin"
        label = "logprob margin (mean_logP_chosen - mean_logP_rejected)"
        sort_fn = lambda r: r[key_field]
    else:
        if "similarity_score" not in train_rows[0]:
            raise SystemExit("Input must contain 'similarity_score' (run Phase 1 scoring first).")
        key_field = "similarity_score"
        label = "similarity"
        sort_fn = lambda r: r[key_field]

    # Curriculum: sort training set (default: descending = Easy → Hard)
    curriculum_train = sorted(train_rows, key=sort_fn, reverse=not reverse)
    save_jsonl(curriculum_train_path, curriculum_train)
    print(f"Wrote {len(curriculum_train)} rows (easy→hard by {label}) to {curriculum_train_path}")

    # Curriculum test: unsorted test set (same for all models)
    save_jsonl(curriculum_test_path, test_rows)
    print(f"Wrote {len(test_rows)} rows (constant test set) to {curriculum_test_path}")

    # Baseline: shuffle training set
    baseline_train = train_rows.copy()
    rng = random.Random(seed)
    rng.shuffle(baseline_train)
    save_jsonl(baseline_train_path, baseline_train)
    print(f"Wrote {len(baseline_train)} rows (shuffled train, seed={seed}) to {baseline_train_path}")

    # Baseline test: same as curriculum test (constant across all models)
    save_jsonl(baseline_test_path, test_rows)
    print(f"Wrote {len(test_rows)} rows (constant test set) to {baseline_test_path}")


def cli() -> None:
    root = _repo_root()
    p = argparse.ArgumentParser(description="Create curriculum and baseline orderings from scored dataset with train/test split support.")
    p.add_argument("--input-train", type=Path, default=root / "data" / "processed" / "scored_dataset_train.jsonl", help="Scored training JSONL (80%)")
    p.add_argument("--input-test", type=Path, default=root / "data" / "processed" / "scored_dataset_test.jsonl", help="Scored test JSONL (20%, constant)")
    p.add_argument("--curriculum-train-out", type=Path, default=root / "data" / "processed" / "curriculum_dataset_train.jsonl", help="Output: easy→hard ordered training")
    p.add_argument("--curriculum-test-out", type=Path, default=root / "data" / "processed" / "curriculum_dataset_test.jsonl", help="Output: constant test set")
    p.add_argument("--baseline-train-out", type=Path, default=root / "data" / "processed" / "baseline_dataset_train.jsonl", help="Output: shuffled training baseline")
    p.add_argument("--baseline-test-out", type=Path, default=root / "data" / "processed" / "baseline_dataset_test.jsonl", help="Output: constant test set")
    p.add_argument("--seed", type=int, default=42, help="Random seed for baseline shuffle")
    p.add_argument("--sort-key", choices=["similarity", "ratio", "margin", "logprob"], default="similarity", help="Sort key: similarity, ratio (sim_chosen/sim_rejected), margin (sim_chosen-sim_rejected), or logprob (mean_logP_chosen-mean_logP_rejected)")
    p.add_argument("--reverse", action="store_true", help="Reverse sort order: Hard → Easy (anti-curriculum)")
    args = p.parse_args()
    run(
        input_train_path=args.input_train,
        input_test_path=args.input_test,
        curriculum_train_path=args.curriculum_train_out,
        curriculum_test_path=args.curriculum_test_out,
        baseline_train_path=args.baseline_train_out,
        baseline_test_path=args.baseline_test_out,
        seed=args.seed,
        sort_key=args.sort_key,
        reverse=args.reverse,
    )


if __name__ == "__main__":
    cli()
