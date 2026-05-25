"""
Download PKU-SafeRLHF from Hugging Face and parse into DPO format.

Uses examples where there is a safety contrast (at least one response is unsafe)
and selects the safer response as "chosen" and the less safe as "rejected".

Two tiers of training signal:
  - "one_safe_one_unsafe": strongest signal (~10k examples)
  - "both_unsafe":         weaker but still useful (~32k examples)
  - "both_safe":           skipped (no safety signal)

Output format: {"prompt": ..., "chosen": ..., "rejected": ...}

Saves:
- data/raw/debug_subset.jsonl (100 samples for local testing)
- data/raw/mini_subset.jsonl  (5k samples for mini runs)
- data/raw/full_parsed.jsonl  (full parsed dataset)
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from datasets import load_dataset


def parse_row(example: dict) -> dict | None:
    """
    Parse one PKU-SafeRLHF example into {"prompt", "chosen", "rejected"}.

    Uses `safer_response_id` to pick chosen/rejected.
    Skips examples where both responses are safe (no safety signal)
    or where chosen == rejected.
    """
    s0 = example["is_response_0_safe"]
    s1 = example["is_response_1_safe"]

    # Skip if both are safe — no safety signal for DPO
    if s0 and s1:
        return None

    safer_id = example["safer_response_id"]
    chosen = example[f"response_{safer_id}"].strip()
    rejected = example[f"response_{1 - safer_id}"].strip()
    prompt = example["prompt"].strip()

    if not prompt or not chosen or not rejected:
        return None
    if chosen == rejected:
        return None

    return {"prompt": prompt, "chosen": chosen, "rejected": rejected}


def get_output_dir(repo_root: Path | None) -> Path:
    if repo_root is None:
        repo_root = Path(__file__).resolve().parent.parent.parent
    return repo_root / "data" / "raw"


def main(
    *,
    repo_root: Path | None = None,
    debug_size: int = 100,
    mini_size: int = 5000,
    split: str = "train",
    seed: int = 42,
) -> None:
    out_dir = get_output_dir(repo_root)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading PKU-Alignment/PKU-SafeRLHF (split={split})...")
    dataset = load_dataset("PKU-Alignment/PKU-SafeRLHF", split=split)

    parsed = []
    skipped_both_safe = 0
    skipped_other = 0
    for ex in dataset:
        row = parse_row(ex)
        if row is not None:
            parsed.append(row)
        elif ex["is_response_0_safe"] and ex["is_response_1_safe"]:
            skipped_both_safe += 1
        else:
            skipped_other += 1

    print(f"Parsed {len(parsed)} valid examples.")
    print(f"Skipped {skipped_both_safe} (both safe, no signal), {skipped_other} (other).")

    # Shuffle before taking subsets so mini/debug get a representative mix
    rng = random.Random(seed)
    rng.shuffle(parsed)

    subsets = {
        "debug_subset.jsonl": debug_size,
        "mini_subset.jsonl": mini_size,
    }
    for filename, size in subsets.items():
        path = out_dir / filename
        with open(path, "w") as f:
            for item in parsed[:size]:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"Wrote {min(size, len(parsed))} samples to {path}")

    full_path = out_dir / "full_parsed.jsonl"
    with open(full_path, "w") as f:
        for item in parsed:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Wrote {len(parsed)} samples to {full_path}")


def cli() -> None:
    p = argparse.ArgumentParser(description="Prepare PKU-SafeRLHF for DPO (prompt/chosen/rejected).")
    p.add_argument("--repo-root", type=Path, default=None, help="Repo root (default: auto-detect)")
    p.add_argument("--debug-size", type=int, default=100, help="Number of samples in debug_subset.jsonl")
    p.add_argument("--mini-size", type=int, default=5000, help="Number of samples in mini_subset.jsonl")
    p.add_argument("--split", default="train", help="Dataset split to use")
    p.add_argument("--seed", type=int, default=42, help="Random seed for shuffling")
    args = p.parse_args()
    main(repo_root=args.repo_root, debug_size=args.debug_size, mini_size=args.mini_size, split=args.split, seed=args.seed)


if __name__ == "__main__":
    cli()
