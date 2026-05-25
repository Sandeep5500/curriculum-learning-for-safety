"""
Create staged or expanding curriculum training files from a pre-sorted curriculum JSONL.

Takes an already difficulty-sorted curriculum_dataset_train.jsonl (easy→hard) and produces
a single pre-concatenated JSONL designed to be consumed with NUM_EPOCHS=1 SEQUENTIAL=1.

Modes
-----
staged:
  Split the sorted data into N equal buckets (easy → hard).
  For each bucket i, emit `epochs_per_stage[i]` independently shuffled passes.
  Training sees: [easy×e1 (shuffled each pass)] → [medium×e2] → [hard×e3] → ...
  Default: 3 stages, 2+2+1 epochs (5 total equivalent epochs across 3 buckets).

expanding:
  Each "epoch" k unlocks a larger prefix of the sorted data.
  emit `expand_fractions` as a list of per-epoch data fractions.
  Training sees: [easiest 33%] → [easiest 66%] → [all] → [all] → [all]
  Default fractions: 0.33 0.66 1.0 1.0 1.0 (~4 equivalent epochs total).

Both methods use independent shuffles per pass for within-stage diversity.

Usage
-----
# Staged (3 stages, 2+2+1 epochs each):
python3 -m src.data.create_staged_curriculum staged \\
  --input  data/processed/exp_073/combined_pku_hh/llama3/curriculum_dataset_train.jsonl \\
  --output data/processed/exp_075/combined_pku_hh/llama3/curriculum_dataset_train.jsonl \\
  --n-stages 3 --epochs-per-stage 2 2 1

# Expanding (5 fractions: 33% 66% 100% 100% 100%):
python3 -m src.data.create_staged_curriculum expanding \\
  --input  data/processed/exp_073/combined_pku_hh/llama3/curriculum_dataset_train.jsonl \\
  --output data/processed/exp_076/combined_pku_hh/llama3/curriculum_dataset_train.jsonl \\
  --expand-fractions 0.33 0.66 1.0 1.0 1.0
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def save_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def shuffled_copy(rows: list[dict], rng: random.Random) -> list[dict]:
    c = rows.copy()
    rng.shuffle(c)
    return c


def create_staged(
    sorted_rows: list[dict],
    n_stages: int,
    epochs_per_stage: list[float],
    seed: int,
) -> list[dict]:
    """
    Divide sorted_rows into n_stages equal buckets.
    For stage i, emit ceil(epochs_per_stage[i]) independently-shuffled passes.
    Fractional epochs are supported: a fraction f means emit f*len(bucket) rows.
    """
    assert len(epochs_per_stage) == n_stages, (
        f"--epochs-per-stage must have exactly {n_stages} values, got {epochs_per_stage}"
    )
    rng = random.Random(seed)
    n = len(sorted_rows)
    stage_size = n // n_stages
    out = []

    for i in range(n_stages):
        start = i * stage_size
        end = (i + 1) * stage_size if i < n_stages - 1 else n
        bucket = sorted_rows[start:end]
        epochs = epochs_per_stage[i]
        full_passes = int(epochs)
        frac = epochs - full_passes

        label = f"stage {i+1}/{n_stages} (rows {start}:{end}, {len(bucket)} examples)"

        # Full passes
        for p in range(full_passes):
            out.extend(shuffled_copy(bucket, rng))

        # Fractional pass (take first frac*len examples after shuffle)
        if frac > 0:
            partial = shuffled_copy(bucket, rng)
            n_partial = max(1, round(frac * len(bucket)))
            out.extend(partial[:n_partial])

        print(f"  {label}: {epochs} epochs → {len(out) - (len(out) - (full_passes * len(bucket) + (round(frac * len(bucket)) if frac > 0 else 0)))} rows appended")

    # Reprint totals cleanly
    print(f"  Total rows in staged file: {len(out)}")
    print(f"  Equivalent full-dataset epochs: {len(out)/n:.2f}")
    return out


def create_expanding(
    sorted_rows: list[dict],
    expand_fractions: list[float],
    seed: int,
) -> list[dict]:
    """
    Each 'epoch' k uses the top (expand_fractions[k] * N) easiest examples, shuffled.
    expand_fractions should be in non-decreasing order (e.g. [0.33, 0.66, 1.0, 1.0, 1.0]).
    """
    rng = random.Random(seed)
    n = len(sorted_rows)
    out = []

    for k, frac in enumerate(expand_fractions):
        cutoff = max(1, round(frac * n))
        prefix = sorted_rows[:cutoff]
        shuffled = shuffled_copy(prefix, rng)
        out.extend(shuffled)
        print(f"  Epoch {k+1}: top {frac*100:.0f}% ({cutoff} examples) → shuffled")

    print(f"  Total rows in expanding file: {len(out)}")
    print(f"  Equivalent full-dataset epochs: {len(out)/n:.2f}")
    return out


def cli() -> None:
    p = argparse.ArgumentParser(description="Create staged or expanding curriculum files.")
    sub = p.add_subparsers(dest="mode", required=True)

    # Shared args
    def add_common(sp):
        sp.add_argument("--input", type=Path, required=True,
                        help="Pre-sorted curriculum_dataset_train.jsonl (easy→hard)")
        sp.add_argument("--output", type=Path, required=True,
                        help="Output pre-concatenated JSONL (use with NUM_EPOCHS=1 SEQUENTIAL=1)")
        sp.add_argument("--seed", type=int, default=42, help="Random seed for shuffles")
        sp.add_argument("--num-training-epochs", type=int, default=1,
                        help="Number of training epochs. Each epoch gets independent within-stage "
                             "shuffles. Sequences are concatenated; train with NUM_EPOCHS=1 SEQUENTIAL=1.")

    # Staged subcommand
    sp_staged = sub.add_parser("staged", help="Fixed-stage curriculum")
    add_common(sp_staged)
    sp_staged.add_argument("--n-stages", type=int, default=3,
                           help="Number of equal-size difficulty stages (default: 3)")
    sp_staged.add_argument("--epochs-per-stage", type=float, nargs="+", default=None,
                           help="Epochs to train per stage (default: equal split of 5 total, e.g. 2 2 1 for 3 stages)")

    # Expanding subcommand
    sp_expanding = sub.add_parser("expanding", help="Expanding/competence-based curriculum")
    add_common(sp_expanding)
    sp_expanding.add_argument("--expand-fractions", type=float, nargs="+",
                               default=[0.33, 0.66, 1.0, 1.0, 1.0],
                               help="Per-epoch data fractions in non-decreasing order "
                                    "(default: 0.33 0.66 1.0 1.0 1.0 ≈ 4 equiv. epochs)")

    args = p.parse_args()

    print(f"Loading sorted curriculum from {args.input} ...")
    rows = load_jsonl(args.input)
    print(f"  Loaded {len(rows)} rows.")

    num_training_epochs = args.num_training_epochs

    if args.mode == "staged":
        n_stages = args.n_stages
        if args.epochs_per_stage is None:
            # Default: distribute 5 total epochs evenly, rounding to nice numbers
            # e.g. 3 stages → [2, 2, 1]
            base = 5 // n_stages
            remainder = 5 % n_stages
            epochs_per_stage = [float(base + (1 if i < remainder else 0)) for i in range(n_stages)]
            print(f"  Auto epochs-per-stage: {epochs_per_stage} (sum={sum(epochs_per_stage):.1f})")
        else:
            epochs_per_stage = args.epochs_per_stage
            print(f"  Epochs-per-stage: {epochs_per_stage} (sum={sum(epochs_per_stage):.1f})")

        print(f"Building staged curriculum ({n_stages} stages, {num_training_epochs} training epoch(s) with independent shuffles) ...")
        out_rows = []
        for epoch in range(num_training_epochs):
            epoch_seed = args.seed + epoch
            print(f"  --- Training epoch {epoch + 1}/{num_training_epochs} (seed={epoch_seed}) ---")
            out_rows.extend(create_staged(rows, n_stages, epochs_per_stage, epoch_seed))

    else:  # expanding
        fracs = args.expand_fractions
        print(f"Building expanding curriculum (fractions: {fracs}, {num_training_epochs} training epoch(s) with independent shuffles) ...")
        out_rows = []
        for epoch in range(num_training_epochs):
            epoch_seed = args.seed + epoch
            print(f"  --- Training epoch {epoch + 1}/{num_training_epochs} (seed={epoch_seed}) ---")
            out_rows.extend(create_expanding(rows, fracs, epoch_seed))

    n = len(rows)
    save_jsonl(args.output, out_rows)
    print(f"Wrote {len(out_rows)} rows to {args.output}")
    print(f"Equivalent full-dataset epochs: {len(out_rows)/n:.2f} ({num_training_epochs} training epoch(s) × {len(out_rows)/n/num_training_epochs:.2f} per epoch)")
    print("Train with: NUM_EPOCHS=1 SEQUENTIAL=1 (pre-concatenated sequence, no repeat)")


if __name__ == "__main__":
    cli()
