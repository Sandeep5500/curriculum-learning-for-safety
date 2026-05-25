"""
Write experiment.json metadata file for a DPO experiment.

Called at the end of train_curriculum_dpo.slurm to record all experiment parameters.
Can also be run standalone for backfilling existing experiments.

Usage (from SLURM):
  python3 -m src.utils.write_experiment_meta \
    --exp-id exp_073 \
    --model-key llama3 \
    --model-id informatiker/Llama-3-8B-Instruct-abliterated \
    --model-short llama3 \
    --dataset-key combined_pku_hh \
    --data-source-exp exp_021 \
    --ordering margin \
    --shuffling sequential \
    --beta 0.1 \
    --num-epochs 5 \
    --lr 5e-5 \
    --output-dir outputs/exp_073/combined_pku_hh/llama3
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def write_meta(
    exp_id: str,
    model_key: str,
    model_id: str,
    model_short: str,
    dataset_key: str,
    data_source_exp: str,
    ordering: str,
    shuffling: str,
    beta: float,
    num_epochs: float,
    lr: float,
    grad_accum: int,
    output_dir: Path,
    notes: str = "",
) -> None:
    meta = {
        "exp_id": exp_id,
        "model": {
            "key": model_key,
            "id": model_id,
            "short": model_short,
        },
        "dataset": {
            "key": dataset_key,
            "data_source_exp": data_source_exp,
        },
        "curriculum": {
            "ordering": ordering,   # baseline | curr | margin | logprob
            "shuffling": shuffling, # shuffled | sequential | staged | expanding
        },
        "training": {
            "beta": beta,
            "num_epochs": num_epochs,
            "learning_rate": lr,
            "grad_accum": grad_accum,
        },
        "notes": notes,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    out = Path(output_dir) / "experiment.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Wrote experiment metadata to {out}")


def cli() -> None:
    p = argparse.ArgumentParser(description="Write experiment.json metadata.")
    p.add_argument("--exp-id", required=True)
    p.add_argument("--model-key", required=True)
    p.add_argument("--model-id", required=True)
    p.add_argument("--model-short", required=True)
    p.add_argument("--dataset-key", required=True)
    p.add_argument("--data-source-exp", required=True)
    p.add_argument("--ordering", required=True,
                   choices=["baseline", "curr", "margin", "logprob"])
    p.add_argument("--shuffling", required=True,
                   choices=["shuffled", "sequential", "staged", "expanding", "competence", "sqrt_pacing", "curri_dpo", "curri_pacing"])
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--num-epochs", type=float, default=5)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--notes", default="")
    args = p.parse_args()

    write_meta(
        exp_id=args.exp_id,
        model_key=args.model_key,
        model_id=args.model_id,
        model_short=args.model_short,
        dataset_key=args.dataset_key,
        data_source_exp=args.data_source_exp,
        ordering=args.ordering,
        shuffling=args.shuffling,
        beta=args.beta,
        num_epochs=args.num_epochs,
        lr=args.lr,
        grad_accum=args.grad_accum,
        output_dir=args.output_dir,
        notes=args.notes,
    )


if __name__ == "__main__":
    cli()
