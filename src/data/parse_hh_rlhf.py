"""
Download and parse the Anthropic HH-RLHF dataset.

Filters to single-turn conversations only, extracts (prompt, chosen, rejected)
in the same format as PKU, then saves to JSONL.

Usage:
  python -m src.data.parse_hh_rlhf \
    --output data/raw/hh_rlhf_single_turn.jsonl \
    --split train
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def parse_conversation(text: str) -> tuple[str, str] | None:
    """
    Parse a HH-RLHF conversation string into (prompt, response).
    Returns None if not single-turn or malformed.
    """
    text = text.strip()

    # Split on Human/Assistant turns
    # Format: \n\nHuman: ...\n\nAssistant: ...
    human_turns = re.split(r'\n\nHuman:\s*', text)
    # First element before any Human: is usually empty
    human_turns = [t.strip() for t in human_turns if t.strip()]

    if len(human_turns) != 1:
        return None  # multi-turn or malformed

    human_part = human_turns[0]

    # Split human turn into prompt and assistant response
    parts = re.split(r'\n\nAssistant:\s*', human_part, maxsplit=1)
    if len(parts) != 2:
        return None

    prompt = parts[0].strip()
    response = parts[1].strip()

    if not prompt or not response:
        return None

    return prompt, response


def run(output_path: Path, split: str) -> None:
    from datasets import load_dataset

    print(f"Loading Anthropic/hh-rlhf split='{split}'...")
    ds = load_dataset("Anthropic/hh-rlhf", split=split)
    print(f"Total examples: {len(ds)}")

    records = []
    skipped_multi = 0
    skipped_malformed = 0

    for ex in ds:
        chosen_parsed = parse_conversation(ex["chosen"])
        rejected_parsed = parse_conversation(ex["rejected"])

        if chosen_parsed is None or rejected_parsed is None:
            # Check if it's multi-turn vs malformed
            n_turns = ex["chosen"].count("\n\nHuman:")
            if n_turns > 1:
                skipped_multi += 1
            else:
                skipped_malformed += 1
            continue

        prompt_chosen, chosen_response = chosen_parsed
        prompt_rejected, rejected_response = rejected_parsed

        # Use chosen prompt (they should be identical for single-turn)
        records.append({
            "prompt": prompt_chosen,
            "chosen": chosen_response,
            "rejected": rejected_response,
        })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\nResults:")
    print(f"  Total:           {len(ds)}")
    print(f"  Single-turn kept:{len(records):>7}  ({len(records)/len(ds):.1%})")
    print(f"  Skipped multi:   {skipped_multi:>7}  ({skipped_multi/len(ds):.1%})")
    print(f"  Skipped malform: {skipped_malformed:>7}  ({skipped_malformed/len(ds):.1%})")
    print(f"  Saved → {output_path}")


def cli() -> None:
    root = _repo_root()
    p = argparse.ArgumentParser(description="Parse HH-RLHF to single-turn JSONL.")
    p.add_argument("--output", type=Path, default=root / "data" / "raw" / "hh_rlhf_single_turn.jsonl")
    p.add_argument("--split", default="train", choices=["train", "test"])
    args = p.parse_args()
    run(args.output, args.split)


if __name__ == "__main__":
    cli()
