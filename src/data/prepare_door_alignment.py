"""
Prepare DOOR-Alignment dataset for DPO training.

DOOR-Alignment provides two parallel JSONL files:
  - Llama-3-8B-Instruct-good.jsonl  : "chosen" (safe refusals)
  - Llama-3-8B-Instruct-bad.jsonl   : "rejected" (unsafe/jailbroken responses)

Both files have the same format:
  {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}

Files are positionally aligned: good[i] and bad[i] have the same user prompt.
The assistant turn differs: good = refusal, bad = harmful compliance.

Reference: https://github.com/wicai24/DOOR-Alignment

Usage:
  python -m src.data.prepare_door_alignment
  python -m src.data.prepare_door_alignment \
    --good-url https://raw.githubusercontent.com/wicai24/DOOR-Alignment/main/dataset/train/Llama-3-8B-Instruct-good.jsonl \
    --bad-url  https://raw.githubusercontent.com/wicai24/DOOR-Alignment/main/dataset/train/Llama-3-8B-Instruct-bad.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import urllib.request
from pathlib import Path


GOOD_URL = "https://raw.githubusercontent.com/wicai24/DOOR-Alignment/main/dataset/train/Llama-3-8B-Instruct-good.jsonl"
BAD_URL = "https://raw.githubusercontent.com/wicai24/DOOR-Alignment/main/dataset/train/Llama-3-8B-Instruct-bad.jsonl"


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def fetch_jsonl_from_url(url: str) -> list[dict]:
    """Download a JSONL file from URL and parse into list of dicts."""
    print(f"  Fetching: {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as response:
        content = response.read().decode("utf-8")

    rows = []
    for line in content.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))

    print(f"  Loaded {len(rows)} rows")
    return rows


def load_jsonl_from_path(path: Path) -> list[dict]:
    """Load local JSONL file."""
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    print(f"  Loaded {len(rows)} rows from {path}")
    return rows


def save_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def extract_messages(row: dict) -> tuple[str, str]:
    """Extract (prompt, response) from messages format.

    For multi-turn: prompt = all messages up to final user turn (as formatted string)
    Response = last assistant message

    Returns:
        (prompt_text, response_text)
    """
    messages = row.get("messages", [])
    if not messages:
        return "", ""

    # Find last assistant message
    last_assistant_idx = -1
    for i in range(len(messages) - 1, -1, -1):
        if messages[i]["role"] == "assistant":
            last_assistant_idx = i
            break

    if last_assistant_idx < 0:
        return "", ""

    response = messages[last_assistant_idx]["content"].strip()

    # Build prompt from all messages before the final assistant turn
    prompt_messages = messages[:last_assistant_idx]
    if not prompt_messages:
        return "", response

    # For single-turn (just one user message), extract as plain string
    if len(prompt_messages) == 1 and prompt_messages[0]["role"] == "user":
        prompt = prompt_messages[0]["content"].strip()
    else:
        # Multi-turn: format as conversation string
        prompt_parts = []
        for msg in prompt_messages:
            role = msg["role"].capitalize()
            content = msg["content"].strip()
            prompt_parts.append(f"{role}: {content}")
        prompt = "\n\n".join(prompt_parts)

    return prompt, response


def parse_pair(good_row: dict, bad_row: dict) -> dict | None:
    """Parse a (good, bad) pair into DPO format.

    Both rows should share the same user prompt.
    Returns {"prompt": ..., "chosen": ..., "rejected": ...} or None if invalid.
    """
    prompt_good, chosen = extract_messages(good_row)
    prompt_bad, rejected = extract_messages(bad_row)

    # Basic validation
    if not prompt_good or not chosen or not rejected:
        return None

    # Prompts should match (same user request, different responses)
    # Use good file's prompt as canonical
    prompt = prompt_good

    # Skip if chosen == rejected (no learning signal)
    if chosen == rejected:
        return None

    return {"prompt": prompt, "chosen": chosen, "rejected": rejected}


def main(
    good_url: str = GOOD_URL,
    bad_url: str = BAD_URL,
    good_path: Path | None = None,
    bad_path: Path | None = None,
    output_dir: Path | None = None,
    debug_size: int = 100,
    mini_size: int = 5000,
    seed: int = 42,
) -> None:
    root = _repo_root()
    raw_dir = root / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    if output_dir is None:
        output_dir = raw_dir

    # Load both files
    print("Loading DOOR-Alignment good (chosen) responses...")
    if good_path and good_path.exists():
        good_rows = load_jsonl_from_path(good_path)
    else:
        good_rows = fetch_jsonl_from_url(good_url)

    print("Loading DOOR-Alignment bad (rejected) responses...")
    if bad_path and bad_path.exists():
        bad_rows = load_jsonl_from_path(bad_path)
    else:
        bad_rows = fetch_jsonl_from_url(bad_url)

    # Validate alignment
    if len(good_rows) != len(bad_rows):
        print(f"WARNING: good ({len(good_rows)}) and bad ({len(bad_rows)}) files have different lengths. Using min.")
        n = min(len(good_rows), len(bad_rows))
        good_rows = good_rows[:n]
        bad_rows = bad_rows[:n]

    print(f"\nParsing {len(good_rows)} pairs into DPO format...")

    parsed = []
    skipped = 0

    for good_row, bad_row in zip(good_rows, bad_rows):
        row = parse_pair(good_row, bad_row)
        if row is not None:
            parsed.append(row)
        else:
            skipped += 1

    print(f"Parsed {len(parsed)} valid pairs")
    print(f"Skipped {skipped} invalid/duplicate pairs")

    # Shuffle for representative subsets
    rng = random.Random(seed)
    rng.shuffle(parsed)

    # Save subsets
    subsets = {
        "door_debug_subset.jsonl": debug_size,
        "door_mini_subset.jsonl": mini_size,
    }
    for filename, size in subsets.items():
        path = output_dir / filename
        save_jsonl(path, parsed[:size])
        print(f"Wrote {min(size, len(parsed))} samples to {path}")

    # Full parsed dataset
    full_path = output_dir / "door_full_parsed.jsonl"
    save_jsonl(full_path, parsed)
    print(f"Wrote {len(parsed)} samples to {full_path}")

    print("\nDone. door_full_parsed.jsonl is ready for Phase 1 scoring — no LlamaGuard filtering needed.")
    print("  (Dataset is pre-cleaned: good=safe refusals, bad=harmful responses.)")


def cli() -> None:
    root = _repo_root()
    p = argparse.ArgumentParser(description="Prepare DOOR-Alignment dataset for DPO training.")
    p.add_argument("--good-url", default=GOOD_URL, help="URL for good (chosen) JSONL file")
    p.add_argument("--bad-url", default=BAD_URL, help="URL for bad (rejected) JSONL file")
    p.add_argument("--good-path", type=Path, default=None, help="Local path for good file (overrides URL)")
    p.add_argument("--bad-path", type=Path, default=None, help="Local path for bad file (overrides URL)")
    p.add_argument("--output-dir", type=Path, default=None, help="Output directory (default: data/raw/)")
    p.add_argument("--debug-size", type=int, default=100, help="Number of samples in debug subset")
    p.add_argument("--mini-size", type=int, default=5000, help="Number of samples in mini subset")
    p.add_argument("--seed", type=int, default=42, help="Random seed for shuffling")
    args = p.parse_args()
    main(
        good_url=args.good_url,
        bad_url=args.bad_url,
        good_path=args.good_path,
        bad_path=args.bad_path,
        output_dir=args.output_dir,
        debug_size=args.debug_size,
        mini_size=args.mini_size,
        seed=args.seed,
    )


if __name__ == "__main__":
    cli()
