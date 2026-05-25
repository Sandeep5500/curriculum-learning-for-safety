# Data

## Files

```
data/processed/
├── combined_pku_hh.jsonl        # Cleaned-PKU-HH-SafeRLHF — main release (10,931 pairs)
├── clean_parsed_gpt.jsonl       # GPT-filtered PKU-SafeRLHF (6,962 pairs)
├── hh_rlhf_clean_gpt.jsonl      # GPT-filtered HH-RLHF single-turn (3,969 pairs)
└── clean_parsed_gpt_stats.json  # PKU filtering statistics
```

All `.jsonl` files share the same schema:

```jsonl
{"prompt": "...", "chosen": "...", "rejected": "..."}
```

`chosen` = safe response, `rejected` = unsafe response.

## Cleaned-PKU-HH-SafeRLHF

The main release dataset used throughout the paper.

| Source | Raw pairs | After filtering |
|--------|----------:|----------------:|
| PKU-SafeRLHF | 43,452 | 6,962 |
| HH-RLHF (single-turn) | 49,388 | 3,969 |
| **Combined** | 92,840 | **10,931** |

Stratified 80/20 train/test split: 8,744 / 2,187.

## Reproducing the cleaning pipeline

```bash
# 1. Filter each source with GPT-4o-mini judge
python -m src.data.clean_with_gpt --input data/raw/full_parsed.jsonl \
       --output data/processed/clean_parsed_gpt.jsonl
python -m src.data.clean_with_gpt --input data/raw/hh_rlhf_single_turn.jsonl \
       --output data/processed/hh_rlhf_clean_gpt.jsonl

# 2. Combine into the release dataset
python -m src.data.create_staged_curriculum --inputs \
       data/processed/clean_parsed_gpt.jsonl \
       data/processed/hh_rlhf_clean_gpt.jsonl \
       --output data/processed/combined_pku_hh.jsonl
```

Requires `OPENAI_API_KEY` for the GPT-4o-mini judge. Raw source files (`data/raw/`) are not redistributed — download from the [PKU-SafeRLHF](https://huggingface.co/datasets/PKU-Alignment/PKU-SafeRLHF) and [HH-RLHF](https://huggingface.co/datasets/Anthropic/hh-rlhf) repositories.

## License

Cleaned-PKU-HH-SafeRLHF inherits its source licenses:
- PKU-SafeRLHF: CC BY-NC 4.0
- HH-RLHF: MIT
