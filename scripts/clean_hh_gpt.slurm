#!/bin/bash
#SBATCH --job-name=clean_hh_gpt
#SBATCH --partition=cpu
#SBATCH --mem=8G
#SBATCH --cpus-per-task=8
#SBATCH --time=4:00:00
#SBATCH --output=/data/user_data/sandeep3/curriculum-safety/logs/clean_hh_gpt_%j.out
#SBATCH --error=/data/user_data/sandeep3/curriculum-safety/logs/clean_hh_gpt_%j.err

# Parse HH-RLHF to single-turn, then clean with GPT-4o-mini safety judge.
# Keeps only pairs where chosen=safe AND rejected=unsafe.
#
# Run: sbatch scripts/clean_hh_gpt.slurm

set -e
REPO_ROOT="/data/user_data/sandeep3/curriculum-safety"
cd "$REPO_ROOT"

VENV_PATH="venv-new"
DATA_SITE="$REPO_ROOT/$VENV_PATH/lib/python3.9/site-packages"
USER_SITE="/home/sandeep3/.local/lib/python3.9/site-packages"

HH_RAW="$REPO_ROOT/data/raw/hh_rlhf_single_turn.jsonl"
HH_CLEAN="$REPO_ROOT/data/processed/hh_rlhf_clean_gpt.jsonl"
HH_STATS="$REPO_ROOT/data/processed/hh_rlhf_clean_gpt_stats.json"

echo "=== Step 1: Parse HH-RLHF (single-turn only) ==="
REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-bundle.crt \
SSL_CERT_FILE=/etc/ssl/certs/ca-bundle.crt \
HF_HOME=/data/user_data/sandeep3/.cache/huggingface \
PYTHONPATH="$REPO_ROOT:$DATA_SITE:$USER_SITE:$PYTHONPATH" \
python3 -m src.data.parse_hh_rlhf \
  --output "$HH_RAW" \
  --split train

echo ""
echo "=== Step 2: GPT safety judge (chosen=safe, rejected=unsafe) ==="
REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-bundle.crt \
SSL_CERT_FILE=/etc/ssl/certs/ca-bundle.crt \
PYTHONPATH="$REPO_ROOT:$DATA_SITE:$USER_SITE:$PYTHONPATH" \
python3 -m src.data.clean_with_gpt \
  --input "$HH_RAW" \
  --output-clean "$HH_CLEAN" \
  --output-stats "$HH_STATS" \
  --openai-key-file "$REPO_ROOT/open-ai-token.txt" \
  --max-workers 16

echo ""
echo "=== HH-RLHF cleaning complete ==="
