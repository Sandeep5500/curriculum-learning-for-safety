#!/bin/bash
#SBATCH --job-name=clean_gpt
#SBATCH --partition=cpu
#SBATCH --mem=8G
#SBATCH --cpus-per-task=8
#SBATCH --time=4:00:00
#SBATCH --output=/data/user_data/sandeep3/curriculum-safety/logs/clean_data_gpt_%j.out
#SBATCH --error=/data/user_data/sandeep3/curriculum-safety/logs/clean_data_gpt_%j.err

# Clean PKU dataset with GPT-4o-mini safety judge.
# Keeps only pairs where chosen=safe AND rejected=unsafe.
# ~43k pairs x2 responses = ~86k GPT calls at 16 workers ≈ 1-2 hours.
#
# Run: sbatch scripts/clean_data_gpt.slurm

set -e
REPO_ROOT="/data/user_data/sandeep3/curriculum-safety"
cd "$REPO_ROOT"

VENV_PATH="venv-new"
DATA_SITE="$REPO_ROOT/$VENV_PATH/lib/python3.9/site-packages"
USER_SITE="/home/sandeep3/.local/lib/python3.9/site-packages"

echo "=== GPT PKU Cleaning ==="
echo "Input:  $REPO_ROOT/data/raw/full_parsed.jsonl"
echo "Output: $REPO_ROOT/data/processed/clean_parsed_gpt.jsonl"

REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-bundle.crt \
SSL_CERT_FILE=/etc/ssl/certs/ca-bundle.crt \
PYTHONPATH="$REPO_ROOT:$DATA_SITE:$USER_SITE:$PYTHONPATH" \
python3 -m src.data.clean_with_gpt \
  --input "$REPO_ROOT/data/raw/full_parsed.jsonl" \
  --output-clean "$REPO_ROOT/data/processed/clean_parsed_gpt.jsonl" \
  --output-stats "$REPO_ROOT/data/processed/clean_parsed_gpt_stats.json" \
  --openai-key-file "$REPO_ROOT/open-ai-token.txt" \
  --max-workers 16

echo ""
echo "=== GPT Cleaning complete ==="
