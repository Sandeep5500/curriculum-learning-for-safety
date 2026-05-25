#!/bin/bash
#SBATCH --job-name=jailbreak_judge
#SBATCH --partition=general
#SBATCH --gres=gpu:A6000:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=8
#SBATCH --time=2:00:00
#SBATCH --output=/data/user_data/sandeep3/curriculum-safety/logs/jailbreak_judge_%j.out
#SBATCH --error=/data/user_data/sandeep3/curriculum-safety/logs/jailbreak_judge_%j.err

# Phase 4b: Jailbreak robustness judging with GPT-4o-mini.
# Scores response JSONL files and reports Attack Success Rate (ASR) per variant and attack type.
#
# Run after all inference jobs complete:
#   MODEL_KEY=llama3 DATASET_KEY=combined_pku_hh sbatch scripts/run_jailbreak_judge.slurm exp_046

set -e
REPO_ROOT="/data/user_data/sandeep3/curriculum-safety"
cd "$REPO_ROOT"

EXP_ID="${1:-exp_007}"
MODEL_KEY="${MODEL_KEY:-llama3}"
DATASET_KEY="${DATASET_KEY:-combined_pku_hh}"
source "$REPO_ROOT/scripts/model_configs.sh"

DATA_SITE="$REPO_ROOT/$VENV_PATH/lib/python3.9/site-packages"
USER_SITE="/home/sandeep3/.local/lib/python3.9/site-packages"

RESULTS_DIR="$REPO_ROOT/results/$EXP_ID/$DATASET_KEY/$MODEL_SHORT/jailbreak"
mkdir -p "$RESULTS_DIR"

echo "=== Jailbreak Judge (GPT-4o-mini): model=$MODEL_KEY exp=$EXP_ID ==="
echo "Responses dir → $RESULTS_DIR"

env -u HF_HUB_OFFLINE -u TRANSFORMERS_OFFLINE \
  REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-bundle.crt \
  SSL_CERT_FILE=/etc/ssl/certs/ca-bundle.crt \
  PYTHONPATH=$REPO_ROOT:$DATA_SITE:$USER_SITE:$PYTHONPATH \
  python3 -m src.eval.judge_jailbreak \
    --responses-dir "$RESULTS_DIR" \
    --variants raw baseline curriculum \
    --attack-types gcg \
    --output "$RESULTS_DIR/summary.json" \
    --openai-key-file "$REPO_ROOT/open-ai-token.txt" \
    --max-workers 16

echo ""
echo "=== Jailbreak Judge complete: $RESULTS_DIR/summary.json ==="
