#!/bin/bash
#SBATCH --job-name=prefill_judge
#SBATCH --partition=cpu
#SBATCH --mem=4G
#SBATCH --cpus-per-task=4
#SBATCH --time=0:30:00
#SBATCH --output=/data/user_data/sandeep3/curriculum-safety/logs/prefill_judge_%j.out
#SBATCH --error=/data/user_data/sandeep3/curriculum-safety/logs/prefill_judge_%j.err

# Prefill attack GPT-4o-mini judge.
# Runs judge_prefill.py over all 4 variant response files and outputs summary_gpt.json.
#
# Usage:
#   MODEL_KEY=qwen3 K=20 sbatch scripts/run_prefill_judge.slurm exp_009
#   MODEL_KEY=llama3 K=20 sbatch scripts/run_prefill_judge.slurm exp_007

set -e
REPO_ROOT="/data/user_data/sandeep3/curriculum-safety"
cd "$REPO_ROOT"

EXP_ID="${1:-exp_009}"
MODEL_KEY="${MODEL_KEY:-qwen3}"
DATASET_KEY="${DATASET_KEY:-pku}"
K="${K:-20}"
source "$REPO_ROOT/scripts/model_configs.sh"

DATA_SITE="$REPO_ROOT/$VENV_PATH/lib/python3.9/site-packages"
USER_SITE="/home/sandeep3/.local/lib/python3.9/site-packages"

RESULTS_DIR="$REPO_ROOT/results/$EXP_ID/$DATASET_KEY/$MODEL_SHORT/prefill/k_${K}"
OPENAI_KEY_FILE="${OPENAI_KEY_FILE:-$REPO_ROOT/open-ai-token.txt}"

echo "=== Prefill Judge: model=$MODEL_KEY k=$K exp=$EXP_ID ==="
echo "Responses dir → $RESULTS_DIR"

REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-bundle.crt \
  SSL_CERT_FILE=/etc/ssl/certs/ca-bundle.crt \
  PYTHONPATH=$REPO_ROOT:$DATA_SITE:$USER_SITE:$PYTHONPATH \
  python3 -m src.eval.judge_prefill \
    --responses-dir "$RESULTS_DIR" \
    --variants raw baseline curriculum ratio \
    --openai-key-file "$OPENAI_KEY_FILE" \
    --output "$RESULTS_DIR/summary_gpt.json" \
    --max-workers 16

echo ""
echo "=== Prefill Judge complete: $RESULTS_DIR/summary_gpt.json ==="
