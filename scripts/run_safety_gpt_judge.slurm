#!/bin/bash
#SBATCH --job-name=safety_gpt_judge
#SBATCH --partition=cpu
#SBATCH --mem=4G
#SBATCH --cpus-per-task=4
#SBATCH --time=2:00:00
#SBATCH --output=/data/user_data/sandeep3/curriculum-safety/logs/safety_gpt_judge_%j.out
#SBATCH --error=/data/user_data/sandeep3/curriculum-safety/logs/safety_gpt_judge_%j.err

# GPT-4o-mini judge for phase3 safety benchmarks (advbench, sorrybench, hexphi).
# Re-scores existing LlamaGuard-evaluated response files with GPT.
#
# Usage:
#   MODEL_KEY=llama3 sbatch scripts/run_safety_gpt_judge.slurm exp_007
#   MODEL_KEY=qwen3  sbatch scripts/run_safety_gpt_judge.slurm exp_009

set -e
REPO_ROOT="/data/user_data/sandeep3/curriculum-safety"
cd "$REPO_ROOT"

EXP_ID="${1:-exp_007}"
MODEL_KEY="${MODEL_KEY:-llama3}"
DATASET_KEY="${DATASET_KEY:-pku}"
source "$REPO_ROOT/scripts/model_configs.sh"

DATA_SITE="$REPO_ROOT/$VENV_PATH/lib/python3.9/site-packages"
USER_SITE="/home/sandeep3/.local/lib/python3.9/site-packages"

PHASE3_DIR="$REPO_ROOT/results/$EXP_ID/$DATASET_KEY/$MODEL_SHORT/phase3"
OPENAI_KEY_FILE="${OPENAI_KEY_FILE:-$REPO_ROOT/open-ai-token.txt}"

echo "=== Safety GPT Judge: model=$MODEL_KEY exp=$EXP_ID ==="
echo "Phase3 dir → $PHASE3_DIR"

REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-bundle.crt \
  SSL_CERT_FILE=/etc/ssl/certs/ca-bundle.crt \
  PYTHONPATH=$REPO_ROOT:$DATA_SITE:$USER_SITE:$PYTHONPATH \
  python3 -m src.eval.judge_safety_gpt \
    --phase3-dir "$PHASE3_DIR" \
    --benchmarks advbench sorrybench hexphi \
    --variants raw baseline curriculum \
    --openai-key-file "$OPENAI_KEY_FILE" \
    --output "$PHASE3_DIR/summary_gpt.json" \
    --max-workers 16

echo ""
echo "=== Safety GPT Judge complete: $PHASE3_DIR/summary_gpt.json ==="
