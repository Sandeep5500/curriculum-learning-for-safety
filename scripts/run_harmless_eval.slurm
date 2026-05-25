#!/bin/bash
#SBATCH --job-name=harmless_eval
#SBATCH --partition=general
#SBATCH --gres=gpu:A6000:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=8:00:00
#SBATCH --output=/data/user_data/sandeep3/curriculum-safety/logs/harmless_eval_%j.out
#SBATCH --error=/data/user_data/sandeep3/curriculum-safety/logs/harmless_eval_%j.err

# Phase 3c: Helpfulness evaluation on Alpaca with GPT-4o-mini refusal scoring.
# Measures whether safety tuning degrades general helpfulness on benign tasks.
# Outputs go to results/{EXP_ID}/{MODEL_SHORT}/phase3/.
#
# Requires: open-ai-token.txt in repo root (or OPENAI_API_KEY env var).
# Run: MODEL_KEY=llama3 DATASET_KEY=pku sbatch scripts/run_harmless_eval.slurm [exp_id]

set -e
REPO_ROOT="/data/user_data/sandeep3/curriculum-safety"
cd "$REPO_ROOT"

EXP_ID="${1:-exp_004}"
MODEL_KEY="${MODEL_KEY:-llama3}"
DATASET_KEY="${DATASET_KEY:-pku}"
source "$REPO_ROOT/scripts/model_configs.sh"

module load cuda-12.1

DATA_SITE="$REPO_ROOT/$VENV_PATH/lib/python3.9/site-packages"
USER_SITE="/home/sandeep3/.local/lib/python3.9/site-packages"
export HF_HOME="/data/user_data/sandeep3/.cache/huggingface"

RESULTS_DIR="$REPO_ROOT/results/$EXP_ID/$DATASET_KEY/$MODEL_SHORT/phase3"
OUTPUTS_DIR="$REPO_ROOT/outputs/$EXP_ID/$DATASET_KEY/$MODEL_SHORT"
RAW_CACHE="$REPO_ROOT/results/_raw/$DATASET_KEY/$MODEL_SHORT"
mkdir -p "$RESULTS_DIR" "$RAW_CACHE"

PYRUN="env -u HF_HUB_OFFLINE -u TRANSFORMERS_OFFLINE HF_HOME=/data/user_data/sandeep3/.cache/huggingface REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-bundle.crt SSL_CERT_FILE=/etc/ssl/certs/ca-bundle.crt PYTHONPATH=$REPO_ROOT:$DATA_SITE:$USER_SITE:$PYTHONPATH python3 -m src.eval.evaluate_harmless"

echo "=== Harmless Eval (Alpaca): model=$MODEL_KEY exp=$EXP_ID ==="
echo "Results → $RESULTS_DIR"

# ── Raw model (no adapter) ───────────────────────────────────────────────────
echo ""
echo "--- Raw model ---"
if [ ! -f "$RAW_CACHE/alpaca_raw.json" ]; then
  $PYRUN \
    --base-model "$MODEL_ID" \
    --output "$RAW_CACHE/alpaca_raw.json"
else
  echo "  Using cached raw result from $RAW_CACHE"
fi
cp "$RAW_CACHE/alpaca_raw.json" "$RESULTS_DIR/alpaca_raw.json"

# ── Baseline DPO ─────────────────────────────────────────────────────────────
if [ -d "$OUTPUTS_DIR/dpo_baseline" ]; then
  echo "--- Baseline DPO ---"
  $PYRUN \
    --base-model "$MODEL_ID" \
    --adapter-path "$OUTPUTS_DIR/dpo_baseline" \
    --output "$RESULTS_DIR/alpaca_baseline.json"
fi

# ── Curriculum DPO (similarity) ───────────────────────────────────────────────
if [ -d "$OUTPUTS_DIR/dpo_curriculum" ]; then
  echo "--- Curriculum DPO (similarity) ---"
  $PYRUN \
    --base-model "$MODEL_ID" \
    --adapter-path "$OUTPUTS_DIR/dpo_curriculum" \
    --output "$RESULTS_DIR/alpaca_curriculum.json"
fi

echo ""
echo "=== Harmless Eval complete: $RESULTS_DIR ==="
