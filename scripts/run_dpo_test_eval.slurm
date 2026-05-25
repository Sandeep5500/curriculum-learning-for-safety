#!/bin/bash
#SBATCH --job-name=dpo_test_eval
#SBATCH --partition=general
#SBATCH --gres=gpu:A6000:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=6:00:00
#SBATCH --output=/data/user_data/sandeep3/curriculum-safety/logs/dpo_test_eval_%j.out
#SBATCH --error=/data/user_data/sandeep3/curriculum-safety/logs/dpo_test_eval_%j.err

# Phase 3b: In-distribution evaluation.
#   - DPO reward accuracy on held-out test split (all 4 model variants)
#   - Training loss curves plot
# Outputs go to results/{EXP_ID}/{MODEL_SHORT}/phase2/.
#
# Run: MODEL_KEY=llama3 DATASET_KEY=pku sbatch scripts/run_dpo_test_eval.slurm [exp_id]

set -e
REPO_ROOT="/data/user_data/sandeep3/curriculum-safety"
cd "$REPO_ROOT"

EXP_ID="${1:-exp_004}"
MODEL_KEY="${MODEL_KEY:-llama3}"
DATASET_KEY="${DATASET_KEY:-pku}"
BASELINE_EXP_ID="${BASELINE_EXP_ID:-}"
source "$REPO_ROOT/scripts/model_configs.sh"

module load cuda-12.1

DATA_SITE="$REPO_ROOT/$VENV_PATH/lib/python3.9/site-packages"
USER_SITE="/home/sandeep3/.local/lib/python3.9/site-packages"

TEST_SET="$PROCESSED_DIR/curriculum_dataset_test.jsonl"
RESULTS_DIR="$REPO_ROOT/results/$EXP_ID/$DATASET_KEY/$MODEL_SHORT/phase2"
OUTPUTS_DIR="$REPO_ROOT/outputs/$EXP_ID/$DATASET_KEY/$MODEL_SHORT"
mkdir -p "$RESULTS_DIR"

PYRUN="env -u HF_HUB_OFFLINE -u TRANSFORMERS_OFFLINE HF_HOME=/data/user_data/sandeep3/.cache/huggingface REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-bundle.crt SSL_CERT_FILE=/etc/ssl/certs/ca-bundle.crt PYTHONPATH=$REPO_ROOT:$DATA_SITE:$USER_SITE:$PYTHONPATH python3"

echo "=== DPO Test Eval + Loss Plot: model=$MODEL_KEY dataset=$DATASET_KEY exp=$EXP_ID ==="
echo "Test set: $TEST_SET"
echo "Results → $RESULTS_DIR"

# ── DPO reward accuracy ───────────────────────────────────────────────────────

echo ""
echo "--- Evaluating raw model ---"
$PYRUN -m src.eval.evaluate_dpo_test \
  --test-path "$TEST_SET" \
  --base-model "$MODEL_ID" \
  --output "$RESULTS_DIR/dpo_test_raw.json"

if [ -d "$OUTPUTS_DIR/dpo_baseline" ]; then
  echo "--- Evaluating baseline DPO ---"
  $PYRUN -m src.eval.evaluate_dpo_test \
    --test-path "$TEST_SET" \
    --base-model "$MODEL_ID" \
    --adapter-path "$OUTPUTS_DIR/dpo_baseline" \
    --output "$RESULTS_DIR/dpo_test_baseline.json"
fi

if [ -d "$OUTPUTS_DIR/dpo_curriculum" ]; then
  # Detect Curri-DPO: if curri_base_model.txt exists, use staged merged model as base
  CURRICULUM_BASE="$MODEL_ID"
  if [ -f "$OUTPUTS_DIR/dpo_curriculum/curri_base_model.txt" ]; then
    CURRICULUM_BASE=$(cat "$OUTPUTS_DIR/dpo_curriculum/curri_base_model.txt")
    echo "  Curri-DPO detected: using base model $CURRICULUM_BASE"
  fi
  echo "--- Evaluating curriculum DPO (similarity) ---"
  $PYRUN -m src.eval.evaluate_dpo_test \
    --test-path "$TEST_SET" \
    --base-model "$CURRICULUM_BASE" \
    --adapter-path "$OUTPUTS_DIR/dpo_curriculum" \
    --output "$RESULTS_DIR/dpo_test_curriculum.json"
fi

# ── Training loss curves ──────────────────────────────────────────────────────

echo ""
echo "--- Generating training loss curves ---"
BASELINE_MARGIN_FLAG=""
if [ -n "$BASELINE_EXP_ID" ]; then
  BASELINE_MARGIN_DIR="$REPO_ROOT/outputs/$BASELINE_EXP_ID/$DATASET_KEY/$MODEL_SHORT/dpo_baseline"
  BASELINE_MARGIN_FLAG="--baseline-margin-dir $BASELINE_MARGIN_DIR"
  echo "  Baseline margin dir (exp $BASELINE_EXP_ID): $BASELINE_MARGIN_DIR"
fi
$PYRUN -m src.eval.plot_training_loss \
  --baseline "$OUTPUTS_DIR/dpo_baseline" \
  --curriculum "$OUTPUTS_DIR/dpo_curriculum" \
  --output "$RESULTS_DIR/training_loss_curves.png" \
  $BASELINE_MARGIN_FLAG

echo ""
echo "=== DPO Test Eval complete: $RESULTS_DIR ==="
