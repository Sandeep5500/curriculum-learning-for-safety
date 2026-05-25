#!/bin/bash
#SBATCH --job-name=quality_eval
#SBATCH --partition=general
#SBATCH --gres=gpu:A6000:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=16:00:00
#SBATCH --output=/data/user_data/sandeep3/curriculum-safety/logs/quality_eval_%j.out
#SBATCH --error=/data/user_data/sandeep3/curriculum-safety/logs/quality_eval_%j.err

# Quality evaluation: MMLU + HellaSwag benchmarks for raw, baseline, curriculum.
# Outputs go to results/{EXP_ID}/{DATASET_KEY}/{MODEL_SHORT}/phase3/.
#
# Run: MODEL_KEY=qwen3 DATASET_KEY=combined_pku_hh sbatch scripts/run_quality_eval.slurm exp_042

set -e
REPO_ROOT="/data/user_data/sandeep3/curriculum-safety"
cd "$REPO_ROOT"

EXP_ID="${1:-exp_004}"
MODEL_KEY="${MODEL_KEY:-llama3}"
DATASET_KEY="${DATASET_KEY:-combined_pku_hh}"
source "$REPO_ROOT/scripts/model_configs.sh"

module load cuda-12.1

DATA_SITE="$REPO_ROOT/$VENV_PATH/lib/python3.9/site-packages"
USER_SITE="/home/sandeep3/.local/lib/python3.9/site-packages"

RESULTS_DIR="$REPO_ROOT/results/$EXP_ID/$DATASET_KEY/$MODEL_SHORT/phase3"
OUTPUTS_DIR="$REPO_ROOT/outputs/$EXP_ID/$DATASET_KEY/$MODEL_SHORT"
CACHE_SUFFIX="${LOGPROBS:-0}"; [ "$CACHE_SUFFIX" = "1" ] && CACHE_SUFFIX="logprobs" || CACHE_SUFFIX="gen"
RAW_CACHE="$REPO_ROOT/results/_raw/$DATASET_KEY/$MODEL_SHORT/$CACHE_SUFFIX"
mkdir -p "$RESULTS_DIR" "$RAW_CACHE"

PYRUN="env -u HF_HUB_OFFLINE -u TRANSFORMERS_OFFLINE HF_HOME=/data/user_data/sandeep3/.cache/huggingface REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-bundle.crt SSL_CERT_FILE=/etc/ssl/certs/ca-bundle.crt PYTHONPATH=$REPO_ROOT:$DATA_SITE:$USER_SITE:$PYTHONPATH python3"

# BENCHMARKS: space-separated list of benchmarks to run (default: mmlu hellaswag)
BENCHMARKS="${BENCHMARKS:-mmlu hellaswag}"
# LOGPROBS=1 to use log-probability scoring (immune to position bias; recommended)
LOGPROBS_FLAG=""
[ "${LOGPROBS:-0}" = "1" ] && LOGPROBS_FLAG="--logprobs"
COT_FLAG=""
[ "${COT:-0}" = "1" ] && COT_FLAG="--cot"

CURRICULUM_BASE="$MODEL_ID"
if [ -f "$OUTPUTS_DIR/dpo_curriculum/curri_base_model.txt" ]; then
  CURRICULUM_BASE=$(cat "$OUTPUTS_DIR/dpo_curriculum/curri_base_model.txt")
fi

echo "=== Quality Eval: model=$MODEL_KEY dataset=$DATASET_KEY exp=$EXP_ID benchmarks=$BENCHMARKS ==="
echo "Results → $RESULTS_DIR"

# --- Raw model (with caching) ---
if [ -f "$RAW_CACHE/quality_raw.json" ]; then
  echo "Using cached raw quality result from $RAW_CACHE"
  cp "$RAW_CACHE/quality_raw.json" "$RESULTS_DIR/quality_raw.json"
else
  echo "--- Quality eval: raw model ---"
  $PYRUN -m src.eval.evaluate_quality \
    --base-model "$MODEL_ID" \
    --benchmarks $BENCHMARKS \
    --max-tokens 10 $LOGPROBS_FLAG $COT_FLAG \
    --output "$RESULTS_DIR/quality_raw.json"
  cp "$RESULTS_DIR/quality_raw.json" "$RAW_CACHE/quality_raw.json"
fi

# --- Baseline DPO ---
if [ -d "$OUTPUTS_DIR/dpo_baseline" ]; then
  echo "--- Quality eval: baseline ---"
  $PYRUN -m src.eval.evaluate_quality \
    --base-model "$MODEL_ID" \
    --adapter-path "$OUTPUTS_DIR/dpo_baseline" \
    --benchmarks $BENCHMARKS \
    --max-tokens 10 $LOGPROBS_FLAG $COT_FLAG \
    --output "$RESULTS_DIR/quality_baseline.json"
else
  echo "Baseline checkpoint not found at $OUTPUTS_DIR/dpo_baseline (skipping)"
fi

# --- Curriculum DPO ---
if [ -d "$OUTPUTS_DIR/dpo_curriculum" ]; then
  echo "--- Quality eval: curriculum ---"
  $PYRUN -m src.eval.evaluate_quality \
    --base-model "$CURRICULUM_BASE" \
    --adapter-path "$OUTPUTS_DIR/dpo_curriculum" \
    --benchmarks $BENCHMARKS \
    --max-tokens 10 $LOGPROBS_FLAG $COT_FLAG \
    --output "$RESULTS_DIR/quality_curriculum.json"
else
  echo "Curriculum checkpoint not found at $OUTPUTS_DIR/dpo_curriculum (skipping)"
fi

echo ""
echo "=== Quality Eval complete: $RESULTS_DIR ==="
