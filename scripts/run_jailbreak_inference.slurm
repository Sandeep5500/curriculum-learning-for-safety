#!/bin/bash
#SBATCH --job-name=jailbreak_inference
#SBATCH --partition=general
#SBATCH --gres=gpu:A6000:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=8:00:00
#SBATCH --output=/data/user_data/sandeep3/curriculum-safety/logs/jailbreak_inference_%j.out
#SBATCH --error=/data/user_data/sandeep3/curriculum-safety/logs/jailbreak_inference_%j.err

# Phase 4a: Jailbreak robustness inference.
# Generates model responses to pre-computed GCG and AutoDAN attack prompts (HarmBench).
#
# Run one job per model variant:
#   MODEL_VARIANT=raw       sbatch scripts/run_jailbreak_inference.slurm exp_007
#   MODEL_VARIANT=baseline  sbatch scripts/run_jailbreak_inference.slurm exp_007
#   MODEL_VARIANT=curriculum sbatch scripts/run_jailbreak_inference.slurm exp_007
#   MODEL_VARIANT=ratio     sbatch scripts/run_jailbreak_inference.slurm exp_007

set -e
REPO_ROOT="/data/user_data/sandeep3/curriculum-safety"
cd "$REPO_ROOT"

EXP_ID="${1:-exp_007}"
MODEL_KEY="${MODEL_KEY:-llama3}"
DATASET_KEY="${DATASET_KEY:-pku}"
MODEL_VARIANT="${MODEL_VARIANT:-raw}"
source "$REPO_ROOT/scripts/model_configs.sh"

module load cuda-12.1

DATA_SITE="$REPO_ROOT/$VENV_PATH/lib/python3.9/site-packages"
USER_SITE="/home/sandeep3/.local/lib/python3.9/site-packages"

OUTPUTS_DIR="$REPO_ROOT/outputs/$EXP_ID/$DATASET_KEY/$MODEL_SHORT"
OUTPUTS_RATIO="$REPO_ROOT/outputs/${EXP_ID}_ratio/$DATASET_KEY/$MODEL_SHORT"
RESULTS_DIR="$REPO_ROOT/results/$EXP_ID/$DATASET_KEY/$MODEL_SHORT/jailbreak"
GCG_CASES="${GCG_CASES:-$REPO_ROOT/data/harmbench/gcg_test_cases.json}"
AUTODAN_CASES="$REPO_ROOT/data/harmbench/autodan_test_cases.json"
mkdir -p "$RESULTS_DIR"

PYRUN="env -u HF_HUB_OFFLINE -u TRANSFORMERS_OFFLINE \
  HF_HOME=/data/user_data/sandeep3/.cache/huggingface \
  REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-bundle.crt \
  SSL_CERT_FILE=/etc/ssl/certs/ca-bundle.crt \
  PYTHONPATH=$REPO_ROOT:$DATA_SITE:$USER_SITE:$PYTHONPATH \
  python3 -m src.eval.evaluate_jailbreak"

# Resolve base model for curri-dpo
CURRICULUM_BASE="$MODEL_ID"
if [ -f "$OUTPUTS_DIR/dpo_curriculum/curri_base_model.txt" ]; then
  CURRICULUM_BASE=$(cat "$OUTPUTS_DIR/dpo_curriculum/curri_base_model.txt")
fi

# Resolve adapter path based on variant
case "$MODEL_VARIANT" in
  raw)
    ADAPTER_ARG=""
    BASE_MODEL="$MODEL_ID"
    ;;
  baseline)
    ADAPTER_ARG="--adapter-path $OUTPUTS_DIR/dpo_baseline"
    BASE_MODEL="$MODEL_ID"
    ;;
  curriculum)
    ADAPTER_ARG="--adapter-path $OUTPUTS_DIR/dpo_curriculum"
    BASE_MODEL="$CURRICULUM_BASE"
    ;;
  ratio)
    ADAPTER_ARG="--adapter-path $OUTPUTS_RATIO/dpo_curriculum"
    BASE_MODEL="$MODEL_ID"
    ;;
  *)
    echo "ERROR: Unknown MODEL_VARIANT='$MODEL_VARIANT'. Use: raw baseline curriculum ratio"
    exit 1
    ;;
esac

echo "=== Jailbreak Inference: model=$MODEL_KEY variant=$MODEL_VARIANT exp=$EXP_ID ==="
echo "Results → $RESULTS_DIR"

# GCG
echo ""
echo "--- GCG attacks ---"
$PYRUN \
  --base-model "$BASE_MODEL" \
  $ADAPTER_ARG \
  --test-cases "$GCG_CASES" \
  --attack-type gcg \
  --output "$RESULTS_DIR/gcg_${MODEL_VARIANT}.jsonl" \
  --batch-size 4 \
  --max-new-tokens 256

# AutoDAN (skip if SKIP_AUTODAN=1)
if [ "${SKIP_AUTODAN:-0}" != "1" ]; then
  echo ""
  echo "--- AutoDAN attacks ---"
  $PYRUN \
    --base-model "$BASE_MODEL" \
    $ADAPTER_ARG \
    --test-cases "$AUTODAN_CASES" \
    --attack-type autodan \
    --output "$RESULTS_DIR/autodan_${MODEL_VARIANT}.jsonl" \
    --batch-size 4 \
    --max-new-tokens 2048
fi

echo ""
echo "=== Jailbreak Inference complete: variant=$MODEL_VARIANT ==="
