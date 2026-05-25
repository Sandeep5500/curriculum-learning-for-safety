#!/bin/bash
#SBATCH --job-name=prefill_inference
#SBATCH --partition=general
#SBATCH --gres=gpu:A6000:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --time=1:00:00
#SBATCH --output=/data/user_data/sandeep3/curriculum-safety/logs/prefill_inference_%j.out
#SBATCH --error=/data/user_data/sandeep3/curriculum-safety/logs/prefill_inference_%j.err

# Prefill attack inference.
# For each record in door_full_parsed.jsonl, builds a prompt with the first k tokens
# of the harmful response as a prefill, then generates the model's continuation.
#
# Run one job per model variant:
#   MODEL_VARIANT=raw       MODEL_KEY=qwen3 K=20 sbatch scripts/run_prefill_inference.slurm exp_009
#   MODEL_VARIANT=baseline  MODEL_KEY=qwen3 K=20 sbatch scripts/run_prefill_inference.slurm exp_009
#   MODEL_VARIANT=curriculum MODEL_KEY=qwen3 K=20 sbatch scripts/run_prefill_inference.slurm exp_009
#   MODEL_VARIANT=ratio     MODEL_KEY=qwen3 K=20 sbatch scripts/run_prefill_inference.slurm exp_009

set -e
REPO_ROOT="/data/user_data/sandeep3/curriculum-safety"
cd "$REPO_ROOT"

EXP_ID="${1:-exp_009}"
MODEL_KEY="${MODEL_KEY:-qwen3}"
DATASET_KEY="${DATASET_KEY:-pku}"
MODEL_VARIANT="${MODEL_VARIANT:-raw}"
K="${K:-20}"
source "$REPO_ROOT/scripts/model_configs.sh"

module load cuda-12.1

DATA_SITE="$REPO_ROOT/$VENV_PATH/lib/python3.9/site-packages"
USER_SITE="/home/sandeep3/.local/lib/python3.9/site-packages"

OUTPUTS_DIR="$REPO_ROOT/outputs/$EXP_ID/$DATASET_KEY/$MODEL_SHORT"
OUTPUTS_RATIO="$REPO_ROOT/outputs/${EXP_ID}_ratio/$DATASET_KEY/$MODEL_SHORT"
RESULTS_DIR="$REPO_ROOT/results/$EXP_ID/$DATASET_KEY/$MODEL_SHORT/prefill/k_${K}"
EVAL_DATA="$REPO_ROOT/data/raw/door_full_parsed.jsonl"
mkdir -p "$RESULTS_DIR"

PYRUN="HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  HF_HOME=/data/user_data/sandeep3/.cache/huggingface \
  REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-bundle.crt \
  SSL_CERT_FILE=/etc/ssl/certs/ca-bundle.crt \
  PYTHONPATH=$REPO_ROOT:$DATA_SITE:$USER_SITE:$PYTHONPATH \
  python3 -m src.eval.evaluate_prefill"

# Resolve base model for curri-dpo (adapter is built on merged stage, not original base)
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

echo "=== Prefill Inference: model=$MODEL_KEY variant=$MODEL_VARIANT k=$K exp=$EXP_ID ==="
echo "Results → $RESULTS_DIR"
echo ""

eval $PYRUN \
  --base-model "$BASE_MODEL" \
  $ADAPTER_ARG \
  --eval-data "$EVAL_DATA" \
  --k "$K" \
  --output "$RESULTS_DIR/prefill_${MODEL_VARIANT}.jsonl" \
  --batch-size 8 \
  --max-new-tokens 256

echo ""
echo "=== Prefill Inference complete: variant=$MODEL_VARIANT k=$K ==="
