#!/bin/bash
#SBATCH --job-name=align_depth
#SBATCH --partition=general
#SBATCH --gres=gpu:A6000:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=3:00:00
#SBATCH --output=/data/user_data/sandeep3/curriculum-safety/logs/align_depth_%j.out
#SBATCH --error=/data/user_data/sandeep3/curriculum-safety/logs/align_depth_%j.err

# Alignment Depth Analysis: measures whether curriculum training produces
# deeper safety alignment (suppression signal distributed across more tokens).
#
# Usage:
#   CURRI_EXP=exp_122 BASELINE_EXP=exp_127 MODEL_KEY=llama3 \
#     sbatch scripts/run_alignment_depth.slurm

set -e
REPO_ROOT="/data/user_data/sandeep3/curriculum-safety"
cd "$REPO_ROOT"

CURRI_EXP="${CURRI_EXP:-exp_122}"
BASELINE_EXP="${BASELINE_EXP:-exp_127}"
MODEL_KEY="${MODEL_KEY:-llama3}"
DATASET_KEY="${DATASET_KEY:-combined_pku_hh}"
DATA_EXP_ID="${DATA_EXP_ID:-exp_073}"
N_SAMPLES="${N_SAMPLES:-200}"

source "$REPO_ROOT/scripts/model_configs.sh"
module load cuda-12.1

DATA_SITE="$REPO_ROOT/$VENV_PATH/lib/python3.9/site-packages"
USER_SITE="/home/sandeep3/.local/lib/python3.9/site-packages"

BASELINE_ADAPTER="$REPO_ROOT/outputs/$BASELINE_EXP/$DATASET_KEY/$MODEL_SHORT/dpo_baseline"
CURRI_ADAPTER="$REPO_ROOT/outputs/$CURRI_EXP/$DATASET_KEY/$MODEL_SHORT/dpo_curriculum"
CURRI_BASE=$(cat "$CURRI_ADAPTER/curri_base_model.txt" 2>/dev/null || echo "$MODEL_ID")
# If the curri base is a local path that no longer exists, fall back to original model
if [ ! -d "$CURRI_BASE" ] && [ "${CURRI_BASE:0:1}" = "/" ]; then
  echo "  WARNING: curri base '$CURRI_BASE' not found on disk, falling back to $MODEL_ID"
  CURRI_BASE="$MODEL_ID"
fi
TEST_PATH="$REPO_ROOT/data/processed/$DATA_EXP_ID/$DATASET_KEY/$MODEL_SHORT/curriculum_dataset_test.jsonl"
OUTPUT_DIR="$REPO_ROOT/results/${CURRI_EXP}_${BASELINE_EXP}/$DATASET_KEY/$MODEL_SHORT/alignment_depth"

echo "=== Alignment Depth Analysis ==="
echo "  curri=$CURRI_EXP ($CURRI_ADAPTER)"
echo "  baseline=$BASELINE_EXP ($BASELINE_ADAPTER)"
echo "  curri base=$CURRI_BASE"
echo "  test=$TEST_PATH"
echo "  output=$OUTPUT_DIR"
echo "  n_samples=$N_SAMPLES"

env HF_HOME=/data/user_data/sandeep3/.cache/huggingface \
    PYTHONPATH="$REPO_ROOT:$DATA_SITE:$USER_SITE:$PYTHONPATH" \
  python3 -m src.interpretability.alignment_depth \
    --base-model "$MODEL_ID" \
    --baseline-adapter "$BASELINE_ADAPTER" \
    --curri-adapter "$CURRI_ADAPTER" \
    --curri-base "$CURRI_BASE" \
    --test-path "$TEST_PATH" \
    --output "$OUTPUT_DIR" \
    --n-samples "$N_SAMPLES"

echo "=== Done: $OUTPUT_DIR ==="

# Copy plots to shared plots directory
PLOTS_DIR="$REPO_ROOT/results/plots"
mkdir -p "$PLOTS_DIR"
for f in "$OUTPUT_DIR"/*.png; do
  [ -f "$f" ] || continue
  fname=$(basename "$f" .png)
  cp "$f" "$PLOTS_DIR/${CURRI_EXP}_${BASELINE_EXP}_${fname}.png"
  echo "Copied to $PLOTS_DIR/${CURRI_EXP}_${BASELINE_EXP}_${fname}.png"
done
