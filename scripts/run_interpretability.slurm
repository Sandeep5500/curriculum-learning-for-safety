#!/bin/bash
#SBATCH --job-name=interpretability
#SBATCH --partition=general
#SBATCH --gres=gpu:A6000:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=4:00:00
#SBATCH --output=/data/user_data/sandeep3/curriculum-safety/logs/interpretability_%j.out
#SBATCH --error=/data/user_data/sandeep3/curriculum-safety/logs/interpretability_%j.err

# Interpretability analysis: linear probing, FDR, UMAP for curri-dpo vs baseline.
# Compares 5 variants: raw, baseline, curri_s0, curri_s1, curri_s2.
#
# Run: MODEL_KEY=llama3 DATASET_KEY=combined_pku_hh sbatch scripts/run_interpretability.slurm exp_109

set -e
REPO_ROOT="/data/user_data/sandeep3/curriculum-safety"
cd "$REPO_ROOT"

EXP_ID="${1:-exp_109}"
BASELINE_EXP="${2:-exp_098}"
MODEL_KEY="${MODEL_KEY:-llama3}"
DATASET_KEY="${DATASET_KEY:-combined_pku_hh}"
DATA_EXP_ID="${DATA_EXP_ID:-exp_073}"
source "$REPO_ROOT/scripts/model_configs.sh"

module load cuda-12.1

DATA_SITE="$REPO_ROOT/$VENV_PATH/lib/python3.9/site-packages"
USER_SITE="/home/sandeep3/.local/lib/python3.9/site-packages"

RESULTS_DIR="$REPO_ROOT/results/$EXP_ID/$DATASET_KEY/$MODEL_SHORT/interpretability"
DATASET_PATH="$REPO_ROOT/data/processed/$DATA_EXP_ID/$DATASET_KEY/$MODEL_SHORT/curriculum_dataset_test.jsonl"
mkdir -p "$RESULTS_DIR"

PYRUN="env -u HF_HUB_OFFLINE -u TRANSFORMERS_OFFLINE HF_HOME=/data/user_data/sandeep3/.cache/huggingface REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-bundle.crt SSL_CERT_FILE=/etc/ssl/certs/ca-bundle.crt PYTHONPATH=$REPO_ROOT:$DATA_SITE:$USER_SITE:$PYTHONPATH python3"

echo "=== Interpretability: model=$MODEL_KEY dataset=$DATASET_KEY exp=$EXP_ID ==="
echo "Dataset: $DATASET_PATH"
echo "Results → $RESULTS_DIR"

N_SAMPLES="${N_SAMPLES:-}"  # empty = use all 2187 pairs; set N_SAMPLES=500 for faster run
UMAP_LAYER="${UMAP_LAYER:-28}"

$PYRUN -m src.interpretability.probe_safety \
  --base-model "$MODEL_ID" \
  --exp-id "$EXP_ID" \
  --baseline-exp "$BASELINE_EXP" \
  --model-short "$MODEL_SHORT" \
  --dataset-path "$DATASET_PATH" \
  --output "$RESULTS_DIR" \
  --layers 0 2 4 6 8 10 12 14 16 18 20 22 24 26 28 30 32 \
  --batch-size 4 \
  --umap-layer "$UMAP_LAYER" \
  ${N_SAMPLES:+--n-samples "$N_SAMPLES"}

echo ""
echo "=== Interpretability complete: $RESULTS_DIR ==="
