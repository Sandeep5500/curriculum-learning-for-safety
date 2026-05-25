#!/bin/bash
#SBATCH --job-name=safety_eval
#SBATCH --partition=general
#SBATCH --gres=gpu:A6000:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=/data/user_data/sandeep3/curriculum-safety/logs/safety_eval_%j.out
#SBATCH --error=/data/user_data/sandeep3/curriculum-safety/logs/safety_eval_%j.err

# Phase 3a: Safety evaluation on AdvBench, SorryBench, HEx-PHI, and XSTest.
# Evaluates 3 model variants (raw, baseline, curriculum).
# Outputs go to results/{EXP_ID}/{MODEL_SHORT}/phase3/.
#
# Run: MODEL_KEY=llama3 DATASET_KEY=pku sbatch scripts/run_safety_eval.slurm [exp_id]

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
export HF_TOKEN=$(cat "$REPO_ROOT/.hf_token")
export HF_HOME="/data/user_data/sandeep3/.cache/huggingface"

RESULTS_DIR="$REPO_ROOT/results/$EXP_ID/$DATASET_KEY/$MODEL_SHORT/phase3"
OUTPUTS_DIR="$REPO_ROOT/outputs/$EXP_ID/$DATASET_KEY/$MODEL_SHORT"
RAW_CACHE="$REPO_ROOT/results/_raw/$DATASET_KEY/$MODEL_SHORT"
mkdir -p "$RESULTS_DIR" "$RAW_CACHE"

PYRUN="env -u HF_HUB_OFFLINE -u TRANSFORMERS_OFFLINE HF_HOME=/data/user_data/sandeep3/.cache/huggingface REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-bundle.crt SSL_CERT_FILE=/etc/ssl/certs/ca-bundle.crt PYTHONPATH=$REPO_ROOT:$DATA_SITE:$USER_SITE:$PYTHONPATH python3 -m src.eval.evaluate_safety"
OPENAI_KEY_FILE="${OPENAI_KEY_FILE:-$REPO_ROOT/open-ai-token.txt}"

NOTHINK_FLAG="--no-think"
if [ "${NO_THINK:-1}" = "0" ]; then
  NOTHINK_FLAG=""
  echo "Thinking ENABLED (NO_THINK=0)"
else
  echo "Thinking DISABLED (default)"
fi

echo "=== Safety Eval: model=$MODEL_KEY exp=$EXP_ID ==="
echo "Results → $RESULTS_DIR"

# Helper: generate responses for one benchmark/variant (LlamaGuard skipped for non-XSTest)
eval_one() {
  local bench="$1"
  local label="$2"
  shift 2
  local adapter_args="$@"
  local out="$RESULTS_DIR/${bench}_${label}.json"
  echo ""
  echo "--- $bench / $label ---"
  if [ -f "$out" ] && python3 -c "import json; d=json.load(open('$out')); exit(0 if d['samples'] else 1)" 2>/dev/null; then
    echo "  Already exists with responses, skipping: $out"
    return 0
  fi
  local skip_flag=""
  [ "$bench" != "xstest" ] && skip_flag="--skip_llamaguard"
  $PYRUN \
    --base_model "$MODEL_ID" \
    $adapter_args \
    --benchmark "$bench" \
    --output "$out" \
    $skip_flag \
    $NOTHINK_FLAG
}

# ── Raw model (no adapter) ──────────────────────────────────────────────────
for bench in advbench sorrybench hexphi xstest; do
  CACHE_FILE="$RAW_CACHE/${bench}_raw.json"
  OUT_FILE="$RESULTS_DIR/${bench}_raw.json"
  if [ -f "$CACHE_FILE" ]; then
    echo "  Copying cached raw result: $bench → $OUT_FILE"
    cp "$CACHE_FILE" "$OUT_FILE"
  else
    eval_one "$bench" "raw"
    cp "$OUT_FILE" "$CACHE_FILE"
  fi
done

# ── Baseline DPO ─────────────────────────────────────────────────────────────
if [ -d "$OUTPUTS_DIR/dpo_baseline" ]; then
  for bench in advbench sorrybench hexphi xstest; do
    eval_one "$bench" "baseline" --adapter_path "$OUTPUTS_DIR/dpo_baseline"
  done
fi

# ── Curriculum DPO ───────────────────────────────────────────────────────────
if [ -d "$OUTPUTS_DIR/dpo_curriculum" ]; then
  # Detect Curri-DPO: if curri_base_model.txt exists, use staged merged model as base
  CURRICULUM_BASE="$MODEL_ID"
  if [ -f "$OUTPUTS_DIR/dpo_curriculum/curri_base_model.txt" ]; then
    CURRICULUM_BASE=$(cat "$OUTPUTS_DIR/dpo_curriculum/curri_base_model.txt")
    echo "  Curri-DPO detected: using base model $CURRICULUM_BASE"
  fi
  _saved_model_id="$MODEL_ID"
  MODEL_ID="$CURRICULUM_BASE"
  for bench in advbench sorrybench hexphi xstest; do
    eval_one "$bench" "curriculum" --adapter_path "$OUTPUTS_DIR/dpo_curriculum"
  done
  MODEL_ID="$_saved_model_id"
fi

# ── GPT judge for advbench/sorrybench/hexphi ─────────────────────────────────
echo ""
echo "=== Running GPT judge on advbench/sorrybench/hexphi ==="
REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-bundle.crt \
  SSL_CERT_FILE=/etc/ssl/certs/ca-bundle.crt \
  PYTHONPATH=$REPO_ROOT:$DATA_SITE:$USER_SITE:$PYTHONPATH \
  python3 -m src.eval.judge_safety_gpt \
    --phase3-dir "$RESULTS_DIR" \
    --benchmarks advbench sorrybench hexphi \
    --variants raw baseline curriculum \
    --openai-key-file "$OPENAI_KEY_FILE" \
    --output "$RESULTS_DIR/summary_gpt.json" \
    --max-workers 16

echo ""
echo "=== Safety Eval complete: $RESULTS_DIR ==="
