#!/bin/bash
# Model + dataset registry.
# Source this file from SLURM scripts: source scripts/model_configs.sh
#
# Usage:
#   EXP_ID=exp_001 MODEL_KEY=llama3 DATASET_KEY=combined_pku_hh
#   source scripts/model_configs.sh
#   echo $MODEL_ID          # informatiker/Llama-3-8B-Instruct-abliterated
#   echo $LORA_MODULES      # q_proj v_proj
#   echo $DATASET_INPUT     # data/processed/combined_pku_hh.jsonl
#
# Supported MODEL_KEY values: llama3, qwen3-8b-v3, yi15_9b, gemma3_4b, qwen3_1b7, qwen3_4b
# Supported DATASET_KEY values: combined_pku_hh (recommended), pku_gpt, hh

MODEL_KEY="${MODEL_KEY:-llama3}"

case "$MODEL_KEY" in
  llama3)
    MODEL_ID="QuixiAI/Llama-3-8B-Instruct-abliterated-v2"
    MODEL_SHORT="llama3"
    LORA_MODULES="${LORA_MODULES:-q_proj v_proj}"
    VENV_PATH="${VENV_PATH:-venv-new}"
    ;;
  qwen3-8b-v3)
    MODEL_ID="Goekdeniz-Guelmez/Josiefied-Qwen3-8B-abliterated-v1"
    MODEL_SHORT="qwen3-8b-v3"
    LORA_MODULES="${LORA_MODULES:-q_proj v_proj}"
    VENV_PATH="${VENV_PATH:-venv-new}"
    ;;
  yi15_9b)
    MODEL_ID="byroneverson/Yi-1.5-9B-Chat-abliterated"
    MODEL_SHORT="yi15_9b"
    LORA_MODULES="${LORA_MODULES:-q_proj v_proj}"
    VENV_PATH="${VENV_PATH:-venv-new}"
    ;;
  gemma3_4b)
    MODEL_ID="mlabonne/gemma-3-4b-it-abliterated"
    MODEL_SHORT="gemma3_4b"
    LORA_MODULES="${LORA_MODULES:-q_proj v_proj}"
    VENV_PATH="${VENV_PATH:-venv-new}"
    ;;
  qwen3_1b7)
    MODEL_ID="mlabonne/Qwen3-1.7B-abliterated"
    MODEL_SHORT="qwen3_1b7"
    LORA_MODULES="${LORA_MODULES:-q_proj v_proj}"
    VENV_PATH="${VENV_PATH:-venv-new}"
    ;;
  qwen3_4b)
    MODEL_ID="mlabonne/Qwen3-4B-abliterated"
    MODEL_SHORT="qwen3_4b"
    LORA_MODULES="${LORA_MODULES:-q_proj v_proj}"
    VENV_PATH="${VENV_PATH:-venv-new}"
    ;;
  *)
    echo "ERROR: Unknown MODEL_KEY='$MODEL_KEY'. Valid values: llama3, qwen3-8b-v3, yi15_9b, gemma3_4b, qwen3_1b7, qwen3_4b"
    exit 1
    ;;
esac

DATASET_KEY="${DATASET_KEY:-combined_pku_hh}"

case "$DATASET_KEY" in
  combined_pku_hh)
    DATASET_INPUT="${REPO_ROOT}/data/processed/combined_pku_hh.jsonl"
    DEFAULT_EPOCHS=5
    ;;
  pku_gpt)
    DATASET_INPUT="${REPO_ROOT}/data/processed/clean_parsed_gpt.jsonl"
    DEFAULT_EPOCHS=5
    ;;
  hh)
    DATASET_INPUT="${REPO_ROOT}/data/processed/hh_rlhf_clean_gpt.jsonl"
    DEFAULT_EPOCHS=5
    ;;
  *)
    echo "ERROR: Unknown DATASET_KEY='$DATASET_KEY'. Valid values: combined_pku_hh, pku_gpt, hh"
    exit 1
    ;;
esac

# DATA_EXP_ID: which experiment's phase1 data to use for training (defaults to EXP_ID).
# Set DATA_EXP_ID=exp_001 to reuse a previous scoring run without re-running phase 1.
DATA_EXP_ID="${DATA_EXP_ID:-$EXP_ID}"
PROCESSED_DIR="${REPO_ROOT}/data/processed/${DATA_EXP_ID}/${DATASET_KEY}/${MODEL_SHORT}"

# DPO_BETA: KL penalty coefficient (lower = stronger preference learning signal).
DPO_BETA="${DPO_BETA:-0.1}"

echo "Model: $MODEL_KEY → $MODEL_ID"
echo "LoRA modules: $LORA_MODULES"
echo "DPO beta: $DPO_BETA"
echo "Venv: $VENV_PATH"
echo "Dataset: $DATASET_KEY → $DATASET_INPUT"
echo "Processed dir: $PROCESSED_DIR (data from $DATA_EXP_ID)"
