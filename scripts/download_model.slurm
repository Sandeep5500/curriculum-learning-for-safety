#!/bin/bash
#SBATCH --job-name=download_model
#SBATCH --output=/data/user_data/sandeep3/curriculum-safety/logs/download_model_%j.out
#SBATCH --error=/data/user_data/sandeep3/curriculum-safety/logs/download_model_%j.err
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --partition=cpu

# Download a model from HuggingFace Hub.
# Run: MODEL_REPO=mlabonne/gemma-3-4b-it-abliterated sbatch scripts/download_model.slurm

set -euo pipefail
REPO_ROOT="/data/user_data/sandeep3/curriculum-safety"
cd "$REPO_ROOT"
source venv-new/bin/activate

MODEL_REPO="${MODEL_REPO:-mlabonne/gemma-3-4b-it-abliterated}"

export HF_TOKEN=$(cat "$REPO_ROOT/.hf_token")
export HF_HOME="/data/user_data/sandeep3/.cache/huggingface"
export REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-bundle.crt
export SSL_CERT_FILE=/etc/ssl/certs/ca-bundle.crt

echo "Downloading $MODEL_REPO ..."
env -u HF_HUB_OFFLINE -u TRANSFORMERS_OFFLINE \
  python3 -c "
from huggingface_hub import snapshot_download
import os
path = snapshot_download(
    repo_id='$MODEL_REPO',
    cache_dir=os.environ['HF_HOME'] + '/hub',
    token=os.environ['HF_TOKEN'],
)
print(f'Downloaded to: {path}')
"
echo "Done."
