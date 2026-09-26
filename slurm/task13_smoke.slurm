#!/bin/bash

#SBATCH --job-name=t13_smoke
#SBATCH --partition=emergency_gpua40
#SBATCH --qos=emergency_gpua40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --output=task13_smoke_%j.out
#SBATCH --error=task13_smoke_%j.err

set -euo pipefail

echo "===== Job information ====="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Start: $(date)"

echo "===== Environment ====="
module load anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate nanogpt
export HF_ENDPOINT=https://hf-mirror.com

cd "$SLURM_SUBMIT_DIR"

echo "Python: $(which python)"
python --version

echo "===== GPU check ====="
nvidia-smi

python - <<'PY'
import torch
print("PyTorch:", torch.__version__)
print("PyTorch CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available inside the Slurm GPU job.")

print("GPU:", torch.cuda.get_device_name(0))
print("VRAM (GB):", torch.cuda.get_device_properties(0).total_memory / 1024**3)
PY

echo "===== Task 13 smoke training ====="

python train.py config/finetune_shakespeare_gpt2.py \
    --out_dir=out-shakespeare-gpt2-smoke \
    --max_iters=5 \
    --eval_interval=2 \
    --eval_iters=2

echo "===== Smoke test finished ====="
echo "End: $(date)"
