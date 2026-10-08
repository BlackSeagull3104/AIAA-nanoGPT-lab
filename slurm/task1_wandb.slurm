#!/bin/bash

#SBATCH --job-name=wandb_task1
#SBATCH --partition=emergency_gpua40
#SBATCH --qos=emergency_gpua40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=task1_wandb_%j.out
#SBATCH --error=task1_wandb_%j.err

set -euo pipefail

echo "===== Job information ====="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"

module load anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate nanogpt

cd "$SLURM_SUBMIT_DIR"

echo "===== GPU check ====="
nvidia-smi


python - <<'PY'
import torch

print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if not torch.cuda.is_available():
    raise RuntimeError("No GPU")

print(torch.cuda.get_device_name(0))
PY


echo "===== Task1 W&B test ====="

python train.py config/train_shakespeare_char.py \
    --wandb_log=True \
    --out_dir=out-wandb-task1 \
    --max_iters=500 \
    --eval_interval=100 \
    --eval_iters=20 \
    --log_interval=1


echo "===== Finished ====="
