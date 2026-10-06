#!/bin/bash

#SBATCH --job-name=task4_gpt2_ft
#SBATCH --partition=emergency_gpua40
#SBATCH --qos=emergency_gpua40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=task4_gpt2_finetune_%j.out
#SBATCH --error=task4_gpt2_finetune_%j.err

set -euo pipefail

module load anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate nanogpt
export HF_ENDPOINT=https://hf-mirror.com

cd "$SLURM_SUBMIT_DIR"

echo "===== Task 4: GPT-2 Shakespeare fine-tuning ====="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Start: $(date)"

nvidia-smi

if [[ ! -f data/shakespeare/train.bin || ! -f data/shakespeare/val.bin ]]; then
    echo "Missing GPT-2 BPE Shakespeare data. Run: python data/shakespeare/prepare.py"
    exit 1
fi

python train.py config/finetune_gpt2_shakespeare_wandb.py

echo "===== Task 4 finished ====="
echo "End: $(date)"
