#!/bin/bash
#SBATCH --job-name=task13_compare
#SBATCH --partition=emergency_gpua40
#SBATCH --qos=emergency_gpua40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --output=task13_compare_%j.out
#SBATCH --error=task13_compare_%j.err

set -e

module load anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate nanogpt

export HF_ENDPOINT=https://hf-mirror.com

cd "$SLURM_SUBMIT_DIR"

echo "===== Job information ====="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Start: $(date)"

echo "===== GPU ====="
nvidia-smi

echo "===== GPT-2 before/after comparison ====="
python scripts/compare_gpt2_finetune.py

echo "===== Finished ====="
echo "End: $(date)"
