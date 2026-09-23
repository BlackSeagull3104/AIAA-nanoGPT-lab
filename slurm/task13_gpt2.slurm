#!/bin/bash

#SBATCH --job-name=task13_gpt2
#SBATCH --partition=emergency_gpua40
#SBATCH --qos=emergency_gpua40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=task13_%j.out
#SBATCH --error=task13_%j.err

set -e

echo "===== Job information ====="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Start: $(date)"

echo "===== GPU ====="
nvidia-smi

echo "===== Training ====="
python train.py config/finetune_shakespeare_gpt2.py

echo "===== Before / After comparison ====="
python scripts/compare_gpt2_finetune.py

echo "===== Finished ====="
echo "End: $(date)"
