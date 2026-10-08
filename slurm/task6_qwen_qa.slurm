#!/bin/bash

#SBATCH --job-name=task6_qwen_qa
#SBATCH --partition=emergency_gpua40
#SBATCH --qos=emergency_gpua40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=task6_qwen_qa_%j.out
#SBATCH --error=task6_qwen_qa_%j.err

set -euo pipefail

module load anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate qwen-task6

# Use the same Hugging Face mirror as Task 4.
export HF_ENDPOINT=https://hf-mirror.com

cd "$SLURM_SUBMIT_DIR"

echo "===== Task 6: GPT-2 vs Qwen Instruct QA comparison ====="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Start: $(date)"

nvidia-smi

if [[ ! -f config/task6_qwen_qa_wandb.py ]]; then
    echo "Missing config/task6_qwen_qa_wandb.py"
    exit 1
fi

if [[ ! -f scripts/task6_qwen_qa_comparison.py ]]; then
    echo "Missing scripts/task6_qwen_qa_comparison.py"
    exit 1
fi

python scripts/task6_qwen_qa_comparison.py \
    config/task6_qwen_qa_wandb.py

echo "===== Task 6 finished ====="
echo "End: $(date)"
