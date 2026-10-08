#!/bin/bash
#SBATCH --job-name=scratch_wandb
#SBATCH --partition=emergency_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=slurm/scratch_%j.out
#SBATCH --error=slurm/scratch_%j.err

echo "===== Job information ====="
echo "Job ID: $SLURM_JOB_ID"
hostname

module load anaconda3

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate nanogpt

cd ~/AIAA-nanoGPT-lab

python train.py config/train_shakespeare_char_scratch_wandb.py
