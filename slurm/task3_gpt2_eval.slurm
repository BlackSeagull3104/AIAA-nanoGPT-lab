#!/bin/bash
#SBATCH --job-name=gpt2_eval
#SBATCH --partition=i64m1tga40u
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=slurm/%x_%j.out
#SBATCH --error=slurm/%x_%j.err

module load anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate nanogpt

cd ~/AIAA-nanoGPT-lab

python train.py config/eval_gpt2_pretrained_wandb.py
