#!/bin/bash

#SBATCH --job-name=task8_gpt2_lora
#SBATCH --partition=emergency_gpua40
#SBATCH --qos=emergency_gpua40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=task8_gpt2_lora_%j.out
#SBATCH --error=task8_gpt2_lora_%j.err

set -euo pipefail

module load anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate nanogpt
export HF_ENDPOINT=https://hf-mirror.com

cd "$SLURM_SUBMIT_DIR"

if [[ ! -f data/shakespeare/train.bin || ! -f data/shakespeare/val.bin ]]; then
    echo "Missing GPT-2 BPE Shakespeare data. Run: python data/shakespeare/prepare.py"
    exit 1
fi

python scripts/test_lora.py

SMOKE_TEST="${SMOKE_TEST:-0}"
if [[ "$SMOKE_TEST" == "1" ]]; then
    OUT_DIR="out-gpt2-lora-smoke"
    python train.py config/finetune_gpt2_shakespeare_lora_wandb.py \
        --out_dir="$OUT_DIR" --max_iters=2 --eval_interval=1 --eval_iters=2 \
        --batch_size=1 --gradient_accumulation_steps=1 --wandb_log=False
    RELOAD_TOKENS=8
else
    OUT_DIR="out-gpt2-lora-shakespeare-wandb"
    python train.py config/finetune_gpt2_shakespeare_lora_wandb.py
    RELOAD_TOKENS=64
fi

python scripts/test_lora_adapter.py \
    --adapter "$OUT_DIR/lora_adapter.pt" \
    --device cuda \
    --max-new-tokens "$RELOAD_TOKENS"
