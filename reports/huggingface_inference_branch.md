# Hugging Face Inference

This branch organizes the Hugging Face inference components of the nanoGPT course project.

## Scope

- Hugging Face GPT-2 loading and inference
- Hugging Face / nanoGPT GPT-2 weight alignment
- Qwen instruct inference comparison
- GPT-2 fine-tuning configuration

## Relevant files

- `scripts/task9_hf_gpt2_inference.py`
- `scripts/task10_hf_nanogpt_alignment.py`
- `scripts/task11_instruct_comparison.py`
- `scripts/task6_qwen_qa_comparison.py`
- `config/task6_qwen_qa_wandb.py`
- `config/finetune_gpt2_shakespeare_wandb.py`
- `reports/hf_gpt2_predictions.jsonl`
- `reports/hf_nanogpt_alignment.json`
- `reports/task11_instruct_comparison.json`

## Validation

Inference and alignment checks should use fixed prompts and lightweight model loading. Full fine-tuning is not required for PR validation.
