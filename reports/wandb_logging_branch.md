# W&B Experiment Tracking

This branch organizes the W&B logging components of the nanoGPT course project.

## Scope

- Training and validation loss logging
- Learning rate and MFU tracking
- Iteration time and throughput
- Gradient norm and parameter statistics
- W&B Tables
- Checkpoint Artifacts
- Experiment summaries

## Relevant files

- `train.py`
- `config/train_shakespeare_char_scratch_wandb.py`
- `config/eval_gpt2_pretrained_wandb.py`
- `config/finetune_gpt2_shakespeare_wandb.py`
- `reports/wandb/experiment_summary.md`
- `reports/wandb/scratch_training_curve.png`

## Validation

W&B integration should be validated with lightweight runs that confirm metric logging, Tables, and Artifact handling without full model training.

The lightweight validation should verify key logged metrics including loss, learning rate, iteration time, and gradient norm.
