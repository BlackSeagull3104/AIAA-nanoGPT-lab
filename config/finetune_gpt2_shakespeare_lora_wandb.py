# Task 8: parameter-efficient GPT-2 fine-tuning with LoRA on combined QKV.

out_dir = 'out-gpt2-lora-shakespeare-wandb'
eval_interval = 10
eval_iters = 50
log_interval = 1
always_save_checkpoint = True
init_from = 'gpt2'

lora_enabled = True
lora_rank = 8
lora_alpha = 16.0
lora_dropout = 0.05
lora_target_modules = 'attn.c_attn'
lora_log_adapter_artifact = True
lora_adapter_artifact_name = 'nanogpt-gpt2-lora-adapter'

wandb_log = True
wandb_project = 'aiaa-nanogpt-task8'
wandb_group = 'task8-lora-vs-full-finetune'
wandb_run_name = 'gpt2-lora-shakespeare'
wandb_tags = 'pretrained,gpt2,lora,parameter-efficient-finetuning'
wandb_prompts = 'ROMEO:|JULIET:|To be, or not to be'
wandb_generate_tokens = 64
wandb_compare_samples = True
wandb_log_generation_speed = True
wandb_before_stage = 'pretrained'
wandb_after_stage = 'lora-finetuned'
wandb_log_model_comparison = True
wandb_eval_tracking = True
wandb_log_checkpoint_artifact = False

# Historical comparison only; the Task 4 run is never modified or rerun.
full_finetune_reference_run = 'ermv6omj'
full_finetune_validation_loss = 3.24544
full_finetune_perplexity = 25.6730
full_finetune_peak_gpu_memory_mb = 4073.75
full_finetune_training_tokens_per_sec = 64226.63

dataset = 'shakespeare'
tokenizer_type = 'gpt2'
gradient_accumulation_steps = 8
batch_size = 4
block_size = 1024

learning_rate = 1e-4
weight_decay = 0.0
max_iters = 100
decay_lr = False

compile = False
seed = 1337
