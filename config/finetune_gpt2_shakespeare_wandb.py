# Task 4: fine-tune pretrained GPT-2 on GPT-2 BPE Tiny Shakespeare.

out_dir = 'out-gpt2-finetune-shakespeare-wandb'
eval_interval = 10
eval_iters = 50
log_interval = 1
always_save_checkpoint = True
init_from = 'gpt2'

wandb_log = True
wandb_project = 'aiaa-nanogpt-task4'
wandb_run_name = 'gpt2-finetune-shakespeare'
wandb_tags = 'pretrained,gpt2,finetuned'
wandb_prompts = 'ROMEO:|JULIET:|To be, or not to be'
wandb_generate_tokens = 64
wandb_compare_samples = True
wandb_log_generation_speed = True
wandb_before_stage = 'pretrained'
wandb_after_stage = 'finetuned'
wandb_eval_tracking = True
wandb_artifact_name = 'nanogpt-gpt2-finetuned-checkpoint'

dataset = 'shakespeare'
tokenizer_type = 'gpt2'
gradient_accumulation_steps = 8
batch_size = 4
block_size = 1024

learning_rate = 3e-5
max_iters = 100
decay_lr = False

compile = False
seed = 1337
