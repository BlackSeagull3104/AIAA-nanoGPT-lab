# Task 2: train a small character-level GPT from scratch with W&B tracking.

out_dir = 'out-shakespeare-char-scratch-wandb'
eval_interval = 100
eval_iters = 50
log_interval = 10
always_save_checkpoint = False
init_from = 'scratch'

wandb_log = True
wandb_project = 'aiaa-nanogpt-task2'
wandb_run_name = 'shakespeare-char-scratch'
wandb_tags = 'scratch,character-tokenizer'
wandb_prompts = 'First Citizen:|ROMEO:|JULIET:|HAMLET:|To be, or not to be'
wandb_generate_tokens = 64
wandb_compare_samples = True
wandb_artifact_name = 'nanogpt-scratch-checkpoint'

dataset = 'shakespeare_char'
tokenizer_type = 'character'
gradient_accumulation_steps = 1
batch_size = 32
block_size = 128

n_layer = 4
n_head = 4
n_embd = 128
dropout = 0.2

learning_rate = 1e-3
max_iters = 5000
lr_decay_iters = 5000
min_lr = 1e-4
beta2 = 0.99
warmup_iters = 100

seed = 1337
