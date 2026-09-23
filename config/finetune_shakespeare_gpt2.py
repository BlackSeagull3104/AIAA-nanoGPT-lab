import time

# output / evaluation
out_dir = 'out-shakespeare-gpt2'
eval_interval = 10
eval_iters = 20
log_interval = 1

wandb_log = False
wandb_project = 'shakespeare'
wandb_run_name = 'gpt2-ft-' + str(time.time())

# data / pretrained model
dataset = 'shakespeare'
init_from = 'gpt2'

# Save checkpoint only when validation loss improves
always_save_checkpoint = False

# GPT-2 124M fine-tuning
block_size = 1024
batch_size = 4
gradient_accumulation_steps = 8
max_iters = 100

# fine-tuning
learning_rate = 3e-5
decay_lr = False
