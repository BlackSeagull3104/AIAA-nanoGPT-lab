# Task 6 release wiring check: deliberately tiny, CPU-only, and two iterations.

out_dir = 'out-task6-repro'
eval_interval = 1
eval_iters = 1
log_interval = 1
always_save_checkpoint = True
init_from = 'scratch'

wandb_log = False
wandb_project = 'aiaa-nanogpt-task6-reproducibility'
wandb_run_name = 'task6-lightweight-reproduction'
wandb_tags = 'task6,reproducibility,cpu-smoke'

dataset = 'shakespeare_char'
tokenizer_type = 'character'
gradient_accumulation_steps = 1
batch_size = 1
block_size = 32

n_layer = 1
n_head = 1
n_embd = 32
dropout = 0.0
bias = True

learning_rate = 1e-3
max_iters = 2
decay_lr = False
warmup_iters = 0

device = 'cpu'
dtype = 'float32'
compile = False
seed = 1337
