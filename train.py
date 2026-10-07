"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python train.py --batch_size=32 --compile=False

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""

import os
import time
import math
import pickle
from contextlib import nullcontext

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from model import GPTConfig, GPT

# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText
# I/O
out_dir = 'out'
eval_interval = 2000
log_interval = 1
eval_iters = 200
eval_only = False # if True, script exits right after the first eval
always_save_checkpoint = True # if True, always save a checkpoint after each eval
init_from = 'scratch' # 'scratch' or 'resume' or 'gpt2*'
# wandb logging
wandb_log = False # disabled by default
wandb_project = 'owt'
wandb_run_name = 'gpt2' # 'run' + str(time.time())
wandb_group = '' # optional W&B group for related runs
wandb_tags = '' # optional comma-separated tags; inferred from init_from when empty
wandb_prompts = 'First Citizen:|ROMEO:|JULIET:' # fixed prompts separated by |
wandb_generate_tokens = 64
wandb_compare_samples = False # collect one before/after table for scratch experiments
wandb_log_generation_speed = False # add generation throughput to comparison tables
wandb_before_stage = 'before'
wandb_after_stage = 'after'
wandb_log_model_comparison = False # log a wide before/after prompt comparison table
wandb_comparison_observation = '' # optional review note for the wide comparison table
wandb_table_top_k = 100
wandb_token_frequency_chunk_size = 1_000_000
wandb_artifact_name = 'nanogpt-best-checkpoint'
wandb_eval_tracking = False # add perplexity, inference speed, and peak memory
wandb_hf_alignment = False # compare nanoGPT logits/token IDs with Hugging Face
wandb_hf_model_name = 'openai-community/gpt2'
wandb_alignment_prompt = 'To be, or not to be'
wandb_alignment_tokens = 8
wandb_log_eval_artifact = False
wandb_eval_artifact_name = 'nanogpt-evaluation'
wandb_log_checkpoint_artifact = True
# LoRA is disabled by default, so existing nanoGPT runs keep identical behavior.
lora_enabled = False
lora_rank = 8
lora_alpha = 16.0
lora_dropout = 0.0
lora_target_modules = 'attn.c_attn'
lora_log_adapter_artifact = False
lora_adapter_artifact_name = 'nanogpt-gpt2-lora-adapter'
# data
dataset = 'openwebtext'
tokenizer_type = 'auto' # 'auto', 'character', or 'gpt2'
gradient_accumulation_steps = 5 * 8 # used to simulate larger batch sizes
batch_size = 12 # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 1024
# model
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
bias = False # do we use bias inside LayerNorm and Linear layers?
# adamw optimizer
learning_rate = 6e-4 # max learning rate
max_iters = 600000 # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True # whether to decay the learning rate
warmup_iters = 2000 # how many steps to warm up for
lr_decay_iters = 600000 # should be ~= max_iters per Chinchilla
min_lr = 6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
# DDP settings
backend = 'nccl' # 'nccl', 'gloo', etc.
# system
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = True # use PyTorch 2.0 to compile the model to be faster
seed = 1337
# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # will be useful for logging
# -----------------------------------------------------------------------------

# various inits, derived attributes, I/O setup
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank # each process gets a different seed
    # world_size number of processes will be training simultaneously, so we can scale
    # down the desired gradient accumulation iterations per process proportionally
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    # if not ddp, we are running on a single gpu, and one process
    master_process = True
    seed_offset = 0
    ddp_world_size = 1
tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")

if master_process:
    os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(seed + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# poor man's data loader
data_dir = os.path.join('data', dataset)
def get_batch(split):
    # We recreate np.memmap every batch to avoid a memory leak, as per
    # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
    if split == 'train':
        data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9

# attempt to derive vocab_size from the dataset
meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
meta = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

# model init
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=None, dropout=dropout) # start with model_args from command line
if init_from == 'scratch':
    # init a new model from scratch
    print("Initializing a new model from scratch")
    # determine the vocab size we'll use for from-scratch training
    if meta_vocab_size is None:
        print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
    model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
elif init_from == 'resume':
    print(f"Resuming training from {out_dir}")
    # resume training from a checkpoint.
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = checkpoint_model_args[k]
    # create the model
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    if lora_enabled:
        if lora_target_modules != 'attn.c_attn':
            raise ValueError(f"unsupported LoRA targets: {lora_target_modules}")
        model.enable_lora(lora_rank, lora_alpha, lora_dropout)
    state_dict = checkpoint['model']
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']
elif init_from.startswith('gpt2'):
    print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
    # initialize from OpenAI GPT-2 weights
    override_args = dict(dropout=dropout)
    model = GPT.from_pretrained(init_from, override_args)
    if lora_enabled:
        if lora_target_modules != 'attn.c_attn':
            raise ValueError(f"unsupported LoRA targets: {lora_target_modules}")
        model.enable_lora(lora_rank, lora_alpha, lora_dropout)
    # read off the created config params, so we can store them into checkpoint correctly
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = getattr(model.config, k)
# crop down the model block size if desired, using model surgery
if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args['block_size'] = block_size # so that the checkpoint will have the right value
if lora_enabled:
    if not (init_from == 'resume' or init_from.startswith('gpt2')):
        raise ValueError("LoRA training requires init_from='gpt2' or a LoRA checkpoint resume")
    model.freeze_non_lora_parameters()
    parameter_stats = model.get_parameter_stats()
    trainable_parameter_names = [
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    ]
    print(
        "LoRA parameters: "
        f"total={parameter_stats['total_params']:,}, "
        f"trainable={parameter_stats['trainable_params']:,} "
        f"({parameter_stats['trainable_percentage']:.4f}%)"
    )
    print("trainable parameter names:")
    for name in trainable_parameter_names:
        print(f"  {name}")
else:
    parameter_stats = model.get_parameter_stats()
num_params = sum(p.numel() for p in model.parameters())
model.to(device)
analysis_model = model # unwrapped model used for read-only W&B analysis

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

# optimizer
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
optimizer_parameter_ids = {
    id(parameter)
    for group in optimizer.param_groups
    for parameter in group['params']
}
trainable_parameter_ids = {
    id(parameter) for parameter in model.parameters() if parameter.requires_grad
}
if optimizer_parameter_ids != trainable_parameter_ids:
    raise RuntimeError("optimizer parameter groups do not exactly match trainable parameters")
if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None # free up memory
best_artifact_val_loss = best_val_loss # logging-only threshold; does not affect checkpoints

# compile the model
if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model) # requires PyTorch 2.0

# wrap model into DDP container
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

# logging
if wandb_log and master_process:
    import wandb

    resolved_tokenizer_type = tokenizer_type
    if resolved_tokenizer_type == 'auto':
        resolved_tokenizer_type = (
            'character'
            if meta is not None and 'stoi' in meta and 'itos' in meta
            else 'gpt2'
        )

    if resolved_tokenizer_type == 'character':
        if meta is None or 'stoi' not in meta or 'itos' not in meta:
            raise ValueError('character tokenizer requires stoi and itos in meta.pkl')
        stoi, itos = meta['stoi'], meta['itos']
        encode_text = lambda text: [stoi[char] for char in text]
        decode_tokens = lambda token_ids: ''.join(itos[token_id] for token_id in token_ids)
        display_token = lambda token_id: repr(itos.get(token_id, f'<{token_id}>'))
    elif resolved_tokenizer_type == 'gpt2':
        import tiktoken
        tokenizer = tiktoken.get_encoding('gpt2')
        encode_text = lambda text: tokenizer.encode(text, allowed_special={'<|endoftext|>'})

        def token_bytes(token_id):
            try:
                return tokenizer.decode_single_token_bytes(token_id)
            except KeyError:
                return f'<token:{token_id}>'.encode()

        decode_tokens = lambda token_ids: b''.join(
            token_bytes(token_id) for token_id in token_ids
        ).decode('utf-8', errors='replace')
        display_token = lambda token_id: repr(
            token_bytes(token_id).decode('utf-8', errors='replace')
        )
    else:
        raise ValueError(f"unsupported tokenizer_type: {resolved_tokenizer_type}")

    if wandb_tags:
        run_tags = [tag.strip() for tag in wandb_tags.split(',') if tag.strip()]
    elif init_from == 'scratch':
        run_tags = ['scratch']
    elif init_from == 'resume':
        run_tags = ['resume']
    elif init_from.startswith('gpt2'):
        run_tags = ['gpt2'] if eval_only else ['gpt2-finetune']
    else:
        run_tags = [init_from]

    wandb_config = config.copy()
    wandb_config.update({
        'model/architecture': 'GPT',
        'model/num_parameters': num_params,
        'model/trainable_parameters': parameter_stats['trainable_params'],
        'model/trainable_percentage': parameter_stats['trainable_percentage'],
        'model/config': model_args,
        'model/n_layer': model_args['n_layer'],
        'model/n_head': model_args['n_head'],
        'model/n_embd': model_args['n_embd'],
        'model/block_size': model_args['block_size'],
        'model/vocab_size': model_args['vocab_size'],
        'model/bias': model_args['bias'],
        'model/dropout': model_args['dropout'],
        'experiment/dataset': dataset,
        'experiment/tokenizer_type': resolved_tokenizer_type,
        'experiment/seed': seed,
        'experiment/batch_size': batch_size,
        'experiment/gradient_accumulation_steps': config['gradient_accumulation_steps'],
        'experiment/learning_rate': learning_rate,
        'experiment/max_iters': max_iters,
        'runtime/requested_device': config['device'],
        'runtime/device': device,
        'runtime/device_type': device_type,
        'runtime/ddp': ddp,
        'runtime/ddp_world_size': ddp_world_size,
        'runtime/gradient_accumulation_steps_per_rank': gradient_accumulation_steps,
        'lora/enabled': lora_enabled,
        'lora/rank': lora_rank if lora_enabled else None,
        'lora/alpha': lora_alpha if lora_enabled else None,
        'lora/dropout': lora_dropout if lora_enabled else None,
        'lora/target_modules': lora_target_modules if lora_enabled else None,
    })
    wandb.init(
        project=wandb_project,
        name=wandb_run_name,
        group=wandb_group or None,
        config=wandb_config,
        tags=run_tags,
    )

    latest_generation_tokens_per_sec = None

    def create_token_frequency_table():
        train_data = np.memmap(
            os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r'
        )
        counts = np.zeros(model_args['vocab_size'], dtype=np.int64)
        for start in range(0, len(train_data), wandb_token_frequency_chunk_size):
            chunk = train_data[start:start + wandb_token_frequency_chunk_size]
            counts += np.bincount(chunk, minlength=model_args['vocab_size'])
        nonzero_ids = np.flatnonzero(counts)
        top_k = max(0, min(wandb_table_top_k, len(nonzero_ids)))
        top_ids = (
            nonzero_ids[np.argsort(counts[nonzero_ids])[-top_k:][::-1]]
            if top_k > 0 else []
        )
        table = wandb.Table(
            columns=['token_id', 'token', 'count', 'relative_frequency']
        )
        total_tokens = len(train_data)
        for token_id in top_ids:
            token_id = int(token_id)
            table.add_data(
                token_id,
                display_token(token_id),
                int(counts[token_id]),
                float(counts[token_id] / total_tokens),
            )
        return table

    def create_embedding_norm_table():
        embedding_norms = torch.linalg.vector_norm(
            analysis_model.transformer.wte.weight.detach().float(), dim=1
        ).cpu()
        top_k = max(0, min(wandb_table_top_k, embedding_norms.numel()))
        top_norms, top_ids = torch.topk(embedding_norms, top_k)
        table = wandb.Table(columns=['token_id', 'token', 'l2_norm'])
        for token_id, norm in zip(top_ids.tolist(), top_norms.tolist()):
            table.add_data(token_id, display_token(token_id), norm)
        return table

    @torch.no_grad()
    def add_generation_samples(table, step, stage=None):
        global latest_generation_tokens_per_sec
        prompts = [prompt for prompt in wandb_prompts.split('|') if prompt]
        generation_speeds = []
        was_training = analysis_model.training
        analysis_model.eval()
        try:
            for prompt in prompts:
                try:
                    prompt_ids = encode_text(prompt)
                except (KeyError, ValueError) as exc:
                    generated_text = f'[prompt cannot be encoded: {exc}]'
                    if stage is None:
                        if wandb_log_generation_speed:
                            table.add_data(prompt, generated_text, step, None)
                        else:
                            table.add_data(prompt, generated_text, step)
                    elif wandb_log_generation_speed:
                        table.add_data(prompt, generated_text, stage, step, None)
                    else:
                        table.add_data(prompt, generated_text, stage, step)
                    continue
                generated = torch.tensor(
                    prompt_ids, dtype=torch.long, device=device
                )[None, ...]
                if wandb_log_generation_speed:
                    if device_type == 'cuda':
                        torch.cuda.synchronize(device)
                    generation_start = time.time()
                for _ in range(wandb_generate_tokens):
                    generated_context = generated[:, -analysis_model.config.block_size:]
                    with ctx:
                        logits, _ = analysis_model(generated_context)
                    next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                    generated = torch.cat((generated, next_token), dim=1)
                if wandb_log_generation_speed:
                    if device_type == 'cuda':
                        torch.cuda.synchronize(device)
                    generation_seconds = time.time() - generation_start
                    generation_tokens_per_sec = (
                        wandb_generate_tokens / generation_seconds
                    )
                    generation_speeds.append(generation_tokens_per_sec)
                generated_text = decode_tokens(generated[0].tolist())
                if stage is None:
                    if wandb_log_generation_speed:
                        table.add_data(
                            prompt, generated_text, step, generation_tokens_per_sec
                        )
                    else:
                        table.add_data(prompt, generated_text, step)
                elif wandb_log_generation_speed:
                    table.add_data(
                        prompt,
                        generated_text,
                        stage,
                        step,
                        generation_tokens_per_sec,
                    )
                else:
                    table.add_data(prompt, generated_text, stage, step)
        finally:
            analysis_model.train(was_training)
        if generation_speeds:
            latest_generation_tokens_per_sec = (
                sum(generation_speeds) / len(generation_speeds)
            )
        return table

    def create_generation_table(step):
        columns = ['prompt', 'generated_text', 'training_step']
        if wandb_log_generation_speed:
            columns.append('generation_tokens_per_sec')
        table = wandb.Table(columns=columns)
        return add_generation_samples(table, step)

    @torch.no_grad()
    def create_hf_alignment_table():
        from transformers import GPT2LMHeadModel

        prompt_ids = encode_text(wandb_alignment_prompt)
        nano_ids = torch.tensor(prompt_ids, dtype=torch.long, device=device)[None, ...]
        hf_ids = nano_ids.cpu()
        hf_model = GPT2LMHeadModel.from_pretrained(wandb_hf_model_name)
        hf_model.eval()

        was_training = analysis_model.training
        analysis_model.eval()
        try:
            nano_logits, _ = analysis_model(nano_ids)
            hf_logits = hf_model(input_ids=hf_ids).logits[:, -1:, :]
            logits_difference = (
                nano_logits.detach().float().cpu() - hf_logits.float()
            ).abs()

            nano_generated = nano_ids
            hf_generated = hf_ids
            for _ in range(wandb_alignment_tokens):
                nano_context = nano_generated[:, -analysis_model.config.block_size:]
                nano_step_logits, _ = analysis_model(nano_context)
                nano_next = torch.argmax(
                    nano_step_logits[:, -1, :], dim=-1, keepdim=True
                )
                nano_generated = torch.cat((nano_generated, nano_next), dim=1)

                hf_step_logits = hf_model(input_ids=hf_generated).logits
                hf_next = torch.argmax(
                    hf_step_logits[:, -1, :], dim=-1, keepdim=True
                )
                hf_generated = torch.cat((hf_generated, hf_next), dim=1)
        finally:
            analysis_model.train(was_training)

        nano_new_tokens = nano_generated[0, len(prompt_ids):].cpu().tolist()
        hf_new_tokens = hf_generated[0, len(prompt_ids):].tolist()
        table = wandb.Table(columns=['metric', 'value'])
        table.add_data('max_logits_diff', str(logits_difference.max().item()))
        table.add_data('mean_logits_diff', str(logits_difference.mean().item()))
        table.add_data('token_match', str(nano_new_tokens == hf_new_tokens))
        return table

    comparison_samples_table = None
    if wandb_compare_samples:
        comparison_columns = [
            'prompt', 'generated_text', 'training_stage', 'training_step'
        ]
        if wandb_log_generation_speed:
            comparison_columns.append('generation_tokens_per_sec')
        comparison_samples_table = wandb.Table(
            columns=comparison_columns
        )
        add_generation_samples(
            comparison_samples_table, iter_num, stage=wandb_before_stage
        )

    wandb.log({'analysis/token_frequency': create_token_frequency_table()})

# training loop
X, Y = get_batch('train') # fetch the very first batch
local_iter_num = 0 # number of iterations in the lifetime of this process
raw_model = model.module if ddp else model # unwrap DDP container if needed
running_mfu = -1.0
total_training_seconds = 0.0
latest_training_tokens_per_sec = None
max_evaluation_gpu_memory_mb = None
while True:

    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # evaluate the loss on train/val sets and write checkpoints
    if iter_num % eval_interval == 0 and master_process:
        measure_evaluation = wandb_log and wandb_eval_tracking
        if measure_evaluation and device_type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
        evaluation_start = time.time() if measure_evaluation else None
        losses = estimate_loss()
        if measure_evaluation and device_type == 'cuda':
            torch.cuda.synchronize(device)
        evaluation_seconds = (
            time.time() - evaluation_start if measure_evaluation else None
        )
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        if wandb_log:
            evaluation_metrics = {
                "iter": iter_num,
                "train/loss": losses['train'],
                "val/loss": losses['val'],
                "lr": lr,
                "mfu": running_mfu*100, # convert to percentage
                "analysis/embedding_norms": create_embedding_norm_table(),
            }
            if wandb_eval_tracking:
                evaluated_tokens = 2 * eval_iters * batch_size * block_size
                val_loss = float(losses['val'])
                evaluation_metrics.update({
                    "eval/perplexity": math.exp(val_loss),
                    "eval/num_parameters": num_params,
                    "eval/inference_seconds": evaluation_seconds,
                    "eval/inference_tokens_per_sec": (
                        evaluated_tokens / evaluation_seconds
                    ),
                })
                if device_type == 'cuda':
                    evaluation_gpu_memory_mb = (
                        torch.cuda.max_memory_allocated(device) / (1024 ** 2)
                    )
                    evaluation_metrics["eval/gpu_peak_memory_mb"] = evaluation_gpu_memory_mb
                    max_evaluation_gpu_memory_mb = max(
                        max_evaluation_gpu_memory_mb or 0.0,
                        evaluation_gpu_memory_mb,
                    )
            if wandb_hf_alignment:
                evaluation_metrics["evaluation/hf_alignment"] = (
                    create_hf_alignment_table()
                )
            if not wandb_compare_samples:
                evaluation_metrics["samples/generated_text"] = create_generation_table(iter_num)
                if latest_generation_tokens_per_sec is not None:
                    evaluation_metrics["generation/tokens_per_sec"] = (
                        latest_generation_tokens_per_sec
                    )
            wandb.log(evaluation_metrics)
            if wandb_log_eval_artifact:
                try:
                    val_loss = float(losses['val'])
                    evaluation_artifact = wandb.Artifact(
                        wandb_eval_artifact_name,
                        type='evaluation',
                        metadata={
                            'model_name': init_from,
                            'dataset': dataset,
                            'tokenizer_type': resolved_tokenizer_type,
                            'num_parameters': num_params,
                            'validation_loss': val_loss,
                        },
                    )
                    artifact_table = wandb.Table(columns=['metric', 'value'])
                    artifact_table.add_data('validation_loss', val_loss)
                    artifact_table.add_data('num_parameters', num_params)
                    evaluation_artifact.add(artifact_table, 'evaluation_metrics')
                    wandb.log_artifact(evaluation_artifact)
                except Exception as exc:
                    print(f"warning: failed to upload evaluation artifact: {exc}")
        is_better_val_loss = losses['val'] < best_val_loss
        is_best_artifact = losses['val'] < best_artifact_val_loss
        if is_better_val_loss or always_save_checkpoint:
            best_val_loss = losses['val']
            if iter_num > 0:
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': config,
                }
                print(f"saving checkpoint to {out_dir}")
                checkpoint_path = os.path.join(out_dir, 'ckpt.pt')
                torch.save(checkpoint, checkpoint_path)
                if wandb_log and is_best_artifact and wandb_log_checkpoint_artifact:
                    try:
                        artifact = wandb.Artifact(
                            wandb_artifact_name,
                            type='model',
                            metadata={
                                'iter': iter_num,
                                'val_loss': float(losses['val']),
                                'dataset': dataset,
                                'num_parameters': num_params,
                                'model_config': model_args,
                            },
                        )
                        artifact.add_file(checkpoint_path, name='ckpt.pt')
                        wandb.log_artifact(artifact, aliases=['best'])
                    except Exception as exc:
                        print(f"warning: failed to upload checkpoint artifact: {exc}")
                if lora_enabled and is_best_artifact:
                    adapter_metadata = {
                        'base_model': 'gpt2',
                        'target_modules': lora_target_modules,
                        'rank': lora_rank,
                        'alpha': lora_alpha,
                        'dropout': lora_dropout,
                        'training_iteration': iter_num,
                        'validation_loss': float(losses['val']),
                        'seed': seed,
                    }
                    adapter_path = os.path.join(out_dir, 'lora_adapter.pt')
                    torch.save({
                        'lora_state_dict': raw_model.lora_state_dict(),
                        'metadata': adapter_metadata,
                    }, adapter_path)
                    if wandb_log and lora_log_adapter_artifact:
                        try:
                            adapter_artifact = wandb.Artifact(
                                lora_adapter_artifact_name,
                                type='model-adapter',
                                metadata=adapter_metadata,
                            )
                            adapter_artifact.add_file(
                                adapter_path, name='lora_adapter.pt'
                            )
                            wandb.log_artifact(
                                adapter_artifact, aliases=['best', 'latest']
                            )
                        except Exception as exc:
                            print(f"warning: failed to upload LoRA adapter artifact: {exc}")
                if is_best_artifact:
                    best_artifact_val_loss = losses['val']
    if iter_num == 0 and eval_only:
        break

    # Start timing after evaluation and W&B analysis so dt measures the optimizer iteration.
    t0 = time.time()
    # forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16
    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            # in DDP training we only need to sync gradients at the last micro step.
            # the official way to do this is with model.no_sync() context manager, but
            # I really dislike that this bloats the code and forces us to repeat code
            # looking at the source of that context manager, it just toggles this variable
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        with ctx:
            logits, loss = model(X, Y)
            loss = loss / gradient_accumulation_steps # scale the loss to account for gradient accumulation
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        X, Y = get_batch('train')
        # backward pass, with gradient scaling if training in fp16
        scaler.scale(loss).backward()
    # clip the gradient
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    elif wandb_log and master_process:
        # GradScaler gradients must be unscaled before their norm is meaningful.
        scaler.unscale_(optimizer)
        grad_norms = [
            torch.linalg.vector_norm(p.grad.detach(), 2)
            for p in model.parameters()
            if p.grad is not None
        ]
        grad_norm = torch.linalg.vector_norm(torch.stack(grad_norms), 2)
    # step the optimizer and scaler if training in fp16
    scaler.step(optimizer)
    scaler.update()
    # flush the gradients as soon as we can, no need for this memory anymore
    optimizer.zero_grad(set_to_none=True)

    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    total_training_seconds += dt
    latest_training_tokens_per_sec = tokens_per_iter / dt
    should_print = iter_num % log_interval == 0 and master_process
    should_log_wandb = wandb_log and master_process
    if should_print or should_log_wandb:
        # get loss as float. note: this is a CPU-GPU sync point
        # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
        lossf = loss.item() * gradient_accumulation_steps
    if should_print:
        if local_iter_num >= 5: # let the training loop settle a bit
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
    if should_log_wandb:
        iteration_metrics = {
            'iter': iter_num,
            'train/iter_loss': lossf,
            'train/grad_norm': grad_norm.item(),
            'performance/iter_time_ms': dt * 1000,
            'performance/tokens_per_sec': latest_training_tokens_per_sec,
        }
        if device_type == 'cuda':
            iteration_metrics['system/gpu_memory_allocated_mb'] = (
                torch.cuda.memory_allocated(device) / (1024 ** 2)
            )
        wandb.log(iteration_metrics)
    iter_num += 1
    local_iter_num += 1

    # termination conditions
    if iter_num > max_iters:
        break

if wandb_log and master_process and wandb_compare_samples:
    add_generation_samples(
        comparison_samples_table, iter_num, stage=wandb_after_stage
    )
    final_metrics = {
        'samples/before_after': comparison_samples_table,
        'performance/total_training_seconds': total_training_seconds,
    }
    if latest_generation_tokens_per_sec is not None:
        final_metrics['generation/tokens_per_sec'] = latest_generation_tokens_per_sec
    if wandb_log_model_comparison:
        prompt_index = comparison_samples_table.columns.index('prompt')
        text_index = comparison_samples_table.columns.index('generated_text')
        stage_index = comparison_samples_table.columns.index('training_stage')
        generated_by_stage = {
            (row[prompt_index], row[stage_index]): row[text_index]
            for row in comparison_samples_table.data
        }
        model_comparison_table = wandb.Table(columns=[
            'fixed_prompt',
            'pretrained_output',
            'finetuned_output',
            'observation',
        ])
        for prompt in [prompt for prompt in wandb_prompts.split('|') if prompt]:
            pretrained_output = generated_by_stage.get(
                (prompt, wandb_before_stage), ''
            )
            finetuned_output = generated_by_stage.get(
                (prompt, wandb_after_stage), ''
            )
            observation = wandb_comparison_observation or (
                'Outputs are identical.'
                if pretrained_output == finetuned_output
                else 'Output changed after fine-tuning.'
            )
            model_comparison_table.add_data(
                prompt,
                pretrained_output,
                finetuned_output,
                observation,
            )
        final_metrics['samples/model_comparison'] = model_comparison_table
    if lora_enabled:
        comparison_table = wandb.Table(columns=[
            'method', 'total_parameters', 'trainable_parameters',
            'trainable_percentage', 'validation_loss', 'perplexity',
            'training_seconds', 'peak_gpu_memory_mb',
            'training_tokens_per_sec',
        ])
        comparison_table.add_data(
            'Full fine-tuning (historical Task 4)',
            124439808, 124439808, 100.0, 3.24544, 25.6730,
            None, 4073.75, 64226.63,
        )
        comparison_table.add_data(
            'LoRA',
            parameter_stats['total_params'],
            parameter_stats['trainable_params'],
            parameter_stats['trainable_percentage'],
            float(best_artifact_val_loss), math.exp(float(best_artifact_val_loss)),
            total_training_seconds, max_evaluation_gpu_memory_mb,
            latest_training_tokens_per_sec,
        )
        final_metrics['comparison/full_finetune_vs_lora'] = comparison_table
        final_metrics.update({
            'lora/total_parameters': parameter_stats['total_params'],
            'lora/trainable_parameters': parameter_stats['trainable_params'],
            'lora/trainable_percentage': parameter_stats['trainable_percentage'],
            'lora/best_validation_loss': float(best_artifact_val_loss),
            'lora/best_perplexity': math.exp(float(best_artifact_val_loss)),
        })
    wandb.log(final_metrics)
elif wandb_log and master_process:
    wandb.log({'performance/total_training_seconds': total_training_seconds})

if ddp:
    destroy_process_group()
