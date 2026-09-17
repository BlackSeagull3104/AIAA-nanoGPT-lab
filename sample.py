"""
Sample from a trained model
"""
import os
import pickle
from contextlib import nullcontext
import torch
import tiktoken
from model import GPTConfig, GPT

# -----------------------------------------------------------------------------
# Task 7: inference with a randomly initialized model
init_from = 'scratch' # 'scratch', 'resume', or a GPT-2 variant
out_dir = 'out-shakespeare-char'

# Fixed generation settings for controlled comparison
start = "First Citizen:"
num_samples = 5
max_new_tokens = 200
temperature = 0.8
top_k = 20
seed = 1337
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
compile = False # use PyTorch 2.0 to compile the model to be faster
exec(open('configurator.py').read()) # overrides from command line or config file
# -----------------------------------------------------------------------------

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# model
if init_from == 'scratch':
    # Construct the same small Shakespeare GPT architecture used in training,
    # but do NOT load any trained weights.
    gptconf = GPTConfig(
        block_size=128,
        vocab_size=65,
        n_layer=4,
        n_head=4,
        n_embd=128,
        dropout=0.0,
        bias=True,
    )
    model = GPT(gptconf)
    print("Initialized a randomly initialized GPT model from scratch.")

elif init_from == 'resume':
    # init from a model saved in a specific directory
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)

elif init_from.startswith('gpt2'):
    # init from a pretrained GPT-2 model
    model = GPT.from_pretrained(init_from, dict(dropout=0.0))

else:
    raise ValueError(f"Unknown init_from: {init_from}")

model.eval()
model.to(device)
if compile:
    model = torch.compile(model) # requires PyTorch 2.0 (optional)

# look for the meta pickle in case it is available in the dataset folder
load_meta = False

if init_from == 'scratch':
    # The scratch model uses the same 65-character Shakespeare vocabulary.
    meta_path = os.path.join('data', 'shakespeare_char', 'meta.pkl')
    load_meta = os.path.exists(meta_path)

    if not load_meta:
        raise FileNotFoundError(
            f"Character tokenizer metadata not found: {meta_path}"
        )

elif init_from == 'resume' and 'config' in checkpoint and 'dataset' in checkpoint['config']:
    meta_path = os.path.join('data', checkpoint['config']['dataset'], 'meta.pkl')
    load_meta = os.path.exists(meta_path)
if load_meta:
    print(f"Loading meta from {meta_path}...")
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    # TODO want to make this more general to arbitrary encoder/decoder schemes
    stoi, itos = meta['stoi'], meta['itos']
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])
else:
    # ok let's assume gpt-2 encodings by default
    print("No meta.pkl found, assuming GPT-2 encodings...")
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode = lambda l: enc.decode(l)

# encode the beginning of the prompt
if start.startswith('FILE:'):
    with open(start[5:], 'r', encoding='utf-8') as f:
        start = f.read()
start_ids = encode(start)
x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])

# run generation
import json
import time

report_dir = "reports"
sample_path = os.path.join(
    report_dir,
    "samples_scratch_before.jsonl"
)
stats_path = os.path.join(
    report_dir,
    "samples_scratch_before_stats.json"
)

os.makedirs(report_dir, exist_ok=True)

# Reset CUDA peak-memory statistics before generation.
if device_type == "cuda":
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

start_time = time.perf_counter()

total_generated_tokens = 0
records = []

with torch.no_grad():
    with ctx:
        for k in range(num_samples):

            # Generate one sample.
            y = model.generate(
                x,
                max_new_tokens,
                temperature=temperature,
                top_k=top_k
            )

            # y contains both the prompt and newly generated tokens.
            generated_token_count = y.size(1) - x.size(1)
            total_generated_tokens += generated_token_count

            generated_text = decode(
                y[0].tolist()
            )

            record = {
                "sample_id": k + 1,
                "seed": seed,
                "prompt": start,
                "temperature": temperature,
                "top_k": top_k,
                "max_new_tokens": max_new_tokens,
                "generated_tokens": generated_token_count,
                "generated_text": generated_text,
            }

            records.append(record)

            print(f"Sample {k + 1}")
            print(generated_text)
            print("---------------")

if device_type == "cuda":
    torch.cuda.synchronize()

elapsed_time = time.perf_counter() - start_time

generation_speed = (
    total_generated_tokens / elapsed_time
    if elapsed_time > 0
    else 0.0
)

if device_type == "cuda":
    peak_vram_bytes = torch.cuda.max_memory_allocated()
    peak_vram_mb = peak_vram_bytes / (1024 ** 2)
else:
    peak_vram_bytes = None
    peak_vram_mb = None

# Save one JSON object per line.
with open(
    sample_path,
    "w",
    encoding="utf-8"
) as f:
    for record in records:
        f.write(
            json.dumps(
                record,
                ensure_ascii=False
            )
            + "\n"
        )

runtime_stats = {
    "device": device,
    "seed": seed,
    "prompt": start,
    "num_samples": num_samples,
    "max_new_tokens": max_new_tokens,
    "temperature": temperature,
    "top_k": top_k,
    "total_generated_tokens": total_generated_tokens,
    "elapsed_seconds": elapsed_time,
    "generation_speed_tokens_per_second": generation_speed,
    "peak_vram_bytes": peak_vram_bytes,
    "peak_vram_mb": peak_vram_mb,
}

with open(
    stats_path,
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        runtime_stats,
        f,
        indent=2,
        ensure_ascii=False
    )

print()
print("=" * 60)
print("Task 7 Runtime Statistics")
print("=" * 60)

print(
    f"Generated tokens: {total_generated_tokens}"
)

print(
    f"Elapsed time: {elapsed_time:.3f} s"
)

print(
    f"Generation speed: {generation_speed:.2f} tokens/s"
)

if peak_vram_mb is not None:
    print(
        f"Peak VRAM: {peak_vram_mb:.2f} MB"
    )
else:
    print(
        "Peak VRAM: N/A (CPU execution)"
    )

print(f"Samples saved to: {sample_path}")
print(f"Statistics saved to: {stats_path}")
