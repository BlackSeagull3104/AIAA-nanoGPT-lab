import os
import sys
import json
import math
import time

import torch
import tiktoken

# Allow importing model.py from repository root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model import GPT, GPTConfig


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

DATA_DIR = "data/shakespeare"
CHECKPOINT = "out-shakespeare-gpt2/ckpt.pt"
REPORT_PATH = "reports/gpt2_before_after_finetune.jsonl"

# Fixed prompts for fair before/after comparison
PROMPTS = [
    "ROMEO:",
    "JULIET:",
    "To be, or not to be",
]

# Fixed generation settings
MAX_NEW_TOKENS = 100
TEMPERATURE = 0.8
TOP_K = 200
SEED = 42

# Validation settings
BLOCK_SIZE = 1024
BATCH_SIZE = 4
EVAL_ITERS = 20


def load_pretrained():
    print("Loading pretrained GPT-2 124M...")
    model = GPT.from_pretrained("gpt2")
    model.eval()
    model.to(DEVICE)
    return model


def load_finetuned():
    print(f"Loading fine-tuned checkpoint: {CHECKPOINT}")

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=DEVICE,
        weights_only=False,
    )

    config = GPTConfig(**checkpoint["model_args"])
    model = GPT(config)

    state_dict = checkpoint["model"]

    # nanoGPT compiled checkpoints may contain this prefix
    unwanted_prefix = "_orig_mod."
    for key in list(state_dict.keys()):
        if key.startswith(unwanted_prefix):
            state_dict[key[len(unwanted_prefix):]] = state_dict.pop(key)

    model.load_state_dict(state_dict)
    model.eval()
    model.to(DEVICE)

    return model, checkpoint


def get_batch(data):
    ix = torch.randint(
        len(data) - BLOCK_SIZE,
        (BATCH_SIZE,)
    )

    x = torch.stack([
        torch.from_numpy(
            data[i:i + BLOCK_SIZE].astype("int64")
        )
        for i in ix
    ])

    y = torch.stack([
        torch.from_numpy(
            data[i + 1:i + 1 + BLOCK_SIZE].astype("int64")
        )
        for i in ix
    ])

    return x.to(DEVICE), y.to(DEVICE)


@torch.no_grad()
def estimate_validation_loss(model):
    import numpy as np

    torch.manual_seed(SEED)
    if DEVICE == "cuda":
        torch.cuda.manual_seed(SEED)

    val_path = os.path.join(DATA_DIR, "val.bin")
    data = np.memmap(val_path, dtype=np.uint16, mode="r")

    losses = []

    for _ in range(EVAL_ITERS):
        x, y = get_batch(data)

        with torch.autocast(
            device_type="cuda",
            dtype=DTYPE,
            enabled=(DEVICE == "cuda"),
        ):
            _, loss = model(x, y)

        losses.append(loss.item())

    return sum(losses) / len(losses)


@torch.no_grad()
def generate(model, prompt, tokenizer):
    torch.manual_seed(SEED)
    if DEVICE == "cuda":
        torch.cuda.manual_seed(SEED)

    ids = tokenizer.encode(prompt)
    x = torch.tensor(ids, dtype=torch.long, device=DEVICE)[None, ...]

    if DEVICE == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()

    y = model.generate(
        x,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        top_k=TOP_K,
    )

    if DEVICE == "cuda":
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - start

    generated_tokens = y.size(1) - x.size(1)
    tokens_per_second = generated_tokens / elapsed

    text = tokenizer.decode(y[0].tolist())

    return {
        "text": text,
        "elapsed_seconds": elapsed,
        "generated_tokens": generated_tokens,
        "tokens_per_second": tokens_per_second,
    }


def evaluate_model(name, model, tokenizer):
    print(f"\n===== {name} =====")

    val_loss = estimate_validation_loss(model)
    perplexity = math.exp(val_loss)

    print(f"validation loss: {val_loss:.4f}")
    print(f"perplexity:      {perplexity:.4f}")

    generations = []

    for prompt in PROMPTS:
        result = generate(model, prompt, tokenizer)

        generations.append({
            "prompt": prompt,
            **result,
        })

        print(f"\nPrompt: {prompt}")
        print(result["text"])
        print(
            f"Speed: {result['tokens_per_second']:.2f} tokens/s"
        )

    return {
        "model": name,
        "validation_loss": val_loss,
        "perplexity": perplexity,
        "generation_parameters": {
            "max_new_tokens": MAX_NEW_TOKENS,
            "temperature": TEMPERATURE,
            "top_k": TOP_K,
            "seed": SEED,
        },
        "generations": generations,
    }


def main():
    os.makedirs("reports", exist_ok=True)

    print(f"Device: {DEVICE}")

    # Same GPT-2 byte-level BPE tokenizer for both models
    tokenizer = tiktoken.get_encoding("gpt2")

    pretrained = load_pretrained()

    before = evaluate_model(
        "gpt2_before_finetune",
        pretrained,
        tokenizer,
    )

    del pretrained

    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    finetuned, checkpoint = load_finetuned()

    after = evaluate_model(
        "gpt2_after_finetune",
        finetuned,
        tokenizer,
    )

    # Include best checkpoint information in the report
    after["checkpoint"] = {
        "path": CHECKPOINT,
        "iter_num": checkpoint.get("iter_num"),
        "best_val_loss": float(checkpoint.get("best_val_loss")),
    }

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(json.dumps(before, ensure_ascii=False) + "\n")
        f.write(json.dumps(after, ensure_ascii=False) + "\n")

    print("\n===== Comparison =====")
    print(
        f"Validation loss: "
        f"{before['validation_loss']:.4f} -> "
        f"{after['validation_loss']:.4f}"
    )
    print(
        f"Perplexity: "
        f"{before['perplexity']:.4f} -> "
        f"{after['perplexity']:.4f}"
    )

    print(f"\nSaved report to {REPORT_PATH}")


if __name__ == "__main__":
    main()
