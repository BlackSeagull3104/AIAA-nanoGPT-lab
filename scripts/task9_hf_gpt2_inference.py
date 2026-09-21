
import json
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


# ============================================================
# Configuration
# ============================================================

MODEL_NAME = "openai-community/gpt2"

SAMPLE_PATH = Path("data/task4_samples.txt")
OUTPUT_PATH = Path("reports/hf_gpt2_predictions.jsonl")

NUM_PROMPTS = 5
MAX_NEW_TOKENS = 50

# Fixed generation settings for reproducibility
SEED = 1337
TEMPERATURE = 0.8
TOP_K = 50


# ============================================================
# Device
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Device: {device}")


# ============================================================
# Load tokenizer and pretrained GPT-2
# ============================================================

print(f"Loading {MODEL_NAME}...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
model = model.to(device)
model.eval()

print("Model loaded.")


# ============================================================
# Load 5 fixed English prompts from Task 4
# ============================================================

text = SAMPLE_PATH.read_text(encoding="utf-8")

samples = [
    block.strip()
    for block in text.split("\n\n")
    if block.strip()
]

if len(samples) < NUM_PROMPTS:
    raise ValueError(
        f"Expected at least {NUM_PROMPTS} samples, "
        f"but found {len(samples)}."
    )

prompts = samples[:NUM_PROMPTS]

print(f"Loaded {len(prompts)} fixed prompts.")


# ============================================================
# Prepare output
# ============================================================

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

results = []


# ============================================================
# Inference
# ============================================================

for i, prompt in enumerate(prompts, start=1):

    print(f"\n{'=' * 60}")
    print(f"Prompt {i}/{NUM_PROMPTS}")
    print(f"{'=' * 60}")

    # Text -> prompt token IDs
    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    ).to(device)

    prompt_ids = inputs["input_ids"][0]
    prompt_length = prompt_ids.shape[0]

    # Fixed seed for reproducible sampling
    torch.manual_seed(SEED + i - 1)

    if device.type == "cuda":
        torch.cuda.manual_seed_all(SEED + i - 1)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    # --------------------------------------------------------
    # Generate
    # --------------------------------------------------------

    start_time = time.perf_counter()

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=TEMPERATURE,
            top_k=TOP_K,
            pad_token_id=tokenizer.eos_token_id,
        )

    if device.type == "cuda":
        torch.cuda.synchronize()

    runtime_seconds = time.perf_counter() - start_time

    # --------------------------------------------------------
    # Separate newly generated tokens from prompt tokens
    # --------------------------------------------------------

    generated_ids = outputs[0][prompt_length:]

    generated_text = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True
    )

    # --------------------------------------------------------
    # Peak VRAM
    # --------------------------------------------------------

    if device.type == "cuda":
        peak_vram_bytes = torch.cuda.max_memory_allocated()
        peak_vram_mb = peak_vram_bytes / (1024 ** 2)
    else:
        peak_vram_bytes = None
        peak_vram_mb = None

    # --------------------------------------------------------
    # Save result
    # --------------------------------------------------------

    result = {
        "prompt_id": i,
        "model": MODEL_NAME,
        "seed": SEED + i - 1,
        "prompt": prompt,
        "prompt_token_ids": prompt_ids.tolist(),
        "generated_token_ids": generated_ids.tolist(),
        "generated_text": generated_text,
        "max_new_tokens": MAX_NEW_TOKENS,
        "temperature": TEMPERATURE,
        "top_k": TOP_K,
        "runtime_seconds": runtime_seconds,
        "peak_vram_bytes": peak_vram_bytes,
        "peak_vram_mb": peak_vram_mb,
    }

    results.append(result)

    print(f"Prompt tokens:    {len(prompt_ids)}")
    print(f"Generated tokens: {len(generated_ids)}")
    print(f"Runtime:          {runtime_seconds:.4f} s")

    if peak_vram_mb is not None:
        print(f"Peak VRAM:        {peak_vram_mb:.2f} MB")

    print("\nGenerated text ONLY:")
    print(generated_text)


# ============================================================
# Write JSONL
# ============================================================

with OUTPUT_PATH.open("w", encoding="utf-8") as f:
    for result in results:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")


print(f"\n{'=' * 60}")
print("Task 9 inference complete.")
print(f"Saved {len(results)} predictions to:")
print(OUTPUT_PATH)
