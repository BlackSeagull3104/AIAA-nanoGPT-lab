
from pathlib import Path
import json
import random

import tiktoken
from transformers import AutoTokenizer


# ============================================================
# Configuration
# ============================================================

REPO_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = REPO_ROOT / "data" / "shakespeare" / "input.txt"
OUTPUT_PATH = REPO_ROOT / "reports" / "task12_tokenizer_consistency.json"

NUM_SAMPLES = 5
SAMPLE_LENGTH = 200
SEED = 1337


# ============================================================
# Load tokenizers
# ============================================================

# tiktoken implementation of GPT-2 encoding
tiktoken_enc = tiktoken.get_encoding("gpt2")

# Hugging Face implementation of GPT-2 tokenizer
hf_tokenizer = AutoTokenizer.from_pretrained(
    "openai-community/gpt2"
)

print("tiktoken vocabulary size:", tiktoken_enc.n_vocab)
print("HF vocabulary size:", hf_tokenizer.vocab_size)


# ============================================================
# Load Shakespeare text
# ============================================================

text = INPUT_PATH.read_text(encoding="utf-8")

random.seed(SEED)

results = []


# ============================================================
# Random sample consistency check
# ============================================================

for i in range(NUM_SAMPLES):

    start = random.randint(
        0,
        len(text) - SAMPLE_LENGTH
    )

    sample = text[
        start:start + SAMPLE_LENGTH
    ]

    # GPT-2 token IDs from tiktoken
    tiktoken_ids = tiktoken_enc.encode_ordinary(sample)

    # GPT-2 token IDs from Hugging Face
    hf_ids = hf_tokenizer.encode(
        sample,
        add_special_tokens=False
    )

    ids_match = tiktoken_ids == hf_ids

    results.append({
        "sample_id": i + 1,
        "start_position": start,
        "text": sample,
        "tiktoken_ids": tiktoken_ids,
        "hf_ids": hf_ids,
        "token_count": len(tiktoken_ids),
        "ids_match": ids_match
    })

    print("\n" + "=" * 60)
    print(f"Sample {i + 1}")
    print("=" * 60)

    print("Text:")
    print(repr(sample))

    print("\ntiktoken IDs:")
    print(tiktoken_ids)

    print("\nHugging Face IDs:")
    print(hf_ids)

    print("\nToken count:", len(tiktoken_ids))
    print("IDs match:", ids_match)


# ============================================================
# Overall result
# ============================================================

all_match = all(
    result["ids_match"]
    for result in results
)

report = {
    "tokenizer": "GPT-2 BPE",
    "num_samples": NUM_SAMPLES,
    "sample_length_characters": SAMPLE_LENGTH,
    "random_seed": SEED,
    "tiktoken_vocab_size": tiktoken_enc.n_vocab,
    "hf_vocab_size": hf_tokenizer.vocab_size,
    "all_ids_match": all_match,
    "samples": results
}


OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_PATH.write_text(
    json.dumps(
        report,
        ensure_ascii=False,
        indent=2
    ),
    encoding="utf-8"
)


print("\n" + "=" * 60)
print("Overall Consistency Check")
print("=" * 60)

print("All token IDs match:", all_match)
print("Saved to:", OUTPUT_PATH)
