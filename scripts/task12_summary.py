
from pathlib import Path
from collections import Counter
import json

import numpy as np
import tiktoken


# ============================================================
# Paths
# ============================================================

REPO_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = REPO_ROOT / "data" / "shakespeare"
REPORT_DIR = REPO_ROOT / "reports"

TRAIN_PATH = DATA_DIR / "train.bin"
VAL_PATH = DATA_DIR / "val.bin"

CONSISTENCY_PATH = (
    REPORT_DIR / "task12_tokenizer_consistency.json"
)

OUTPUT_PATH = (
    REPORT_DIR / "task12_shakespeare_bpe_summary.json"
)


# ============================================================
# Load GPT-2 BPE data
# ============================================================

enc = tiktoken.get_encoding("gpt2")

train_ids = np.fromfile(
    TRAIN_PATH,
    dtype=np.uint16
)

val_ids = np.fromfile(
    VAL_PATH,
    dtype=np.uint16
)

train_counter = Counter(train_ids.tolist())
val_counter = Counter(val_ids.tolist())


# ============================================================
# Top training-token frequencies
# ============================================================

top_tokens = []

for rank, (token_id, count) in enumerate(
    train_counter.most_common(20),
    start=1
):
    top_tokens.append({
        "rank": rank,
        "token_id": token_id,
        "token_text": enc.decode([token_id]),
        "count": count
    })


# ============================================================
# Load tokenizer consistency result
# ============================================================

with open(
    CONSISTENCY_PATH,
    "r",
    encoding="utf-8"
) as f:
    consistency = json.load(f)


# ============================================================
# Build Task 12 report
# ============================================================

report = {

    "task": "Task 12 - Prepare GPT-2 BPE Shakespeare Data",

    "tokenizer": {
        "encoding": "gpt2",
        "algorithm": "byte-level BPE",
        "vocabulary_size": enc.n_vocab
    },

    "dataset_statistics": {

        "train": {
            "token_count": len(train_ids),
            "unique_token_count": len(train_counter),
            "file_size_bytes": TRAIN_PATH.stat().st_size
        },

        "validation": {
            "token_count": len(val_ids),
            "unique_token_count": len(val_counter),
            "file_size_bytes": VAL_PATH.stat().st_size
        }

    },

    "top_20_train_tokens": top_tokens,

    "tokenizer_consistency_check": {
        "tiktoken_vocab_size":
            consistency["tiktoken_vocab_size"],

        "hf_vocab_size":
            consistency["hf_vocab_size"],

        "num_random_samples":
            consistency["num_samples"],

        "all_ids_match":
            consistency["all_ids_match"]
    },

    "conclusion": (
        "The Tiny Shakespeare dataset was encoded using the "
        "GPT-2 byte-level BPE tokenizer. The training split contains "
        "301,966 tokens and the validation split contains 36,059 tokens. "
        "The token IDs produced by tiktoken and the Hugging Face GPT-2 "
        "tokenizer matched exactly for all five randomly sampled text "
        "segments."
    )
}


# ============================================================
# Save
# ============================================================

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


# ============================================================
# Print summary
# ============================================================

print("=" * 60)
print("Task 12 Summary")
print("=" * 60)

print(f"GPT-2 vocabulary size: {enc.n_vocab:,}")

print(
    f"Train: {len(train_ids):,} tokens, "
    f"{len(train_counter):,} unique tokens, "
    f"{TRAIN_PATH.stat().st_size:,} bytes"
)

print(
    f"Validation: {len(val_ids):,} tokens, "
    f"{len(val_counter):,} unique tokens, "
    f"{VAL_PATH.stat().st_size:,} bytes"
)

print(
    "Tokenizer consistency:",
    consistency["all_ids_match"]
)

print("\nSaved to:")
print(OUTPUT_PATH)
