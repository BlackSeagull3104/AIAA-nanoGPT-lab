
from pathlib import Path
from collections import Counter

import numpy as np
import tiktoken


# ============================================================
# Paths
# ============================================================

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data" / "shakespeare"

TRAIN_PATH = DATA_DIR / "train.bin"
VAL_PATH = DATA_DIR / "val.bin"

enc = tiktoken.get_encoding("gpt2")


# ============================================================
# Load token IDs
# ============================================================

train_ids = np.fromfile(TRAIN_PATH, dtype=np.uint16)
val_ids = np.fromfile(VAL_PATH, dtype=np.uint16)

print("========================================")
print("Dataset Statistics")
print("========================================")

print(f"Train token count: {len(train_ids):,}")
print(f"Validation token count: {len(val_ids):,}")

print(f"Train file size: {TRAIN_PATH.stat().st_size:,} bytes")
print(f"Validation file size: {VAL_PATH.stat().st_size:,} bytes")


# ============================================================
# Token frequency
# ============================================================

train_counter = Counter(train_ids.tolist())
val_counter = Counter(val_ids.tolist())

print(f"\nUnique tokens in train: {len(train_counter):,}")
print(f"Unique tokens in validation: {len(val_counter):,}")


# ============================================================
# Most frequent training tokens
# ============================================================

print("\n========================================")
print("Top 20 Training Tokens")
print("========================================")

for rank, (token_id, count) in enumerate(
    train_counter.most_common(20),
    start=1
):
    token_text = enc.decode([token_id])

    print(
        f"{rank:2d}. "
        f"ID={token_id:<6} "
        f"count={count:<7} "
        f"token={repr(token_text)}"
    )
