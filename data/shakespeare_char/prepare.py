import os
import pickle
from collections import Counter

import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 1. Load tiny Shakespeare dataset
# ============================================================

input_file_path = os.path.join(
    os.path.dirname(__file__),
    "input.txt"
)

if not os.path.exists(input_file_path):
    data_url = (
        "https://raw.githubusercontent.com/"
        "karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    )

    print("input.txt not found. Downloading tiny Shakespeare...")

    response = requests.get(data_url)
    response.raise_for_status()

    with open(input_file_path, "w", encoding="utf-8") as f:
        f.write(response.text)

    print("Download complete.")

with open(input_file_path, "r", encoding="utf-8") as f:
    data = f.read()


# ============================================================
# 2. Basic dataset statistics
# ============================================================

num_characters = len(data)

chars = sorted(list(set(data)))
vocab_size = len(chars)

print("\n=== Dataset Statistics ===")
print(f"Number of characters: {num_characters:,}")
print(f"Vocabulary size: {vocab_size}")

print("\nUnique characters:")
print("".join(chars))


# ============================================================
# 3. Character tokenizer
# ============================================================

stoi = {
    ch: i
    for i, ch in enumerate(chars)
}

itos = {
    i: ch
    for i, ch in enumerate(chars)
}


def encode(text):
    return [stoi[ch] for ch in text]


def decode(ids):
    return "".join(itos[i] for i in ids)


# ============================================================
# 4. Train / validation split
# ============================================================

split_index = int(len(data) * 0.9)

train_data = data[:split_index]
val_data = data[split_index:]

train_ids = encode(train_data)
val_ids = encode(val_data)

train_token_count = len(train_ids)
val_token_count = len(val_ids)

total_token_count = train_token_count + val_token_count

train_ratio = train_token_count / total_token_count
val_ratio = val_token_count / total_token_count

print("\n=== Train / Validation Split ===")
print(f"Train token count: {train_token_count:,}")
print(f"Validation token count: {val_token_count:,}")
print(
    f"Split ratio: train {train_ratio:.2%}, "
    f"validation {val_ratio:.2%}"
)


# ============================================================
# 5. Save train.bin and val.bin
# ============================================================

train_ids_np = np.array(train_ids, dtype=np.uint16)
val_ids_np = np.array(val_ids, dtype=np.uint16)

train_bin_path = os.path.join(
    os.path.dirname(__file__),
    "train.bin"
)

val_bin_path = os.path.join(
    os.path.dirname(__file__),
    "val.bin"
)

train_ids_np.tofile(train_bin_path)
val_ids_np.tofile(val_bin_path)


# ============================================================
# 6. Save tokenizer metadata
# ============================================================

meta = {
    "vocab_size": vocab_size,
    "itos": itos,
    "stoi": stoi,
}

meta_path = os.path.join(
    os.path.dirname(__file__),
    "meta.pkl"
)

with open(meta_path, "wb") as f:
    pickle.dump(meta, f)


# ============================================================
# 7. Token frequency statistics
# ============================================================

all_ids = encode(data)

token_counts = Counter(all_ids)

token_stats = []

for token_id, count in token_counts.items():
    character = itos[token_id]
    relative_frequency = count / len(all_ids)

    token_stats.append(
        {
            "token_id": token_id,
            "character": character,
            "count": count,
            "relative_frequency": relative_frequency,
        }
    )

token_stats = sorted(
    token_stats,
    key=lambda row: row["count"],
    reverse=True,
)


# ============================================================
# 8. Print Top 30 tokens
# ============================================================

top30 = token_stats[:30]

print("\n=== Top 30 Character Tokens ===")

print(
    f"{'Token ID':<10}"
    f"{'Character':<15}"
    f"{'Count':<12}"
    f"{'Relative Frequency'}"
)

for row in top30:
    print(
        f"{row['token_id']:<10}"
        f"{repr(row['character']):<15}"
        f"{row['count']:<12}"
        f"{row['relative_frequency']:.6f}"
    )


# ============================================================
# 9. Create reports directory
# ============================================================

reports_dir = "reports"

os.makedirs(
    reports_dir,
    exist_ok=True
)


# ============================================================
# 10. Save complete token statistics CSV
# ============================================================

df = pd.DataFrame(token_stats)

csv_path = os.path.join(
    reports_dir,
    "token_frequency_char.csv"
)

df.to_csv(
    csv_path,
    index=False,
    encoding="utf-8"
)

print("\nSaved CSV:", csv_path)


# ============================================================
# 11. Plot Top 30 token frequencies
# ============================================================

labels = [
    repr(row["character"])
    for row in top30
]

counts = [
    row["count"]
    for row in top30
]

plt.figure(figsize=(12, 6))

plt.bar(labels, counts)

plt.xlabel("Character Token")
plt.ylabel("Count")
plt.title("Top 30 Character Token Frequencies")

plt.xticks(rotation=45)
plt.tight_layout()

png_path = os.path.join(
    reports_dir,
    "token_frequency_char.png"
)

plt.savefig(
    png_path,
    dpi=150
)

plt.close()

print("Saved bar chart:", png_path)


# ============================================================
# 12. Encode / decode round-trip test
# ============================================================

test_text = "First Citizen:"

encoded = encode(test_text)
decoded = decode(encoded)

round_trip_success = decoded == test_text

print("\n=== Encode / Decode Test ===")
print("Original text:", repr(test_text))
print("Encoded IDs:", encoded)
print("Decoded text:", repr(decoded))
print("Round-trip successful:", round_trip_success)


# ============================================================
# 13. Final summary
# ============================================================

print("\n=== Task 3 Complete ===")
print("Generated files:")
print("-", csv_path)
print("-", png_path)
print("-", train_bin_path)
print("-", val_bin_path)
print("-", meta_path)
