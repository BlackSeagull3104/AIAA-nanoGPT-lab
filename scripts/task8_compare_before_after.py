
import csv
import json
import re
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt


# ============================================================
# Paths
# ============================================================

BEFORE_PATH = Path("reports/samples_scratch_before.jsonl")
AFTER_PATH = Path("reports/samples_scratch_after.jsonl")

JSON_PATH = Path("reports/task8_before_after_comparison.json")
CSV_PATH = Path("reports/task8_character_distribution.csv")
FIGURE_PATH = Path("reports/task8_character_distribution.png")


# ============================================================
# Load generated samples
# ============================================================

def load_jsonl(path):
    samples = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))

    return samples


def get_text(sample):
    """Support several possible field names."""
    for key in ["text", "generated_text", "sample"]:
        if key in sample:
            return sample[key]

    raise KeyError(
        f"Cannot find generated text field. "
        f"Available keys: {list(sample.keys())}"
    )


# ============================================================
# Metrics
# ============================================================

def character_distribution(text):
    """
    Relative frequency of each character.
    """
    counts = Counter(text)
    total = len(text)

    if total == 0:
        return {}

    return {
        char: count / total
        for char, count in sorted(counts.items())
    }


def repetition_rate(text):
    """
    Adjacent-character repetition rate.

    Number of positions where the current character is identical
    to the previous character, divided by all adjacent pairs.
    """
    if len(text) < 2:
        return 0.0

    repeated = sum(
        text[i] == text[i - 1]
        for i in range(1, len(text))
    )

    return repeated / (len(text) - 1)


def average_word_length(text):
    """
    Average length of alphabetic word sequences.
    """
    words = re.findall(r"[A-Za-z]+", text)

    if not words:
        return 0.0

    return sum(len(word) for word in words) / len(words)


def readability_metrics(text):
    """
    Simple structural proxies for readability.
    These are descriptive metrics, not a standard readability score.
    """
    if not text:
        return {
            "alphabetic_ratio": 0.0,
            "space_ratio": 0.0,
            "newline_ratio": 0.0,
        }

    n = len(text)

    return {
        "alphabetic_ratio": sum(c.isalpha() for c in text) / n,
        "space_ratio": text.count(" ") / n,
        "newline_ratio": text.count("\n") / n,
    }


def analyze(samples):
    texts = [get_text(sample) for sample in samples]
    combined = "\n".join(texts)

    return {
        "num_samples": len(texts),
        "total_characters": len(combined),
        "character_distribution": character_distribution(combined),
        "repetition_rate": repetition_rate(combined),
        "average_word_length": average_word_length(combined),
        "readability": readability_metrics(combined),
    }


# ============================================================
# Character distribution output
# ============================================================

def display_char(char):
    if char == " ":
        return "<space>"
    if char == "\n":
        return "<newline>"
    if char == "\t":
        return "<tab>"
    return char


def save_character_csv(before_dist, after_dist):
    characters = sorted(
        set(before_dist.keys()) | set(after_dist.keys())
    )

    with CSV_PATH.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)

        writer.writerow([
            "character",
            "before_frequency",
            "after_frequency"
        ])

        for char in characters:
            writer.writerow([
                display_char(char),
                before_dist.get(char, 0),
                after_dist.get(char, 0)
            ])


def save_character_figure(before_dist, after_dist):
    characters = sorted(
        set(before_dist.keys()) | set(after_dist.keys())
    )

    before_values = [
        before_dist.get(char, 0)
        for char in characters
    ]

    after_values = [
        after_dist.get(char, 0)
        for char in characters
    ]

    x = list(range(len(characters)))
    width = 0.4

    plt.figure(figsize=(16, 6))

    plt.bar(
        [i - width / 2 for i in x],
        before_values,
        width=width,
        label="Before training"
    )

    plt.bar(
        [i + width / 2 for i in x],
        after_values,
        width=width,
        label="After training"
    )

    plt.xticks(
        x,
        [display_char(c) for c in characters],
        rotation=90
    )

    plt.xlabel("Character")
    plt.ylabel("Relative frequency")
    plt.title("Character Distribution Before vs After Training")
    plt.legend()

    plt.tight_layout()
    plt.savefig(FIGURE_PATH, dpi=200)
    plt.close()


# ============================================================
# Main analysis
# ============================================================

before_samples = load_jsonl(BEFORE_PATH)
after_samples = load_jsonl(AFTER_PATH)

before = analyze(before_samples)
after = analyze(after_samples)

comparison = {
    "metric_definitions": {
        "repetition_rate":
            "Proportion of adjacent character pairs that are identical.",
        "average_word_length":
            "Mean length of contiguous alphabetic sequences.",
        "readability":
            "Descriptive structural proxies rather than a standard readability score."
    },
    "before_training": before,
    "after_training": after
}

# Save JSON summary
with JSON_PATH.open("w", encoding="utf-8") as f:
    json.dump(
        comparison,
        f,
        indent=2,
        ensure_ascii=False
    )

# Save character distribution
save_character_csv(
    before["character_distribution"],
    after["character_distribution"]
)

save_character_figure(
    before["character_distribution"],
    after["character_distribution"]
)


# ============================================================
# Console summary
# ============================================================

print("=== Task 8: Before vs After Training ===")

print("\nRepetition rate")
print(f"Before: {before['repetition_rate']:.4f}")
print(f"After : {after['repetition_rate']:.4f}")

print("\nAverage word length")
print(f"Before: {before['average_word_length']:.4f}")
print(f"After : {after['average_word_length']:.4f}")

print("\nReadability proxies")

for metric in before["readability"]:
    print(
        f"{metric:20s} "
        f"Before: {before['readability'][metric]:.4f} | "
        f"After: {after['readability'][metric]:.4f}"
    )

print("\nCharacter distribution")
print(
    f"{'space':20s} "
    f"Before: {before['character_distribution'].get(' ', 0):.4f} | "
    f"After: {after['character_distribution'].get(' ', 0):.4f}"
)
print(
    f"{'newline':20s} "
    f"Before: {before['character_distribution'].get(chr(10), 0):.4f} | "
    f"After: {after['character_distribution'].get(chr(10), 0):.4f}"
)

print("\nSaved:")
print("-", JSON_PATH)
print("-", CSV_PATH)
print("-", FIGURE_PATH)
