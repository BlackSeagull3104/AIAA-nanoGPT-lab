# Task 4: Compare character tokenizer and GPT-2 BPE

sample_path = "data/task4_samples.txt"

with open(sample_path, "r", encoding="utf-8") as f:
    text = f.read()

samples = text.strip().split("\n\n")

print("Number of samples:", len(samples))

for i, sample in enumerate(samples, start=1):
    print(f"\n--- Sample {i} ---")
    print(sample)
