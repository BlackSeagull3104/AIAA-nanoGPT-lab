import os
import sys
import json
import pickle
import torch

# ============================================================
# Paths
# ============================================================

ROOT = "/content/AIAA-nanoGPT-lab"
os.chdir(ROOT)
sys.path.insert(0, ROOT)

from model import GPTConfig, GPT

checkpoint_path = "out-shakespeare-char/ckpt.pt"
meta_path = "data/shakespeare_char/meta.pkl"
output_path = "reports/samples_scratch_after.jsonl"

# ============================================================
# Generation settings
# Must be identical to Task 7
# ============================================================

seed = 1337
prompt = "First Citizen:"
num_samples = 5
max_new_tokens = 200
temperature = 0.8
top_k = 20

device = "cuda" if torch.cuda.is_available() else "cpu"

torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)

# ============================================================
# 1. Load character tokenizer
# ============================================================

with open(meta_path, "rb") as f:
    meta = pickle.load(f)

stoi = meta["stoi"]
itos = meta["itos"]

def encode(text):
    return [stoi[c] for c in text]

def decode(ids):
    return "".join(itos[i] for i in ids)

print("Tokenizer loaded.")
print("Vocabulary size:", len(stoi))

# ============================================================
# 2. Load Task 2 best checkpoint
# ============================================================

checkpoint = torch.load(
    checkpoint_path,
    map_location=device
)

print("\nCheckpoint loaded.")
print("Iteration:", checkpoint["iter_num"])
print("Best validation loss:", checkpoint["best_val_loss"])
print("Model args:", checkpoint["model_args"])

# ============================================================
# 3. Reconstruct the same GPT architecture
# ============================================================

gptconf = GPTConfig(**checkpoint["model_args"])
model = GPT(gptconf)

# ============================================================
# 4. Load learned parameters
# ============================================================

state_dict = checkpoint["model"]

# torch.compile may add "_orig_mod." to parameter names.
# Remove it before loading into the normal GPT model.
prefix = "_orig_mod."

clean_state_dict = {}

for key, value in state_dict.items():
    if key.startswith(prefix):
        key = key[len(prefix):]

    clean_state_dict[key] = value

model.load_state_dict(clean_state_dict)

model.to(device)
model.eval()

print("\nTrained model restored successfully.")
print("Device:", device)

# ============================================================
# 5. Encode prompt
# ============================================================

start_ids = encode(prompt)

x = torch.tensor(
    start_ids,
    dtype=torch.long,
    device=device
)[None, ...]

print("Prompt:", repr(prompt))
print("Prompt token IDs:", start_ids)

# ============================================================
# 6. Generate five samples
# ============================================================

os.makedirs("reports", exist_ok=True)

results = []

with torch.no_grad():
    for sample_id in range(1, num_samples + 1):

        y = model.generate(
            x,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k
        )

        generated_text = decode(y[0].tolist())

        result = {
            "sample_id": sample_id,
            "seed": seed,
            "prompt": prompt,
            "temperature": temperature,
            "top_k": top_k,
            "max_new_tokens": max_new_tokens,
            "text": generated_text
        }

        results.append(result)

        print(f"\n===== Sample {sample_id} =====")
        print(generated_text)

# ============================================================
# 7. Save JSONL
# ============================================================

with open(output_path, "w", encoding="utf-8") as f:
    for result in results:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")

print("\n===================================")
print("Task 8 inference complete.")
print("Saved:", output_path)
print("===================================")
