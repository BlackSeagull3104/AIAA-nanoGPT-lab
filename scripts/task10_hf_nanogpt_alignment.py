import json
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


# ============================================================
# Import nanoGPT model.py from repository root
# ============================================================

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from model import GPT


# ============================================================
# Configuration
# ============================================================

MODEL_NAME = "openai-community/gpt2"

SAMPLE_PATH = REPO_ROOT / "data/task4_samples.txt"
OUTPUT_PATH = REPO_ROOT / "reports/hf_nanogpt_alignment.json"

NUM_PROMPTS = 5
MAX_NEW_TOKENS = 50

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Repository root:", REPO_ROOT)
print("nanoGPT model.py:", REPO_ROOT / "model.py")
print("Device:", device)


# ============================================================
# Load tokenizer
# ============================================================

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


# ============================================================
# Load Hugging Face GPT-2
# ============================================================

print("\nLoading Hugging Face GPT-2...")

hf_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME
).to(device)

hf_model.eval()


# ============================================================
# Load the same GPT-2 weights into nanoGPT implementation
# ============================================================

print("\nLoading GPT-2 weights into nanoGPT...")

nanogpt_model = GPT.from_pretrained(
    "gpt2",
    dict(dropout=0.0)
).to(device)

nanogpt_model.eval()


# ============================================================
# Load same 5 prompts as Task 9
# ============================================================

text = SAMPLE_PATH.read_text(encoding="utf-8")

samples = [
    block.strip()
    for block in text.split("\n\n")
    if block.strip()
]

prompts = samples[:NUM_PROMPTS]

assert len(prompts) == NUM_PROMPTS


# ============================================================
# 1. Parameter count
# ============================================================

hf_param_count = sum(
    p.numel() for p in hf_model.parameters()
)

nanogpt_param_count = sum(
    p.numel() for p in nanogpt_model.parameters()
)

parameter_count_match = (
    hf_param_count == nanogpt_param_count
)

print("\n=== Parameter Count ===")
print("HF GPT-2:      ", f"{hf_param_count:,}")
print("nanoGPT GPT-2: ", f"{nanogpt_param_count:,}")
print("Match:         ", parameter_count_match)


# ============================================================
# 2. Weight shape comparison
# ============================================================

hf_sd = hf_model.state_dict()
nano_sd = nanogpt_model.state_dict()

# Ignore non-learned attention mask buffers.
hf_keys = [
    k for k in hf_sd.keys()
    if not k.endswith(".attn.bias")
    and not k.endswith(".attn.masked_bias")
]

# HF GPT-2 Conv1D stores these matrices in the opposite
# orientation from nanoGPT nn.Linear.
transposed_suffixes = (
    "attn.c_attn.weight",
    "attn.c_proj.weight",
    "mlp.c_fc.weight",
    "mlp.c_proj.weight",
)

shape_comparison = []
missing_in_nanogpt = []

for k in hf_keys:

    if k not in nano_sd:
        missing_in_nanogpt.append(k)
        continue

    hf_shape = tuple(hf_sd[k].shape)
    nano_shape = tuple(nano_sd[k].shape)

    needs_transpose = any(
        k.endswith(s)
        for s in transposed_suffixes
    )

    aligned_hf_shape = (
        tuple(reversed(hf_shape))
        if needs_transpose
        else hf_shape
    )

    shape_comparison.append({
        "name": k,
        "hf_shape": list(hf_shape),
        "nanogpt_shape": list(nano_shape),
        "transpose_required": needs_transpose,
        "aligned_shape_match":
            aligned_hf_shape == nano_shape,
    })


all_shapes_match = (
    len(missing_in_nanogpt) == 0
    and all(
        item["aligned_shape_match"]
        for item in shape_comparison
    )
)

print("\n=== Weight Shape Alignment ===")
print("HF tensors considered:", len(hf_keys))
print("Compared tensors:     ", len(shape_comparison))
print("Missing in nanoGPT:   ", len(missing_in_nanogpt))
print("All aligned shapes match:", all_shapes_match)


# ============================================================
# 3. Forward logits alignment
# ============================================================

# Use Prompt 1.
prompt = prompts[0]

inputs = tokenizer(
    prompt,
    return_tensors="pt"
).to(device)

input_ids = inputs["input_ids"]


with torch.no_grad():

    # HF returns logits for ALL positions:
    # (B, T, V)
    hf_logits = hf_model(
        input_ids=input_ids
    ).logits

    # nanoGPT inference returns only the LAST position:
    # (B, 1, V)
    nano_logits, _ = nanogpt_model(
        input_ids
    )


# IMPORTANT:
# Compare the SAME token position.
hf_last_logits = hf_logits[:, -1:, :]

assert (
    hf_last_logits.shape == nano_logits.shape
), (
    f"Shape mismatch: "
    f"HF last={hf_last_logits.shape}, "
    f"nanoGPT={nano_logits.shape}"
)


abs_error = (
    hf_last_logits.float()
    - nano_logits.float()
).abs()

max_abs_error = abs_error.max().item()
mean_abs_error = abs_error.mean().item()


print("\n=== Forward Logits Alignment ===")
print("HF full logits shape: ", tuple(hf_logits.shape))
print("HF last logits shape: ", tuple(hf_last_logits.shape))
print("nanoGPT logits shape: ", tuple(nano_logits.shape))
print("Max absolute error:   ", max_abs_error)
print("Mean absolute error:  ", mean_abs_error)


# ============================================================
# 4. Deterministic generation alignment
# ============================================================

# Task 10 asks for deterministic generation.
# Therefore greedy decoding is used instead of Task 9 sampling.
#
# max_new_tokens remains the same as Task 9.
# temperature/top_k are not applicable when sampling is disabled.

generation_results = []

print("\n=== Deterministic Generation Alignment ===")


for i, prompt in enumerate(prompts, start=1):

    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    ).to(device)

    input_ids = inputs["input_ids"]
    prompt_length = input_ids.shape[1]


    with torch.no_grad():

        # ----------------------------------------------------
        # Hugging Face greedy decoding
        # ----------------------------------------------------

        hf_output = hf_model.generate(
            input_ids=input_ids,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )


        # ----------------------------------------------------
        # nanoGPT greedy decoding
        # ----------------------------------------------------

        x = input_ids.clone()

        for _ in range(MAX_NEW_TOKENS):

            x_cond = (
                x
                if x.size(1)
                <= nanogpt_model.config.block_size
                else x[
                    :,
                    -nanogpt_model.config.block_size:
                ]
            )

            logits, _ = nanogpt_model(x_cond)

            last_logits = logits[:, -1, :]

            next_token = torch.argmax(
                last_logits,
                dim=-1,
                keepdim=True
            )

            x = torch.cat(
                (x, next_token),
                dim=1
            )

        nano_output = x


    hf_generated_ids = (
        hf_output[0][prompt_length:].tolist()
    )

    nano_generated_ids = (
        nano_output[0][prompt_length:].tolist()
    )

    ids_match = (
        hf_generated_ids
        == nano_generated_ids
    )


    generation_results.append({
        "prompt_id": i,
        "prompt": prompt,

        "hf_generated_token_ids":
            hf_generated_ids,

        "nanogpt_generated_token_ids":
            nano_generated_ids,

        "token_ids_match":
            ids_match,

        "hf_generated_text":
            tokenizer.decode(
                hf_generated_ids,
                skip_special_tokens=True
            ),

        "nanogpt_generated_text":
            tokenizer.decode(
                nano_generated_ids,
                skip_special_tokens=True
            ),
    })


    print(
        f"Prompt {i}: "
        f"generated token IDs match = {ids_match}"
    )


all_generation_match = all(
    item["token_ids_match"]
    for item in generation_results
)


# ============================================================
# 5. Save report
# ============================================================

report = {

    "model": "gpt2",

    "implementations": {
        "huggingface":
            "openai-community/gpt2",

        "nanogpt":
            "GPT.from_pretrained('gpt2')",
    },

    "parameter_count": {
        "huggingface":
            hf_param_count,

        "nanogpt":
            nanogpt_param_count,

        "match":
            parameter_count_match,
    },

    "weight_shape_alignment": {
        "hf_tensor_count":
            len(hf_keys),

        "compared_tensor_count":
            len(shape_comparison),

        "missing_in_nanogpt":
            missing_in_nanogpt,

        "all_aligned_shapes_match":
            all_shapes_match,

        "details":
            shape_comparison,
    },

    "forward_logits_alignment": {
        "prompt_id": 1,

        "hf_full_logits_shape":
            list(hf_logits.shape),

        "hf_last_logits_shape":
            list(hf_last_logits.shape),

        "nanogpt_logits_shape":
            list(nano_logits.shape),

        "comparison":
            "last token position",

        "max_absolute_error":
            max_abs_error,

        "mean_absolute_error":
            mean_abs_error,
    },

    "deterministic_generation": {
        "method":
            "greedy decoding",

        "max_new_tokens":
            MAX_NEW_TOKENS,

        "all_token_ids_match":
            all_generation_match,

        "results":
            generation_results,
    },
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


# ============================================================
# Final summary
# ============================================================

print("\n========================================")
print("Task 10 alignment complete.")
print("========================================")

print(
    "Parameter count match:",
    parameter_count_match
)

print(
    "Weight shapes match:",
    all_shapes_match
)

print(
    "Max logits absolute error:",
    max_abs_error
)

print(
    "Mean logits absolute error:",
    mean_abs_error
)

print(
    "All deterministic generations match:",
    all_generation_match
)

print("\nSaved to:")
print(OUTPUT_PATH)
