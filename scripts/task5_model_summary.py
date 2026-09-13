import json
import os

import torch

from model import GPTConfig, GPT


# ============================================================
# 1. Build model from GPTConfig
# ============================================================

config = GPTConfig()
model = GPT(config)


# ============================================================
# 2. Read basic model configuration
# ============================================================

config_summary = {
    "block_size": config.block_size,
    "vocab_size": config.vocab_size,
    "n_layer": config.n_layer,
    "n_head": config.n_head,
    "n_embd": config.n_embd,
    "dropout": config.dropout,
    "bias": config.bias,
}


# ============================================================
# 3. Total parameter count
# ============================================================

total_params = sum(
    p.numel()
    for p in model.parameters()
)


# ============================================================
# 4. Parameter memory
# ============================================================

total_memory_bytes = sum(
    p.numel() * p.element_size()
    for p in model.parameters()
)

total_memory_mb = (
    total_memory_bytes
    / 1024
    / 1024
)


# ============================================================
# 5. Component-wise parameter counts
# ============================================================

# Token embedding
token_embedding_params = sum(
    p.numel()
    for p in model.transformer.wte.parameters()
)

# Position embedding
position_embedding_params = sum(
    p.numel()
    for p in model.transformer.wpe.parameters()
)


# Attention parameters across all Transformer blocks
attention_params = 0

for block in model.transformer.h:
    attention_params += sum(
        p.numel()
        for p in block.attn.parameters()
    )


# MLP parameters across all Transformer blocks
mlp_params = 0

for block in model.transformer.h:
    mlp_params += sum(
        p.numel()
        for p in block.mlp.parameters()
    )


# LayerNorm parameters
layernorm_params = 0

for block in model.transformer.h:
    layernorm_params += sum(
        p.numel()
        for p in block.ln_1.parameters()
    )

    layernorm_params += sum(
        p.numel()
        for p in block.ln_2.parameters()
    )

layernorm_params += sum(
    p.numel()
    for p in model.transformer.ln_f.parameters()
)


# ============================================================
# 6. Weight sharing check
# ============================================================

embedding_weight = model.transformer.wte.weight
lm_head_weight = model.lm_head.weight

same_parameter_object = (
    embedding_weight
    is lm_head_weight
)

same_memory_location = (
    embedding_weight.data_ptr()
    == lm_head_weight.data_ptr()
)

same_values = torch.equal(
    embedding_weight,
    lm_head_weight
)


# ============================================================
# 7. Build summary
# ============================================================

summary = {
    "config": config_summary,

    "parameters": {
        "total_parameters": total_params,
        "parameter_memory_bytes": total_memory_bytes,
        "parameter_memory_mb": total_memory_mb,
    },

    "component_parameters": {
        "token_embedding": token_embedding_params,
        "position_embedding": position_embedding_params,
        "attention": attention_params,
        "mlp": mlp_params,
        "layernorm": layernorm_params,
    },

    "weight_tying": {
        "token_embedding_and_lm_head_share_parameter_object":
            same_parameter_object,

        "token_embedding_and_lm_head_share_memory":
            same_memory_location,

        "token_embedding_and_lm_head_values_equal":
            same_values,
    },
}


# ============================================================
# 8. Print summary
# ============================================================

print("\n=== GPTConfig ===")

for key, value in config_summary.items():
    print(f"{key}: {value}")


print("\n=== Total Parameters ===")

print(
    f"Total parameters: "
    f"{total_params:,}"
)

print(
    f"Parameter memory: "
    f"{total_memory_mb:.2f} MB"
)


print("\n=== Component Parameter Counts ===")

print(
    f"Token embedding: "
    f"{token_embedding_params:,}"
)

print(
    f"Position embedding: "
    f"{position_embedding_params:,}"
)

print(
    f"Attention: "
    f"{attention_params:,}"
)

print(
    f"MLP: "
    f"{mlp_params:,}"
)

print(
    f"LayerNorm: "
    f"{layernorm_params:,}"
)


print("\n=== Weight Tying Check ===")

print(
    "Same parameter object:",
    same_parameter_object
)

print(
    "Same memory location:",
    same_memory_location
)

print(
    "Same values:",
    same_values
)


# ============================================================
# 9. Save JSON
# ============================================================

os.makedirs(
    "reports",
    exist_ok=True
)

output_path = (
    "reports/model_summary.json"
)

with open(
    output_path,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        summary,
        f,
        indent=2,
    )


print(
    "\nSaved:",
    output_path
)
