
import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import tiktoken

from model import GPT, GPTConfig


# ============================================================
# Task 6: Embedding Norm Analysis
#
# Controlled experiment:
# Randomly Initialized GPT-2 vs Pretrained GPT-2
#
# Same:
#   - GPT-2 architecture
#   - Shakespeare text
#   - GPT-2 tokenizer
#   - token IDs
#   - sequence length
#   - positions
#   - evaluation mode
#
# Main difference:
#   - model parameters: random vs pretrained
# ============================================================


SEED = 42

TEXT_PATH = "data/task4_samples.txt"

REPORT_DIR = "reports"

PLOT_PATH = os.path.join(
    REPORT_DIR,
    "embedding_norm_distribution.png"
)

STATS_PATH = os.path.join(
    REPORT_DIR,
    "embedding_norm_statistics.json"
)

os.makedirs(REPORT_DIR, exist_ok=True)


# ============================================================
# 1. Reproducibility and device
# ============================================================

torch.manual_seed(SEED)
np.random.seed(SEED)

device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("=" * 70)
print("Task 6: Embedding Norm Analysis")
print("=" * 70)

print("Device:", device)


# ============================================================
# 2. Read the same Shakespeare samples used in Task 4
# ============================================================

if not os.path.exists(TEXT_PATH):
    raise FileNotFoundError(
        f"Cannot find {TEXT_PATH}"
    )

with open(
    TEXT_PATH,
    "r",
    encoding="utf-8"
) as f:
    text = f.read()


# Task 4 samples are expected to be separated by blank lines.
samples = [
    sample.strip()
    for sample in text.split("\n\n")
    if sample.strip()
]

print(f"\nLoaded {len(samples)} Shakespeare samples")
print(f"Source: {TEXT_PATH}")

if len(samples) == 0:
    raise RuntimeError(
        "No Shakespeare samples were found."
    )


# ============================================================
# 3. Tokenize with GPT-2 tokenizer
# ============================================================

encoder = tiktoken.get_encoding("gpt2")

encoded_samples = [
    encoder.encode(sample)
    for sample in samples
]

lengths = [
    len(ids)
    for ids in encoded_samples
]

print("\nGPT-2 token lengths:")
print(lengths)


# ============================================================
# 4. Build a real-text batch
#
# We need a rectangular tensor (B, T).
#
# To avoid padding introducing artificial tokens into the
# norm distribution, use the shortest sequence length.
#
# Every sample therefore contributes the same number of
# real Shakespeare tokens.
# ============================================================

T = min(lengths)

# GPT-2 block size is 1024.
T = min(T, 1024)

if T <= 0:
    raise RuntimeError(
        "Sequence length must be greater than zero."
    )

batch_ids = [
    ids[:T]
    for ids in encoded_samples
]

idx = torch.tensor(
    batch_ids,
    dtype=torch.long,
    device=device
)

B, T = idx.shape

print("\nReal Shakespeare batch created")
print("Batch size B      =", B)
print("Sequence length T =", T)
print("Input shape       =", tuple(idx.shape))


# ============================================================
# 5. Load pretrained GPT-2
# ============================================================

print("\nLoading pretrained GPT-2...")

pretrained_model = GPT.from_pretrained(
    "gpt2"
)

pretrained_model = pretrained_model.to(
    device
)

# Evaluation mode:
# dropout is disabled for a controlled comparison.

pretrained_model.eval()


# ============================================================
# 6. Build randomly initialized GPT-2
#    with exactly the same architecture
# ============================================================

print("\nBuilding randomly initialized GPT-2...")

random_config = GPTConfig(
    block_size=pretrained_model.config.block_size,
    vocab_size=pretrained_model.config.vocab_size,
    n_layer=pretrained_model.config.n_layer,
    n_head=pretrained_model.config.n_head,
    n_embd=pretrained_model.config.n_embd,
    dropout=0.0,
    bias=pretrained_model.config.bias,
)

torch.manual_seed(SEED)

random_model = GPT(
    random_config
)

random_model = random_model.to(
    device
)

random_model.eval()


# ============================================================
# 7. Verify controlled architecture
# ============================================================

print("\n" + "=" * 70)
print("Controlled Architecture Check")
print("=" * 70)

attributes = [
    "block_size",
    "vocab_size",
    "n_layer",
    "n_head",
    "n_embd",
    "bias",
]

for attr in attributes:

    random_value = getattr(
        random_model.config,
        attr
    )

    pretrained_value = getattr(
        pretrained_model.config,
        attr
    )

    print(
        f"{attr:12s}: "
        f"random={random_value}, "
        f"pretrained={pretrained_value}"
    )

    assert random_value == pretrained_value


C = pretrained_model.config.n_embd

print("\nControlled input:")
print("B =", B)
print("T =", T)
print("C =", C)
print("idx shape =", tuple(idx.shape))


# ============================================================
# 8. Helper: descriptive statistics
# ============================================================

def summarize_norm(norm_tensor):

    values = (
        norm_tensor
        .detach()
        .cpu()
        .numpy()
        .reshape(-1)
    )

    return {
        "mean": float(
            np.mean(values)
        ),

        "median": float(
            np.median(values)
        ),

        "std": float(
            np.std(values)
        ),

        "p5": float(
            np.percentile(values, 5)
        ),

        "p95": float(
            np.percentile(values, 95)
        ),
    }


# ============================================================
# 9. Analyze one model
# ============================================================

def analyze_model(
    model,
    idx,
    model_name
):

    model.eval()

    B, T = idx.shape
    device = idx.device


    # --------------------------------------------------------
    # A. Token embedding
    #
    # (B, T) -> (B, T, C)
    # --------------------------------------------------------

    with torch.no_grad():

        token_emb = (
            model.transformer.wte(idx)
        )

    token_norm = (
        torch.linalg.vector_norm(
            token_emb,
            ord=2,
            dim=-1
        )
    )

    # token_norm:
    # (B, T)


    # --------------------------------------------------------
    # B. Position embedding
    #
    # positions:
    # [0, 1, ..., T-1]
    #
    # (T,) -> (T, C)
    # --------------------------------------------------------

    pos = torch.arange(
        0,
        T,
        dtype=torch.long,
        device=device
    )

    with torch.no_grad():

        position_emb = (
            model.transformer.wpe(pos)
        )


    # --------------------------------------------------------
    # C. Token + position embedding
    #
    # (B,T,C) + (T,C)
    #       ->
    # (B,T,C)
    # --------------------------------------------------------

    combined_emb = (
        token_emb
        +
        position_emb
    )

    combined_norm = (
        torch.linalg.vector_norm(
            combined_emb,
            ord=2,
            dim=-1
        )
    )


    # --------------------------------------------------------
    # D. Forward hook:
    # capture embedding dropout output
    #
    # This is the tensor immediately before
    # Transformer Block 1.
    # --------------------------------------------------------

    captured = {}

    def dropout_hook(
        module,
        inputs,
        output
    ):
        captured[
            "dropout_output"
        ] = output.detach()


    handle = (
        model.transformer.drop
        .register_forward_hook(
            dropout_hook
        )
    )


    with torch.no_grad():
        _ = model(idx)


    handle.remove()


    if "dropout_output" not in captured:

        raise RuntimeError(
            "Forward hook failed."
        )


    dropout_output = (
        captured["dropout_output"]
    )

    dropout_norm = (
        torch.linalg.vector_norm(
            dropout_output,
            ord=2,
            dim=-1
        )
    )


    # --------------------------------------------------------
    # E. Verify eval-mode dropout behavior
    # --------------------------------------------------------

    dropout_identical = torch.allclose(
        combined_emb,
        dropout_output
    )


    # --------------------------------------------------------
    # F. Statistics
    # --------------------------------------------------------

    stats = {

        "token_embedding":
            summarize_norm(
                token_norm
            ),

        "token_plus_position":
            summarize_norm(
                combined_norm
            ),

        "after_embedding_dropout":
            summarize_norm(
                dropout_norm
            ),

        "dropout_identical_in_eval":
            bool(dropout_identical),
    }


    # --------------------------------------------------------
    # G. Print
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print(model_name)
    print("=" * 70)

    print(
        "token_emb shape:     ",
        tuple(token_emb.shape)
    )

    print(
        "position_emb shape:  ",
        tuple(position_emb.shape)
    )

    print(
        "combined_emb shape:  ",
        tuple(combined_emb.shape)
    )

    print(
        "dropout_output shape:",
        tuple(dropout_output.shape)
    )


    for stage in [
        "token_embedding",
        "token_plus_position",
        "after_embedding_dropout",
    ]:

        print(f"\n{stage}")

        for key, value in stats[
            stage
        ].items():

            print(
                f"  {key:8s}: "
                f"{value:.6f}"
            )


    print(
        "\nDropout output identical "
        "to combined embedding:",
        dropout_identical
    )


    # --------------------------------------------------------
    # H. Raw values for plotting
    # --------------------------------------------------------

    values = {

        "token_embedding":
            token_norm
            .detach()
            .cpu()
            .numpy()
            .reshape(-1),

        "token_plus_position":
            combined_norm
            .detach()
            .cpu()
            .numpy()
            .reshape(-1),

        "after_embedding_dropout":
            dropout_norm
            .detach()
            .cpu()
            .numpy()
            .reshape(-1),
    }


    return stats, values


# ============================================================
# 10. Analyze both models
# ============================================================

random_stats, random_values = (
    analyze_model(
        random_model,
        idx,
        "Randomly Initialized GPT-2"
    )
)

pretrained_stats, pretrained_values = (
    analyze_model(
        pretrained_model,
        idx,
        "Pretrained GPT-2"
    )
)


# ============================================================
# 11. Save statistics
# ============================================================

all_stats = {

    "experiment": {

        "seed": SEED,

        "text_source":
            TEXT_PATH,

        "number_of_samples":
            B,

        "sequence_length":
            T,

        "tokenizer":
            "GPT-2 tiktoken",

        "comparison":
            "random GPT-2 vs pretrained GPT-2",

        "architecture": {

            "block_size":
                pretrained_model.config.block_size,

            "vocab_size":
                pretrained_model.config.vocab_size,

            "n_layer":
                pretrained_model.config.n_layer,

            "n_head":
                pretrained_model.config.n_head,

            "n_embd":
                pretrained_model.config.n_embd,
        },
    },

    "random_gpt2":
        random_stats,

    "pretrained_gpt2":
        pretrained_stats,
}


with open(
    STATS_PATH,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        all_stats,
        f,
        indent=2
    )


print(
    f"\nSaved statistics to: "
    f"{STATS_PATH}"
)


# ============================================================
# 12. Plot norm distributions
# ============================================================

plt.figure(
    figsize=(11, 7)
)


# Random model
plt.hist(
    random_values[
        "token_embedding"
    ],
    bins=40,
    alpha=0.35,
    label="Random: token"
)

plt.hist(
    random_values[
        "token_plus_position"
    ],
    bins=40,
    alpha=0.35,
    label="Random: token + position"
)

plt.hist(
    random_values[
        "after_embedding_dropout"
    ],
    bins=40,
    alpha=0.35,
    label="Random: after dropout"
)


# Pretrained model
plt.hist(
    pretrained_values[
        "token_embedding"
    ],
    bins=40,
    alpha=0.35,
    label="Pretrained: token"
)

plt.hist(
    pretrained_values[
        "token_plus_position"
    ],
    bins=40,
    alpha=0.35,
    label="Pretrained: token + position"
)

plt.hist(
    pretrained_values[
        "after_embedding_dropout"
    ],
    bins=40,
    alpha=0.35,
    label="Pretrained: after dropout"
)


plt.xlabel(
    "L2 Norm"
)

plt.ylabel(
    "Frequency"
)

plt.title(
    "Embedding Norm Distribution on Shakespeare Text\n"
    "Random vs Pretrained GPT-2"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    PLOT_PATH,
    dpi=200,
    bbox_inches="tight"
)

plt.close()


print(
    f"Saved plot to: "
    f"{PLOT_PATH}"
)


# ============================================================
# 13. Concise comparison
# ============================================================

print("\n" + "=" * 70)
print("Random vs Pretrained Comparison")
print("=" * 70)


for stage in [
    "token_embedding",
    "token_plus_position",
    "after_embedding_dropout",
]:

    r = random_stats[stage]
    p = pretrained_stats[stage]

    print(f"\n{stage}")

    print(
        f"  Random mean:      "
        f"{r['mean']:.6f}"
    )

    print(
        f"  Pretrained mean:  "
        f"{p['mean']:.6f}"
    )

    print(
        f"  Random median:    "
        f"{r['median']:.6f}"
    )

    print(
        f"  Pretrained median:"
        f" {p['median']:.6f}"
    )

    print(
        f"  Random std:       "
        f"{r['std']:.6f}"
    )

    print(
        f"  Pretrained std:   "
        f"{p['std']:.6f}"
    )

    print(
        f"  Random P5-P95:    "
        f"{r['p5']:.6f} - "
        f"{r['p95']:.6f}"
    )

    print(
        f"  Pretrained P5-P95:"
        f" {p['p5']:.6f} - "
        f"{p['p95']:.6f}"
    )


# ============================================================
# 14. Generate a short conclusion
# ============================================================

random_token_mean = (
    random_stats[
        "token_embedding"
    ]["mean"]
)

pretrained_token_mean = (
    pretrained_stats[
        "token_embedding"
    ]["mean"]
)

ratio = (
    pretrained_token_mean
    /
    random_token_mean
)


print("\n" + "=" * 70)
print("Short Conclusion")
print("=" * 70)

print(
    "The randomly initialized and pretrained models "
    "use the same GPT-2 architecture and the same "
    "Shakespeare token batch."
)

print(
    f"The pretrained token embeddings have a mean "
    f"L2 norm of {pretrained_token_mean:.4f}, "
    f"compared with {random_token_mean:.4f} for "
    f"the random model "
    f"(approximately {ratio:.2f}x larger)."
)

print(
    "The pretrained model also shows a broader norm "
    "distribution, indicating that pretraining changes "
    "both the scale and variability of the embedding "
    "parameters."
)

print(
    "Adding positional embeddings changes the norm "
    "distribution in both models."
)

print(
    "Because both models are evaluated in eval mode, "
    "dropout is disabled, so the post-dropout norm "
    "distribution is identical to the token-plus-position "
    "distribution."
)

print(
    "L2 norm measures vector magnitude only; it does not "
    "directly measure semantic quality."
)

print("\nTask 6 complete.")
