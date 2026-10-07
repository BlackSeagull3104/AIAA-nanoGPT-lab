"""Reload an adapter-only checkpoint on GPT-2 and run the fixed prompts."""

import argparse
import sys
from pathlib import Path

import tiktoken
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model import GPT


PROMPTS = ["ROMEO:", "JULIET:", "To be, or not to be"]


@torch.inference_mode()
def greedy_generate(model, tokens, max_new_tokens):
    for _ in range(max_new_tokens):
        context = tokens[:, -model.config.block_size:]
        logits, _ = model(context)
        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        tokens = torch.cat((tokens, next_token), dim=1)
    return tokens


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    args = parser.parse_args()

    payload = torch.load(args.adapter, map_location="cpu", weights_only=True)
    metadata = payload["metadata"]
    if metadata["target_modules"] != "attn.c_attn":
        raise ValueError(f"unsupported adapter targets: {metadata['target_modules']}")
    model = GPT.from_pretrained(metadata["base_model"])
    model.enable_lora(
        metadata["rank"], metadata["alpha"], metadata["dropout"]
    )
    model.freeze_non_lora_parameters()
    model.load_lora_state_dict(payload["lora_state_dict"])
    model.to(args.device).eval()

    tokenizer = tiktoken.get_encoding("gpt2")
    torch.manual_seed(metadata["seed"])
    if args.device.startswith("cuda"):
        torch.cuda.manual_seed_all(metadata["seed"])
    for prompt in PROMPTS:
        prompt_ids = tokenizer.encode(prompt)
        tokens = torch.tensor(prompt_ids, dtype=torch.long, device=args.device)[None]
        generated = greedy_generate(model, tokens, args.max_new_tokens)
        print(f"\nPrompt: {prompt}\n{tokenizer.decode(generated[0].tolist())}")
    print("\nLoRA adapter reload and fixed-prompt inference passed.")


if __name__ == "__main__":
    main()
