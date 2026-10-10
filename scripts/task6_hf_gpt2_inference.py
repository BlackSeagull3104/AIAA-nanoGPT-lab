"""Run one lightweight Hugging Face GPT-2 prompt without training or W&B."""

import argparse
import json
import time
from pathlib import Path


MODEL_ID = "openai-community/gpt2"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", default="To be, or not to be")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Use the local Hugging Face cache and fail instead of downloading.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.max_new_tokens < 1:
        raise ValueError("--max-new-tokens must be at least 1")

    # Keep imports inside main so --help and static tests never load a model.
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device("cpu")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        local_files_only=args.local_files_only,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        local_files_only=args.local_files_only,
    ).to(device)
    model.eval()

    inputs = tokenizer(args.prompt, return_tensors="pt").to(device)
    prompt_tokens = inputs["input_ids"].shape[1]
    started = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    elapsed = time.perf_counter() - started
    generated_ids = output[0, prompt_tokens:]

    result = {
        "model_id": MODEL_ID,
        "device": str(device),
        "prompt": args.prompt,
        "max_new_tokens": args.max_new_tokens,
        "generated_tokens": int(generated_ids.numel()),
        "generated_text": tokenizer.decode(generated_ids, skip_special_tokens=True),
        "runtime_seconds": elapsed,
        "local_files_only": args.local_files_only,
    }
    rendered = json.dumps(result, indent=2, ensure_ascii=False)
    print(rendered)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
