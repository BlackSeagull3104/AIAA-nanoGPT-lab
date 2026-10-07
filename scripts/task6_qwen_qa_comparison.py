"""Log a behavior-only GPT-2/Qwen Instruct QA comparison to W&B.

GPT-2 text is reused from the existing Task 11 report. Qwen inference is run
once. Historical GPT-2 resource metrics are read from the completed Task 3 run
and are explicitly labelled because its evaluation setup differs from Task 6.
"""

import argparse
import importlib.util
import json
import time
from pathlib import Path

import torch
import wandb
from transformers import AutoModelForCausalLM, AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_config(path):
    spec = importlib.util.spec_from_file_location("task6_config", path)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    return config


def load_gpt2_outputs(config):
    report_path = REPO_ROOT / config.gpt2_report_path
    report = json.loads(report_path.read_text(encoding="utf-8"))
    outputs = {row["question"]: row["answer"] for row in report["gpt2_results"]}
    missing = [prompt for prompt in config.prompts if prompt not in outputs]
    if missing:
        raise ValueError(f"GPT-2 report is missing configured prompts: {missing}")
    return outputs, str(report_path)


def historical_gpt2_metrics(config):
    source = wandb.Api().run(config.gpt2_source_run)
    summary = source.summary
    required = {
        "parameter_count": "eval/num_parameters",
        "runtime_seconds": "eval/inference_seconds",
        "peak_gpu_memory_mb": "eval/gpu_peak_memory_mb",
    }
    values = {name: summary.get(key) for name, key in required.items()}
    missing = [required[name] for name, value in values.items() if value is None]
    if missing:
        raise RuntimeError(
            "Historical GPT-2 run lacks required summary metrics: "
            + ", ".join(missing)
        )
    return values


@torch.inference_mode()
def run_qwen(config, device):
    tokenizer = AutoTokenizer.from_pretrained(config.qwen_model_name)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        config.qwen_model_name, torch_dtype=dtype
    ).to(device)
    model.eval()
    parameter_count = sum(parameter.numel() for parameter in model.parameters())

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    outputs = {}
    generated_tokens = 0
    for prompt in config.prompts:
        messages = [{"role": "user", "content": prompt}]
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(formatted, return_tensors="pt").to(device)
        result = model.generate(
            **inputs,
            max_new_tokens=config.max_new_tokens,
            do_sample=config.do_sample,
            pad_token_id=tokenizer.eos_token_id,
        )
        new_ids = result[0, inputs["input_ids"].shape[1]:]
        generated_tokens += new_ids.numel()
        outputs[prompt] = tokenizer.decode(
            new_ids, skip_special_tokens=True
        ).strip()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    runtime_seconds = time.perf_counter() - started
    peak_memory_mb = (
        torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        if device.type == "cuda" else None
    )
    return outputs, {
        "parameter_count": parameter_count,
        "runtime_seconds": runtime_seconds,
        "peak_gpu_memory_mb": peak_memory_mb,
        "generated_tokens": generated_tokens,
        "tokens_per_second": generated_tokens / runtime_seconds,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config", nargs="?", default="config/task6_qwen_qa_wandb.py"
    )
    args = parser.parse_args()
    config = load_config(REPO_ROOT / args.config)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    gpt2_outputs, gpt2_report = load_gpt2_outputs(config)
    gpt2_metrics = historical_gpt2_metrics(config)
    qwen_outputs, qwen_metrics = run_qwen(config, device)

    generation_config = {
        "seed": config.seed,
        "max_new_tokens": config.max_new_tokens,
        "do_sample": config.do_sample,
        "decoding": "greedy" if not config.do_sample else "sampling",
        "qwen_chat_template": True,
    }
    run = wandb.init(
        entity=config.wandb_entity,
        project=config.wandb_project,
        name=config.wandb_run_name,
        tags=config.wandb_tags,
        config={
            "qwen_model": config.qwen_model_name,
            "gpt2_model": config.gpt2_model_name,
            "gpt2_output_source": gpt2_report,
            "gpt2_metric_source_run": config.gpt2_source_run,
            "gpt2_runtime_kind": "historical Task 3 evaluation duration",
            "qwen_runtime_kind": "Task 6 total QA generation duration",
            "device": str(device),
            "generation": generation_config,
            "comparison_scope": "generation behavior only; not a rigorous benchmark",
        },
    )
    table = wandb.Table(columns=[
        "prompt", "GPT-2 output", "Qwen output", "instruction following",
        "Chinese expression", "answer completeness",
    ])
    for prompt, observation in zip(config.prompts, config.observations):
        table.add_data(prompt, gpt2_outputs[prompt], qwen_outputs[prompt], *observation)

    metrics = {
        config.wandb_table_key: table,
        "gpt2/parameter_count": gpt2_metrics["parameter_count"],
        "gpt2/runtime_seconds": gpt2_metrics["runtime_seconds"],
        "gpt2/peak_gpu_memory_mb": gpt2_metrics["peak_gpu_memory_mb"],
        "qwen/parameter_count": qwen_metrics["parameter_count"],
        "qwen/runtime_seconds": qwen_metrics["runtime_seconds"],
        "qwen/generated_tokens": qwen_metrics["generated_tokens"],
        "qwen/tokens_per_second": qwen_metrics["tokens_per_second"],
    }
    if qwen_metrics["peak_gpu_memory_mb"] is not None:
        metrics["qwen/peak_gpu_memory_mb"] = qwen_metrics["peak_gpu_memory_mb"]
    wandb.log(metrics)
    run.finish()


if __name__ == "__main__":
    main()
