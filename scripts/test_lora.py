"""CPU-safe correctness tests for the native nanoGPT LoRA implementation."""

import copy
import sys
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model import GPT, GPTConfig


def main():
    torch.manual_seed(1234)
    config = GPTConfig(
        block_size=16, vocab_size=64, n_layer=2, n_head=2, n_embd=16,
        dropout=0.0, bias=True,
    )
    base = GPT(config).eval()
    unchanged = copy.deepcopy(base).eval()
    tokens = torch.randint(0, config.vocab_size, (2, 8))

    with torch.no_grad():
        expected, _ = base(tokens)
        unchanged_output, _ = unchanged(tokens)
    torch.testing.assert_close(unchanged_output, expected)

    base.enable_lora(rank=4, alpha=8.0, dropout=0.0)
    base.eval()
    with torch.no_grad():
        initial_lora_output, _ = base(tokens)
    torch.testing.assert_close(initial_lora_output, expected, rtol=1e-6, atol=1e-6)

    base.freeze_non_lora_parameters()
    stats = base.get_parameter_stats()
    trainable = [name for name, p in base.named_parameters() if p.requires_grad]
    assert trainable and all(name.endswith(("lora_A", "lora_B")) for name in trainable)
    assert stats["trainable_params"] < stats["total_params"]

    base.train()
    _, loss = base(tokens, tokens)
    loss.backward()
    for name, parameter in base.named_parameters():
        if name.endswith(("lora_A", "lora_B")):
            assert parameter.grad is not None, f"missing LoRA gradient: {name}"
        else:
            assert parameter.grad is None, f"frozen base gradient present: {name}"

    with torch.no_grad():
        for name, parameter in base.named_parameters():
            if name.endswith("lora_B"):
                parameter.add_(0.01)
    adapter = base.lora_state_dict()
    with tempfile.NamedTemporaryFile(suffix=".pt") as handle:
        torch.save(adapter, handle.name)
        saved_adapter = torch.load(handle.name, map_location="cpu", weights_only=True)

    reloaded = copy.deepcopy(unchanged)
    reloaded.enable_lora(rank=4, alpha=8.0, dropout=0.0)
    reloaded.freeze_non_lora_parameters()
    reloaded.load_lora_state_dict(saved_adapter)
    for name, tensor in adapter.items():
        torch.testing.assert_close(reloaded.state_dict()[name], tensor)

    base.eval()
    reloaded.eval()
    with torch.no_grad():
        adapted_output, _ = base(tokens)
        reloaded_output, _ = reloaded(tokens)
    torch.testing.assert_close(reloaded_output, adapted_output)

    print(
        "LoRA tests passed: "
        f"total={stats['total_params']:,}, "
        f"trainable={stats['trainable_params']:,}, "
        f"trainable_percentage={stats['trainable_percentage']:.4f}%"
    )
    print("Trainable tensors:")
    for name in trainable:
        print(f"  {name}")


if __name__ == "__main__":
    main()
