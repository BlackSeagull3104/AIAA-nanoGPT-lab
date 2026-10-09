import ast
import random
from dataclasses import fields
from pathlib import Path

import numpy as np
import pytest
import torch

from model import GPT, GPTConfig


SEED = 1337
REPO_ROOT = Path(__file__).resolve().parents[1]
CHAR_PREPARE_PATH = REPO_ROOT / "data" / "shakespeare_char" / "prepare.py"


def generate_values(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    return {
        "python": random.random(),
        "numpy": np.random.rand(4),
        "torch": torch.rand(4),
    }


def test_same_seed_reproduces_same_values():
    first = generate_values(SEED)
    second = generate_values(SEED)

    assert first["python"] == second["python"]
    assert np.array_equal(first["numpy"], second["numpy"])
    assert torch.equal(first["torch"], second["torch"])


def test_different_seed_changes_values():
    first = generate_values(SEED)
    second = generate_values(SEED + 1)

    assert first["python"] != second["python"]
    assert not np.array_equal(first["numpy"], second["numpy"])
    assert not torch.equal(first["torch"], second["torch"])


def test_character_encode_decode_round_trip():
    """Exercise prepare.py's tokenizer functions without importing the script."""
    source = CHAR_PREPARE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(CHAR_PREPARE_PATH))
    tokenizer_functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in {"encode", "decode"}
    ]
    assert {node.name for node in tokenizer_functions} == {"encode", "decode"}

    namespace = {}
    functions_only = ast.Module(body=tokenizer_functions, type_ignores=[])
    exec(compile(functions_only, str(CHAR_PREPARE_PATH), "exec"), namespace)

    text = "First Citizen:\n"
    characters = sorted(set(text))
    namespace["stoi"] = {character: index for index, character in enumerate(characters)}
    namespace["itos"] = {index: character for character, index in namespace["stoi"].items()}

    assert namespace["decode"](namespace["encode"](text)) == text


def test_gpt2_tokenizer_ids_match_offline(monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")

    tiktoken = pytest.importorskip("tiktoken")
    transformers = pytest.importorskip("transformers")
    tiktoken_load = pytest.importorskip("tiktoken.load")

    class MissingLocalTokenizerAsset(Exception):
        pass

    def reject_download(path):
        raise MissingLocalTokenizerAsset(path)

    # tiktoken reads its on-disk cache before calling read_file. Replacing only
    # that final loader lets cached assets work while making a cache miss safe.
    monkeypatch.setattr(tiktoken_load, "read_file", reject_download)

    try:
        tiktoken_encoder = tiktoken.get_encoding("gpt2")
    except MissingLocalTokenizerAsset:
        pytest.skip("tiktoken GPT-2 assets are not available in the local cache")

    try:
        hf_tokenizer = transformers.AutoTokenizer.from_pretrained(
            "openai-community/gpt2",
            local_files_only=True,
        )
    except OSError:
        pytest.skip("Hugging Face GPT-2 tokenizer is not available in the local cache")

    text = "First Citizen: reproducible tokenization!"
    assert tiktoken_encoder.encode_ordinary(text) == hf_tokenizer.encode(
        text,
        add_special_tokens=False,
    )


def test_tiny_gpt_forward_output_shape():
    torch.manual_seed(SEED)
    config = GPTConfig(
        block_size=8,
        vocab_size=32,
        n_layer=1,
        n_head=2,
        n_embd=8,
        dropout=0.0,
        bias=True,
    )
    model = GPT(config).eval()
    inputs = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]], dtype=torch.long)

    with torch.inference_mode():
        logits, loss = model(inputs, targets=inputs)

    assert logits.shape == (2, 4, config.vocab_size)
    assert loss.shape == ()


def test_gpt_config_has_required_fields():
    required_fields = {
        "block_size",
        "vocab_size",
        "n_layer",
        "n_head",
        "n_embd",
        "dropout",
        "bias",
    }
    assert required_fields <= {field.name for field in fields(GPTConfig)}
