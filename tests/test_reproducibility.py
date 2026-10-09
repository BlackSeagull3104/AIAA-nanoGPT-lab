import random

import numpy as np
import torch


SEED = 1337


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
