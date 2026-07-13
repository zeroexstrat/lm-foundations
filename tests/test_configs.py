import random
from types import SimpleNamespace

import numpy as np
import torch

from llm_from_scratch.configs import (
    ModelConfig,
    RuntimeProfile,
    TrainConfig,
    get_device,
    profile_config,
    set_seed,
)


def test_default_model_config_is_tiny_enough_for_tests() -> None:
    cfg = ModelConfig(vocab_size=32)
    assert cfg.vocab_size == 32
    assert cfg.block_size <= 128
    assert cfg.n_embd % cfg.n_head == 0


def test_runtime_profiles_have_ordered_training_sizes() -> None:
    quick_model, quick_train = profile_config("quick")
    study_model, study_train = profile_config("study")
    stretch_model, stretch_train = profile_config("stretch")

    assert quick_model.n_layer <= study_model.n_layer <= stretch_model.n_layer
    assert quick_train.max_steps < study_train.max_steps < stretch_train.max_steps


def test_get_device_returns_torch_device() -> None:
    device = get_device()
    assert isinstance(device, torch.device)
    assert device.type in {"cpu", "cuda", "mps"}


def test_set_seed_reproducible_for_python_numpy_and_torch() -> None:
    set_seed(123)
    first = (random.random(), np.random.rand(), torch.rand(1).item())
    set_seed(123)
    second = (random.random(), np.random.rand(), torch.rand(1).item())
    assert first == second


def test_set_seed_enables_deterministic_torch_behavior(monkeypatch) -> None:
    calls = []

    def record_use_deterministic_algorithms(enabled: bool, warn_only: bool = False) -> None:
        calls.append((enabled, warn_only))

    monkeypatch.setattr(torch, "use_deterministic_algorithms", record_use_deterministic_algorithms)
    monkeypatch.setattr(torch, "manual_seed", lambda seed: None)
    monkeypatch.setattr(torch, "cuda", SimpleNamespace(is_available=lambda: False))
    monkeypatch.setattr(torch, "backends", SimpleNamespace(cudnn=SimpleNamespace()))

    set_seed(7)

    assert calls == [(True, True)]
    assert torch.backends.cudnn.deterministic is True
    assert torch.backends.cudnn.benchmark is False


def test_runtime_profile_literal_values() -> None:
    assert RuntimeProfile.QUICK.value == "quick"
    assert RuntimeProfile.STUDY.value == "study"
    assert RuntimeProfile.STRETCH.value == "stretch"
