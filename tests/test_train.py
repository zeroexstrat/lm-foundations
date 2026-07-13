from dataclasses import replace
from types import SimpleNamespace

import llm_from_scratch.evaluate as evaluate_module
import llm_from_scratch.train as train_module
import pytest
import torch
from torch import nn

from llm_from_scratch.configs import ModelConfig, TrainConfig
from llm_from_scratch.evaluate import estimate_loss, perplexity
from llm_from_scratch.model import TinyGPT
from llm_from_scratch.train import (
    load_checkpoint,
    overfit_tiny_batch,
    save_checkpoint,
    train_steps,
)


class LossSequenceModel(nn.Module):
    def __init__(self, losses: list[float], block_size: int = 4) -> None:
        super().__init__()
        self.config = SimpleNamespace(block_size=block_size)
        self.losses = losses

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        loss = self.losses.pop(0)
        logits = torch.zeros(idx.size(0), idx.size(1), 1)
        return logits, torch.tensor(loss)


def test_perplexity_is_exp_loss() -> None:
    assert abs(perplexity(0.0) - 1.0) < 1e-8


def test_estimate_loss_averages_over_requested_eval_batches(monkeypatch) -> None:
    calls: list[tuple[int, int, torch.device]] = []

    def fake_get_batch(
        tokens: torch.Tensor,
        block_size: int,
        batch_size: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        calls.append((block_size, batch_size, device))
        x = torch.zeros((batch_size, block_size), dtype=torch.long)
        y = torch.ones((batch_size, block_size), dtype=torch.long)
        return x, y

    monkeypatch.setattr("llm_from_scratch.evaluate.get_batch", fake_get_batch)
    model = LossSequenceModel([1.0, 3.0, 5.0, 7.0])
    model.train()
    metrics = estimate_loss(
        model,
        train_tokens=torch.arange(20),
        val_tokens=torch.arange(20),
        batch_size=3,
        eval_batches=2,
        device=torch.device("cpu"),
    )
    assert metrics == {"train": 2.0, "val": 6.0}
    assert calls == [(4, 3, torch.device("cpu"))] * 4
    assert model.training is True


def test_overfit_tiny_batch_reduces_loss() -> None:
    torch.manual_seed(0)
    cfg = ModelConfig(vocab_size=8, block_size=4, n_embd=16, n_head=4, n_layer=1, dropout=0.0)
    model = TinyGPT(cfg)
    x = torch.randint(0, cfg.vocab_size, (4, cfg.block_size))
    y = torch.randint(0, cfg.vocab_size, (4, cfg.block_size))
    first, last = overfit_tiny_batch(model, x, y, steps=40, lr=1e-2)
    assert last < first


def test_checkpoint_round_trip_preserves_nested_metadata(tmp_path, monkeypatch) -> None:
    cfg = ModelConfig(vocab_size=8, block_size=4, n_embd=16, n_head=4, n_layer=1, dropout=0.0)
    model = TinyGPT(cfg)
    path = tmp_path / "nested" / "dir" / "checkpoint.pt"
    metadata = {"step": 7, "notes": ["kept"], "nested": {"enabled": True}}
    save_checkpoint(path, model, metadata=metadata)
    original_load = torch.load
    observed: dict[str, object] = {}

    def recording_load(*args, **kwargs):
        observed.update(kwargs)
        return original_load(*args, **kwargs)

    monkeypatch.setattr(train_module.torch, "load", recording_load)
    checkpoint = load_checkpoint(path)

    assert path.exists()
    assert set(checkpoint) == {"model_state_dict", "metadata"}
    assert checkpoint["metadata"] == metadata
    assert checkpoint["model_state_dict"].keys() == model.state_dict().keys()
    assert observed["weights_only"] is True


def test_train_steps_records_eval_history() -> None:
    torch.manual_seed(0)
    cfg = ModelConfig(vocab_size=8, block_size=4, n_embd=16, n_head=4, n_layer=1, dropout=0.0)
    model = TinyGPT(cfg)
    tokens = torch.randint(0, cfg.vocab_size, (32,))
    train_cfg = TrainConfig(
        batch_size=2,
        max_steps=4,
        eval_interval=2,
        eval_batches=2,
        learning_rate=1e-2,
        weight_decay=0.0,
        grad_clip=1.0,
    )
    history = train_steps(model, tokens, tokens, train_cfg, torch.device("cpu"))
    assert len(history) == 2
    assert history[0]["step"] == 0.0
    assert history[1]["step"] == 2.0
    for entry in history:
        assert set(entry) == {"step", "train", "val"}
        assert entry["train"] >= 0.0
        assert entry["val"] >= 0.0


def test_train_steps_seed_controls_eval_and_train_sampling(monkeypatch) -> None:
    cfg = ModelConfig(vocab_size=128, block_size=4, n_embd=16, n_head=4, n_layer=1, dropout=0.0)
    tokens = torch.arange(64, dtype=torch.long)
    device = torch.device("cpu")
    train_cfg = TrainConfig(
        batch_size=2,
        max_steps=1,
        eval_interval=1,
        eval_batches=2,
        learning_rate=1e-2,
        weight_decay=0.0,
        grad_clip=1.0,
        seed=0,
    )

    original_eval_get_batch = evaluate_module.get_batch
    original_train_get_batch = train_module.get_batch

    def run_with_seed(seed: int, noise_seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        captured: dict[str, torch.Tensor] = {}

        def record_first(kind: str, original_get_batch):
            def wrapper(
                batch_tokens: torch.Tensor,
                block_size: int,
                batch_size: int,
                batch_device: torch.device,
            ) -> tuple[torch.Tensor, torch.Tensor]:
                x, y = original_get_batch(batch_tokens, block_size, batch_size, batch_device)
                captured.setdefault(kind, x.cpu().clone())
                return x, y

            return wrapper

        monkeypatch.setattr(train_module, "get_batch", record_first("train", original_train_get_batch))
        monkeypatch.setattr(evaluate_module, "get_batch", record_first("eval", original_eval_get_batch))

        torch.manual_seed(0)
        model = TinyGPT(cfg)
        torch.manual_seed(noise_seed)
        seeded_cfg = replace(train_cfg, seed=seed)
        train_steps(model, tokens, tokens, seeded_cfg, device)
        return captured["eval"], captured["train"]

    same_eval_a, same_train_a = run_with_seed(seed=11, noise_seed=101)
    same_eval_b, same_train_b = run_with_seed(seed=11, noise_seed=909)
    diff_eval, diff_train = run_with_seed(seed=29, noise_seed=101)

    assert torch.equal(same_eval_a, same_eval_b)
    assert torch.equal(same_train_a, same_train_b)
    assert not torch.equal(same_eval_a, diff_eval)
    assert not torch.equal(same_train_a, diff_train)
