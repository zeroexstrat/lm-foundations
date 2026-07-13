from types import SimpleNamespace

import torch
from torch import nn

from llm_from_scratch.configs import ModelConfig
from llm_from_scratch.generate import generate, sample_next_token, top_k_filter, top_p_filter
from llm_from_scratch.model import TinyGPT


class RecordingGeneratorModel(nn.Module):
    def __init__(self, block_size: int, vocab_size: int) -> None:
        super().__init__()
        self.config = SimpleNamespace(block_size=block_size)
        self.vocab_size = vocab_size
        self.seen_contexts: list[torch.Tensor] = []

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, None]:
        self.seen_contexts.append(idx.clone())
        logits = torch.full((idx.size(0), idx.size(1), self.vocab_size), float("-inf"))
        logits[:, :, 0] = 0.0
        return logits, None


def test_top_k_filter_keeps_only_k_logits() -> None:
    logits = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    filtered = top_k_filter(logits, k=2)
    assert torch.isneginf(filtered[0, 0])
    assert torch.isneginf(filtered[0, 1])
    assert filtered[0, 2] == 3.0
    assert filtered[0, 3] == 4.0


def test_top_p_filter_keeps_only_nucleus_tokens() -> None:
    logits = torch.tensor([[4.0, 3.0, 0.0, -10.0]])
    filtered = top_p_filter(logits, p=0.7)
    assert torch.isfinite(filtered[0, 0])
    assert torch.isneginf(filtered[0, 1])
    assert torch.isneginf(filtered[0, 2])
    assert torch.isneginf(filtered[0, 3])


def test_top_p_filter_keeps_prefix_until_probability_threshold() -> None:
    logits = torch.tensor([[2.0, 2.0, 0.0]])
    filtered = top_p_filter(logits, p=0.8)
    assert torch.isfinite(filtered[0, 0])
    assert torch.isfinite(filtered[0, 1])
    assert torch.isneginf(filtered[0, 2])


def test_sample_next_token_respects_top_k_filtering() -> None:
    logits = torch.tensor([[0.0, 8.0, 1.0]])
    next_token = sample_next_token(logits, top_k=1)
    assert next_token.shape == (1, 1)
    assert next_token.item() == 1


def test_generate_appends_tokens() -> None:
    cfg = ModelConfig(vocab_size=10, block_size=8, n_embd=16, n_head=4, n_layer=1, dropout=0.0)
    model = TinyGPT(cfg)
    idx = torch.zeros((1, 3), dtype=torch.long)
    out = generate(model, idx, max_new_tokens=2)
    assert out.shape == (1, 5)


def test_generate_crops_context_to_block_size_and_restores_training_mode() -> None:
    model = RecordingGeneratorModel(block_size=3, vocab_size=5)
    model.train()
    idx = torch.tensor([[1, 2, 3, 4, 1]])
    out = generate(model, idx, max_new_tokens=2)
    assert out.shape == (1, 7)
    assert [context.shape[1] for context in model.seen_contexts] == [3, 3]
    assert torch.equal(model.seen_contexts[0], torch.tensor([[3, 4, 1]]))
    assert torch.equal(model.seen_contexts[1], torch.tensor([[4, 1, 0]]))
    assert model.training is True


def test_generate_preserves_eval_mode() -> None:
    model = RecordingGeneratorModel(block_size=2, vocab_size=4)
    model.eval()
    generate(model, torch.tensor([[1, 2, 3]]), max_new_tokens=1)
    assert model.training is False
