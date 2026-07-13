import torch

from llm_from_scratch.configs import ModelConfig
from llm_from_scratch.functional import cross_entropy_from_logits
from llm_from_scratch.model import GPTBlock, TinyGPT, count_parameters


def test_tiny_gpt_forward_without_targets_returns_logits() -> None:
    cfg = ModelConfig(vocab_size=20, block_size=8, n_embd=16, n_head=4, n_layer=1)
    model = TinyGPT(cfg)
    idx = torch.randint(0, cfg.vocab_size, (2, 6))
    logits, loss = model(idx)
    assert logits.shape == (2, 6, cfg.vocab_size)
    assert loss is None


def test_gpt_block_preserves_shape() -> None:
    cfg = ModelConfig(vocab_size=20, block_size=8, n_embd=16, n_head=4, n_layer=1)
    block = GPTBlock(cfg)
    x = torch.randn(2, 6, cfg.n_embd)
    out = block(x)
    assert out.shape == x.shape


def test_tiny_gpt_forward_with_targets_matches_reference_cross_entropy() -> None:
    cfg = ModelConfig(vocab_size=20, block_size=8, n_embd=16, n_head=4, n_layer=1)
    model = TinyGPT(cfg)
    idx = torch.randint(0, cfg.vocab_size, (2, 6))
    targets = torch.randint(0, cfg.vocab_size, (2, 6))
    logits, loss = model(idx, targets)
    expected_loss = cross_entropy_from_logits(logits, targets)
    assert logits.shape == (2, 6, cfg.vocab_size)
    assert loss is not None
    assert loss.ndim == 0
    assert torch.allclose(loss, expected_loss)


def test_tiny_gpt_rejects_too_long_context() -> None:
    cfg = ModelConfig(vocab_size=20, block_size=4, n_embd=16, n_head=4, n_layer=1)
    model = TinyGPT(cfg)
    idx = torch.randint(0, cfg.vocab_size, (2, 5))
    try:
        model(idx)
    except ValueError as exc:
        assert "block_size" in str(exc)
    else:
        raise AssertionError("model should reject sequences longer than block_size")


def test_count_parameters_positive() -> None:
    cfg = ModelConfig(vocab_size=20, block_size=8, n_embd=16, n_head=4, n_layer=1)
    model = TinyGPT(cfg)
    assert count_parameters(model) > 0


def test_tiny_gpt_ties_token_embedding_and_lm_head_weights() -> None:
    cfg = ModelConfig(vocab_size=20, block_size=8, n_embd=16, n_head=4, n_layer=1)
    model = TinyGPT(cfg)
    assert model.lm_head.weight is model.token_embedding.weight
    assert model.lm_head.weight.data_ptr() == model.token_embedding.weight.data_ptr()
