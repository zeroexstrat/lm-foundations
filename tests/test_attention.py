import torch

from llm_from_scratch.attention import CausalSelfAttention
from llm_from_scratch.configs import ModelConfig


def test_causal_self_attention_preserves_shape() -> None:
    cfg = ModelConfig(vocab_size=16, block_size=8, n_embd=12, n_head=3, n_layer=1)
    attn = CausalSelfAttention(cfg)
    x = torch.randn(2, 5, 12)
    out = attn(x)
    assert out.shape == x.shape


def test_causal_self_attention_blocks_future_token_influence() -> None:
    cfg = ModelConfig(
        vocab_size=16,
        block_size=4,
        n_embd=2,
        n_head=1,
        n_layer=1,
        dropout=0.0,
        bias=False,
    )
    attn = CausalSelfAttention(cfg)

    with torch.no_grad():
        attn.qkv.weight.zero_()
        attn.qkv.weight[:2] = torch.eye(2)
        attn.qkv.weight[2:4] = torch.eye(2)
        attn.qkv.weight[4:6] = torch.eye(2)
        attn.proj.weight.copy_(torch.eye(2))

    x_base = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]])
    x_changed = x_base.clone()
    x_changed[0, 2] = torch.tensor([10.0, -10.0])

    out_base = attn(x_base)
    out_changed = attn(x_changed)

    assert torch.allclose(out_base[:, :2], out_changed[:, :2])
    assert not torch.allclose(out_base[:, 2], out_changed[:, 2])
