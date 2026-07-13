from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from llm_from_scratch.configs import ModelConfig
from llm_from_scratch.functional import causal_mask


class CausalSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        if config.n_embd % config.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")

        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

        mask = causal_mask(config.block_size)
        self.register_buffer("mask", mask.view(1, 1, config.block_size, config.block_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, n_embd = x.shape
        q, k, v = self.qkv(x).split(n_embd, dim=-1)

        q = q.view(batch, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq_len, self.n_head, self.head_dim).transpose(1, 2)

        scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(~self.mask[:, :, :seq_len, :seq_len], float("-inf"))
        weights = F.softmax(scores, dim=-1)
        weights = self.dropout(weights)

        out = weights @ v
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, n_embd)
        return self.proj(out)
