from __future__ import annotations

import math

import torch

from llm_from_scratch.data import get_batch


def perplexity(loss: float) -> float:
    return math.exp(loss)


@torch.no_grad()
def estimate_loss(
    model: torch.nn.Module,
    train_tokens: torch.Tensor,
    val_tokens: torch.Tensor,
    batch_size: int,
    eval_batches: int,
    device: torch.device,
) -> dict[str, float]:
    was_training = model.training
    model.eval()

    losses: dict[str, float] = {}
    for split_name, tokens in (("train", train_tokens), ("val", val_tokens)):
        split_losses: list[float] = []
        for _ in range(eval_batches):
            x, y = get_batch(tokens, model.config.block_size, batch_size, device)
            _, loss = model(x, y)
            assert loss is not None
            split_losses.append(loss.item())
        losses[split_name] = float(sum(split_losses) / len(split_losses))

    if was_training:
        model.train()
    return losses
