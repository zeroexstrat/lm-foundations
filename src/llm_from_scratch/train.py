from __future__ import annotations

from pathlib import Path

import torch

from llm_from_scratch.configs import TrainConfig, set_seed
from llm_from_scratch.data import get_batch
from llm_from_scratch.evaluate import estimate_loss


def train_steps(
    model: torch.nn.Module,
    train_tokens: torch.Tensor,
    val_tokens: torch.Tensor,
    train_config: TrainConfig,
    device: torch.device,
) -> list[dict[str, float]]:
    set_seed(train_config.seed)
    model.to(device)
    model.train()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
    )

    history: list[dict[str, float]] = []
    for step in range(train_config.max_steps):
        if step % train_config.eval_interval == 0:
            metrics = estimate_loss(
                model,
                train_tokens,
                val_tokens,
                batch_size=train_config.batch_size,
                eval_batches=train_config.eval_batches,
                device=device,
            )
            history.append(
                {
                    "step": float(step),
                    "train": metrics["train"],
                    "val": metrics["val"],
                }
            )

        x, y = get_batch(train_tokens, model.config.block_size, train_config.batch_size, device)
        _, loss = model(x, y)
        assert loss is not None

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if train_config.grad_clip > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.grad_clip)
        optimizer.step()

    return history


def overfit_tiny_batch(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    steps: int = 80,
    lr: float = 1e-2,
) -> tuple[float, float]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()

    _, first_loss_tensor = model(x, y)
    assert first_loss_tensor is not None
    first_loss = float(first_loss_tensor.item())

    last_loss = first_loss
    for _ in range(steps):
        _, loss = model(x, y)
        assert loss is not None
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        last_loss = float(loss.item())

    return first_loss, last_loss


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    metadata: dict[str, object] | None = None,
) -> None:
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "metadata": {} if metadata is None else metadata,
    }
    torch.save(checkpoint, checkpoint_path)


def load_checkpoint(
    path: str | Path,
    map_location: str | torch.device = "cpu",
) -> dict[str, object]:
    checkpoint = torch.load(Path(path), map_location=map_location, weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError("checkpoint payload must be a dictionary")
    return checkpoint
