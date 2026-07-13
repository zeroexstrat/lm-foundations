#!/usr/bin/env python3
"""Generate real data traces for the interactive textbook visualizations.

Trains the toy TinyGPT on tiny_corpus.txt (CPU, ~1 minute), then exports
everything the HTML interactives need to output/viz_data.json:

  - the real training loss curve (per-step batch loss + periodic train/val)
  - p(correct next token) per position of a probe sentence, over training
  - trained attention weights per layer/head on the probe, token-labeled
  - a real generation trace (full next-token distribution at each step)
  - a 2D PCA projection of the trained token-embedding table
  - a real weight tensor slice for the quantization explorer

Run from the repo root with the project venv:

    .venv/bin/python scripts/viz_data.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import torch
import torch.nn.functional as F

from llm_from_scratch.configs import ModelConfig, set_seed
from llm_from_scratch.data import get_batch, split_tokens
from llm_from_scratch.evaluate import estimate_loss
from llm_from_scratch.model import TinyGPT
from llm_from_scratch.tokenization import CharTokenizer

OUT_PATH = ROOT / "output" / "viz_data.json"

SEED = 123
MAX_STEPS = 400
EVAL_EVERY = 20
EVAL_BATCHES = 8
BATCH_SIZE = 16
LR = 3e-3


def r4(x):
    """Round floats (recursively) to 4 decimals to keep the JSON small."""
    if isinstance(x, float):
        return round(x, 4)
    if isinstance(x, list):
        return [r4(v) for v in x]
    return x


def capture_attention(model: TinyGPT, idx: torch.Tensor):
    """Recompute per-layer, per-head attention on `idx`.

    Replicates CausalSelfAttention.forward exactly (dropout is 0 in eval).
    Returns (weights, raw_scores) with shapes [layer][head][t][s]; raw scores
    are the UNSCALED, UNMASKED dot products q_t . k_s, so the client can
    re-apply / remove the 1/sqrt(D) temperature and the causal mask live.
    """
    model.eval()
    weights_out: list = []
    raw_out: list = []

    def hook(module, inputs, output):
        (x,) = inputs
        batch, seq_len, n_embd = x.shape
        q, k, v = module.qkv(x).split(n_embd, dim=-1)
        q = q.view(batch, seq_len, module.n_head, module.head_dim).transpose(1, 2)
        k = k.view(batch, seq_len, module.n_head, module.head_dim).transpose(1, 2)
        raw = q @ k.transpose(-2, -1)
        raw_out.append(raw[0].detach())
        scores = raw / math.sqrt(module.head_dim)
        scores = scores.masked_fill(~module.mask[:, :, :seq_len, :seq_len], float("-inf"))
        weights_out.append(F.softmax(scores, dim=-1)[0].detach())

    handles = [block.attn.register_forward_hook(hook) for block in model.blocks]
    with torch.no_grad():
        model(idx)
    for h in handles:
        h.remove()
    return [w.tolist() for w in weights_out], [r.tolist() for r in raw_out]


def capture_flow(model: TinyGPT, idx: torch.Tensor) -> list[dict]:
    """Per-stage residual-stream norms on `idx`: mean and per-token L2 norms."""
    model.eval()
    stages: list[dict] = []

    def record(name: str, x: torch.Tensor) -> None:
        norms = x[0].norm(dim=-1)
        stages.append({"name": name, "mean": norms.mean().item(),
                       "per_token": norms.tolist()})

    with torch.no_grad():
        _, seq_len = idx.shape
        positions = torch.arange(seq_len)
        x = model.token_embedding(idx) + model.position_embedding(positions)
        record("In(x) = W_E[x] + W_P", x)
        for li, block in enumerate(model.blocks):
            x = x + block.attn(block.ln_1(x))
            record(f"block {li+1}: + Attn(LN(x))", x)
            x = x + block.ff(block.ln_2(x))
            record(f"block {li+1}: + MLP(LN(x))", x)
        x = model.ln_f(x)
        record("LN_f(x)", x)
    return stages


def main() -> None:
    set_seed(SEED)
    device = torch.device("cpu")

    text = (ROOT / "data" / "tiny_corpus.txt").read_text(encoding="utf-8")
    tok = CharTokenizer.from_text(text)
    tokens = torch.tensor(tok.encode(text), dtype=torch.long)
    train_tokens, val_tokens = split_tokens(tokens, 0.9)

    cfg = ModelConfig(
        vocab_size=tok.vocab_size, block_size=16,
        n_embd=32, n_head=4, n_layer=2, dropout=0.0,
    )
    model = TinyGPT(cfg).to(device)

    # Probe sentence: the first block of the corpus.
    probe_text = text[: cfg.block_size + 1]
    probe_ids = tok.encode(probe_text)
    probe_x = torch.tensor([probe_ids[:-1]], dtype=torch.long)
    probe_y = probe_ids[1:]

    @torch.no_grad()
    def probe_p_correct() -> list[float]:
        model.eval()
        logits, _ = model(probe_x)
        probs = F.softmax(logits[0], dim=-1)
        model.train()
        return [probs[t, probe_y[t]].item() for t in range(len(probe_y))]

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    model.train()

    step_loss: list[float] = []
    eval_steps: list[int] = []
    eval_train: list[float] = []
    eval_val: list[float] = []
    p_correct_traj: list[list[float]] = []

    print(f"Training TinyGPT ({cfg.n_layer}L/{cfg.n_head}H/C={cfg.n_embd}, V={cfg.vocab_size}) "
          f"for {MAX_STEPS} steps on CPU...")
    for step in range(MAX_STEPS):
        if step % EVAL_EVERY == 0:
            metrics = estimate_loss(model, train_tokens, val_tokens,
                                    batch_size=BATCH_SIZE, eval_batches=EVAL_BATCHES, device=device)
            eval_steps.append(step)
            eval_train.append(metrics["train"])
            eval_val.append(metrics["val"])
            p_correct_traj.append(probe_p_correct())
            print(f"  step {step:4d}  train {metrics['train']:.4f}  val {metrics['val']:.4f}")

        x, y = get_batch(train_tokens, cfg.block_size, BATCH_SIZE, device)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        step_loss.append(loss.item())

    # Final eval point.
    metrics = estimate_loss(model, train_tokens, val_tokens,
                            batch_size=BATCH_SIZE, eval_batches=EVAL_BATCHES, device=device)
    eval_steps.append(MAX_STEPS)
    eval_train.append(metrics["train"])
    eval_val.append(metrics["val"])
    p_correct_traj.append(probe_p_correct())
    print(f"  final     train {metrics['train']:.4f}  val {metrics['val']:.4f}")

    # ── Attention weights + raw scores + flow norms on the probe ────
    attention, raw_scores = capture_attention(model, probe_x)
    flow = capture_flow(model, probe_x)

    # ── Generation trace: full distribution at each step ────────────
    set_seed(SEED)
    model.eval()
    gen_prompt = "The model "
    gen_ids = tok.encode(gen_prompt)
    gen_steps = []
    with torch.no_grad():
        for _ in range(14):
            ctx = torch.tensor([gen_ids[-cfg.block_size:]], dtype=torch.long)
            logits, _ = model(ctx)
            z = logits[0, -1, :]
            probs = F.softmax(z, dim=-1)
            chosen = int(torch.multinomial(probs, num_samples=1).item())
            gen_steps.append({
                "context": tok.decode(gen_ids),
                "logits": r4(z.tolist()),
                "chosen": chosen,
            })
            gen_ids.append(chosen)

    # ── Embedding PCA (token embedding = tied lm_head) ──────────────
    W_E = model.token_embedding.weight.detach()
    centered = W_E - W_E.mean(dim=0, keepdim=True)
    U, S, _ = torch.linalg.svd(centered, full_matrices=False)
    xy = (U[:, :2] * S[:2]).tolist()
    var_explained = (S**2 / (S**2).sum())[:2].tolist()

    # ── Real weight slice for the quantization explorer ─────────────
    quant_weights = model.blocks[0].attn.qkv.weight.detach().flatten()[:96].tolist()

    data = {
        "meta": {
            "seed": SEED, "max_steps": MAX_STEPS,
            "config": {"vocab_size": cfg.vocab_size, "block_size": cfg.block_size,
                       "n_embd": cfg.n_embd, "n_head": cfg.n_head, "n_layer": cfg.n_layer},
            "final_train": r4(eval_train[-1]), "final_val": r4(eval_val[-1]),
        },
        "itos": {str(i): tok.itos[i] for i in range(tok.vocab_size)},
        "loss": {
            "step_loss": r4(step_loss),
            "eval_steps": eval_steps,
            "eval_train": r4(eval_train),
            "eval_val": r4(eval_val),
        },
        "probe": {
            "text": probe_text,
            "input_ids": probe_ids[:-1],
            "target_ids": probe_y,
            "eval_steps": eval_steps,
            "p_correct": r4(p_correct_traj),
        },
        "attention": {
            "tokens": [tok.itos[i] for i in probe_ids[:-1]],
            "weights": r4(attention),          # [layer][head][t][s], softmaxed
            "raw_scores": r4(raw_scores),      # [layer][head][t][s], unscaled+unmasked q.k
            "head_dim": cfg.n_embd // cfg.n_head,
        },
        "flow": [{"name": s["name"], "mean": r4(s["mean"]), "per_token": r4(s["per_token"])}
                 for s in flow],
        "generation": {"prompt": gen_prompt, "steps": gen_steps},
        "embedding": {
            "labels": [tok.itos[i] for i in range(tok.vocab_size)],
            "xy": r4(xy),
            "var_explained": r4(var_explained),
        },
        "quant": {"weights": r4(quant_weights), "name": "blocks[0].attn.qkv.weight (first 96 values)"},
    }

    OUT_PATH.parent.mkdir(exist_ok=True)
    OUT_PATH.write_text(json.dumps(data), encoding="utf-8")
    size_kb = OUT_PATH.stat().st_size // 1024
    print(f"\n✅ Wrote {OUT_PATH} ({size_kb} KB)")
    print("   Rebuild the textbook to embed it: python3 scripts/build_interactive_textbook.py")


if __name__ == "__main__":
    main()
