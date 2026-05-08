#!/usr/bin/env python3
"""
One-to-one SST-2 parity check (single batch, single optimizer step).

Workflow:
  1) Run Rustral dump:
     cargo run --release -p rustral-runtime --features training --example sst2_classifier -- \
       --overfit-32 --num-layers 0 --parity-one-step --parity-batch 8 --parity-dump out/parity.json

  2) Run this script with torch env:
     python3 scripts/eval/sst2_parity_one_step.py out/parity.json

It checks:
  - logits (flat)
  - loss
  - gradient signs / norms (head + embeddings)
  - head parameter update after one Adam step
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _l2(x: List[float]) -> float:
    return math.sqrt(sum(float(v) * float(v) for v in x))


def _max_abs_diff(a: List[float], b: List[float]) -> float:
    n = min(len(a), len(b))
    if n == 0:
        return 0.0
    return max(abs(float(a[i]) - float(b[i])) for i in range(n))


def _reshape_logits(flat: List[float], bsz: int) -> "torch.Tensor":
    import torch  # type: ignore

    t = torch.tensor(flat, dtype=torch.float32)
    return t.view(bsz, 2)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dump", help="path to Rustral parity dump json")
    ap.add_argument("--atol", type=float, default=1e-4)
    ap.add_argument("--rtol", type=float, default=1e-4)
    args = ap.parse_args()

    dump_path = Path(args.dump)
    obj: Dict[str, Any] = json.loads(dump_path.read_text(encoding="utf-8"))

    # torch
    import torch  # type: ignore
    import torch.nn.functional as F  # type: ignore

    bsz = int(obj["batch_size"])
    seq_len = int(obj["seq_len"])
    d_model = int(obj["d_model"])
    lr = float(obj["lr"])
    vocab: List[str] = list(obj["vocab"])
    vocab_size = len(vocab)

    batch_ids = torch.tensor(obj["batch_ids"], dtype=torch.long)
    assert batch_ids.shape == (bsz, seq_len)
    batch_y = torch.tensor(obj["batch_labels"], dtype=torch.long)
    assert batch_y.shape == (bsz,)

    # Build the exact same simple model for parity phase: embeddings + mean pool + linear.
    # (Rustral dump uses whatever num_layers was passed; we only use weights provided.)
    tok_embed = torch.nn.Embedding(vocab_size, d_model)
    pos_embed = torch.nn.Embedding(seq_len, d_model)
    head = torch.nn.Linear(d_model, 2, bias=True)

    # Load weights from dump (Rustral tensors are stored flat, row-major).
    def load_emb(emb: torch.nn.Embedding, flat: List[float]) -> None:
        w = torch.tensor(flat, dtype=torch.float32).view(emb.num_embeddings, emb.embedding_dim)
        emb.weight.data.copy_(w)

    def load_linear(linear: torch.nn.Linear, w_flat: List[float], b_flat: List[float]) -> None:
        w = torch.tensor(w_flat, dtype=torch.float32).view(2, d_model)
        b = torch.tensor(b_flat, dtype=torch.float32).view(2)
        linear.weight.data.copy_(w)
        linear.bias.data.copy_(b)

    load_emb(tok_embed, obj["tok_embed_table"])
    load_emb(pos_embed, obj["pos_embed_table"])
    load_linear(head, obj["head_weight"], obj["head_bias"])

    # Forward
    pos = torch.arange(seq_len).unsqueeze(0).expand(bsz, seq_len)
    x = tok_embed(batch_ids) + pos_embed(pos)  # [B,T,C]
    pooled = x.mean(dim=1)  # [B,C]
    logits = head(pooled)  # [B,2]

    loss = F.cross_entropy(logits, batch_y, reduction="mean")

    # Compare logits/loss vs Rustral.
    rustral_logits = list(map(float, obj["logits_flat"]))
    torch_logits = logits.detach().cpu().view(-1).tolist()
    max_logit_diff = _max_abs_diff(torch_logits, rustral_logits)

    rustral_loss = float(obj["loss"])
    torch_loss = float(loss.detach().cpu().item())
    loss_diff = abs(torch_loss - rustral_loss)

    # Backward
    opt = torch.optim.Adam(
        list(tok_embed.parameters()) + list(pos_embed.parameters()) + list(head.parameters()),
        lr=lr,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
    )
    opt.zero_grad(set_to_none=True)
    logits.retain_grad()
    loss.backward()

    # Compare dL/dlogits directly (retain_grad() makes it available).
    dlogits = logits.grad.detach().cpu().view(-1).tolist()
    rustral_dlogits = list(map(float, obj["dlogits_flat"]))

    # Extract grads (flat)
    tok_g = tok_embed.weight.grad.detach().cpu().view(-1).tolist()
    pos_g = pos_embed.weight.grad.detach().cpu().view(-1).tolist()
    hw_g = head.weight.grad.detach().cpu().view(-1).tolist()
    hb_g = head.bias.grad.detach().cpu().view(-1).tolist()

    # Compare grad norms and max abs diff.
    report = {
        "logits_max_abs_diff": max_logit_diff,
        "loss_diff": loss_diff,
        "dlogits_max_abs_diff": _max_abs_diff(dlogits, rustral_dlogits),
        "tok_embed_grad_l2": _l2(tok_g),
        "pos_embed_grad_l2": _l2(pos_g),
        "head_weight_grad_l2": _l2(hw_g),
        "head_bias_grad_l2": _l2(hb_g),
        "tok_embed_grad_max_abs_diff": _max_abs_diff(tok_g, list(map(float, obj["tok_embed_grad"]))),
        "pos_embed_grad_max_abs_diff": _max_abs_diff(pos_g, list(map(float, obj["pos_embed_grad"]))),
        "head_weight_grad_max_abs_diff": _max_abs_diff(hw_g, list(map(float, obj["head_weight_grad"]))),
        "head_bias_grad_max_abs_diff": _max_abs_diff(hb_g, list(map(float, obj["head_bias_grad"]))),
    }

    # One optimizer step and compare head params after.
    opt.step()
    hw_after = head.weight.detach().cpu().view(-1).tolist()
    hb_after = head.bias.detach().cpu().view(-1).tolist()
    report["head_weight_after_max_abs_diff"] = _max_abs_diff(hw_after, list(map(float, obj["head_weight_after"])))
    report["head_bias_after_max_abs_diff"] = _max_abs_diff(hb_after, list(map(float, obj["head_bias_after"])))

    print(json.dumps(report, indent=2, sort_keys=False))

    ok = True
    # Logits/loss are the strongest early signals.
    if report["logits_max_abs_diff"] > 1e-3:
        ok = False
    if report["loss_diff"] > 1e-4:
        ok = False
    if report["head_weight_after_max_abs_diff"] > 1e-4:
        ok = False
    if report["head_bias_after_max_abs_diff"] > 1e-5:
        ok = False

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

