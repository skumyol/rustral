#!/usr/bin/env python3
"""
PyTorch SST-2 baseline that mirrors the Rustral transformer architecture.

Model:
  token embedding + learned positional embedding
  2x pre-LN transformer encoder layer (MHA + FFN ReLU)
  mean pool over sequence
  linear head to 2 classes

Tokenization:
  word-level whitespace split, lowercased
  vocabulary loaded from a Rustral-produced vocab.txt (token -> id)

Dataset:
  HuggingFace SetFit/sst2 JSONL mirror (train/dev).

Outputs:
  benchmarks/runs/v0.1.0/nlp/sst2_pytorch.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import statistics
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[2]
VERSION = "0.1.0"

SST2_TRAIN_URL = "https://huggingface.co/datasets/SetFit/sst2/resolve/main/train.jsonl"
SST2_TRAIN_SHA256 = "7a4b1cfdd65be1dc48339404db86528bb2427e1d8772860ef838b76b8c38c4a8"
SST2_DEV_URL = "https://huggingface.co/datasets/SetFit/sst2/resolve/main/dev.jsonl"
SST2_DEV_SHA256 = "573c3ed18d96aa0a79a6e5980a544b80543317a319f18bd4f1660c16b2f6b939"


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256_hex(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(1024 * 1024)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def cache_root() -> Path:
    if os.environ.get("RUSTRAL_CACHE_DIR"):
        return Path(os.environ["RUSTRAL_CACHE_DIR"]).expanduser().resolve()
    home = os.environ.get("HOME", "")
    if home:
        return (Path(home) / ".cache" / "rustral").resolve()
    return (Path(".cache") / "rustral").resolve()


def fetch_url(url: str, expected_sha256: str, cache_subdir: str) -> Path:
    target_dir = cache_root() / "datasets" / cache_subdir
    target_dir.mkdir(parents=True, exist_ok=True)
    basename = url.rsplit("/", 1)[-1] or "download.bin"
    target = target_dir / basename

    if target.exists():
        print(f"[cache] using existing {basename}")
        actual = sha256_hex(target)
        if actual == expected_sha256:
            return target
        target.unlink()

    print(f"[download] {basename}")
    tmp = target.with_suffix(target.suffix + ".partial")
    with urllib.request.urlopen(url) as r, tmp.open("wb") as f:
        f.write(r.read())
    actual = sha256_hex(tmp)
    if actual != expected_sha256:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"checksum mismatch for {url}: expected {expected_sha256}, got {actual}")
    tmp.replace(target)
    print(f"[download] complete {basename}")
    return target


def load_vocab(vocab_path: Path) -> Tuple[Dict[str, int], List[str]]:
    tokens = vocab_path.read_text(encoding="utf-8").splitlines()
    tok_to_id = {t: i for i, t in enumerate(tokens)}
    return tok_to_id, tokens


def encode_sentence(tok_to_id: Dict[str, int], text: str, seq_len: int, pad_id: int, unk_id: int) -> List[int]:
    toks = text.lower().split()
    ids = [tok_to_id.get(t, unk_id) for t in toks]
    if len(ids) > seq_len:
        ids = ids[:seq_len]
    else:
        ids = ids + [pad_id] * (seq_len - len(ids))
    return ids


def load_sst2_jsonl(cache_subdir: str = "sst2") -> Tuple[List[Tuple[str, int]], List[Tuple[str, int]]]:
    train_path = fetch_url(SST2_TRAIN_URL, SST2_TRAIN_SHA256, cache_subdir)
    dev_path = fetch_url(SST2_DEV_URL, SST2_DEV_SHA256, cache_subdir)

    def read(path: Path) -> List[Tuple[str, int]]:
        out: List[Tuple[str, int]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            obj = json.loads(line)
            out.append((obj["text"], int(obj["label"])))
        return out

    return read(train_path), read(dev_path)


class PreLnEncoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, ffn_dim: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model, eps=1e-5)
        self.attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model, eps=1e-5)
        self.fc1 = nn.Linear(d_model, ffn_dim, bias=True)
        self.fc2 = nn.Linear(ffn_dim, d_model, bias=True)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        # x: [B, T, C]
        n1 = self.ln1(x)
        a, _ = self.attn(n1, n1, n1, attn_mask=attn_mask, need_weights=False)
        x = x + a
        n2 = self.ln2(x)
        f = self.fc2(F.relu(self.fc1(n2)))
        return x + f


class Sst2Transformer(nn.Module):
    def __init__(self, vocab_size: int, seq_len: int, d_model: int, num_heads: int, ffn_dim: int, num_layers: int):
        super().__init__()
        self.seq_len = seq_len
        self.tok_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(seq_len, d_model)
        self.layers = nn.ModuleList(
            [PreLnEncoderLayer(d_model=d_model, num_heads=num_heads, ffn_dim=ffn_dim) for _ in range(num_layers)]
        )
        self.head = nn.Linear(d_model, 2, bias=True)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        # ids: [B, T]
        b, t = ids.shape
        assert t == self.seq_len
        pos = torch.arange(t, device=ids.device).unsqueeze(0).expand(b, t)
        x = self.tok_embed(ids) + self.pos_embed(pos)  # [B, T, C]
        for layer in self.layers:
            x = layer(x, attn_mask=None)
        pooled = x.mean(dim=1)  # [B, C]
        logits = self.head(pooled)  # [B, 2]
        return logits


@dataclass(frozen=True)
class EpochStats:
    epoch: int
    mean_loss: float
    elapsed_sec: float


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_one_seed(
    seed: int,
    vocab_path: Path,
    device: str,
    seq_len: int,
    d_model: int,
    num_heads: int,
    ffn_dim: int,
    num_layers: int,
    lr: float,
    batch_size: int,
    epochs: int,
    train_examples_cap: Optional[int] = None,
) -> Dict[str, Any]:
    set_seed(seed)
    print(f"[seed {seed}] loading data")
    tok_to_id, tokens = load_vocab(vocab_path)
    pad_id = tok_to_id.get("<pad>", 0)
    unk_id = tok_to_id.get("<unk>", 1)

    train_raw, dev_raw = load_sst2_jsonl()
    if train_examples_cap is not None:
        train_raw = train_raw[:train_examples_cap]
    print(f"[seed {seed}] encoding sentences")
    train_ids = [encode_sentence(tok_to_id, s, seq_len, pad_id, unk_id) for (s, _) in train_raw]
    train_y = [int(y) for (_, y) in train_raw]
    dev_ids = [encode_sentence(tok_to_id, s, seq_len, pad_id, unk_id) for (s, _) in dev_raw]
    dev_y = [int(y) for (_, y) in dev_raw]

    x_train = torch.tensor(train_ids, dtype=torch.long)
    y_train = torch.tensor(train_y, dtype=torch.long)
    x_dev = torch.tensor(dev_ids, dtype=torch.long)
    y_dev = torch.tensor(dev_y, dtype=torch.long)

    model = Sst2Transformer(
        vocab_size=len(tokens),
        seq_len=seq_len,
        d_model=d_model,
        num_heads=num_heads,
        ffn_dim=ffn_dim,
        num_layers=num_layers,
    ).to(device)

    print(f"[seed {seed}] starting training")
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    epoch_stats: List[EpochStats] = []

    t0 = time.time()
    for ep in range(epochs):
        ep_t0 = time.time()
        model.train()
        # Shuffle deterministically per epoch to match Rustral's trainer protocol:
        # seed ^ (epoch * 0x9E37).
        g = torch.Generator(device="cpu")
        g.manual_seed(int(seed) ^ (int(ep) * 0x9E37))
        idx = torch.randperm(x_train.shape[0], generator=g)
        x_train_shuf = x_train[idx]
        y_train_shuf = y_train[idx]

        losses: List[float] = []
        for i in range(0, x_train_shuf.shape[0], batch_size):
            xb = x_train_shuf[i : i + batch_size].to(device)
            yb = y_train_shuf[i : i + batch_size].to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu().item()))

        epoch_stats.append(EpochStats(epoch=ep, mean_loss=float(np.mean(losses)), elapsed_sec=time.time() - ep_t0))
        print(f"[seed {seed}] epoch {ep+1}/{epochs} loss: {np.mean(losses):.4f}")

    train_elapsed_sec = time.time() - t0
    print(f"[seed {seed}] training complete: {train_elapsed_sec:.1f}s")

    # Dev eval.
    model.eval()
    with torch.no_grad():
        logits = []
        for i in range(0, x_dev.shape[0], batch_size):
            xb = x_dev[i : i + batch_size].to(device)
            logits.append(model(xb).cpu())
        logits_all = torch.cat(logits, dim=0)
        dev_loss = float(F.cross_entropy(logits_all, y_dev).item())
        preds = logits_all.argmax(dim=1)
        dev_acc = float((preds == y_dev).float().mean().item())

    total_params = int(sum(p.numel() for p in model.parameters()))
    samples_per_sec = float((len(train_raw) * epochs) / max(train_elapsed_sec, 1e-9))

    return {
        "task": "sst2_classifier",
        "framework": "pytorch",
        "torch_version": torch.__version__,
        "seed": int(seed),
        "seq_len": int(seq_len),
        "d_model": int(d_model),
        "num_heads": int(num_heads),
        "ffn_dim": int(ffn_dim),
        "num_layers": int(num_layers),
        "total_params": int(total_params),
        "vocab_size": int(len(tokens)),
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "learning_rate": float(lr),
        "train_examples": int(len(train_raw)),
        "quick_mode": train_examples_cap is not None,
        "dev_examples": int(len(dev_raw)),
        "dev_loss": float(dev_loss),
        "dev_accuracy": float(dev_acc),
        "samples_per_sec": float(samples_per_sec),
        "train_elapsed_sec": float(train_elapsed_sec),
        "epoch_stats": [{"epoch": e.epoch, "mean_loss": e.mean_loss, "elapsed_sec": e.elapsed_sec} for e in epoch_stats],
        "dataset": "SST-2 (binary, HuggingFace SetFit/sst2 mirror)",
        "tokenizer": "Rustral vocab.txt + whitespace lowercased split",
    }


def mean_std(xs: List[float]) -> Tuple[float, float]:
    if not xs:
        return (0.0, 0.0)
    if len(xs) == 1:
        return (float(xs[0]), 0.0)
    return (float(statistics.mean(xs)), float(statistics.stdev(xs)))


def main() -> int:
    print("[main] starting")
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--device", default="cpu", help="cpu or mps (default: cpu)")
    ap.add_argument(
        "--vocab-path",
        default="benchmarks/data/sst2_quick_vocab.txt",
        help="path to vocab.txt (token per line, line index = token id)",
    )
    ap.add_argument("--seq-len", type=int, default=32)
    ap.add_argument("--d-model", type=int, default=64)
    ap.add_argument("--num-heads", type=int, default=4)
    ap.add_argument("--ffn-dim", type=int, default=128)
    ap.add_argument("--num-layers", type=int, default=2)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument(
        "--out-json",
        default=f"benchmarks/runs/v{VERSION}/nlp/sst2_pytorch.json",
    )
    ap.add_argument(
        "--benchmark",
        action="store_true",
        help="tiny model + 1 epoch for fast CPU runs (matches run_nlp_real.py --benchmark)",
    )
    args = ap.parse_args()

    if args.benchmark:
        args.seq_len = 16
        args.d_model = 32
        args.num_heads = 2
        args.ffn_dim = 64
        args.num_layers = 1
        args.epochs = 1

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    vocab_path = (REPO_ROOT / args.vocab_path).resolve()
    if not vocab_path.exists():
        raise SystemExit(f"missing vocab: {vocab_path}")

    device = args.device
    if device == "mps" and not torch.backends.mps.is_available():
        raise SystemExit("mps requested but not available")

    runs: List[Dict[str, Any]] = []
    train_cap: Optional[int] = 256 if args.benchmark else None
    for seed in seeds:
        print(f"[sst2 pytorch] seed={seed} device={device}")
        manifest = train_one_seed(
            seed=seed,
            vocab_path=vocab_path,
            device=device,
            seq_len=args.seq_len,
            d_model=args.d_model,
            num_heads=args.num_heads,
            ffn_dim=args.ffn_dim,
            num_layers=args.num_layers,
            lr=args.lr,
            batch_size=args.batch_size,
            epochs=args.epochs,
            train_examples_cap=train_cap,
        )
        runs.append({"seed": seed, "manifest": manifest})

    accs = [float(r["manifest"]["dev_accuracy"]) for r in runs]
    mean, std = mean_std(accs)
    obj: Dict[str, Any] = {
        "schema_version": 1,
        "created_at": now_iso(),
        "version": VERSION,
        "task": "sst2_classifier",
        "metric": "dev_accuracy",
        "aggregate": {"mean": mean, "std": std, "n": len(accs)},
        "runs": runs,
    }

    out_path = (REPO_ROOT / args.out_json).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

