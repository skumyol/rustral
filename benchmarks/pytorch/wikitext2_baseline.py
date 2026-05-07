#!/usr/bin/env python3
"""
PyTorch WikiText-2 baseline that mirrors the Rustral transformer LM architecture.

Model:
  token embedding + learned positional embedding
  2x pre-LN transformer encoder layer with causal additive mask
  take last position hidden state, project to vocab logits

Tokenization:
  word-level whitespace split, lowercased
  vocabulary loaded from a Rustral-produced vocab.txt (token -> id)

Dataset:
  WikiText-2 raw v1 zip (smerity.com mirror).

Outputs:
  benchmarks/runs/v0.1.0/nlp/wikitext2_pytorch.json
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
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[2]
VERSION = "0.1.0"

WIKITEXT2_RAW_URL = "https://wikitext.smerity.com/wikitext-2-raw-v1.zip"
WIKITEXT2_RAW_SHA256 = "ef7edb566e3e2b2d31b29c1fdb0c89a4cc683597484c3dc2517919c615435a11"


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
        actual = sha256_hex(target)
        if actual == expected_sha256:
            return target
        target.unlink()

    tmp = target.with_suffix(target.suffix + ".partial")
    with urllib.request.urlopen(url) as r, tmp.open("wb") as f:
        f.write(r.read())
    actual = sha256_hex(tmp)
    if actual != expected_sha256:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"checksum mismatch for {url}: expected {expected_sha256}, got {actual}")
    tmp.replace(target)
    return target


def extract_splits(zip_path: Path) -> Tuple[str, str, str]:
    with zipfile.ZipFile(zip_path, "r") as z:
        # Mirror may contain either wikitext-2-raw/ or wikitext-2-raw-v1/ dir.
        candidates = ["wikitext-2-raw", "wikitext-2-raw-v1"]
        base = None
        for c in candidates:
            name = f"{c}/wiki.train.raw"
            if name in z.namelist():
                base = c
                break
        if base is None:
            raise RuntimeError("zip did not contain wiki.train.raw under expected directory")
        train = z.read(f"{base}/wiki.train.raw").decode("utf-8", errors="replace")
        valid = z.read(f"{base}/wiki.valid.raw").decode("utf-8", errors="replace")
        test = z.read(f"{base}/wiki.test.raw").decode("utf-8", errors="replace")
        return train, valid, test


def load_vocab(vocab_path: Path) -> Tuple[Dict[str, int], List[str]]:
    tokens = vocab_path.read_text(encoding="utf-8").splitlines()
    tok_to_id = {t: i for i, t in enumerate(tokens)}
    return tok_to_id, tokens


def encode_text(tok_to_id: Dict[str, int], text: str, unk_id: int) -> List[int]:
    toks = text.lower().split()
    return [tok_to_id.get(t, unk_id) for t in toks]


def build_windows(ids: List[int], block_size: int) -> Tuple[np.ndarray, np.ndarray]:
    # Returns X: [N, block], Y: [N]
    if len(ids) <= block_size:
        return np.zeros((0, block_size), dtype=np.int64), np.zeros((0,), dtype=np.int64)
    n = len(ids) - block_size
    x = np.empty((n, block_size), dtype=np.int64)
    y = np.empty((n,), dtype=np.int64)
    for i in range(n):
        x[i, :] = ids[i : i + block_size]
        y[i] = ids[i + block_size]
    return x, y


class PreLnEncoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, ffn_dim: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model, eps=1e-5)
        self.attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model, eps=1e-5)
        self.fc1 = nn.Linear(d_model, ffn_dim, bias=True)
        self.fc2 = nn.Linear(ffn_dim, d_model, bias=True)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        n1 = self.ln1(x)
        a, _ = self.attn(n1, n1, n1, attn_mask=attn_mask, need_weights=False)
        x = x + a
        n2 = self.ln2(x)
        f = self.fc2(F.relu(self.fc1(n2)))
        return x + f


class WikiTextTransformerLm(nn.Module):
    def __init__(self, vocab_size: int, block_size: int, d_model: int, num_heads: int, ffn_dim: int, num_layers: int):
        super().__init__()
        self.block_size = block_size
        self.tok_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(block_size, d_model)
        self.layers = nn.ModuleList(
            [PreLnEncoderLayer(d_model=d_model, num_heads=num_heads, ffn_dim=ffn_dim) for _ in range(num_layers)]
        )
        self.head = nn.Linear(d_model, vocab_size, bias=True)

        # Causal mask: True/1 means disallow in PyTorch additive mask form -> use float -inf.
        # MultiheadAttention expects attn_mask shape [T, T] with float -inf for masked positions.
        mask = torch.triu(torch.ones(block_size, block_size), diagonal=1).bool()
        attn = torch.zeros(block_size, block_size, dtype=torch.float32)
        attn[mask] = -1.0e9
        self.register_buffer("causal_mask", attn, persistent=False)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        # ids: [B, T]
        b, t = ids.shape
        assert t == self.block_size
        pos = torch.arange(t, device=ids.device).unsqueeze(0).expand(b, t)
        x = self.tok_embed(ids) + self.pos_embed(pos)  # [B, T, C]
        for layer in self.layers:
            x = layer(x, attn_mask=self.causal_mask)
        last = x[:, -1, :]  # [B, C]
        return self.head(last)  # [B, V]


@dataclass(frozen=True)
class EpochStats:
    epoch: int
    mean_loss: float
    elapsed_sec: float


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def mean_std(xs: List[float]) -> Tuple[float, float]:
    if not xs:
        return (0.0, 0.0)
    if len(xs) == 1:
        return (float(xs[0]), 0.0)
    return (float(statistics.mean(xs)), float(statistics.stdev(xs)))


def train_one_seed(
    seed: int,
    vocab_path: Path,
    device: str,
    block_size: int,
    d_model: int,
    num_heads: int,
    ffn_dim: int,
    num_layers: int,
    lr: float,
    batch_size: int,
    epochs: int,
    train_tokens_cap: int,
    train_windows_cap: int,
    eval_windows_cap: int,
) -> Dict[str, Any]:
    set_seed(seed)
    tok_to_id, tokens = load_vocab(vocab_path)
    unk_id = tok_to_id.get("<unk>", 1)

    zip_path = fetch_url(WIKITEXT2_RAW_URL, WIKITEXT2_RAW_SHA256, "wikitext-2")
    train_text, valid_text, _test_text = extract_splits(zip_path)

    train_ids = encode_text(tok_to_id, train_text, unk_id)
    valid_ids = encode_text(tok_to_id, valid_text, unk_id)
    if len(train_ids) > train_tokens_cap:
        train_ids = train_ids[:train_tokens_cap]

    x_train_np, y_train_np = build_windows(train_ids, block_size)
    x_valid_np, y_valid_np = build_windows(valid_ids, block_size)

    if train_windows_cap > 0:
        n = min(train_windows_cap, x_train_np.shape[0])
        x_train_np = x_train_np[:n]
        y_train_np = y_train_np[:n]

    if eval_windows_cap > 0:
        n = min(eval_windows_cap, x_valid_np.shape[0])
        x_valid_np = x_valid_np[:n]
        y_valid_np = y_valid_np[:n]

    x_train = torch.from_numpy(x_train_np).long()
    y_train = torch.from_numpy(y_train_np).long()
    x_valid = torch.from_numpy(x_valid_np).long()
    y_valid = torch.from_numpy(y_valid_np).long()

    model = WikiTextTransformerLm(
        vocab_size=len(tokens),
        block_size=block_size,
        d_model=d_model,
        num_heads=num_heads,
        ffn_dim=ffn_dim,
        num_layers=num_layers,
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    epoch_stats: List[EpochStats] = []

    t0 = time.time()
    for ep in range(epochs):
        ep_t0 = time.time()
        model.train()
        idx = torch.randperm(x_train.shape[0]) if x_train.shape[0] > 0 else torch.arange(0)
        x_shuf = x_train[idx]
        y_shuf = y_train[idx]
        losses: List[float] = []
        for i in range(0, x_shuf.shape[0], batch_size):
            xb = x_shuf[i : i + batch_size].to(device)
            yb = y_shuf[i : i + batch_size].to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu().item()))
        epoch_stats.append(EpochStats(ep, float(np.mean(losses)) if losses else 0.0, time.time() - ep_t0))

    train_elapsed_sec = time.time() - t0
    windows_per_sec = float((max(1, x_train.shape[0]) * max(1, epochs)) / max(train_elapsed_sec, 1e-9))

    # Dev eval.
    model.eval()
    with torch.no_grad():
        losses: List[float] = []
        for i in range(0, x_valid.shape[0], batch_size):
            xb = x_valid[i : i + batch_size].to(device)
            yb = y_valid[i : i + batch_size].to(device)
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            losses.append(float(loss.detach().cpu().item()))
        dev_loss = float(np.mean(losses)) if losses else 0.0
        dev_ppl = float(np.exp(dev_loss)) if dev_loss > 0 else 1.0

    total_params = int(sum(p.numel() for p in model.parameters()))

    return {
        "task": "wikitext2_word_lm",
        "framework": "pytorch",
        "torch_version": torch.__version__,
        "seed": int(seed),
        "block_size": int(block_size),
        "d_model": int(d_model),
        "num_heads": int(num_heads),
        "ffn_dim": int(ffn_dim),
        "num_layers": int(num_layers),
        "total_params": int(total_params),
        "vocab_size": int(len(tokens)),
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "learning_rate": float(lr),
        "train_tokens_used": int(min(len(encode_text(tok_to_id, train_text, unk_id)), train_tokens_cap)),
        "train_windows_used": int(x_train.shape[0]),
        "eval_windows_used": int(x_valid.shape[0]),
        "dev_loss_nats": float(dev_loss),
        "dev_perplexity": float(dev_ppl),
        "windows_per_sec": float(windows_per_sec),
        "train_elapsed_sec": float(train_elapsed_sec),
        "epoch_stats": [{"epoch": e.epoch, "mean_loss": e.mean_loss, "elapsed_sec": e.elapsed_sec} for e in epoch_stats],
        "dataset": "WikiText-2 raw v1 (smerity.com mirror)",
        "tokenizer": "Rustral vocab.txt + whitespace lowercased split",
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--device", default="cpu", help="cpu or mps (default: cpu)")
    ap.add_argument(
        "--vocab-path",
        default="out/nlp_real/wikitext2/seed_0/vocab.txt",
        help="path to Rustral-produced vocab.txt",
    )
    ap.add_argument("--block-size", type=int, default=32)
    ap.add_argument("--d-model", type=int, default=64)
    ap.add_argument("--num-heads", type=int, default=4)
    ap.add_argument("--ffn-dim", type=int, default=128)
    ap.add_argument("--num-layers", type=int, default=2)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--train-tokens", type=int, default=50_000)
    ap.add_argument("--train-windows", type=int, default=2_000)
    ap.add_argument("--eval-windows", type=int, default=20_000)
    ap.add_argument(
        "--out-json",
        default=f"benchmarks/runs/v{VERSION}/nlp/wikitext2_pytorch.json",
    )
    args = ap.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    vocab_path = (REPO_ROOT / args.vocab_path).resolve()
    if not vocab_path.exists():
        raise SystemExit(f"missing vocab: {vocab_path}")

    device = args.device
    if device == "mps" and not torch.backends.mps.is_available():
        raise SystemExit("mps requested but not available")

    runs: List[Dict[str, Any]] = []
    for seed in seeds:
        print(f"[wikitext2 pytorch] seed={seed} device={device}")
        manifest = train_one_seed(
            seed=seed,
            vocab_path=vocab_path,
            device=device,
            block_size=args.block_size,
            d_model=args.d_model,
            num_heads=args.num_heads,
            ffn_dim=args.ffn_dim,
            num_layers=args.num_layers,
            lr=args.lr,
            batch_size=args.batch_size,
            epochs=args.epochs,
            train_tokens_cap=args.train_tokens,
            train_windows_cap=args.train_windows,
            eval_windows_cap=args.eval_windows,
        )
        runs.append({"seed": seed, "manifest": manifest})

    ppls = [float(r["manifest"]["dev_perplexity"]) for r in runs]
    mean, std = mean_std(ppls)
    obj: Dict[str, Any] = {
        "schema_version": 1,
        "created_at": now_iso(),
        "version": VERSION,
        "task": "wikitext2_word_lm",
        "metric": "dev_perplexity",
        "aggregate": {"mean": mean, "std": std, "n": len(ppls)},
        "runs": runs,
    }

    out_path = (REPO_ROOT / args.out_json).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

