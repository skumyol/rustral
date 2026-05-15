#!/usr/bin/env python3
import argparse
import json
import os
import platform
import socket
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Optional

_BENCH_ROOT = Path(__file__).resolve().parent.parent
if str(_BENCH_ROOT) not in sys.path:
    sys.path.insert(0, str(_BENCH_ROOT))
import stats_harness

SCHEMA_VERSION = "2.0.0"

@dataclass
class Token:
    text: str
    id: int
    start: int
    end: int
    pos: Optional[str] = None

@dataclass
class Sentence:
    tokens: List[Token]
    dependency_graph: Optional[Dict] = None

@dataclass
class Entity:
    start: int
    end: int
    label: str
    score: Optional[float] = None

@dataclass
class Document:
    text: str
    sentences: List[Sentence]
    entities: List[Entity]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=1)
    args = ap.parse_args()

    try:
        import torch
        import torch.nn as nn
    except ImportError:
        print("Torch not found", file=sys.stderr)
        sys.exit(2)

    device = torch.device("cpu")
    d_model = 128
    nhead = 4
    num_layers = 2
    dim_feedforward = 512

    # Simple Transformer Encoder
    encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, batch_first=True)
    transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers).to(device)
    embedding = nn.Embedding(1000, d_model).to(device)

    transformer_encoder.eval()

    samples = []

    # 1. Pipeline Benchmark
    runs = []
    for _ in range(args.warmup + args.repeats):
        t0 = time.perf_counter()

        # Symbolic overhead
        tokens = [Token("The", 0, 0, 3, "DT"), Token("model", 1, 4, 9, "NN"),
                  Token("learns", 2, 10, 16, "VBZ"), Token("quickly", 3, 17, 24, "RB"),
                  Token(".", 4, 24, 25, ".")]
        _doc = Document("The model learns quickly.", [Sentence(tokens)],
                        [Entity(4, 9, "COMP", 0.9)])

        # Model forward
        input_ids = torch.tensor([[1, 2, 3, 4, 5]], device=device)
        with torch.no_grad():
            x = embedding(input_ids)
            _y = transformer_encoder(x)

        dt = (time.perf_counter() - t0) * 1000.0
        runs.append(dt)

    runs = runs[args.warmup:]
    samples.append({
        "name": "nlp.full_pipeline",
        "backend": "pytorch-cpu",
        "device": "cpu",
        "dtype": "f32",
        "params": {"d_model": str(d_model), "seq_len": "5"},
        "runs_ms": runs,
        "mean_ms": statistics.mean(runs),
        "std_ms": statistics.stdev(runs) if len(runs) > 1 else 0.0,
        "min_ms": min(runs),
        "max_ms": max(runs),
        "p50_ms": statistics.median(runs),
    })

    out = {
        "suite": "pytorch-nlp",
        "schema_version": SCHEMA_VERSION,
        "machine": {
            "os": platform.system().lower(),
            "arch": platform.machine().lower(),
            "hostname": socket.gethostname(),
        },
        "samples": samples,
    }
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
