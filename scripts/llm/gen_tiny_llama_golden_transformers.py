#!/usr/bin/env python3
"""Generate reference logits for the tiny_llama test fixture using Hugging Face Transformers.

Weights match the synthetic scheme in crates/llm/tests/llama_fixture_integration.rs.

Usage (from repo root, with torch+transformers in a venv):
  .venv-golden/bin/python scripts/llm/gen_tiny_llama_golden_transformers.py

Writes: crates/llm/tests/fixtures/tiny_llama/transformers_golden_logits.json
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from transformers import LlamaConfig, LlamaForCausalLM


def synthetic_weights_torch(model: LlamaForCausalLM, cfg: LlamaConfig) -> None:
    """Populate model weights like Rust `synthetic_meta_state_dict`."""
    d = cfg.hidden_size
    inter = cfg.intermediate_size
    vocab = cfg.vocab_size
    n_layer = cfg.num_hidden_layers
    n_kv = cfg.num_key_value_heads
    head_dim = d // cfg.num_attention_heads
    kv_dim = n_kv * head_dim

    m = model.model

    with torch.no_grad():
        m.embed_tokens.weight.copy_(
            torch.tensor(
                [(i) * 1e-5 for i in range(vocab * d)], dtype=torch.float32
            ).view(vocab, d)
        )
        m.norm.weight.copy_(torch.ones(d, dtype=torch.float32))
        model.lm_head.weight.copy_(torch.full((vocab, d), 0.01, dtype=torch.float32))

        for layer_idx in range(n_layer):
            layer = m.layers[layer_idx]
            layer.input_layernorm.weight.copy_(torch.ones(d, dtype=torch.float32))
            layer.post_attention_layernorm.weight.copy_(torch.ones(d, dtype=torch.float32))

            for proj, rows in (
                ("q_proj", d),
                ("k_proj", kv_dim),
                ("v_proj", kv_dim),
                ("o_proj", d),
            ):
                lin = getattr(layer.self_attn, proj)
                lin.weight.copy_(
                    torch.full((rows, d), 0.001, dtype=torch.float32)
                )

            layer.mlp.gate_proj.weight.copy_(
                torch.full((inter, d), 0.002, dtype=torch.float32)
            )
            layer.mlp.up_proj.weight.copy_(
                torch.full((inter, d), 0.002, dtype=torch.float32)
            )
            layer.mlp.down_proj.weight.copy_(
                torch.full((d, inter), 0.002, dtype=torch.float32)
            )


def main() -> None:
    repo = Path(__file__).resolve().parents[2]
    cfg_path = repo / "crates/llm/tests/fixtures/tiny_llama/config.json"
    out_path = repo / "crates/llm/tests/fixtures/tiny_llama/transformers_golden_logits.json"

    with cfg_path.open() as f:
        raw = json.load(f)

    cfg = LlamaConfig(
        vocab_size=raw["vocab_size"],
        hidden_size=raw["hidden_size"],
        intermediate_size=raw["intermediate_size"],
        num_hidden_layers=raw["num_hidden_layers"],
        num_attention_heads=raw["num_attention_heads"],
        num_key_value_heads=raw.get("num_key_value_heads", raw["num_attention_heads"]),
        rms_norm_eps=raw["rms_norm_eps"],
        rope_theta=raw["rope_theta"],
        max_position_embeddings=raw["max_position_embeddings"],
        tie_word_embeddings=False,
    )

    model = LlamaForCausalLM(cfg)
    synthetic_weights_torch(model, cfg)
    model.eval()

    input_ids = [3, 11, 7, 2, 19]
    ids = torch.tensor([input_ids], dtype=torch.long)
    with torch.no_grad():
        out = model(input_ids=ids)
        logits = out.logits[0].contiguous().view(-1).tolist()

    payload = {
        "schema": "rustral.llm.tiny_llama.transformers_golden.v1",
        "framework": "transformers",
        "input_ids": input_ids,
        "shape": [1, len(input_ids), cfg.vocab_size],
        "logits": logits,
    }
    out_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {out_path} ({len(logits)} floats)")


if __name__ == "__main__":
    main()
