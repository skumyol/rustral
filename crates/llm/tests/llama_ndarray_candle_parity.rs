//! [`LlamaDecoder`] forward on `CpuBackend` vs `CandleBackend::cpu()` with identical checkpoint weights.

use std::collections::HashMap;
use std::path::PathBuf;

use rustral_candle_backend::CandleBackend;
use rustral_core::{Backend, ForwardCtx, Mode};
use rustral_io::{MetaStateDict, TensorEntry};
use rustral_llm::{
    apply_llama_flat_map_to_decoder, build_llama_flat_map, detect_llama_state_dict_root, HfLlamaConfig,
};
use rustral_ndarray_backend::CpuBackend;
use rustral_nn::LlamaDecoder;
use safetensors::Dtype;

fn fixture_config_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny_llama").join("config.json")
}

fn tensor_entry(name: &str, shape: Vec<usize>, data: Vec<f32>) -> TensorEntry {
    let mut bytes = Vec::with_capacity(data.len() * 4);
    for x in data {
        bytes.extend_from_slice(&x.to_le_bytes());
    }
    TensorEntry { name: name.to_string(), shape, dtype: Dtype::F32, data: bytes }
}

fn synthetic_meta_state_dict(cfg: &HfLlamaConfig) -> MetaStateDict {
    let d = cfg.hidden_size;
    let inter = cfg.intermediate_size;
    let vocab = cfg.vocab_size;
    let n_layer = cfg.num_hidden_layers;
    let head_dim = d / cfg.num_attention_heads;
    let n_kv = cfg.num_key_value_heads.unwrap_or(cfg.num_attention_heads);
    let kv_dim = n_kv * head_dim;

    let mut tensors = HashMap::new();
    let root = "model";

    tensors.insert(
        format!("{root}.embed_tokens.weight"),
        tensor_entry(
            "embed_tokens.weight",
            vec![vocab, d],
            (0..vocab * d).map(|i| (i as f32) * 1e-5).collect(),
        ),
    );
    tensors.insert(format!("{root}.norm.weight"), tensor_entry("norm.weight", vec![d], vec![1.0f32; d]));
    tensors.insert(
        "lm_head.weight".to_string(),
        tensor_entry("lm_head.weight", vec![vocab, d], vec![0.01f32; vocab * d]),
    );

    for layer in 0..n_layer {
        let h = format!("{root}.layers.{layer}");
        tensors.insert(format!("{h}.input_layernorm.weight"), tensor_entry("iln", vec![d], vec![1.0f32; d]));
        tensors.insert(
            format!("{h}.post_attention_layernorm.weight"),
            tensor_entry("paln", vec![d], vec![1.0f32; d]),
        );
        for (pn, rows) in [("q_proj", d), ("k_proj", kv_dim), ("v_proj", kv_dim), ("o_proj", d)] {
            tensors.insert(
                format!("{h}.self_attn.{pn}.weight"),
                tensor_entry(pn, vec![rows, d], vec![0.001f32; rows * d]),
            );
        }
        tensors.insert(
            format!("{h}.mlp.gate_proj.weight"),
            tensor_entry("gate", vec![inter, d], vec![0.002f32; inter * d]),
        );
        tensors.insert(
            format!("{h}.mlp.up_proj.weight"),
            tensor_entry("up", vec![inter, d], vec![0.002f32; inter * d]),
        );
        tensors.insert(
            format!("{h}.mlp.down_proj.weight"),
            tensor_entry("down", vec![d, inter], vec![0.002f32; d * inter]),
        );
        tensors.insert(
            format!("{h}.self_attn.rotary_emb.inv_freq"),
            tensor_entry("inv_freq", vec![d / 2], vec![1.0f32; d / 2]),
        );
    }

    MetaStateDict { tensors }
}

fn assert_logits_close(cpu: &[f32], candle: &[f32], atol: f32) {
    assert_eq!(cpu.len(), candle.len(), "logits length mismatch");
    for (i, (a, b)) in cpu.iter().zip(candle.iter()).enumerate() {
        let d = (*a - *b).abs();
        assert!(d <= atol, "logits differ at {i}: cpu={a} candle={b} diff={d}");
    }
}

#[test]
fn llama_decoder_forward_cpu_matches_candle_cpu() {
    let cfg = HfLlamaConfig::from_json_file(fixture_config_path()).expect("fixture");
    let meta = synthetic_meta_state_dict(&cfg);
    let dec_cfg = cfg.to_decoder_config();
    let root = detect_llama_state_dict_root(&meta).expect("root");
    let (flat, _) = build_llama_flat_map(&meta, &cfg, root).expect("flat");

    let cpu_backend = CpuBackend::default();
    let candle_backend = CandleBackend::cpu();

    let mut cpu_dec =
        LlamaDecoder::new(&cpu_backend, dec_cfg.clone(), cfg.vocab_size, 42).expect("cpu decoder");
    apply_llama_flat_map_to_decoder(&mut cpu_dec, &cpu_backend, &flat).expect("apply cpu");

    let mut candle_dec =
        LlamaDecoder::new(&candle_backend, dec_cfg, cfg.vocab_size, 999).expect("candle decoder");
    apply_llama_flat_map_to_decoder(&mut candle_dec, &candle_backend, &flat).expect("apply candle");

    let input_ids: Vec<usize> = vec![3, 11, 7, 2, 19];
    let mut ctx_cpu = ForwardCtx::new(&cpu_backend, Mode::Inference);
    let mut ctx_candle = ForwardCtx::new(&candle_backend, Mode::Inference);

    let logits_cpu = cpu_dec.forward(input_ids.clone(), &mut ctx_cpu).expect("cpu");
    let logits_candle = candle_dec.forward(input_ids, &mut ctx_candle).expect("candle");

    let cpu_vec = cpu_backend.ops().tensor_to_vec(&logits_cpu).expect("cpu vec");
    let candle_vec = candle_backend.ops().tensor_to_vec(&logits_candle).expect("candle vec");

    assert_logits_close(&cpu_vec, &candle_vec, 5e-4);

    let tok_cpu = cpu_dec.generate_token(vec![3, 11, 7], &mut ctx_cpu).expect("cpu tok");
    let tok_candle = candle_dec.generate_token(vec![3, 11, 7], &mut ctx_candle).expect("candle tok");
    assert_eq!(tok_cpu, tok_candle, "greedy token should match");
}
