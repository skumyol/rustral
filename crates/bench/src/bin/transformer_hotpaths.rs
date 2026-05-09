//! Transformer Hotpaths Benchmark
//!
//! Standalone benchmark for transformer-specific operations and hot paths.
//! Focuses on the most critical transformer operations:
//! - Self-attention (multi-head)
//! - Feed-forward networks (linear + activation + linear)
//! - Layer normalization
//! - Residual connections
//! - Position embeddings
//!
//! This complements the general workload benchmark by providing deeper
//! insight into transformer-specific performance characteristics.

use std::env;

use rustral_bench::{samples_to_json, time_runs, Sample};
use rustral_core::{Backend, ForwardCtx, Mode, Module};
use rustral_ndarray_backend::CpuBackend;
use rustral_nn::{
    LayerNorm, LayerNormConfig, Linear, LinearConfig, MultiHeadAttention, SelfAttentionConfig,
    TransformerEncoder, TransformerEncoderConfig,
};

const BACKEND: &str = "ndarray-cpu";

fn parse_arg(args: &[String], name: &str, default: usize) -> usize {
    for w in args.windows(2) {
        if w[0] == name {
            if let Ok(v) = w[1].parse::<usize>() {
                return v;
            }
        }
    }
    default
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let repeats = parse_arg(&args, "--repeats", 10);
    let warmup = parse_arg(&args, "--warmup", 2);

    let backend = CpuBackend::default();
    let mut samples: Vec<Sample> = Vec::new();

    println!("Running transformer hotpaths benchmark...");
    println!("  Backend: {}", BACKEND);
    println!("  Repeats: {}", repeats);
    println!("  Warmup: {}", warmup);

    // Attention hotpaths
    bench_self_attention(&backend, repeats, warmup, &mut samples);
    bench_multi_head_attention(&backend, repeats, warmup, &mut samples);

    // FFN hotpaths (using linear layers)
    bench_feed_forward_gelu(&backend, repeats, warmup, &mut samples);
    bench_feed_forward_relu(&backend, repeats, warmup, &mut samples);

    // Layer normalization hotpaths
    bench_layer_norm(&backend, repeats, warmup, &mut samples);

    // Transformer layer hotpaths
    bench_transformer_encoder_layer(&backend, repeats, warmup, &mut samples);
    bench_residual_connection(&backend, repeats, warmup, &mut samples);

    // Position embedding hotpaths
    bench_position_embedding(&backend, repeats, warmup, &mut samples);

    println!("\nBenchmark complete. Outputting JSON...");
    print!("{}", samples_to_json("transformer_hotpaths", &samples));
}

/// Benchmark self-attention with various sequence lengths and dimensions
fn bench_self_attention(backend: &CpuBackend, repeats: usize, warmup: usize, out: &mut Vec<Sample>) {
    for &(seq_len, d_model) in &[(64, 128), (128, 256), (256, 512), (512, 768)] {
        let config = SelfAttentionConfig::new(d_model, 4).with_dropout(0.0);
        let attention = MultiHeadAttention::new(backend, config, 42).unwrap();

        let x = backend.tensor_from_vec(vec![0.01f32; seq_len * d_model], &[seq_len, d_model]).unwrap();

        let runs = time_runs(
            || {
                let mut ctx = ForwardCtx::new(backend, Mode::Inference);
                let _ = attention.forward(x.clone(), &mut ctx).unwrap();
            },
            warmup,
            repeats,
        );

        out.push(Sample::cpu_f32(
            "self_attention",
            BACKEND,
            vec![
                ("seq_len".into(), seq_len.to_string()),
                ("d_model".into(), d_model.to_string()),
                ("heads".into(), "4".to_string()),
            ],
            runs,
        ));
    }
}

/// Benchmark multi-head attention with different head configurations
fn bench_multi_head_attention(backend: &CpuBackend, repeats: usize, warmup: usize, out: &mut Vec<Sample>) {
    let seq_len = 128;
    let d_model = 512;

    for &num_heads in &[4, 8, 16] {
        let config = SelfAttentionConfig::new(d_model, num_heads).with_dropout(0.0);
        let attention = MultiHeadAttention::new(backend, config, 42).unwrap();

        let x = backend.tensor_from_vec(vec![0.01f32; seq_len * d_model], &[seq_len, d_model]).unwrap();

        let runs = time_runs(
            || {
                let mut ctx = ForwardCtx::new(backend, Mode::Inference);
                let _ = attention.forward(x.clone(), &mut ctx).unwrap();
            },
            warmup,
            repeats,
        );

        out.push(Sample::cpu_f32(
            "multi_head_attention",
            BACKEND,
            vec![
                ("seq_len".into(), seq_len.to_string()),
                ("d_model".into(), d_model.to_string()),
                ("heads".into(), num_heads.to_string()),
            ],
            runs,
        ));
    }
}

/// Benchmark feed-forward network with GELU activation (linear -> gelu -> linear)
fn bench_feed_forward_gelu(backend: &CpuBackend, repeats: usize, warmup: usize, out: &mut Vec<Sample>) {
    let batch = 8;
    let seq_len = 128;
    let d_model = 512;

    for &d_ff in &[1024, 2048, 4096] {
        let config1 = LinearConfig::new(d_model, d_ff).with_bias(true);
        let config2 = LinearConfig::new(d_ff, d_model).with_bias(true);
        let linear1 = Linear::new(backend, config1).unwrap();
        let linear2 = Linear::new(backend, config2).unwrap();

        let x = backend.tensor_from_vec(vec![0.01f32; batch * seq_len * d_model], &[batch, seq_len, d_model]).unwrap();

        let runs = time_runs(
            || {
                let mut ctx = ForwardCtx::new(backend, Mode::Inference);
                let hidden = linear1.forward(x.clone(), &mut ctx).unwrap();
                let activated = backend.ops().gelu(&hidden).unwrap();
                let _ = linear2.forward(activated, &mut ctx).unwrap();
            },
            warmup,
            repeats,
        );

        out.push(Sample::cpu_f32(
            "feed_forward_gelu",
            BACKEND,
            vec![
                ("batch".into(), batch.to_string()),
                ("seq_len".into(), seq_len.to_string()),
                ("d_model".into(), d_model.to_string()),
                ("d_ff".into(), d_ff.to_string()),
            ],
            runs,
        ));
    }
}

/// Benchmark feed-forward network with ReLU activation
fn bench_feed_forward_relu(backend: &CpuBackend, repeats: usize, warmup: usize, out: &mut Vec<Sample>) {
    let batch = 8;
    let seq_len = 128;
    let d_model = 512;

    for &d_ff in &[1024, 2048, 4096] {
        let config1 = LinearConfig::new(d_model, d_ff).with_bias(true);
        let config2 = LinearConfig::new(d_ff, d_model).with_bias(true);
        let linear1 = Linear::new(backend, config1).unwrap();
        let linear2 = Linear::new(backend, config2).unwrap();

        let x = backend.tensor_from_vec(vec![0.01f32; batch * seq_len * d_model], &[batch, seq_len, d_model]).unwrap();

        let runs = time_runs(
            || {
                let mut ctx = ForwardCtx::new(backend, Mode::Inference);
                let hidden = linear1.forward(x.clone(), &mut ctx).unwrap();
                let activated = backend.ops().relu(&hidden).unwrap();
                let _ = linear2.forward(activated, &mut ctx).unwrap();
            },
            warmup,
            repeats,
        );

        out.push(Sample::cpu_f32(
            "feed_forward_relu",
            BACKEND,
            vec![
                ("batch".into(), batch.to_string()),
                ("seq_len".into(), seq_len.to_string()),
                ("d_model".into(), d_model.to_string()),
                ("d_ff".into(), d_ff.to_string()),
            ],
            runs,
        ));
    }
}

/// Benchmark layer normalization
fn bench_layer_norm(backend: &CpuBackend, repeats: usize, warmup: usize, out: &mut Vec<Sample>) {
    for &(batch, seq_len, d_model) in &[(4, 64, 256), (8, 128, 512), (16, 256, 768)] {
        let config = LayerNormConfig::new(vec![d_model]);
        let layer_norm = LayerNorm::new(backend, config, 42).unwrap();

        let x = backend.tensor_from_vec(vec![0.01f32; batch * seq_len * d_model], &[batch, seq_len, d_model]).unwrap();

        let runs = time_runs(
            || {
                let mut ctx = ForwardCtx::new(backend, Mode::Inference);
                let _ = layer_norm.forward(x.clone(), &mut ctx).unwrap();
            },
            warmup,
            repeats,
        );

        out.push(Sample::cpu_f32(
            "layer_norm",
            BACKEND,
            vec![
                ("batch".into(), batch.to_string()),
                ("seq_len".into(), seq_len.to_string()),
                ("d_model".into(), d_model.to_string()),
            ],
            runs,
        ));
    }
}

/// Benchmark complete transformer encoder layer
fn bench_transformer_encoder_layer(backend: &CpuBackend, repeats: usize, warmup: usize, out: &mut Vec<Sample>) {
    for &(batch, seq_len, d_model, num_heads, d_ff) in &[
        (4, 64, 256, 4, 1024),
        (8, 128, 512, 8, 2048),
        (16, 256, 768, 12, 3072),
    ] {
        let config = TransformerEncoderConfig::new(d_model, num_heads, d_ff, 6).with_dropout(0.1);
        let encoder = TransformerEncoder::new(backend, config, 1000, 42).unwrap();

        let input_positions: Vec<usize> = (0..seq_len).collect();

        let runs = time_runs(
            || {
                let mut ctx = ForwardCtx::new(backend, Mode::Inference);
                let _ = encoder.forward(input_positions.clone(), &mut ctx).unwrap();
            },
            warmup,
            repeats,
        );

        out.push(Sample::cpu_f32(
            "transformer_encoder_layer",
            BACKEND,
            vec![
                ("batch".into(), batch.to_string()),
                ("seq_len".into(), seq_len.to_string()),
                ("d_model".into(), d_model.to_string()),
                ("heads".into(), num_heads.to_string()),
                ("d_ff".into(), d_ff.to_string()),
                ("layers".into(), "6".to_string()),
            ],
            runs,
        ));
    }
}

/// Benchmark residual connection (add + layer norm)
fn bench_residual_connection(backend: &CpuBackend, repeats: usize, warmup: usize, out: &mut Vec<Sample>) {
    let batch = 8;
    let seq_len = 128;
    let d_model = 512;

    let config = LayerNormConfig::new(vec![d_model]);
    let layer_norm = LayerNorm::new(backend, config, 42).unwrap();

    let x = backend.tensor_from_vec(vec![0.01f32; batch * seq_len * d_model], &[batch, seq_len, d_model]).unwrap();
    let residual = backend.tensor_from_vec(vec![0.02f32; batch * seq_len * d_model], &[batch, seq_len, d_model]).unwrap();

    let runs = time_runs(
        || {
            let added = backend.ops().add(&x, &residual).unwrap();
            let mut ctx = ForwardCtx::new(backend, Mode::Inference);
            let _ = layer_norm.forward(added, &mut ctx).unwrap();
        },
        warmup,
        repeats,
    );

    out.push(Sample::cpu_f32(
        "residual_connection",
        BACKEND,
        vec![
            ("batch".into(), batch.to_string()),
            ("seq_len".into(), seq_len.to_string()),
            ("d_model".into(), d_model.to_string()),
        ],
        runs,
    ));
}

/// Benchmark position embedding lookup
fn bench_position_embedding(backend: &CpuBackend, repeats: usize, warmup: usize, out: &mut Vec<Sample>) {
    for &(seq_len, d_model) in &[(64, 256), (128, 512), (256, 768), (512, 1024)] {
        let max_seq_len = 2048;

        let runs = time_runs(
            || {
                // Simulate position embedding lookup
                let mut embedded = vec![0.0f32; seq_len * d_model];
                for i in 0..seq_len {
                    for j in 0..d_model {
                        embedded[i * d_model + j] = ((i as f32) * (j as f32) * 0.001).sin();
                    }
                }
                let _ = backend.tensor_from_vec(embedded, &[seq_len, d_model]).unwrap();
            },
            warmup,
            repeats,
        );

        out.push(Sample::cpu_f32(
            "position_embedding",
            BACKEND,
            vec![
                ("seq_len".into(), seq_len.to_string()),
                ("d_model".into(), d_model.to_string()),
                ("max_seq_len".into(), max_seq_len.to_string()),
            ],
            runs,
        ));
    }
}
