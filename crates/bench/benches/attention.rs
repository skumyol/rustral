//! Benchmark: Self-Attention Forward Pass
//!
//! Tests attention mechanism performance.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use mnr_core::{Backend, ForwardCtx, Mode, Module};
use mnr_ndarray_backend::CpuBackend;
use mnr_nn::{MultiHeadAttention, SelfAttentionConfig};

fn bench_attention(c: &mut Criterion) {
    let backend = CpuBackend::default();

    let mut group = c.benchmark_group("attention");

    let configs = vec![
        ("small", 64, 4, 32),    // d_model=64, heads=4, seq=32
        ("medium", 256, 8, 128), // d_model=256, heads=8, seq=128
        ("large", 512, 8, 256),  // d_model=512, heads=8, seq=256
    ];

    for &(name, d_model, num_heads, seq_len) in &configs {
        let config = SelfAttentionConfig::new(d_model, num_heads);
        let mha = MultiHeadAttention::new(&backend, config, 42).unwrap();

        let input =
            backend.tensor_from_vec(vec![1.0f32; 1 * seq_len * d_model], &[1, seq_len, d_model]).unwrap();

        group.bench_with_input(
            BenchmarkId::new("forward", name),
            &(d_model, num_heads, seq_len),
            |bencher, _| {
                bencher.iter(|| {
                    let mut ctx = ForwardCtx::new(&backend, Mode::Inference);
                    let result = mha.forward(black_box(input.clone()), &mut ctx).unwrap();
                    black_box(result);
                });
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_attention);
criterion_main!(benches);
