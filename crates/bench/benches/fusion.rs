//! Benchmark: Fused Operations
//!
//! Compares performance of fused linear+bias+activation vs unfused operations.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use rustral_core::{Backend, ForwardCtx, Mode, Module};
use rustral_candle_backend::CandleBackend;
use rustral_nn::{Linear, LinearConfig, LinearReLU, LinearGELU};

fn bench_fusion_relu(c: &mut Criterion) {
    let backend = CandleBackend::new();
    let mut ctx = ForwardCtx::new(&backend, Mode::Inference);

    let mut group = c.benchmark_group("fusion_relu");

    let sizes = vec![(64, 128), (128, 256), (256, 512), (512, 1024)];
    let batch_size = 32;

    for &(in_dim, out_dim) in &sizes {
        let config = LinearConfig { in_dim, out_dim, bias: true };
        
        // Create layers
        let linear_relu = LinearReLU::new(&backend, config.clone()).unwrap();
        let linear = Linear::new(&backend, config.clone()).unwrap();

        let input = backend.tensor_from_vec(vec![1.0f32; batch_size * in_dim], &[batch_size, in_dim]).unwrap();

        // Benchmark fused operation
        group.bench_with_input(
            BenchmarkId::new("fused", format!("{}x{}", in_dim, out_dim)),
            &(in_dim, out_dim),
            |bencher, _| {
                bencher.iter(|| {
                    let result = linear_relu.forward(black_box(input.clone()), black_box(&mut ctx)).unwrap();
                    black_box(result);
                });
            },
        );

        // Benchmark unfused operation
        group.bench_with_input(
            BenchmarkId::new("unfused", format!("{}x{}", in_dim, out_dim)),
            &(in_dim, out_dim),
            |bencher, _| {
                bencher.iter(|| {
                    let result = linear.forward(black_box(input.clone()), black_box(&mut ctx)).unwrap();
                    let result = backend.ops().relu(black_box(&result)).unwrap();
                    black_box(result);
                });
            },
        );
    }

    group.finish();
}

fn bench_fusion_gelu(c: &mut Criterion) {
    let backend = CandleBackend::new();
    let mut ctx = ForwardCtx::new(&backend, Mode::Inference);

    let mut group = c.benchmark_group("fusion_gelu");

    let sizes = vec![(64, 128), (128, 256), (256, 512), (512, 1024)];
    let batch_size = 32;

    for &(in_dim, out_dim) in &sizes {
        let config = LinearConfig { in_dim, out_dim, bias: true };
        
        // Create layers
        let linear_gelu = LinearGELU::new(&backend, config.clone()).unwrap();
        let linear = Linear::new(&backend, config.clone()).unwrap();

        let input = backend.tensor_from_vec(vec![1.0f32; batch_size * in_dim], &[batch_size, in_dim]).unwrap();

        // Benchmark fused operation
        group.bench_with_input(
            BenchmarkId::new("fused", format!("{}x{}", in_dim, out_dim)),
            &(in_dim, out_dim),
            |bencher, _| {
                bencher.iter(|| {
                    let result = linear_gelu.forward(black_box(input.clone()), black_box(&mut ctx)).unwrap();
                    black_box(result);
                });
            },
        );

        // Benchmark unfused operation
        group.bench_with_input(
            BenchmarkId::new("unfused", format!("{}x{}", in_dim, out_dim)),
            &(in_dim, out_dim),
            |bencher, _| {
                bencher.iter(|| {
                    let result = linear.forward(black_box(input.clone()), black_box(&mut ctx)).unwrap();
                    let result = backend.ops().gelu(black_box(&result)).unwrap();
                    black_box(result);
                });
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_fusion_relu, bench_fusion_gelu);
criterion_main!(benches);
