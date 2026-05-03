//! Benchmark: Matrix Multiplication
//!
//! Compares matmul performance across different matrix sizes.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use mnr_core::{Backend, TensorOps};
use mnr_ndarray_backend::CpuBackend;

fn bench_matmul(c: &mut Criterion) {
    let backend = CpuBackend::default();
    let ops = backend.ops();

    let mut group = c.benchmark_group("matmul");

    let sizes = vec![
        (64, 64, 64),
        (128, 128, 128),
        (256, 256, 256),
        (512, 512, 512),
        (1024, 1024, 1024),
    ];

    for &(m, k, n) in &sizes {
        let a = backend
            .tensor_from_vec(vec![1.0f32; m * k], &[m, k])
            .unwrap();
        let b = backend
            .tensor_from_vec(vec![1.0f32; k * n], &[k, n])
            .unwrap();

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}x{}x{}", m, k, n)),
            &(m, k, n),
            |bencher, _| {
                bencher.iter(|| {
                    let result = ops.matmul(black_box(&a), black_box(&b)).unwrap();
                    black_box(result);
                });
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_matmul);
criterion_main!(benches);
