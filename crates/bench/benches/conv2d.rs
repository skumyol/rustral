//! Benchmark: 2D Convolution
//!
//! Tests Conv2d forward pass performance.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use mnr_core::{Backend, ForwardCtx, Mode, Module};
use mnr_ndarray_backend::CpuBackend;
use mnr_nn::{Conv2d, Conv2dConfig};

fn bench_conv2d(c: &mut Criterion) {
    let backend = CpuBackend::default();

    let mut group = c.benchmark_group("conv2d");

    let configs = vec![
        ("small", vec![1, 6, 28, 28], vec![6, 1, 5, 5]),
        ("medium", vec![4, 16, 32, 32], vec![16, 16, 3, 3]),
        ("large", vec![8, 64, 64, 64], vec![64, 64, 3, 3]),
    ];

    for &(name, ref input_shape, ref filter_shape) in &configs {
        let input = backend
            .tensor_from_vec(vec![1.0f32; input_shape.iter().product()], &input_shape)
            .unwrap();

        let out_channels = filter_shape[0];
        let kernel_h = filter_shape[2];
        let kernel_w = filter_shape[3];

        let conv = Conv2d::new(
            &backend,
            Conv2dConfig::new(out_channels, kernel_h, kernel_w),
        )
        .unwrap();

        group.bench_with_input(
            BenchmarkId::new("forward", name),
            &name,
            |bencher, _| {
                bencher.iter(|| {
                    let mut ctx = ForwardCtx::new(&backend, Mode::Inference);
                    let result = conv.forward(black_box(input.clone()), &mut ctx).unwrap();
                    black_box(result);
                });
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_conv2d);
criterion_main!(benches);
