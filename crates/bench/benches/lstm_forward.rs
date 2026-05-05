//! Benchmark: LSTM Forward Pass
//!
//! Tests LSTM sequence processing performance.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use mnr_core::{Backend, ForwardCtx, Mode, Module, StatefulModule};
use mnr_ndarray_backend::CpuBackend;
use mnr_nn::{LstmCell, LstmConfig};

fn bench_lstm_forward(c: &mut Criterion) {
    let backend = CpuBackend::default();

    let mut group = c.benchmark_group("lstm_forward");

    let configs = vec![
        ("small", 128, 10),  // 128 hidden, 10 steps
        ("medium", 256, 50), // 256 hidden, 50 steps
        ("large", 512, 100), // 512 hidden, 100 steps
    ];

    for &(name, hidden_size, seq_len) in &configs {
        let lstm = LstmCell::new(&backend, LstmConfig::new(hidden_size)).unwrap();

        let input = backend.tensor_from_vec(vec![1.0f32; hidden_size], &[hidden_size]).unwrap();

        group.bench_with_input(
            BenchmarkId::new("sequence", name),
            &(hidden_size, seq_len),
            |bencher, &(hidden, steps)| {
                bencher.iter(|| {
                    let mut ctx = ForwardCtx::new(&backend, Mode::Inference);
                    let mut state = lstm.initial_state(&mut ctx).unwrap();

                    for _ in 0..steps {
                        let input = backend.tensor_from_vec(vec![1.0f32; hidden], &[hidden]).unwrap();
                        let result = lstm.forward((state.clone(), input), &mut ctx).unwrap();
                        state = result;
                        black_box(state.clone());
                    }
                });
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_lstm_forward);
criterion_main!(benches);
