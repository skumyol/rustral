use std::time::Instant;

fn main() {
    println!("=== Backend Benchmark: candle-core vs ndarray ===\n");

    // Test configurations
    let configs = vec![("small", 256usize, 256, 128), ("medium", 1024, 512, 256), ("large", 128, 4096, 2048)];

    for (name, batch, in_features, out_features) in configs {
        println!("--- {} ({}x{} -> {}x{}) ---", name, batch, in_features, batch, out_features);

        // ndarray backend
        {
            use rustral_core::{ForwardCtx, Mode, Module};
            use rustral_ndarray_backend::CpuBackend;
            use rustral_nn::{Linear, LinearConfig};

            let backend = CpuBackend::default();
            let linear = Linear::new(&backend, LinearConfig::new(in_features, out_features)).unwrap();
            let input =
                backend.tensor_from_vec(vec![0.5f32; batch * in_features], &[batch, in_features]).unwrap();

            let start = Instant::now();
            for _ in 0..5 {
                let mut ctx = ForwardCtx::new(&backend, Mode::Inference);
                let _ = linear.forward(input.clone(), &mut ctx).unwrap();
            }
            let elapsed = start.elapsed();
            println!("  ndarray: {:?} ({} iterations)", elapsed, 5);
        }

        // candle backend
        {
            use rustral_candle_backend::CandleBackend;
            use rustral_core::{ForwardCtx, Mode, Module};
            use rustral_nn::{Linear, LinearConfig};

            let backend = CandleBackend::cpu();
            let linear = Linear::new(&backend, LinearConfig::new(in_features, out_features)).unwrap();
            let input =
                backend.tensor_from_vec(vec![0.5f32; batch * in_features], &[batch, in_features]).unwrap();

            let start = Instant::now();
            for _ in 0..5 {
                let mut ctx = ForwardCtx::new(&backend, Mode::Inference);
                let _ = linear.forward(input.clone(), &mut ctx).unwrap();
            }
            let elapsed = start.elapsed();
            println!("  candle:  {:?} ({} iterations)", elapsed, 5);
        }
        println!();
    }
}
