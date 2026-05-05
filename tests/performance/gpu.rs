use crate::common::{run_performance_test, PerfConfig, TestRunner};
use std::time::Instant;

pub fn benchmark_gpu_smoke(runner: &mut TestRunner, config: &PerfConfig) {
    runner.run_test("perf_gpu_smoke", || {
        if std::env::var("RUSTRAL_RUN_GPU_PERF").ok().as_deref() != Some("1") {
            println!("  Skipped. Set RUSTRAL_RUN_GPU_PERF=1 to run GPU benchmarks.");
            return Ok(());
        }

        #[cfg(feature = "cuda")]
        {
            use rustral_candle_backend::CandleBackend;
            use rustral_core::{Backend, TensorOps};

            let backend = CandleBackend::cuda(0).map_err(|e| format!("CUDA backend init failed: {:?}", e))?;
            let ops = backend.ops();

            // Heavy matmul workload: [batch, k] x [k, n] -> [batch, n]
            let batch = 4096usize;
            let k = 4096usize;
            let n = 4096usize;

            let a = ops
                .tensor_from_vec(vec![0.01f32; batch * k], &[batch, k])
                .map_err(|e| format!("Create A failed: {:?}", e))?;
            let b = ops
                .tensor_from_vec(vec![0.01f32; k * n], &[k, n])
                .map_err(|e| format!("Create B failed: {:?}", e))?;

            // Warmup + perf loop. Avoid host<->device transfers inside the loop.
            let mut cfg = config.clone();
            cfg.warmup_iterations = cfg.warmup_iterations.max(2);
            cfg.test_iterations = cfg.test_iterations.max(5);
            cfg.max_duration_ms = cfg.max_duration_ms.max(20_000);

            let result = run_performance_test(&cfg, || {
                let y = ops.matmul(&a, &b).unwrap();
                // Ensure the GPU work is actually completed before we stop the timer.
                // Without a sync point, CUDA matmuls can appear "too fast" due to async dispatch.
                let y_sum = ops.sum_all(&y).unwrap();
                let _ = ops.tensor_to_vec(&y_sum).unwrap();
                // Prevent the compiler from eliding the work.
                std::hint::black_box(&y);
            });

            // Approx FLOPs: 2 * batch * k * n per matmul
            let flops = 2.0f64 * batch as f64 * k as f64 * n as f64;
            let gflops = (flops / (result.mean_ms / 1000.0)) / 1e9;

            println!(
                "  GPU matmul ({}x{} @ {}x{}): mean={:.2}ms, ~{:.1} GFLOP/s",
                batch, k, k, n, result.mean_ms, gflops
            );

            // Soft sanity threshold: we mainly want to ensure we're not on CPU accidentally.
            // If CUDA is actually used, this should generally be far better than a CPU fallback.
            if gflops < 50.0 {
                return Err(format!("GPU GFLOP/s too low ({:.1}); likely not utilizing GPU well", gflops));
            }
            Ok(())
        }

        #[cfg(not(feature = "cuda"))]
        {
            println!("  Skipped (built without --features cuda).");
            Ok(())
        }
    });
}

