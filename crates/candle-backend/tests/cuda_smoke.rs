//! CUDA device smoke tests. Built only with `--features cuda`.
//!
//! ```bash
//! ./scripts/check_cuda_env.sh
//! RUSTRAL_TEST_GPU=1 cargo test -p rustral-candle-backend --features cuda --test cuda_smoke -- --nocapture
//! ```

#![cfg(feature = "cuda")]

use rustral_candle_backend::CandleBackend;
use rustral_core::Backend;

#[test]
fn cuda_device_0_initializes_when_opted_in() {
    if std::env::var("RUSTRAL_TEST_GPU").ok().as_deref() != Some("1") {
        eprintln!("skip cuda_device_0_initializes_when_opted_in (set RUSTRAL_TEST_GPU=1 for real GPU)");
        return;
    }

    let backend = CandleBackend::cuda(0).expect("CUDA device 0 init failed (driver / CUDA_VISIBLE_DEVICES?)");
    let ops = backend.ops();
    let a = ops.tensor_from_vec(vec![1.0f32, 2.0, 3.0, 4.0], &[2, 2]).expect("tensor a");
    let b = ops.tensor_from_vec(vec![5.0f32, 6.0, 7.0, 8.0], &[2, 2]).expect("tensor b");
    let c = ops.matmul(&a, &b).expect("matmul");
    let data = backend.to_vec(&c);
    assert_eq!(data, vec![19.0, 22.0, 43.0, 50.0]);

    // Sync: ensure work finished on device before test ends
    let s = ops.sum_all(&c).expect("sum");
    let _ = ops.tensor_to_vec(&s).expect("readback");
}
