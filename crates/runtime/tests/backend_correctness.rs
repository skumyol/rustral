#![cfg(feature = "training")]

use rustral_core::{Backend, ForwardCtx, Mode, Module};
use rustral_nn::{chain, LinearBuilder};
use rustral_runtime::{load_model, save_model};

/// Run the same model weights on two backends and assert outputs match.
///
/// This is the core "correctness as a feature" workflow:
/// verify on the reference backend, then deploy on an accelerated backend.
#[test]
fn outputs_match_between_ndarray_and_candle() {
    let cpu = rustral_ndarray_backend::CpuBackend::default();
    let candle = rustral_candle_backend::CandleBackend::cpu();

    // Build on the reference backend, then serialize weights.
    let model_cpu = chain(
        LinearBuilder::new(8, 16).with_bias(true).seed(1).build(&cpu).unwrap(),
        LinearBuilder::new(16, 4).with_bias(true).seed(2).build(&cpu).unwrap(),
    );
    let bytes = save_model(&model_cpu, &cpu).unwrap();

    // Build the same shape on Candle and load the reference weights.
    let mut model_candle = chain(
        LinearBuilder::new(8, 16).with_bias(true).seed(999).build(&candle).unwrap(),
        LinearBuilder::new(16, 4).with_bias(true).seed(999).build(&candle).unwrap(),
    );
    load_model(&mut model_candle, &candle, &bytes).unwrap();

    // Same input data on both backends.
    let x: Vec<f32> = (0..8).map(|i| (i as f32) * 0.01 - 0.03).collect();
    let x_cpu = cpu.tensor_from_vec(x.clone(), &[1, 8]).unwrap();
    let x_candle = candle.tensor_from_vec(x, &[1, 8]).unwrap();

    let mut ctx_cpu = ForwardCtx::new(&cpu, Mode::Inference);
    let mut ctx_candle = ForwardCtx::new(&candle, Mode::Inference);

    let y_cpu = model_cpu.forward(x_cpu, &mut ctx_cpu).unwrap();
    let y_candle = model_candle.forward(x_candle, &mut ctx_candle).unwrap();

    let y_cpu_vec = cpu.ops().tensor_to_vec(&y_cpu).unwrap();
    let y_candle_vec = candle.ops().tensor_to_vec(&y_candle).unwrap();

    assert_eq!(y_cpu_vec.len(), y_candle_vec.len());
    for (a, b) in y_cpu_vec.iter().zip(y_candle_vec.iter()) {
        let diff = (a - b).abs();
        assert!(diff <= 1e-6, "ndarray vs candle mismatch: a={a} b={b} diff={diff}");
    }
}
