#![cfg(feature = "autodiff")]

//! Tape FFN (Linear → GELU → Linear) vs the same ops in eager `Module` forwards.

use rustral_autodiff::Tape;
use rustral_core::{Backend, ForwardCtx, Mode, Module};
use rustral_ndarray_backend::CpuBackend;
use rustral_nn::tape::TapeModule;
use rustral_nn::tape_transformer::{TapeFeedForward, TapeTransformerEncoderConfig};

fn assert_vec_close(a: &[f32], b: &[f32], atol: f32) {
    assert_eq!(a.len(), b.len());
    for i in 0..a.len() {
        assert!((a[i] - b[i]).abs() < atol, "idx {} a={} b={}", i, a[i], b[i]);
    }
}

#[test]
fn tape_feedforward_matches_eager_linears_and_gelu() {
    let backend = CpuBackend::default();
    let ops = backend.ops();
    let cfg = TapeTransformerEncoderConfig::new(8, 2, 32);
    let ffn = TapeFeedForward::new(&backend, &cfg, 7).unwrap();

    let mut data = vec![0.0f32; 4 * 8];
    for (i, v) in data.iter_mut().enumerate() {
        *v = ((i as f32) * 0.07).sin() - 0.1;
    }
    let x_tensor = ops.tensor_from_vec(data.clone(), &[4, 8]).unwrap();

    let mut ctx = ForwardCtx::new(&backend, Mode::Inference);
    let mut tape = Tape::<CpuBackend>::new();
    let x_id = tape.watch(x_tensor.clone());
    let tape_out = ffn.forward_tape(x_id, &mut tape, &mut ctx).unwrap();
    let tape_vals = ops.tensor_to_vec(tape.value(tape_out).unwrap()).unwrap();

    let mut ctx2 = ForwardCtx::new(&backend, Mode::Inference);
    let h = ffn.fc1.forward(x_tensor, &mut ctx2).unwrap();
    let a = ops.gelu(&h).unwrap();
    let eager_out = ffn.fc2.forward(a, &mut ctx2).unwrap();
    let eager_vals = ops.tensor_to_vec(&eager_out).unwrap();

    assert_vec_close(&tape_vals, &eager_vals, 1e-5);
}
