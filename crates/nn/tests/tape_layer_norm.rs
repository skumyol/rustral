#![cfg(feature = "autodiff")]

use rustral_autodiff::{GradExtFromStore, Tape};
use rustral_core::{Backend, ForwardCtx, Mode};
use rustral_ndarray_backend::CpuBackend;
use rustral_nn::tape::TapeModule;
use rustral_nn::{LayerNorm, LayerNormConfig};

#[test]
fn layer_norm_tape_produces_weight_and_bias_gradients() {
    let backend = CpuBackend::default();
    let ops = backend.ops();

    let ln = LayerNorm::new(&backend, LayerNormConfig::new(vec![4]).with_eps(1e-5), 7).unwrap();

    let input = backend.tensor_from_vec(vec![1.0, 2.0, 3.0, 4.0, 0.5, -0.5, 1.0, 2.0], &[2, 4]).unwrap();

    let mut ctx = ForwardCtx::new(&backend, Mode::Train);
    let mut tape = Tape::<CpuBackend>::new();

    let input_id = tape.watch(input);
    let out_id = ln.forward_tape(input_id, &mut tape, &mut ctx).unwrap();
    let loss_id = tape.sum_all_tape(out_id, &mut ctx).unwrap();

    let param_map = tape.param_map().clone();
    let make_ones = |data: Vec<f32>, shape: &[usize]| ops.tensor_from_vec(data, shape);
    let grads = tape.backward(loss_id, make_ones, ops).unwrap();

    let gw = ln.weight().gradient_from_store(&grads, &param_map).expect("missing layer norm weight grad");
    let gb = ln.bias().gradient_from_store(&grads, &param_map).expect("missing layer norm bias grad");

    assert!(ops.tensor_to_vec(gw).unwrap().iter().any(|v| v.abs() > 0.0));
    assert!(ops.tensor_to_vec(gb).unwrap().iter().any(|v| v.abs() > 0.0));
}
