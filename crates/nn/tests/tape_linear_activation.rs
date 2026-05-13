#![cfg(feature = "autodiff")]

use rustral_autodiff::{GradExtFromStore, Tape};
use rustral_core::{Backend, ForwardCtx, Mode};
use rustral_ndarray_backend::CpuBackend;
use rustral_nn::tape::TapeModule;
use rustral_nn::{LinearConfig, LinearGELU, LinearReLU};

#[test]
fn linear_relu_tape_fused_op_produces_weight_and_bias_grads() {
    let backend = CpuBackend::default();
    let ops = backend.ops();
    let layer = LinearReLU::new(&backend, LinearConfig::new(3, 2).with_bias(true)).unwrap();

    let x = ops.tensor_from_vec(vec![0.1, -0.2, 0.3], &[1, 3]).unwrap();
    let mut ctx = ForwardCtx::new(&backend, Mode::Train);
    let mut tape = Tape::<CpuBackend>::new();
    let x_id = tape.watch(x);

    let y_id = layer.forward_tape(x_id, &mut tape, &mut ctx).unwrap();
    let loss_id = tape.sum_all_tape(y_id, &mut ctx).unwrap();
    let param_map = tape.param_map().clone();
    let grads = tape.backward(loss_id, |data, shape| ops.tensor_from_vec(data, shape), ops).unwrap();

    let gw = layer.weight().gradient_from_store(&grads, &param_map).expect("missing weight grad");
    let gb = layer.bias().gradient_from_store(&grads, &param_map).expect("missing bias grad");

    let gwv = ops.tensor_to_vec(gw).unwrap();
    let gbv = ops.tensor_to_vec(gb).unwrap();
    assert!(gwv.iter().any(|v| v.abs() > 0.0));
    assert!(gbv.iter().any(|v| v.abs() > 0.0));
}

#[test]
fn linear_gelu_tape_fused_op_produces_weight_and_bias_grads() {
    let backend = CpuBackend::default();
    let ops = backend.ops();
    let layer = LinearGELU::new(&backend, LinearConfig::new(3, 2).with_bias(true)).unwrap();

    let x = ops.tensor_from_vec(vec![0.05, 0.0, -0.1], &[1, 3]).unwrap();
    let mut ctx = ForwardCtx::new(&backend, Mode::Train);
    let mut tape = Tape::<CpuBackend>::new();
    let x_id = tape.watch(x);

    let y_id = layer.forward_tape(x_id, &mut tape, &mut ctx).unwrap();
    let loss_id = tape.sum_all_tape(y_id, &mut ctx).unwrap();
    let param_map = tape.param_map().clone();
    let grads = tape.backward(loss_id, |data, shape| ops.tensor_from_vec(data, shape), ops).unwrap();

    let gw = layer.weight().gradient_from_store(&grads, &param_map).expect("missing weight grad");
    let gb = layer.bias().gradient_from_store(&grads, &param_map).expect("missing bias grad");

    let gwv = ops.tensor_to_vec(gw).unwrap();
    let gbv = ops.tensor_to_vec(gb).unwrap();
    assert!(gwv.iter().any(|v| v.abs() > 0.0));
    assert!(gbv.iter().any(|v| v.abs() > 0.0));
}
