use rustral_autodiff::{GradExtFromStore, Tape};
use rustral_core::{Backend, ForwardCtx, Mode};
use rustral_ndarray_backend::CpuBackend;
use rustral_nn::{tape::TapeModule, Linear, LinearConfig};

#[test]
fn linear_tape_produces_weight_gradients() {
    let backend = CpuBackend::default();
    let ops = backend.ops();

    // Biasless linear so gradients stay correct with the current tape integration.
    let linear = Linear::new(&backend, LinearConfig { in_dim: 3, out_dim: 2, bias: false }).unwrap();

    let x = ops.tensor_from_vec(vec![1.0, 2.0, 3.0], &[1, 3]).unwrap();

    let mut ctx = ForwardCtx::new(&backend, Mode::Train);
    let mut tape = Tape::<CpuBackend>::new();
    let x_id = tape.watch(x);

    let y_id = linear.forward_tape(x_id, &mut tape, &mut ctx).unwrap();

    // Scalar loss = sum_all(y) inside tape so gradients flow to weights.
    let loss_id = tape.sum_all_tape(y_id, &mut ctx).unwrap();

    let param_map = tape.param_map().clone();
    let make_ones = |data: Vec<f32>, shape: &[usize]| ops.tensor_from_vec(data, shape);
    let grads = tape.backward(loss_id, make_ones, ops).unwrap();

    let w = linear.weight();
    let g = w.gradient_from_store(&grads, &param_map).expect("missing weight grad");
    let g_vals = ops.tensor_to_vec(g).unwrap();

    // Gradient should not be all zeros.
    assert!(g_vals.iter().any(|v| v.abs() > 0.0));
}

#[test]
fn linear_tape_with_bias_produces_bias_gradients() {
    let backend = CpuBackend::default();
    let ops = backend.ops();

    let linear = Linear::new(&backend, LinearConfig { in_dim: 3, out_dim: 2, bias: true }).unwrap();

    let x = ops.tensor_from_vec(vec![1.0, 2.0, 3.0], &[1, 3]).unwrap();

    let mut ctx = ForwardCtx::new(&backend, Mode::Train);
    let mut tape = Tape::<CpuBackend>::new();
    let x_id = tape.watch(x);

    let y_id = linear.forward_tape(x_id, &mut tape, &mut ctx).unwrap();
    let loss_id = tape.sum_all_tape(y_id, &mut ctx).unwrap();

    let param_map = tape.param_map().clone();
    let make_ones = |data: Vec<f32>, shape: &[usize]| ops.tensor_from_vec(data, shape);
    let grads = tape.backward(loss_id, make_ones, ops).unwrap();

    let b = linear.bias().expect("bias should exist");
    let g = b.gradient_from_store(&grads, &param_map).expect("missing bias grad");
    let g_vals = ops.tensor_to_vec(g).unwrap();
    assert_eq!(g_vals.len(), 2);
    assert!(g_vals.iter().any(|v| v.abs() > 0.0));
}

