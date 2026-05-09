#![cfg(feature = "autodiff")]

//! Tape backward tests for `TapeTransformerEncoderLayer`.
//!
//! These don't compare against finite differences (the layer is too composite for that to
//! be cheap); they just assert that every named parameter receives a non-zero gradient
//! when we backprop a sum-loss through the encoder. That is enough to catch routing bugs
//! where a parameter accidentally gets dropped from the tape graph.

use rustral_autodiff::{GradExtFromStore, ParameterMap, Tape};
use rustral_core::{Backend, ForwardCtx, Mode, NamedParameters, Parameter};
use rustral_ndarray_backend::CpuBackend;
use rustral_nn::tape::TapeModule;
use rustral_nn::tape_transformer::{
    causal_mask_tape, TapeTransformerEncoderConfig, TapeTransformerEncoderLayer,
};

fn make_input(backend: &CpuBackend, seq_len: usize, d_model: usize) -> <CpuBackend as Backend>::Tensor {
    let ops = backend.ops();
    let mut data = vec![0.0_f32; seq_len * d_model];
    for (i, v) in data.iter_mut().enumerate() {
        *v = ((i as f32) * 0.13).sin() + 0.05;
    }
    ops.tensor_from_vec(data, &[seq_len, d_model]).unwrap()
}

fn collect_named_params<B: Backend, M: NamedParameters<B>>(model: &M) -> Vec<(String, &Parameter<B>)> {
    let mut out: Vec<(String, &Parameter<B>)> = Vec::new();
    model.visit_parameters(&mut |name, p| {
        out.push((name.to_string(), unsafe {
            // Lifetime-extending the &Parameter<B> reference is safe here because `model` outlives
            // `out`. The visitor closure borrows for the duration of the closure call only, so we
            // re-borrow through a raw pointer.
            &*(p as *const _)
        }))
    });
    out
}

fn assert_all_params_have_nonzero_grads<B: Backend, M: NamedParameters<B>>(
    backend: &B,
    model: &M,
    grads: &std::collections::HashMap<rustral_autodiff::TensorId, B::Tensor>,
    param_map: &ParameterMap,
) {
    let mut zero_or_missing: Vec<String> = Vec::new();
    let names = collect_named_params::<B, M>(model);
    for (name, param) in names {
        match param.gradient_from_store(grads, param_map) {
            Some(g) => {
                let v = backend.ops().tensor_to_vec(g).unwrap();
                let any_nonzero = v.iter().any(|x| x.abs() > 1e-12);
                if !any_nonzero {
                    zero_or_missing.push(format!("{name} (all zeros, len={})", v.len()));
                }
            }
            None => zero_or_missing.push(format!("{name} (missing from grad store)")),
        }
    }
    assert!(zero_or_missing.is_empty(), "parameters with zero/missing gradients: {:?}", zero_or_missing);
}

#[test]
fn encoder_layer_unmasked_backward_touches_every_parameter() {
    let backend = CpuBackend::default();
    let cfg = TapeTransformerEncoderConfig::new(16, 4, 32);
    let layer = TapeTransformerEncoderLayer::new(&backend, cfg, 11).unwrap();

    let mut ctx = ForwardCtx::new(&backend, Mode::Train);
    let mut tape = Tape::<CpuBackend>::new();
    let x = tape.watch(make_input(&backend, 6, 16));

    let out = layer.forward_tape(x, &mut tape, &mut ctx).unwrap();
    let loss = tape.sum_all_tape(out, &mut ctx).unwrap();

    let param_map = tape.param_map().clone();
    let ops = backend.ops();
    let make_ones = |data: Vec<f32>, shape: &[usize]| ops.tensor_from_vec(data, shape);
    let grads = tape.backward(loss, make_ones, ops).unwrap();

    assert_all_params_have_nonzero_grads(&backend, &layer, &grads, &param_map);
}

#[test]
fn encoder_layer_with_causal_mask_backward_touches_every_parameter() {
    let backend = CpuBackend::default();
    let cfg = TapeTransformerEncoderConfig::new(16, 4, 32);
    let layer = TapeTransformerEncoderLayer::new(&backend, cfg, 13).unwrap();

    let seq_len = 5_usize;
    let mut ctx = ForwardCtx::new(&backend, Mode::Train);
    let mut tape = Tape::<CpuBackend>::new();
    let x = tape.watch(make_input(&backend, seq_len, 16));
    let mask = causal_mask_tape::<CpuBackend>(seq_len, &mut tape, &mut ctx).unwrap();

    let out = layer.forward_tape_with_mask(x, Some(mask), &mut tape, &mut ctx).unwrap();
    let loss = tape.sum_all_tape(out, &mut ctx).unwrap();

    let param_map = tape.param_map().clone();
    let ops = backend.ops();
    let make_ones = |data: Vec<f32>, shape: &[usize]| ops.tensor_from_vec(data, shape);
    let grads = tape.backward(loss, make_ones, ops).unwrap();

    assert_all_params_have_nonzero_grads(&backend, &layer, &grads, &param_map);
}

#[test]
fn encoder_layer_single_head_works() {
    // d_model == num_heads * d_head, with num_heads = 1, exercises the single-head fast path.
    let backend = CpuBackend::default();
    let cfg = TapeTransformerEncoderConfig::new(8, 1, 16);
    let layer = TapeTransformerEncoderLayer::new(&backend, cfg, 17).unwrap();

    let mut ctx = ForwardCtx::new(&backend, Mode::Train);
    let mut tape = Tape::<CpuBackend>::new();
    let x = tape.watch(make_input(&backend, 4, 8));
    let out = layer.forward_tape(x, &mut tape, &mut ctx).unwrap();
    let loss = tape.sum_all_tape(out, &mut ctx).unwrap();

    let param_map = tape.param_map().clone();
    let ops = backend.ops();
    let make_ones = |data: Vec<f32>, shape: &[usize]| ops.tensor_from_vec(data, shape);
    let grads = tape.backward(loss, make_ones, ops).unwrap();

    assert_all_params_have_nonzero_grads(&backend, &layer, &grads, &param_map);
}
