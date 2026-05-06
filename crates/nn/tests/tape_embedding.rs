#![cfg(feature = "autodiff")]

use rustral_autodiff::{GradExtFromStore, Tape};
use rustral_core::{Backend, ForwardCtx, Mode};
use rustral_ndarray_backend::CpuBackend;
use rustral_nn::tape::TapeModule;
use rustral_nn::{Embedding, EmbeddingConfig};

#[test]
fn embedding_tape_produces_table_gradients() {
    let backend = CpuBackend::default();
    let ops = backend.ops();

    let embed = Embedding::new(&backend, EmbeddingConfig::new(8, 4), 42).unwrap();

    // Indices as a tensor (f32 ids) since Tape currently models indices this way.
    let ids = backend.tensor_from_vec(vec![1.0, 3.0, 1.0], &[3]).unwrap();

    let mut ctx = ForwardCtx::new(&backend, Mode::Train);
    let mut tape = Tape::<CpuBackend>::new();

    let ids_id = tape.watch(ids);
    let out_id = embed.forward_tape(ids_id, &mut tape, &mut ctx).unwrap();
    let loss_id = tape.sum_all_tape(out_id, &mut ctx).unwrap();

    let param_map = tape.param_map().clone();
    let make_ones = |data: Vec<f32>, shape: &[usize]| ops.tensor_from_vec(data, shape);
    let grads = tape.backward(loss_id, make_ones, ops).unwrap();

    let g = embed.table().gradient_from_store(&grads, &param_map).expect("missing embedding table grad");
    let g_vals = ops.tensor_to_vec(g).unwrap();

    // Rows 1 and 3 were used; gradient should be non-zero somewhere.
    assert!(g_vals.iter().any(|v| v.abs() > 0.0));
}
