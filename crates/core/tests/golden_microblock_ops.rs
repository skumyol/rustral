//! Golden micro-blocks: CPU backend vs f64-derived reference vectors.

#[path = "common/mod.rs"]
mod common;
#[path = "golden/mod.rs"]
mod golden;

use common::numeric::{assert_allclose, default_tolerance, BackendFlavor, OpFamily};
use golden::{softmax_last_dim_reference, sum_dim0_reference};
use rustral_ndarray_backend::CpuBackend;
use rustral_core::Backend;

#[test]
fn golden_softmax_dim1_matches_f64_reference() {
    let backend = CpuBackend::default();
    let ops = backend.ops();
    let rows = 2usize;
    let cols = 4usize;
    let flat = vec![0.0f32, 1.0, 2.0, 3.0, -1.0, 0.5, 0.25, 2.0];
    let x = ops.tensor_from_vec(flat.clone(), &[rows, cols]).unwrap();
    let y = ops.softmax_dim(&x, 1).unwrap();
    let got = ops.tensor_to_vec(&y).unwrap();
    let expected = softmax_last_dim_reference(&flat, rows, cols);
    let tol = default_tolerance(BackendFlavor::Cpu, OpFamily::Softmax);
    assert_allclose("softmax_dim(1)", &got, &expected, tol);
}

#[test]
fn golden_sum_dim0_matches_reference() {
    let backend = CpuBackend::default();
    let ops = backend.ops();
    let rows = 3usize;
    let cols = 2usize;
    let flat = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let x = ops.tensor_from_vec(flat.clone(), &[rows, cols]).unwrap();
    let y = ops.sum_dim0(&x).unwrap();
    let got = ops.tensor_to_vec(&y).unwrap();
    let expected = sum_dim0_reference(&flat, rows, cols);
    let tol = default_tolerance(BackendFlavor::Cpu, OpFamily::Elementwise);
    assert_allclose("sum_dim0", &got, &expected, tol);
}
