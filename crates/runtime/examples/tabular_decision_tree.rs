//! Decision Tree inference using Rustral primitives.
use rustral_core::Backend;
use rustral_ndarray_backend::CpuBackend;
fn main() -> anyhow::Result<()> {
    let backend = CpuBackend::default();
    let ops = backend.ops();
    let x = backend.tensor_from_vec(vec![0.5, 0.2], &[2]).unwrap();
    let f0 = ops.tensor_element(&x, 0)?;
    let f1 = ops.tensor_element(&x, 1)?;
    let prediction = if f0 < 0.6 { if f1 > 0.1 { 1.0 } else { 0.0 } } else { 2.0 };
    println!("Input: [0.5, 0.2]");
    println!("Decision Tree Prediction: {}", prediction);
    let features = backend.tensor_from_vec(vec![0.1, 0.9, 0.4, 0.3, 0.7], &[5]).unwrap();
    let bins = ops.mul_scalar(&features, 10.0)?;
    let hist = ops.bincount(&bins, 10)?;
    println!("Feature Histogram (bins 0-9): {:?}", ops.tensor_to_vec(&hist)?);
    Ok(())
}
