//! K-Nearest Neighbors prototype using Rustral primitives.
use rustral_core::Backend;
use rustral_ndarray_backend::CpuBackend;
fn main() -> anyhow::Result<()> {
    let backend = CpuBackend::default();
    let ops = backend.ops();
    let ref_data = backend.tensor_from_vec(vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.5, 0.5, 0.0], &[4, 3]).unwrap();
    let query = backend.tensor_from_vec(vec![0.4, 0.6, 0.1], &[1, 3]).unwrap();
    println!("Finding 2-NN for query: [0.4, 0.6, 0.1]");
    let mut distances = Vec::new();
    for i in 0..4 {
        let item = ops.slice(&ref_data, i, i + 1)?;
        let d = ops.dist_l2(&query, &item)?;
        distances.push(ops.tensor_element(&d, 0)?);
    }
    let dist_tensor = backend.tensor_from_vec(distances, &[4]).unwrap();
    let (top_vals, top_indices) = ops.topk(&dist_tensor, 2, 0, false)?;
    println!("Top-2 Distances: {:?}", ops.tensor_to_vec(&top_vals)?);
    println!("Top-2 Indices: {:?}", ops.tensor_to_vec(&top_indices)?);
    Ok(())
}
