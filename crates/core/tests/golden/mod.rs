//! Reference numerics for golden micro-block tests (f64 softmax, then cast to f32).

/// Row-major `rows x cols`: softmax over the last dimension (each row sums to 1).
pub fn softmax_last_dim_reference(flat: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    assert_eq!(flat.len(), rows * cols);
    let mut out = vec![0.0f32; flat.len()];
    for r in 0..rows {
        let base = r * cols;
        let mut max = f64::NEG_INFINITY;
        for c in 0..cols {
            max = max.max(flat[base + c] as f64);
        }
        let mut sum = 0.0f64;
        let mut tmp = vec![0.0f64; cols];
        for c in 0..cols {
            let e = ((flat[base + c] as f64) - max).exp();
            tmp[c] = e;
            sum += e;
        }
        let inv = 1.0 / sum;
        for c in 0..cols {
            out[base + c] = (tmp[c] * inv) as f32;
        }
    }
    out
}

/// Row-major `rows x cols`: sum over dim 0 → length `cols` (column sums).
pub fn sum_dim0_reference(flat: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    assert_eq!(flat.len(), rows * cols);
    let mut out = vec![0.0f32; cols];
    for r in 0..rows {
        for c in 0..cols {
            out[c] += flat[r * cols + c];
        }
    }
    out
}
