//! Small CPU backend used as a deterministic reference implementation.
//!
//! This backend is intentionally simple. It exists to test module contracts and
//! examples before plugging in a production backend such as Burn or Candle.
//!
//! # Platform-Specific Optimizations
//!
//! When the `matrixmultiply` feature is enabled, matrix operations use the
//! `matrixmultiply` crate which provides:
//! - **x86/x86_64**: AVX2 and FMA instructions
//! - **ARM/AArch64**: NEON SIMD instructions (when `neon` feature enabled)
//! - **Fallback**: Portable scalar implementation

use mnr_core::{Backend, CoreError, Parameter, Result, ShapeExt, TensorOps};
use rand::{rngs::StdRng, Rng, SeedableRng};
use rand::thread_rng;
use serde::{Deserialize, Serialize};

/// Dense row-major CPU tensor used by the reference backend.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CpuTensor {
    values: Vec<f32>,
    shape: Vec<usize>,
}

impl CpuTensor {
    /// Create a tensor from flat values and an explicit shape.
    pub fn new(values: Vec<f32>, shape: &[usize]) -> Result<Self> {
        let expected = shape.elem_count();
        if expected != values.len() {
            return Err(CoreError::ShapeMismatch { expected: vec![expected], actual: vec![values.len()] });
        }
        Ok(Self { values, shape: shape.to_vec() })
    }

    /// Borrow tensor values in row-major order.
    pub fn values(&self) -> &[f32] {
        &self.values
    }

    /// Borrow tensor shape.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Consume the tensor and return flat row-major values.
    pub fn into_values(self) -> Vec<f32> {
        self.values
    }
}

impl AsRef<[f32]> for CpuTensor {
    fn as_ref(&self) -> &[f32] {
        &self.values
    }
}

impl mnr_core::TensorShape for CpuTensor {
    fn shape(&self) -> &[usize] {
        &self.shape
    }
}

impl From<Vec<f32>> for CpuTensor {
    fn from(values: Vec<f32>) -> Self {
        Self {
            values: values.clone(),
            shape: vec![values.len()],
        }
    }
}

/// CPU device marker for the reference backend.
#[derive(Clone, Debug, Default)]
pub struct CpuDevice;

/// Deterministic CPU backend.
#[derive(Clone, Default)]
pub struct CpuBackend {
    ops: CpuOps,
}

impl CpuBackend {
    /// Create a tensor using this backend's operation table.
    pub fn tensor_from_vec(&self, values: Vec<f32>, shape: &[usize]) -> Result<CpuTensor> {
        self.ops.tensor_from_vec(values, shape)
    }

}

impl Backend for CpuBackend {
    type Tensor = CpuTensor;
    type Device = CpuDevice;

    /// Return the CPU device marker.
    fn device(&self) -> Self::Device {
        CpuDevice
    }

    /// Return tensor operations for this backend.
    fn ops(&self) -> &dyn TensorOps<Self> {
        &self.ops
    }

    fn normal_parameter(
        &self,
        name: &str,
        shape: &[usize],
        seed: u64,
        scale: f32,
    ) -> Result<Parameter<Self>> {
        let mut rng = StdRng::seed_from_u64(seed);
        let values = if scale > 0.0 {
            (0..shape.elem_count())
                .map(|_| rng.gen_range(-scale..scale))
                .collect::<Vec<_>>()
        } else {
            vec![0.0; shape.elem_count()]
        };
        Ok(Parameter::new(name, CpuTensor::new(values, shape)?))
    }

    fn parameter_from_vec(
        &self,
        name: &str,
        values: Vec<f32>,
        shape: &[usize],
    ) -> Result<Parameter<Self>> {
        Ok(Parameter::new(name, CpuTensor::new(values, shape)?))
    }
}

/// Operation table for the CPU backend.
#[derive(Clone, Default)]
pub struct CpuOps;

impl CpuOps {
    fn ensure_rank(shape: &[usize], rank: usize) -> Result<()> {
        if shape.len() != rank {
            return Err(CoreError::InvalidShape { shape: shape.to_vec(), reason: format!("expected rank {rank}") });
        }
        Ok(())
    }
}

impl TensorOps<CpuBackend> for CpuOps {
    /// Return a copy of the tensor shape.
    fn shape(&self, x: &CpuTensor) -> Vec<usize> {
        x.shape.clone()
    }

    /// Build a CPU tensor from flat row-major values.
    fn tensor_from_vec(&self, values: Vec<f32>, shape: &[usize]) -> Result<CpuTensor> {
        CpuTensor::new(values, shape)
    }

    /// Allocate a zero-filled CPU tensor.
    fn zeros(&self, shape: &[usize]) -> Result<CpuTensor> {
        CpuTensor::new(vec![0.0; shape.elem_count()], shape)
    }

    /// Matrix multiplication for rank-2 CPU tensors.
    ///
    /// Uses matrixmultiply for optimized BLAS-like performance when the feature
    /// is enabled; falls back to a naive implementation otherwise.
    fn matmul(&self, a: &CpuTensor, b: &CpuTensor) -> Result<CpuTensor> {
        Self::ensure_rank(&a.shape, 2)?;
        Self::ensure_rank(&b.shape, 2)?;
        let (m, k) = (a.shape[0], a.shape[1]);
        let (k2, n) = (b.shape[0], b.shape[1]);
        if k != k2 {
            return Err(CoreError::ShapeMismatch { expected: vec![k], actual: vec![k2] });
        }

        #[cfg(feature = "matrixmultiply")]
        {
            use matrixmultiply::sgemm;

            let mut out = vec![0.0; m * n];

            // sgemm(m, k, n, alpha, a, rsa, csa, b, rsb, csb, beta, c, rsc, csc)
            // rsa/csa = row/column stride for a
            // For row-major: stride between rows = k, between columns = 1
            unsafe {
                sgemm(
                    m, k, n,              // dimensions
                    1.0,                  // alpha
                    a.values.as_ptr(),    // A
                    k as isize,           // rsa (row stride)
                    1,                    // csa (col stride)
                    b.values.as_ptr(),    // B
                    n as isize,           // rsb
                    1,                    // csb
                    0.0,                  // beta (initialize C to 1.0 * C + 0.0)
                    out.as_mut_ptr(),     // C
                    n as isize,           // rsc
                    1,                    // csc
                );
            }

            CpuTensor::new(out, &[m, n])
        }

        #[cfg(not(feature = "matrixmultiply"))]
        {
            // Fallback naive implementation for environments without matrixmultiply
            let mut out = vec![0.0; m * n];
            for i in 0..m {
                for j in 0..n {
                    let mut sum = 0.0;
                    for kk in 0..k {
                        sum += a.values[i * k + kk] * b.values[kk * n + j];
                    }
                    out[i * n + j] = sum;
                }
            }
            CpuTensor::new(out, &[m, n])
        }
    }

    /// Transpose a rank-2 CPU tensor.
    fn transpose(&self, x: &CpuTensor) -> Result<CpuTensor> {
        Self::ensure_rank(&x.shape, 2)?;
        let (m, n) = (x.shape[0], x.shape[1]);
        let mut out = vec![0.0; m * n];
        for i in 0..m {
            for j in 0..n {
                out[j * m + i] = x.values[i * n + j];
            }
        }
        CpuTensor::new(out, &[n, m])
    }

    /// Element-wise addition for identical CPU tensor shapes.
    fn add(&self, a: &CpuTensor, b: &CpuTensor) -> Result<CpuTensor> {
        if a.shape != b.shape {
            return Err(CoreError::ShapeMismatch { expected: a.shape.clone(), actual: b.shape.clone() });
        }
        CpuTensor::new(a.values.iter().zip(&b.values).map(|(x, y)| x + y).collect(), &a.shape)
    }

    /// Add a row vector to every row of a rank-2 CPU tensor.
    fn add_row_vector(&self, a: &CpuTensor, row: &CpuTensor) -> Result<CpuTensor> {
        Self::ensure_rank(&a.shape, 2)?;
        Self::ensure_rank(&row.shape, 1)?;
        let (m, n) = (a.shape[0], a.shape[1]);
        if row.shape[0] != n {
            return Err(CoreError::ShapeMismatch { expected: vec![n], actual: row.shape.clone() });
        }
        let mut out = a.values.clone();
        for i in 0..m {
            for j in 0..n {
                out[i * n + j] += row.values[j];
            }
        }
        CpuTensor::new(out, &a.shape)
    }

    /// Apply ReLU element-wise.
    fn relu(&self, x: &CpuTensor) -> Result<CpuTensor> {
        CpuTensor::new(x.values.iter().map(|v| v.max(0.0)).collect(), &x.shape)
    }

    /// Apply softmax over all tensor values.
    fn softmax(&self, x: &CpuTensor) -> Result<CpuTensor> {
        if x.values.is_empty() {
            return Err(CoreError::InvalidArgument("softmax of empty tensor".into()));
        }
        let max = x.values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exps = x.values.iter().map(|v| (v - max).exp()).collect::<Vec<_>>();
        let sum: f32 = exps.iter().sum();
        CpuTensor::new(exps.into_iter().map(|v| v / sum).collect(), &x.shape)
    }

    /// Return the flat index of the largest value.
    fn argmax(&self, x: &CpuTensor) -> Result<usize> {
        x.values
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .ok_or_else(|| CoreError::InvalidArgument("argmax of empty tensor".into()))
    }

    /// Apply log-softmax for numerical stability.
    ///
    /// Computes: x - log(sum(exp(x)))
    fn log_softmax(&self, x: &CpuTensor) -> Result<CpuTensor> {
        if x.values.is_empty() {
            return Err(CoreError::InvalidArgument("log_softmax of empty tensor".into()));
        }
        let max = x.values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exps = x.values.iter().map(|v| (v - max).exp()).collect::<Vec<_>>();
        let sum: f32 = exps.iter().sum();
        let log_sum = sum.ln();
        CpuTensor::new(x.values.iter().map(|v| *v - max - log_sum).collect(), &x.shape)
    }

    /// Gather rows from a rank-2 CPU parameter tensor.
    fn gather_rows(&self, table: &Parameter<CpuBackend>, ids: &[usize]) -> Result<CpuTensor> {
        let t = table.tensor();
        Self::ensure_rank(&t.shape, 2)?;
        let rows = t.shape[0];
        let cols = t.shape[1];
        let mut values = Vec::with_capacity(ids.len() * cols);
        for &id in ids {
            if id >= rows {
                return Err(CoreError::InvalidArgument(format!("row id {id} out of bounds {rows}")));
            }
            values.extend_from_slice(&t.values[id * cols..(id + 1) * cols]);
        }
        CpuTensor::new(values, &[ids.len(), cols])
    }

    /// Apply an affine projection using CPU tensors.
    fn linear(
        &self,
        input: &CpuTensor,
        weight: &Parameter<CpuBackend>,
        bias: Option<&Parameter<CpuBackend>>,
    ) -> Result<CpuTensor> {
        let w = weight.tensor();
        Self::ensure_rank(&w.shape, 2)?;
        let x = match input.shape.len() {
            1 => CpuTensor::new(input.values.clone(), &[1, input.shape[0]])?,
            2 => input.clone(),
            _ => return Err(CoreError::InvalidShape { shape: input.shape.clone(), reason: "linear expects vector or matrix input".into() }),
        };
        if x.shape[1] != w.shape[1] {
            return Err(CoreError::ShapeMismatch { expected: vec![w.shape[1]], actual: vec![x.shape[1]] });
        }
        let mut out = vec![0.0; x.shape[0] * w.shape[0]];
        for batch in 0..x.shape[0] {
            for out_dim in 0..w.shape[0] {
                let mut sum = 0.0;
                for in_dim in 0..w.shape[1] {
                    sum += x.values[batch * x.shape[1] + in_dim] * w.values[out_dim * w.shape[1] + in_dim];
                }
                out[batch * w.shape[0] + out_dim] = sum;
            }
        }
        let y = CpuTensor::new(out, &[x.shape[0], w.shape[0]])?;
        if let Some(b) = bias { self.add_row_vector(&y, b.tensor()) } else { Ok(y) }
    }

    /// Apply sigmoid element-wise.
    fn sigmoid(&self, x: &CpuTensor) -> Result<CpuTensor> {
        CpuTensor::new(x.values.iter().map(|&v| 1.0 / (1.0 + (-v).exp())).collect(), &x.shape)
    }

    /// Apply tanh element-wise.
    fn tanh(&self, x: &CpuTensor) -> Result<CpuTensor> {
        CpuTensor::new(x.values.iter().map(|&v| v.tanh()).collect(), &x.shape)
    }

    /// Element-wise multiplication (Hadamard product).
    fn mul(&self, a: &CpuTensor, b: &CpuTensor) -> Result<CpuTensor> {
        if a.shape != b.shape {
            return Err(CoreError::ShapeMismatch { expected: a.shape.clone(), actual: b.shape.clone() });
        }
        CpuTensor::new(a.values.iter().zip(&b.values).map(|(x, y)| x * y).collect(), &a.shape)
    }

    /// Apply dropout to tensor.
    fn dropout(&self, x: &CpuTensor, p: f32, training: bool) -> Result<CpuTensor> {
        if !training || p <= 0.0 {
            return Ok(x.clone());
        }
        if p >= 1.0 {
            return CpuTensor::new(vec![0.0; x.values.len()], &x.shape);
        }
        let scale = 1.0 / (1.0 - p);
        let mut rng = thread_rng();
        CpuTensor::new(
            x.values.iter().map(|&v| {
                if rng.gen::<f32>() < p { 0.0 } else { v * scale }
            }).collect(),
            &x.shape
        )
    }

    /// Concatenate tensors along a dimension (only dim 0 supported for now).
    fn concat(&self, tensors: &[&CpuTensor], dim: usize) -> Result<CpuTensor> {
        if tensors.is_empty() {
            return Err(CoreError::InvalidArgument("concat requires at least one tensor".into()));
        }
        if dim != 0 {
            return Err(CoreError::InvalidArgument("concat only supports dim=0".into()));
        }
        let first_shape = &tensors[0].shape;
        for t in tensors.iter().skip(1) {
            if t.shape.len() != first_shape.len() {
                return Err(CoreError::ShapeMismatch { expected: first_shape.clone(), actual: t.shape.clone() });
            }
            for (i, (&expected, &actual)) in first_shape.iter().zip(t.shape.iter()).enumerate() {
                if i != dim && expected != actual {
                    return Err(CoreError::ShapeMismatch { expected: first_shape.clone(), actual: t.shape.clone() });
                }
            }
        }
        let total_dim0: usize = tensors.iter().map(|t| t.shape[0]).sum();
        let mut new_shape = first_shape.clone();
        new_shape[0] = total_dim0;
        let mut values = Vec::new();
        for t in tensors {
            values.extend_from_slice(&t.values);
        }
        CpuTensor::new(values, &new_shape)
    }

    /// Slice a tensor along dimension 0.
    fn slice(&self, x: &CpuTensor, start: usize, end: usize) -> Result<CpuTensor> {
        if start >= end || end > x.shape[0] {
            return Err(CoreError::InvalidArgument(format!("invalid slice {}..{} for dim 0 of size {}", start, end, x.shape[0])));
        }
        let slice_len = end - start;
        let elem_per_row: usize = x.shape.iter().skip(1).product();
        let start_idx = start * elem_per_row;
        let end_idx = end * elem_per_row;
        let mut new_shape = x.shape.clone();
        new_shape[0] = slice_len;
        CpuTensor::new(x.values[start_idx..end_idx].to_vec(), &new_shape)
    }

    /// Reshape a tensor to a new shape.
    fn reshape(&self, x: &CpuTensor, shape: &[usize]) -> Result<CpuTensor> {
        let new_len: usize = shape.iter().product();
        if new_len != x.values.len() {
            return Err(CoreError::InvalidArgument(format!(
                "reshape: total elements mismatch: {} vs {}",
                x.values.len(), new_len
            )));
        }
        CpuTensor::new(x.values.clone(), shape)
    }

    /// Element-wise addition with scalar.
    fn add_scalar(&self, x: &CpuTensor, scalar: f32) -> Result<CpuTensor> {
        CpuTensor::new(x.values.iter().map(|&v| v + scalar).collect(), &x.shape)
    }

    /// Element-wise multiplication by scalar.
    fn mul_scalar(&self, x: &CpuTensor, scalar: f32) -> Result<CpuTensor> {
        CpuTensor::new(x.values.iter().map(|&v| v * scalar).collect(), &x.shape)
    }

    /// Broadcast tensor to a new shape.
    fn broadcast(&self, x: &CpuTensor, shape: &[usize]) -> Result<CpuTensor> {
        let new_len: usize = shape.iter().product();
        if x.values.len() == 1 {
            // Broadcasting a scalar
            CpuTensor::new(vec![x.values[0]; new_len], shape)
        } else if x.values.len() == new_len {
            // Already correct shape
            Ok(x.clone())
        } else {
            // Try to broadcast by repeating dimensions
            let old_shape = &x.shape;
            let old_len = x.values.len();

            // Check if broadcasting is possible: each dimension must be equal or 1
            if old_shape.len() > shape.len() {
                return Err(CoreError::InvalidArgument(
                    format!("broadcast: cannot broadcast from {:?} to {:?}", old_shape, shape)
                ));
            }

            // Pad old_shape with leading 1s to match target length
            let mut padded_old = vec![1usize; shape.len() - old_shape.len()];
            padded_old.extend_from_slice(old_shape);

            // Validate compatibility
            for (old_dim, new_dim) in padded_old.iter().zip(shape.iter()) {
                if *old_dim != *new_dim && *old_dim != 1 {
                    return Err(CoreError::InvalidArgument(
                        format!("broadcast: cannot broadcast from {:?} to {:?}", old_shape, shape)
                    ));
                }
            }

            // Check total size consistency
            let expected_new_len: usize = shape.iter().product();
            let expanded_len: usize = padded_old.iter().product();
            if expanded_len != old_len {
                // The padded shape product should equal actual data length
                return Err(CoreError::InvalidArgument(
                    format!("broadcast: size mismatch {:?} vs {:?}", old_shape, shape)
                ));
            }

            // Perform broadcasting by repeating elements
            let mut result = x.values.clone();
            for (axis_idx, (old_dim, new_dim)) in padded_old.iter().zip(shape.iter()).enumerate() {
                if *old_dim == 1 && *new_dim > 1 {
                    // Repeat along this axis
                    let repeat = *new_dim;
                    let inner_size: usize = shape[axis_idx + 1..].iter().product();
                    let outer_size = result.len() / inner_size;

                    let mut new_result = Vec::with_capacity(outer_size * repeat * inner_size);
                    for chunk in result.chunks(inner_size) {
                        for _ in 0..repeat {
                            new_result.extend_from_slice(chunk);
                        }
                    }
                    result = new_result;
                }
            }

            if result.len() != expected_new_len {
                return Err(CoreError::InvalidArgument(
                    format!("broadcast: result size mismatch {} vs {}", result.len(), expected_new_len)
                ));
            }

            CpuTensor::new(result, shape)
        }
    }

    /// Element-wise negation.
    fn neg(&self, x: &CpuTensor) -> Result<CpuTensor> {
        CpuTensor::new(x.values.iter().map(|&v| -v).collect(), &x.shape)
    }

    /// Element-wise subtraction.
    fn sub(&self, a: &CpuTensor, b: &CpuTensor) -> Result<CpuTensor> {
        if a.shape != b.shape {
            return Err(CoreError::ShapeMismatch { expected: a.shape.clone(), actual: b.shape.clone() });
        }
        CpuTensor::new(
            a.values.iter().zip(b.values.iter()).map(|(x, y)| x - y).collect(),
            &a.shape,
        )
    }

    /// Element-wise square root.
    fn sqrt(&self, x: &CpuTensor) -> Result<CpuTensor> {
        // Check for negative values
        if x.values.iter().any(|&v| v < 0.0) {
            return Err(CoreError::InvalidArgument("sqrt of negative number".into()));
        }
        CpuTensor::new(x.values.iter().map(|&v| v.sqrt()).collect(), &x.shape)
    }

    /// Element-wise division.
    fn div(&self, a: &CpuTensor, b: &CpuTensor) -> Result<CpuTensor> {
        if a.shape != b.shape {
            return Err(CoreError::ShapeMismatch { expected: a.shape.clone(), actual: b.shape.clone() });
        }
        // Check for division by zero
        if b.values.iter().any(|&v| v == 0.0) {
            return Err(CoreError::InvalidArgument("division by zero".into()));
        }
        CpuTensor::new(
            a.values.iter().zip(b.values.iter()).map(|(x, y)| x / y).collect(),
            &a.shape,
        )
    }

    /// Element-wise exponential: `exp(x)`.
    fn exp(&self, x: &CpuTensor) -> Result<CpuTensor> {
        CpuTensor::new(x.values.iter().map(|&v| v.exp()).collect(), &x.shape)
    }

    /// Element-wise natural logarithm: `ln(x)`.
    fn log(&self, x: &CpuTensor) -> Result<CpuTensor> {
        // Check for non-positive values
        if x.values.iter().any(|&v| v <= 0.0) {
            return Err(CoreError::InvalidArgument("log of non-positive number".into()));
        }
        CpuTensor::new(x.values.iter().map(|&v| v.ln()).collect(), &x.shape)
    }

    /// Element-wise maximum: `max(a, b)`.
    fn maximum(&self, a: &CpuTensor, b: &CpuTensor) -> Result<CpuTensor> {
        if a.shape != b.shape {
            return Err(CoreError::ShapeMismatch { expected: a.shape.clone(), actual: b.shape.clone() });
        }
        CpuTensor::new(
            a.values.iter().zip(b.values.iter()).map(|(x, y)| x.max(*y)).collect(),
            &a.shape,
        )
    }

    /// Element-wise greater-than with scalar: returns 1.0 if x > scalar else 0.0.
    fn gt_scalar(&self, x: &CpuTensor, scalar: f32) -> Result<CpuTensor> {
        CpuTensor::new(
            x.values.iter().map(|&v| if v > scalar { 1.0 } else { 0.0 }).collect(),
            &x.shape,
        )
    }

    /// Sum all elements of a tensor into a scalar.
    fn sum_all(&self, x: &CpuTensor) -> Result<CpuTensor> {
        let sum: f32 = x.values.iter().sum();
        CpuTensor::new(vec![sum], &[1])
    }

    /// Extract a single scalar element from a tensor at a flat index.
    fn tensor_element(&self, x: &CpuTensor, index: usize) -> Result<f32> {
        x.values.get(index).copied()
            .ok_or_else(|| CoreError::InvalidArgument(format!("index {} out of bounds for tensor with {} elements", index, x.values.len())))
    }
}

impl mnr_core::TensorView<CpuBackend> for CpuOps {
    fn as_slice_f32<'a>(&self, tensor: &'a CpuTensor) -> Result<&'a [f32]> {
        Ok(&tensor.values)
    }
}

impl mnr_core::TensorInPlaceOps<CpuBackend> for CpuOps {
    fn add_assign(&self, tensor: &mut CpuTensor, other: &CpuTensor) -> Result<()> {
        if tensor.shape != other.shape {
            return Err(CoreError::ShapeMismatch {
                expected: tensor.shape.clone(),
                actual: other.shape.clone(),
            });
        }
        for (t, o) in tensor.values.iter_mut().zip(&other.values) {
            *t += o;
        }
        Ok(())
    }

    fn mul_assign(&self, tensor: &mut CpuTensor, scalar: f32) -> Result<()> {
        for t in &mut tensor.values {
            *t *= scalar;
        }
        Ok(())
    }

    fn axpy(&self, y: &mut CpuTensor, a: f32, x: &CpuTensor) -> Result<()> {
        if y.shape != x.shape {
            return Err(CoreError::ShapeMismatch {
                expected: y.shape.clone(),
                actual: x.shape.clone(),
            });
        }
        for (yi, xi) in y.values.iter_mut().zip(&x.values) {
            *yi += a * xi;
        }
        Ok(())
    }
}
