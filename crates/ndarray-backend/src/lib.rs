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

use rand::thread_rng;
use rand::{rngs::StdRng, Rng, SeedableRng};
use rustral_core::{
    parallel_reductions_enabled, Backend, BackendCapabilities, ConvLayout, CoreError, FusionOps, Parameter,
    Result, ShapeExt, TensorOps, TrainingDtype,
};
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

impl rustral_core::TensorShape for CpuTensor {
    fn shape(&self) -> &[usize] {
        &self.shape
    }
}

impl From<Vec<f32>> for CpuTensor {
    fn from(values: Vec<f32>) -> Self {
        Self { values: values.clone(), shape: vec![values.len()] }
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

    fn fusion_ops(&self) -> Option<&dyn FusionOps<Self>> {
        Some(&self.ops)
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn capabilities(&self) -> BackendCapabilities {
        // CPU backend capabilities - updated to reflect actual implementation
        BackendCapabilities {
            supports_fp16: false,
            supports_bf16: false,
            tensor_cores: false,
            optimal_batch_size: 32,
            optimal_chunk_size: 4096, // Matches parallelization threshold in elementwise ops
            max_allocation_size: usize::MAX,
            prefers_contiguous: true,
            supports_in_place: true, // CpuOps implements TensorInPlaceOps
            supports_mixed_precision: false,
            recommended_training_dtype: TrainingDtype::F32,
            supports_fast_fp16_tensor_cores: false,
            preferred_conv_layout: ConvLayout::NCHW, // CPU default
            supports_strided_layouts: true,          // ndarray supports strided views
            supports_packed_layouts: false,
        }
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
            (0..shape.elem_count()).map(|_| rng.gen_range(-scale..scale)).collect::<Vec<_>>()
        } else {
            vec![0.0; shape.elem_count()]
        };
        Ok(Parameter::new(name, CpuTensor::new(values, shape)?))
    }

    fn parameter_from_vec(&self, name: &str, values: Vec<f32>, shape: &[usize]) -> Result<Parameter<Self>> {
        Ok(Parameter::new(name, CpuTensor::new(values, shape)?))
    }
}

/// Operation table for the CPU backend.
#[derive(Clone, Default)]
pub struct CpuOps;

impl CpuOps {
    fn ensure_rank(shape: &[usize], rank: usize) -> Result<()> {
        if shape.len() != rank {
            return Err(CoreError::InvalidShape {
                shape: shape.to_vec(),
                reason: format!("expected rank {rank}"),
            });
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
    /// is enabled and the matrix is large enough to benefit from it.
    /// Falls back to a naive implementation for tiny matrices (overhead dominates)
    fn matmul(&self, a: &CpuTensor, b: &CpuTensor) -> Result<CpuTensor> {
        Self::ensure_rank(&a.shape, 2)?;
        Self::ensure_rank(&b.shape, 2)?;
        let (m, k) = (a.shape[0], a.shape[1]);
        let (k2, n) = (b.shape[0], b.shape[1]);
        if k != k2 {
            return Err(CoreError::ShapeMismatch { expected: vec![k], actual: vec![k2] });
        }

        // Heuristic: use naive implementation for tiny matrices where BLAS overhead dominates.
        // Threshold should be measured with actual benchmarks on target hardware.
        // Current threshold of 64 elements total is a conservative default.
        const TINY_MATMUL_THRESHOLD: usize = 64;
        let total_elements = m * k * n;

        #[cfg(feature = "matrixmultiply")]
        {
            if total_elements < TINY_MATMUL_THRESHOLD {
                // Use naive for tiny matrices
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
                return CpuTensor::new(out, &[m, n]);
            }

            use matrixmultiply::sgemm;

            let mut out = vec![0.0; m * n];

            // sgemm(m, k, n, alpha, a, rsa, csa, b, rsb, csb, beta, c, rsc, csc)
            // rsa/csa = row/column stride for a
            // For row-major: stride between rows = k, between columns = 1
            unsafe {
                sgemm(
                    m,
                    k,
                    n,                 // dimensions
                    1.0,               // alpha
                    a.values.as_ptr(), // A
                    k as isize,        // rsa (row stride)
                    1,                 // csa (col stride)
                    b.values.as_ptr(), // B
                    n as isize,        // rsb
                    1,                 // csb
                    0.0,               // beta (initialize C to 1.0 * C + 0.0)
                    out.as_mut_ptr(),  // C
                    n as isize,        // rsc
                    1,                 // csc
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
        use wide::f32x8;

        let len = a.values.len();
        let mut out = Vec::with_capacity(len);

        // Process 8 elements at a time using f32x8
        let chunks = a.values.chunks_exact(8).zip(b.values.chunks_exact(8));
        let a_remainder = a.values.chunks_exact(8).remainder();
        let b_remainder = b.values.chunks_exact(8).remainder();

        for (a_chunk, b_chunk) in chunks {
            let mut a_arr = [0.0f32; 8];
            let mut b_arr = [0.0f32; 8];
            a_arr.copy_from_slice(a_chunk);
            b_arr.copy_from_slice(b_chunk);
            let a_vec = f32x8::new(a_arr);
            let b_vec = f32x8::new(b_arr);
            let sum = a_vec + b_vec;
            let sum_arr = sum.to_array();
            out.extend_from_slice(&sum_arr);
        }

        // Scalar fallback for remainder
        for (&a_val, &b_val) in a_remainder.iter().zip(b_remainder.iter()) {
            out.push(a_val + b_val);
        }

        CpuTensor::new(out, &a.shape)
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
        if m > 64 {
            use rayon::prelude::*;
            out.par_chunks_exact_mut(n).for_each(|chunk| {
                for j in 0..n {
                    chunk[j] += row.values[j];
                }
            });
        } else {
            for i in 0..m {
                for j in 0..n {
                    out[i * n + j] += row.values[j];
                }
            }
        }
        CpuTensor::new(out, &a.shape)
    }

    /// Apply ReLU element-wise.
    fn relu(&self, x: &CpuTensor) -> Result<CpuTensor> {
        use wide::f32x8;

        let len = x.values.len();
        let mut out = Vec::with_capacity(len);

        // Process 8 elements at a time using f32x8
        let chunks = x.values.chunks_exact(8);
        let zero = f32x8::splat(0.0);

        for chunk in chunks {
            let mut arr = [0.0f32; 8];
            arr.copy_from_slice(chunk);
            let vec = f32x8::new(arr);
            let relu_vec = vec.max(zero);
            let relu_arr = relu_vec.to_array();
            out.extend_from_slice(&relu_arr);
        }

        // Scalar fallback for remainder
        let remainder = x.values.chunks_exact(8).remainder();
        for &v in remainder.iter() {
            out.push(v.max(0.0));
        }

        CpuTensor::new(out, &x.shape)
    }

    /// Apply softmax over all tensor values.
    fn softmax(&self, x: &CpuTensor) -> Result<CpuTensor> {
        if x.values.is_empty() {
            return Err(CoreError::InvalidArgument("softmax of empty tensor".into()));
        }
        use wide::f32x8;

        // Find max using scalar (reduction not straightforward with SIMD)
        let max = x.values.iter().copied().fold(f32::NEG_INFINITY, f32::max);

        // Compute exp(x - max) using SIMD
        let len = x.values.len();
        let mut exps = Vec::with_capacity(len);
        let max_vec = f32x8::splat(max);

        let chunks = x.values.chunks_exact(8);
        for chunk in chunks {
            let mut arr = [0.0f32; 8];
            arr.copy_from_slice(chunk);
            let vec = f32x8::new(arr);
            let shifted = vec - max_vec;
            let exp_vec = shifted.exp();
            let exp_arr = exp_vec.to_array();
            exps.extend_from_slice(&exp_arr);
        }

        // Scalar fallback for remainder
        let remainder = x.values.chunks_exact(8).remainder();
        for &v in remainder.iter() {
            exps.push((v - max).exp());
        }

        // Sum and normalize using scalar
        let sum: f32 = exps.iter().sum();
        let inv_sum = 1.0 / sum;
        let scale_vec = f32x8::splat(inv_sum);

        let mut out = Vec::with_capacity(len);
        let chunks = exps.chunks_exact(8);
        for chunk in chunks {
            let mut arr = [0.0f32; 8];
            arr.copy_from_slice(chunk);
            let vec = f32x8::new(arr);
            let scaled = vec * scale_vec;
            let scaled_arr = scaled.to_array();
            out.extend_from_slice(&scaled_arr);
        }

        // Scalar fallback for remainder
        let remainder = exps.chunks_exact(8).remainder();
        for &v in remainder.iter() {
            out.push(v * inv_sum);
        }

        CpuTensor::new(out, &x.shape)
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
        let exps: Vec<f32> = x.values.iter().map(|v| (v - max).exp()).collect();
        let sum: f32 = exps.iter().sum();
        let log_sum = sum.ln();
        let offset = max + log_sum;
        if x.values.len() > 4096 {
            use rayon::prelude::*;
            let mut out = x.values.clone();
            out.par_iter_mut().for_each(|v| *v -= offset);
            CpuTensor::new(out, &x.shape)
        } else {
            CpuTensor::new(x.values.iter().map(|v| *v - offset).collect(), &x.shape)
        }
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
            _ => {
                return Err(CoreError::InvalidShape {
                    shape: input.shape.clone(),
                    reason: "linear expects vector or matrix input".into(),
                })
            }
        };
        if x.shape[1] != w.shape[1] {
            return Err(CoreError::ShapeMismatch { expected: vec![w.shape[1]], actual: vec![x.shape[1]] });
        }
        let mut out = vec![0.0; x.shape[0] * w.shape[0]];
        let x_values = &x.values;
        let w_values = &w.values;
        let (batch_size, in_dim, out_dim) = (x.shape[0], x.shape[1], w.shape[0]);

        // Parallelize over (batch, out) pairs for large matrices.
        if batch_size * out_dim > 512 {
            use rayon::prelude::*;
            out.par_chunks_exact_mut(out_dim).enumerate().for_each(|(batch, chunk)| {
                for o in 0..out_dim {
                    let mut sum = 0.0;
                    for i in 0..in_dim {
                        sum += x_values[batch * in_dim + i] * w_values[o * in_dim + i];
                    }
                    chunk[o] = sum;
                }
            });
        } else {
            for batch in 0..batch_size {
                for o in 0..out_dim {
                    let mut sum = 0.0;
                    for i in 0..in_dim {
                        sum += x_values[batch * in_dim + i] * w_values[o * in_dim + i];
                    }
                    out[batch * out_dim + o] = sum;
                }
            }
        }

        let y = CpuTensor::new(out, &[batch_size, out_dim])?;
        if let Some(b) = bias {
            self.add_row_vector(&y, b.tensor())
        } else {
            Ok(y)
        }
    }

    /// Apply sigmoid element-wise.
    fn sigmoid(&self, x: &CpuTensor) -> Result<CpuTensor> {
        if x.values.len() > 4096 {
            use rayon::prelude::*;
            let mut out = x.values.clone();
            out.par_iter_mut().for_each(|v| *v = 1.0 / (1.0 + (-*v).exp()));
            CpuTensor::new(out, &x.shape)
        } else {
            CpuTensor::new(x.values.iter().map(|&v| 1.0 / (1.0 + (-v).exp())).collect(), &x.shape)
        }
    }

    /// Apply tanh element-wise.
    fn tanh(&self, x: &CpuTensor) -> Result<CpuTensor> {
        if x.values.len() > 4096 {
            use rayon::prelude::*;
            let mut out = x.values.clone();
            out.par_iter_mut().for_each(|v| *v = v.tanh());
            CpuTensor::new(out, &x.shape)
        } else {
            CpuTensor::new(x.values.iter().map(|&v| v.tanh()).collect(), &x.shape)
        }
    }

    /// Apply GELU activation element-wise.
    ///
    /// GELU (Gaussian Error Linear Unit) is a smooth, non-monotonic activation function
    /// that tends to work better than ReLU for transformer models.
    /// Approximation: `0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))`
    fn gelu(&self, x: &CpuTensor) -> Result<CpuTensor> {
        // GELU is complex for SIMD due to tanh; using scalar for now
        let sqrt_2_pi = (2.0_f32 / std::f32::consts::PI).sqrt();
        if x.values.len() > 4096 {
            use rayon::prelude::*;
            let out: Vec<f32> = x
                .values
                .par_iter()
                .map(|&v| {
                    let x_cubed = v * v * v;
                    let x_plus = v + 0.044715 * x_cubed;
                    let tanh_arg = sqrt_2_pi * x_plus;
                    let tanh_val = tanh_arg.tanh();
                    let one_plus = 1.0 + tanh_val;
                    0.5 * v * one_plus
                })
                .collect();
            CpuTensor::new(out, &x.shape)
        } else {
            let out: Vec<f32> = x
                .values
                .iter()
                .map(|&v| {
                    let x_cubed = v * v * v;
                    let x_plus = v + 0.044715 * x_cubed;
                    let tanh_arg = sqrt_2_pi * x_plus;
                    let tanh_val = tanh_arg.tanh();
                    let one_plus = 1.0 + tanh_val;
                    0.5 * v * one_plus
                })
                .collect();
            CpuTensor::new(out, &x.shape)
        }
    }

    /// Element-wise multiplication (Hadamard product).
    fn mul(&self, a: &CpuTensor, b: &CpuTensor) -> Result<CpuTensor> {
        if a.shape != b.shape {
            return Err(CoreError::ShapeMismatch { expected: a.shape.clone(), actual: b.shape.clone() });
        }
        use wide::f32x8;

        let len = a.values.len();
        let mut out = Vec::with_capacity(len);

        // Process 8 elements at a time using f32x8
        let chunks = a.values.chunks_exact(8).zip(b.values.chunks_exact(8));
        let a_remainder = a.values.chunks_exact(8).remainder();
        let b_remainder = b.values.chunks_exact(8).remainder();

        for (a_chunk, b_chunk) in chunks {
            let mut a_arr = [0.0f32; 8];
            let mut b_arr = [0.0f32; 8];
            a_arr.copy_from_slice(a_chunk);
            b_arr.copy_from_slice(b_chunk);
            let a_vec = f32x8::new(a_arr);
            let b_vec = f32x8::new(b_arr);
            let product = a_vec * b_vec;
            let product_arr = product.to_array();
            out.extend_from_slice(&product_arr);
        }

        // Scalar fallback for remainder
        for (&a_val, &b_val) in a_remainder.iter().zip(b_remainder.iter()) {
            out.push(a_val * b_val);
        }

        CpuTensor::new(out, &a.shape)
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
            x.values.iter().map(|&v| if rng.gen::<f32>() < p { 0.0 } else { v * scale }).collect(),
            &x.shape,
        )
    }

    fn dropout_with_seed(&self, x: &CpuTensor, p: f32, seed: u64, training: bool) -> Result<CpuTensor> {
        if !training || p <= 0.0 {
            return Ok(x.clone());
        }
        if p >= 1.0 {
            return CpuTensor::new(vec![0.0; x.values.len()], &x.shape);
        }
        let scale = 1.0 / (1.0 - p);
        let mut rng = StdRng::seed_from_u64(seed);
        CpuTensor::new(
            x.values.iter().map(|&v| if rng.gen::<f32>() < p { 0.0 } else { v * scale }).collect(),
            &x.shape,
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
                return Err(CoreError::ShapeMismatch {
                    expected: first_shape.clone(),
                    actual: t.shape.clone(),
                });
            }
            for (i, (&expected, &actual)) in first_shape.iter().zip(t.shape.iter()).enumerate() {
                if i != dim && expected != actual {
                    return Err(CoreError::ShapeMismatch {
                        expected: first_shape.clone(),
                        actual: t.shape.clone(),
                    });
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
    fn dist_l2(&self, a: &CpuTensor, b: &CpuTensor) -> Result<CpuTensor> {
        let mut out = vec![0.0f32; a.values.len()];
        for i in 0..a.values.len() {
            let d = a.values[i] - b.values[i];
            out[i] = d * d;
        }
        let sum: f32 = out.iter().sum();
        CpuTensor::new(vec![sum.sqrt()], &[1])
    }
    fn less(&self, a: &CpuTensor, b: &CpuTensor) -> Result<CpuTensor> {
        let mut out = vec![0.0f32; a.values.len()];
        for i in 0..a.values.len() {
            out[i] = if a.values[i] < b.values[i] { 1.0 } else { 0.0 };
        }
        CpuTensor::new(out, &a.shape)
    }
    fn greater(&self, a: &CpuTensor, b: &CpuTensor) -> Result<CpuTensor> {
        let mut out = vec![0.0f32; a.values.len()];
        for i in 0..a.values.len() {
            out[i] = if a.values[i] > b.values[i] { 1.0 } else { 0.0 };
        }
        CpuTensor::new(out, &a.shape)
    }
    fn ones(&self, shape: &[usize]) -> Result<CpuTensor> {
        CpuTensor::new(vec![1.0; shape.iter().product()], shape)
    }
    fn argmax_dim(&self, x: &CpuTensor, dim: usize, keepdim: bool) -> Result<CpuTensor> {
        let shape = self.shape(x);
        if dim >= shape.len() {
            return Err(CoreError::InvalidArgument(format!("argmax_dim: dim {} out of range", dim)));
        }
        let mut out_shape = shape.clone();
        if keepdim {
            out_shape[dim] = 1;
        } else {
            out_shape.remove(dim);
        }
        let outer: usize = shape.iter().take(dim).product();
        let axis_size = shape[dim];
        let inner: usize = shape.iter().skip(dim + 1).product();
        let mut res = Vec::with_capacity(outer * inner);
        for i in 0..outer {
            for k in 0..inner {
                let mut max_val = f32::NEG_INFINITY;
                let mut max_idx = 0;
                for j in 0..axis_size {
                    let val = x.values[(i * axis_size + j) * inner + k];
                    if val > max_val {
                        max_val = val;
                        max_idx = j;
                    }
                }
                res.push(max_idx as f32);
            }
        }
        CpuTensor::new(res, &out_shape)
    }
    fn clamp(&self, x: &CpuTensor, min: f32, max: f32) -> Result<CpuTensor> {
        let mut out = vec![0.0f32; x.values.len()];
        for i in 0..x.values.len() {
            out[i] = x.values[i].clamp(min, max);
        }
        CpuTensor::new(out, &x.shape)
    }
    fn abs(&self, x: &CpuTensor) -> Result<CpuTensor> {
        let mut out = vec![0.0f32; x.values.len()];
        for i in 0..x.values.len() {
            out[i] = x.values[i].abs();
        }
        CpuTensor::new(out, &x.shape)
    }
    fn sign(&self, x: &CpuTensor) -> Result<CpuTensor> {
        let mut out = vec![0.0f32; x.values.len()];
        for i in 0..x.values.len() {
            out[i] = if x.values[i] > 0.0 {
                1.0
            } else if x.values[i] < 0.0 {
                -1.0
            } else {
                0.0
            };
        }
        CpuTensor::new(out, &x.shape)
    }
    fn index_add(
        &self,
        input: &CpuTensor,
        indices: &CpuTensor,
        source: &CpuTensor,
        axis: usize,
    ) -> Result<CpuTensor> {
        if axis != 0 {
            return Err(CoreError::Other("CpuOps only supports index_add on axis 0".into()));
        }
        let mut out_vals = input.values.clone();
        let elem_per_row: usize = input.shape.iter().skip(1).product();
        for (i, &idx) in indices.values.iter().enumerate() {
            let target_idx = idx as usize;
            let src_offset = i * elem_per_row;
            let dst_offset = target_idx * elem_per_row;
            for j in 0..elem_per_row {
                out_vals[dst_offset + j] += source.values[src_offset + j];
            }
        }
        CpuTensor::new(out_vals, &input.shape)
    }
    fn argsort(&self, x: &CpuTensor, axis: usize, descending: bool) -> Result<CpuTensor> {
        if axis != 0 || x.shape.len() != 1 {
            return Err(CoreError::Other("CpuOps only supports argsort on 1D tensors axis 0".into()));
        }
        let mut indices: Vec<usize> = (0..x.values.len()).collect();
        if descending {
            indices.sort_by(|&a, &b| x.values[b].partial_cmp(&x.values[a]).unwrap());
        } else {
            indices.sort_by(|&a, &b| x.values[a].partial_cmp(&x.values[b]).unwrap());
        }
        CpuTensor::new(indices.into_iter().map(|i| i as f32).collect(), &x.shape)
    }
    fn bincount(&self, x: &CpuTensor, minlength: usize) -> Result<CpuTensor> {
        use rayon::prelude::*;
        let max_val = x.values.iter().fold(0.0f32, |m, &v| m.max(v)) as usize;
        let len = (max_val + 1).max(minlength);
        if x.values.len() > 16384 {
            let num_threads = rayon::current_num_threads();
            let chunk_size = (x.values.len() + num_threads - 1) / num_threads;
            let histograms: Vec<Vec<f32>> = x
                .values
                .par_chunks(chunk_size)
                .map(|chunk| {
                    let mut local_hist = vec![0.0f32; len];
                    for &v in chunk {
                        local_hist[v as usize] += 1.0;
                    }
                    local_hist
                })
                .collect();
            let mut out = vec![0.0f32; len];
            for h in histograms {
                for i in 0..len {
                    out[i] += h[i];
                }
            }
            CpuTensor::new(out, &[len])
        } else {
            let mut out = vec![0.0f32; len];
            for &v in &x.values {
                out[v as usize] += 1.0;
            }
            CpuTensor::new(out, &[len])
        }
    }
    fn topk(&self, x: &CpuTensor, k: usize, axis: usize, largest: bool) -> Result<(CpuTensor, CpuTensor)> {
        let shape = self.shape(x);
        if axis != shape.len() - 1 {
            return Err(CoreError::Other("Ndarray topk only supports last axis".into()));
        }
        let outer: usize = shape.iter().take(axis).product();
        let axis_size = shape[axis];
        let mut out_values = Vec::with_capacity(outer * k);
        let mut out_indices = Vec::with_capacity(outer * k);
        for i in 0..outer {
            let mut items: Vec<(f32, usize)> =
                (0..axis_size).map(|j| (x.values[i * axis_size + j], j)).collect();
            if largest {
                items.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
            } else {
                items.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
            }
            for j in 0..k {
                out_values.push(items[j].0);
                out_indices.push(items[j].1 as f32);
            }
        }
        let mut out_shape = shape.clone();
        out_shape[axis] = k;
        Ok((CpuTensor::new(out_values, &out_shape)?, CpuTensor::new(out_indices, &out_shape)?))
    }
    fn unique_with_counts(&self, x: &CpuTensor) -> Result<(CpuTensor, CpuTensor)> {
        if x.values.is_empty() {
            return Ok((CpuTensor::new(vec![], &[0])?, CpuTensor::new(vec![], &[0])?));
        }
        let mut sorted_vals = x.values.clone();
        sorted_vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let mut unique_vals = Vec::new();
        let mut unique_counts = Vec::new();

        let mut current_val = sorted_vals[0];
        let mut current_count = 0.0f32;

        for &v in &sorted_vals {
            if (v - current_val).abs() < 1e-9 {
                current_count += 1.0;
            } else {
                unique_vals.push(current_val);
                unique_counts.push(current_count);
                current_val = v;
                current_count = 1.0;
            }
        }
        unique_vals.push(current_val);
        unique_counts.push(current_count);

        let n = unique_vals.len();
        let m = unique_counts.len();
        Ok((CpuTensor::new(unique_vals, &[n])?, CpuTensor::new(unique_counts, &[m])?))
    }
    fn slice(&self, x: &CpuTensor, start: usize, end: usize) -> Result<CpuTensor> {
        if start >= end || end > x.shape[0] {
            return Err(CoreError::InvalidArgument(format!(
                "invalid slice {}..{} for dim 0 of size {}",
                start, end, x.shape[0]
            )));
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
                x.values.len(),
                new_len
            )));
        }
        CpuTensor::new(x.values.clone(), shape)
    }

    /// Element-wise addition with scalar.
    fn add_scalar(&self, x: &CpuTensor, scalar: f32) -> Result<CpuTensor> {
        if x.values.len() > 4096 {
            use rayon::prelude::*;
            let mut out = x.values.clone();
            out.par_iter_mut().for_each(|v| *v += scalar);
            CpuTensor::new(out, &x.shape)
        } else {
            CpuTensor::new(x.values.iter().map(|&v| v + scalar).collect(), &x.shape)
        }
    }

    /// Element-wise multiplication by scalar.
    fn mul_scalar(&self, x: &CpuTensor, scalar: f32) -> Result<CpuTensor> {
        if x.values.len() > 4096 {
            use rayon::prelude::*;
            let mut out = x.values.clone();
            out.par_iter_mut().for_each(|v| *v *= scalar);
            CpuTensor::new(out, &x.shape)
        } else {
            CpuTensor::new(x.values.iter().map(|&v| v * scalar).collect(), &x.shape)
        }
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
                return Err(CoreError::InvalidArgument(format!(
                    "broadcast: cannot broadcast from {:?} to {:?}",
                    old_shape, shape
                )));
            }

            // Pad old_shape with leading 1s to match target length
            let mut padded_old = vec![1usize; shape.len() - old_shape.len()];
            padded_old.extend_from_slice(old_shape);

            // Validate compatibility
            for (old_dim, new_dim) in padded_old.iter().zip(shape.iter()) {
                if *old_dim != *new_dim && *old_dim != 1 {
                    return Err(CoreError::InvalidArgument(format!(
                        "broadcast: cannot broadcast from {:?} to {:?}",
                        old_shape, shape
                    )));
                }
            }

            // Check total size consistency
            let expected_new_len: usize = shape.iter().product();
            let expanded_len: usize = padded_old.iter().product();
            if expanded_len != old_len {
                // The padded shape product should equal actual data length
                return Err(CoreError::InvalidArgument(format!(
                    "broadcast: size mismatch {:?} vs {:?}",
                    old_shape, shape
                )));
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
                return Err(CoreError::InvalidArgument(format!(
                    "broadcast: result size mismatch {} vs {}",
                    result.len(),
                    expected_new_len
                )));
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
        CpuTensor::new(a.values.iter().zip(b.values.iter()).map(|(x, y)| x - y).collect(), &a.shape)
    }

    /// Element-wise square root.
    fn sqrt(&self, x: &CpuTensor) -> Result<CpuTensor> {
        // Check for negative values
        if x.values.iter().any(|&v| v < 0.0) {
            return Err(CoreError::InvalidArgument("sqrt of negative number".into()));
        }
        use wide::f32x8;

        let len = x.values.len();
        let mut out = Vec::with_capacity(len);

        // Process 8 elements at a time using f32x8
        let chunks = x.values.chunks_exact(8);

        for chunk in chunks {
            let mut arr = [0.0f32; 8];
            arr.copy_from_slice(chunk);
            let vec = f32x8::new(arr);
            let sqrt_vec = vec.sqrt();
            let sqrt_arr = sqrt_vec.to_array();
            out.extend_from_slice(&sqrt_arr);
        }

        // Scalar fallback for remainder
        let remainder = x.values.chunks_exact(8).remainder();
        for &v in remainder.iter() {
            out.push(v.sqrt());
        }

        CpuTensor::new(out, &x.shape)
    }

    /// Element-wise division.
    fn div(&self, a: &CpuTensor, b: &CpuTensor) -> Result<CpuTensor> {
        if a.shape != b.shape {
            return Err(CoreError::ShapeMismatch { expected: a.shape.clone(), actual: b.shape.clone() });
        }
        // Check for division by zero
        if b.values.contains(&0.0) {
            return Err(CoreError::InvalidArgument("division by zero".into()));
        }
        use wide::f32x8;

        let len = a.values.len();
        let mut out = Vec::with_capacity(len);

        // Process 8 elements at a time using f32x8
        let chunks = a.values.chunks_exact(8).zip(b.values.chunks_exact(8));
        let a_remainder = a.values.chunks_exact(8).remainder();
        let b_remainder = b.values.chunks_exact(8).remainder();

        for (a_chunk, b_chunk) in chunks {
            let mut a_arr = [0.0f32; 8];
            let mut b_arr = [0.0f32; 8];
            a_arr.copy_from_slice(a_chunk);
            b_arr.copy_from_slice(b_chunk);
            let a_vec = f32x8::new(a_arr);
            let b_vec = f32x8::new(b_arr);
            let div_vec = a_vec / b_vec;
            let div_arr = div_vec.to_array();
            out.extend_from_slice(&div_arr);
        }

        // Scalar fallback for remainder
        for (&a_val, &b_val) in a_remainder.iter().zip(b_remainder.iter()) {
            out.push(a_val / b_val);
        }

        CpuTensor::new(out, &a.shape)
    }

    /// Element-wise exponential: `exp(x)`.
    fn exp(&self, x: &CpuTensor) -> Result<CpuTensor> {
        use wide::f32x8;

        let len = x.values.len();
        let mut out = Vec::with_capacity(len);

        // Process 8 elements at a time using f32x8
        let chunks = x.values.chunks_exact(8);

        for chunk in chunks {
            let mut arr = [0.0f32; 8];
            arr.copy_from_slice(chunk);
            let vec = f32x8::new(arr);
            let exp_vec = vec.exp();
            let exp_arr = exp_vec.to_array();
            out.extend_from_slice(&exp_arr);
        }

        // Scalar fallback for remainder
        let remainder = x.values.chunks_exact(8).remainder();
        for &v in remainder.iter() {
            out.push(v.exp());
        }

        CpuTensor::new(out, &x.shape)
    }

    /// Element-wise natural logarithm: `ln(x)`.
    fn log(&self, x: &CpuTensor) -> Result<CpuTensor> {
        // Check for non-positive values
        if x.values.iter().any(|&v| v <= 0.0) {
            return Err(CoreError::InvalidArgument("log of non-positive number".into()));
        }
        use wide::f32x8;

        let len = x.values.len();
        let mut out = Vec::with_capacity(len);

        // Process 8 elements at a time using f32x8
        let chunks = x.values.chunks_exact(8);

        for chunk in chunks {
            let mut arr = [0.0f32; 8];
            arr.copy_from_slice(chunk);
            let vec = f32x8::new(arr);
            let log_vec = vec.ln();
            let log_arr = log_vec.to_array();
            out.extend_from_slice(&log_arr);
        }

        // Scalar fallback for remainder
        let remainder = x.values.chunks_exact(8).remainder();
        for &v in remainder.iter() {
            out.push(v.ln());
        }

        CpuTensor::new(out, &x.shape)
    }

    /// Element-wise maximum: `max(a, b)`.
    fn maximum(&self, a: &CpuTensor, b: &CpuTensor) -> Result<CpuTensor> {
        if a.shape != b.shape {
            return Err(CoreError::ShapeMismatch { expected: a.shape.clone(), actual: b.shape.clone() });
        }
        use wide::f32x8;

        let len = a.values.len();
        if len > 4096 {
            use rayon::prelude::*;
            let out: Vec<f32> = a
                .values
                .par_chunks(8)
                .zip(b.values.par_chunks(8))
                .flat_map(|(a_chunk, b_chunk)| {
                    let mut result = [0.0f32; 8];
                    let chunk_len = a_chunk.len();
                    if chunk_len == 8 {
                        let mut a_arr = [0.0f32; 8];
                        let mut b_arr = [0.0f32; 8];
                        a_arr.copy_from_slice(a_chunk);
                        b_arr.copy_from_slice(b_chunk);
                        let a_vec = f32x8::new(a_arr);
                        let b_vec = f32x8::new(b_arr);
                        let m = a_vec.max(b_vec);
                        let arr = m.to_array();
                        result.copy_from_slice(&arr);
                    } else {
                        for (i, (&x, &y)) in a_chunk.iter().zip(b_chunk.iter()).enumerate() {
                            result[i] = x.max(y);
                        }
                    }
                    result[0..chunk_len].to_vec()
                })
                .collect();
            CpuTensor::new(out, &a.shape)
        } else {
            let mut out = Vec::with_capacity(len);
            let a_chunks = a.values.chunks_exact(8);
            let b_chunks = b.values.chunks_exact(8);
            let a_remainder = a_chunks.remainder();
            let b_remainder = b_chunks.remainder();

            for (a_chunk, b_chunk) in a_chunks.zip(b_chunks) {
                let mut a_arr = [0.0f32; 8];
                let mut b_arr = [0.0f32; 8];
                a_arr.copy_from_slice(a_chunk);
                b_arr.copy_from_slice(b_chunk);
                let a_vec = f32x8::new(a_arr);
                let b_vec = f32x8::new(b_arr);
                let m = a_vec.max(b_vec);
                let arr = m.to_array();
                out.extend_from_slice(&arr);
            }

            for (&x, &y) in a_remainder.iter().zip(b_remainder.iter()) {
                out.push(x.max(y));
            }

            CpuTensor::new(out, &a.shape)
        }
    }

    /// Element-wise greater-than with scalar: returns 1.0 if x > scalar else 0.0.
    fn gt_scalar(&self, x: &CpuTensor, scalar: f32) -> Result<CpuTensor> {
        CpuTensor::new(x.values.iter().map(|&v| if v > scalar { 1.0 } else { 0.0 }).collect(), &x.shape)
    }

    /// Sum all elements of a tensor into a scalar.
    fn sum_all(&self, x: &CpuTensor) -> Result<CpuTensor> {
        use wide::f32x8;

        let par = parallel_reductions_enabled();

        let values = &x.values;

        // Deterministic SIMD reduction: process chunks in fixed order
        let total: f32 = if par && values.len() > 32_768 {
            // Deterministic parallel reduction: parallel chunk sums, then sequential fold.
            use rayon::prelude::*;
            let chunk = 4096usize;
            let partials: Vec<f32> = values.par_chunks(chunk).map(|c| c.iter().sum::<f32>()).collect();
            partials.into_iter().sum()
        } else {
            let mut sum = f32x8::splat(0.0);
            let chunks = values.chunks_exact(8);
            for chunk in chunks {
                let mut arr = [0.0f32; 8];
                arr.copy_from_slice(chunk);
                let vec = f32x8::new(arr);
                sum += vec;
            }
            let sum_array = sum.to_array();
            let mut total: f32 = sum_array.iter().sum();
            let remainder = values.chunks_exact(8).remainder();
            for &v in remainder {
                total += v;
            }
            total
        };

        CpuTensor::new(vec![total], &[1])
    }

    fn sum_dim0(&self, x: &CpuTensor) -> Result<CpuTensor> {
        if x.shape.len() != 2 {
            return Err(CoreError::InvalidArgument(format!(
                "sum_dim0 expects rank-2 tensor [batch, features], got shape {:?}",
                x.shape
            )));
        }
        let batch = x.shape[0];
        let features = x.shape[1];
        let mut out = vec![0.0f32; features];
        let par = parallel_reductions_enabled();

        // Use SIMD for feature-wise summation when features >= 8
        if features >= 8 {
            use wide::f32x8;
            let features_aligned = (features / 8) * 8;

            if par && batch * features > 32_768 {
                // Deterministic parallelization across feature blocks.
                use rayon::prelude::*;
                out.par_chunks_exact_mut(8).enumerate().for_each(|(blk, out_chunk)| {
                    let j = blk * 8;
                    let mut acc = f32x8::splat(0.0);
                    for b in 0..batch {
                        let offset = b * features + j;
                        let mut arr = [0.0f32; 8];
                        arr.copy_from_slice(&x.values[offset..offset + 8]);
                        acc += f32x8::new(arr);
                    }
                    let arr = acc.to_array();
                    out_chunk.copy_from_slice(&arr);
                });
                // Tail features (if any) in deterministic scalar order.
                for j in features_aligned..features {
                    let mut acc = 0.0f32;
                    for b in 0..batch {
                        acc += x.values[b * features + j];
                    }
                    out[j] = acc;
                }
            } else {
                for b in 0..batch {
                    let offset = b * features;

                    for j in (0..features_aligned).step_by(8) {
                        let mut arr = [0.0f32; 8];
                        for k in 0..8 {
                            arr[k] = x.values[offset + j + k];
                        }
                        let vec = f32x8::new(arr);
                        let out_vec = f32x8::new([
                            out[j],
                            out[j + 1],
                            out[j + 2],
                            out[j + 3],
                            out[j + 4],
                            out[j + 5],
                            out[j + 6],
                            out[j + 7],
                        ]);
                        let sum = out_vec + vec;
                        let sum_arr = sum.to_array();
                        out[j..(j + 8)].copy_from_slice(&sum_arr);
                    }

                    for j in features_aligned..features {
                        out[j] += x.values[offset + j];
                    }
                }
            }
        } else {
            // Small feature dimension: use scalar (deterministic)
            if par && batch * features > 32_768 {
                use rayon::prelude::*;
                out.par_iter_mut().enumerate().for_each(|(j, slot)| {
                    let mut acc = 0.0f32;
                    for b in 0..batch {
                        acc += x.values[b * features + j];
                    }
                    *slot = acc;
                });
            } else {
                for b in 0..batch {
                    let offset = b * features;
                    for j in 0..features {
                        out[j] += x.values[offset + j];
                    }
                }
            }
        }

        CpuTensor::new(out, &[features])
    }

    /// Read all tensor values into a flat row-major vector.
    fn tensor_to_vec(&self, x: &CpuTensor) -> Result<Vec<f32>> {
        Ok(x.values.clone())
    }

    /// Extract a single scalar element from a tensor at a flat index.
    fn tensor_element(&self, x: &CpuTensor, index: usize) -> Result<f32> {
        x.values.get(index).copied().ok_or_else(|| {
            CoreError::InvalidArgument(format!(
                "index {} out of bounds for tensor with {} elements",
                index,
                x.values.len()
            ))
        })
    }

    fn softmax_dim(&self, x: &CpuTensor, dim: usize) -> Result<CpuTensor> {
        let shape = &x.shape;
        if dim >= shape.len() {
            return Err(CoreError::InvalidArgument(format!(
                "softmax_dim: dim {} out of range for shape {:?}",
                dim, shape
            )));
        }
        let outer: usize = shape[..dim].iter().product();
        let axis = shape[dim];
        let inner: usize = shape[dim + 1..].iter().product();
        if axis == 0 {
            return Err(CoreError::InvalidArgument("softmax_dim: axis size is 0".into()));
        }
        let mut out = vec![0.0f32; x.values.len()];

        for o in 0..outer {
            for i in 0..inner {
                let mut max = f32::NEG_INFINITY;
                for a in 0..axis {
                    let idx = o * axis * inner + a * inner + i;
                    max = max.max(x.values[idx]);
                }
                let mut sum = 0.0f32;
                for a in 0..axis {
                    let idx = o * axis * inner + a * inner + i;
                    let e = (x.values[idx] - max).exp();
                    out[idx] = e;
                    sum += e;
                }
                let inv_sum = 1.0 / sum;
                for a in 0..axis {
                    let idx = o * axis * inner + a * inner + i;
                    out[idx] *= inv_sum;
                }
            }
        }

        CpuTensor::new(out, shape)
    }

    fn log_softmax_dim(&self, x: &CpuTensor, dim: usize) -> Result<CpuTensor> {
        let shape = &x.shape;
        if dim >= shape.len() {
            return Err(CoreError::InvalidArgument(format!(
                "log_softmax_dim: dim {} out of range for shape {:?}",
                dim, shape
            )));
        }
        let outer: usize = shape[..dim].iter().product();
        let axis = shape[dim];
        let inner: usize = shape[dim + 1..].iter().product();
        if axis == 0 {
            return Err(CoreError::InvalidArgument("log_softmax_dim: axis size is 0".into()));
        }
        let mut out = vec![0.0f32; x.values.len()];

        for o in 0..outer {
            for i in 0..inner {
                let mut max = f32::NEG_INFINITY;
                for a in 0..axis {
                    let idx = o * axis * inner + a * inner + i;
                    max = max.max(x.values[idx]);
                }
                let mut sum = 0.0f32;
                for a in 0..axis {
                    let idx = o * axis * inner + a * inner + i;
                    sum += (x.values[idx] - max).exp();
                }
                let log_denom = max + sum.ln();
                for a in 0..axis {
                    let idx = o * axis * inner + a * inner + i;
                    out[idx] = x.values[idx] - log_denom;
                }
            }
        }

        CpuTensor::new(out, shape)
    }

    fn sum_dim(&self, x: &CpuTensor, dim: usize, keepdim: bool) -> Result<CpuTensor> {
        let shape = &x.shape;
        if dim >= shape.len() {
            return Err(CoreError::InvalidArgument(format!(
                "sum_dim: dim {} out of range for shape {:?}",
                dim, shape
            )));
        }
        let outer: usize = shape[..dim].iter().product();
        let axis = shape[dim];
        let inner: usize = shape[dim + 1..].iter().product();

        let mut out_shape = shape.clone();
        if keepdim {
            out_shape[dim] = 1;
        } else {
            out_shape.remove(dim);
            if out_shape.is_empty() {
                out_shape.push(1);
            }
        }
        let out_len: usize = out_shape.iter().product();
        let mut out = vec![0.0f32; out_len];
        let par = parallel_reductions_enabled();

        // Use SIMD for axis reduction when axis >= 8 (deterministic fixed-order)
        if axis >= 8 {
            use wide::f32x8;
            let axis_aligned = (axis / 8) * 8;

            if par && outer * inner * axis > 32_768 {
                use rayon::prelude::*;
                out.par_iter_mut().enumerate().for_each(|(out_idx, slot)| {
                    let o = out_idx / inner;
                    let i = out_idx % inner;
                    let mut sum = f32x8::splat(0.0);
                    for a in (0..axis_aligned).step_by(8) {
                        let mut arr = [0.0f32; 8];
                        for k in 0..8 {
                            let idx = o * axis * inner + (a + k) * inner + i;
                            arr[k] = x.values[idx];
                        }
                        sum += f32x8::new(arr);
                    }
                    let sum_array = sum.to_array();
                    let mut s: f32 = sum_array.iter().sum();
                    for a in axis_aligned..axis {
                        let idx = o * axis * inner + a * inner + i;
                        s += x.values[idx];
                    }
                    *slot = s;
                });
            } else {
                for o in 0..outer {
                    for i in 0..inner {
                        let mut sum = f32x8::splat(0.0);
                        for a in (0..axis_aligned).step_by(8) {
                            let mut arr = [0.0f32; 8];
                            for k in 0..8 {
                                let idx = o * axis * inner + (a + k) * inner + i;
                                arr[k] = x.values[idx];
                            }
                            let vec = f32x8::new(arr);
                            sum += vec;
                        }
                        let sum_array = sum.to_array();
                        let mut s: f32 = sum_array.iter().sum();
                        for a in axis_aligned..axis {
                            let idx = o * axis * inner + a * inner + i;
                            s += x.values[idx];
                        }
                        let out_idx = o * inner + i;
                        out[out_idx] = s;
                    }
                }
            }
        } else {
            // Small axis dimension: use scalar (deterministic)
            if par && outer * inner * axis > 32_768 {
                use rayon::prelude::*;
                out.par_iter_mut().enumerate().for_each(|(out_idx, slot)| {
                    let o = out_idx / inner;
                    let i = out_idx % inner;
                    let mut s = 0.0f32;
                    for a in 0..axis {
                        let idx = o * axis * inner + a * inner + i;
                        s += x.values[idx];
                    }
                    *slot = s;
                });
            } else {
                for o in 0..outer {
                    for i in 0..inner {
                        let mut s = 0.0f32;
                        for a in 0..axis {
                            let idx = o * axis * inner + a * inner + i;
                            s += x.values[idx];
                        }
                        let out_idx = o * inner + i;
                        out[out_idx] = s;
                    }
                }
            }
        }

        CpuTensor::new(out, &out_shape)
    }

    fn mean_dim(&self, x: &CpuTensor, dim: usize, keepdim: bool) -> Result<CpuTensor> {
        let shape = &x.shape;
        if dim >= shape.len() {
            return Err(CoreError::InvalidArgument(format!(
                "mean_dim: dim {} out of range for shape {:?}",
                dim, shape
            )));
        }
        let axis = shape[dim];
        if axis == 0 {
            return Err(CoreError::InvalidArgument("mean_dim: axis size is 0".into()));
        }
        let sum = self.sum_dim(x, dim, keepdim)?;
        self.mul_scalar(&sum, 1.0 / axis as f32)
    }

    fn var_dim(&self, x: &CpuTensor, dim: usize, unbiased: bool, keepdim: bool) -> Result<CpuTensor> {
        let shape = &x.shape;
        if dim >= shape.len() {
            return Err(CoreError::InvalidArgument(format!(
                "var_dim: dim {} out of range for shape {:?}",
                dim, shape
            )));
        }
        let axis = shape[dim];
        if axis == 0 {
            return Err(CoreError::InvalidArgument("var_dim: axis size is 0".into()));
        }
        if unbiased && axis < 2 {
            return Err(CoreError::InvalidArgument("var_dim: unbiased variance requires axis >= 2".into()));
        }

        let mean_keep = self.mean_dim(x, dim, true)?;
        let mean_b = self.broadcast_to(&mean_keep, shape)?;
        let diff = self.sub(x, &mean_b)?;
        let sq = self.mul(&diff, &diff)?;
        let mut var = self.sum_dim(&sq, dim, keepdim)?;
        let denom = if unbiased { (axis - 1) as f32 } else { axis as f32 };
        var = self.mul_scalar(&var, 1.0 / denom)?;
        Ok(var)
    }

    fn layer_norm(&self, x: &CpuTensor, gamma: &CpuTensor, beta: &CpuTensor, eps: f32) -> Result<CpuTensor> {
        let shape = &x.shape;
        let ndim = shape.len();
        if ndim == 0 {
            return Ok(x.clone());
        }
        let norm_elem_count = shape[ndim - 1];
        if norm_elem_count == 0 {
            return Ok(x.clone());
        }

        let mut out = x.values.clone();
        let gamma_vec = &gamma.values;
        let beta_vec = &beta.values;

        if out.len() > 4096 {
            use rayon::prelude::*;
            out.par_chunks_exact_mut(norm_elem_count).for_each(|group| {
                let mean = group.iter().sum::<f32>() / norm_elem_count as f32;
                let var = group.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / norm_elem_count as f32;
                let inv_std = 1.0 / (var + eps).sqrt();
                for i in 0..norm_elem_count {
                    let g = if gamma_vec.len() == 1 { gamma_vec[0] } else { gamma_vec[i] };
                    let b = if beta_vec.len() == 1 { beta_vec[0] } else { beta_vec[i] };
                    group[i] = (group[i] - mean) * inv_std * g + b;
                }
            });
        } else {
            for group in out.chunks_exact_mut(norm_elem_count) {
                let mean = group.iter().sum::<f32>() / norm_elem_count as f32;
                let var = group.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / norm_elem_count as f32;
                let inv_std = 1.0 / (var + eps).sqrt();
                for i in 0..norm_elem_count {
                    let g = if gamma_vec.len() == 1 { gamma_vec[0] } else { gamma_vec[i] };
                    let b = if beta_vec.len() == 1 { beta_vec[0] } else { beta_vec[i] };
                    group[i] = (group[i] - mean) * inv_std * g + b;
                }
            }
        }

        CpuTensor::new(out, shape)
    }

    fn cross_entropy_with_indices(&self, logits: &CpuTensor, targets: &[usize]) -> Result<CpuTensor> {
        let shape = self.shape(logits);
        let num_classes = shape[1];
        let batch_size = targets.len();

        // LogSoftmax calculation
        let log_probs = self.log_softmax(logits)?;
        let lp_values = log_probs.values();

        let mut nll_sum = 0.0f32;
        for (i, &target) in targets.iter().enumerate() {
            if target < num_classes {
                nll_sum -= lp_values[i * num_classes + target];
            }
        }

        CpuTensor::new(vec![nll_sum / batch_size as f32], &[1])
    }

    fn matmul_batched(&self, a: &CpuTensor, b: &CpuTensor) -> Result<CpuTensor> {
        let a_shape = self.shape(a);
        let b_shape = self.shape(b);
        if a_shape.len() != 3 || b_shape.len() != 3 {
            return Err(CoreError::InvalidShape {
                shape: a_shape,
                reason: "matmul_batched expects rank-3".into(),
            });
        }
        let (batch, m, k) = (a_shape[0], a_shape[1], a_shape[2]);
        let n = b_shape[2];

        let mut out = vec![0.0f32; batch * m * n];

        if batch * m * n > 4096 {
            use rayon::prelude::*;
            out.par_chunks_exact_mut(m * n).enumerate().for_each(|(i, chunk)| {
                let ai = &a.values()[i * m * k..(i + 1) * m * k];
                let bi = &b.values()[i * k * n..(i + 1) * k * n];

                // sgemm for each batch item
                for ii in 0..m {
                    for jj in 0..n {
                        let mut sum = 0.0f32;
                        for kk in 0..k {
                            sum += ai[ii * k + kk] * bi[kk * n + jj];
                        }
                        chunk[ii * n + jj] = sum;
                    }
                }
            });
        } else {
            for i in 0..batch {
                let ai = &a.values()[i * m * k..(i + 1) * m * k];
                let bi = &b.values()[i * k * n..(i + 1) * k * n];
                let offset = i * m * n;
                for ii in 0..m {
                    for jj in 0..n {
                        let mut sum = 0.0f32;
                        for kk in 0..k {
                            sum += ai[ii * k + kk] * bi[kk * n + jj];
                        }
                        out[offset + ii * n + jj] = sum;
                    }
                }
            }
        }
        CpuTensor::new(out, &[batch, m, n])
    }
    fn transpose_axes(&self, x: &CpuTensor, dim0: usize, dim1: usize) -> Result<CpuTensor> {
        let shape = self.shape(x);
        if shape.len() == 4 {
            // Basic 4D transpose [B, S, H, D] -> [B, H, S, D]
            if dim0 == 1 && dim1 == 2 {
                let b = shape[0];
                let s = shape[1];
                let h = shape[2];
                let d = shape[3];
                let mut out = vec![0.0f32; b * s * h * d];
                let val = x.values();
                for i in 0..b {
                    for j in 0..s {
                        for k in 0..h {
                            for l in 0..d {
                                let old_idx = ((i * s + j) * h + k) * d + l;
                                let new_idx = ((i * h + k) * s + j) * d + l;
                                out[new_idx] = val[old_idx];
                            }
                        }
                    }
                }
                return CpuTensor::new(out, &[b, h, s, d]);
            }
        }
        // Fallback or Error
        if dim0 == 0 && dim1 == 1 && shape.len() == 2 {
            return self.transpose(x);
        }
        Err(CoreError::Other(format!(
            "transpose_axes not implemented for dims {}/{} on shape {:?}",
            dim0, dim1, shape
        )))
    }

    fn broadcast_to(&self, x: &CpuTensor, shape: &[usize]) -> Result<CpuTensor> {
        self.broadcast(x, shape)
    }
}

impl FusionOps<CpuBackend> for CpuOps {
    fn fused_linear_bias(
        &self,
        input: &CpuTensor,
        weight: &Parameter<CpuBackend>,
        bias: &Parameter<CpuBackend>,
    ) -> Result<CpuTensor> {
        let w = weight.tensor();
        let b = bias.tensor();
        Self::ensure_rank(&w.shape, 2)?;
        Self::ensure_rank(&b.shape, 1)?;

        let x = match input.shape.len() {
            1 => CpuTensor::new(input.values.clone(), &[1, input.shape[0]])?,
            2 => input.clone(),
            _ => {
                return Err(CoreError::InvalidShape {
                    shape: input.shape.clone(),
                    reason: "fused_linear_bias expects vector or matrix input".into(),
                })
            }
        };

        if x.shape[1] != w.shape[1] {
            return Err(CoreError::ShapeMismatch { expected: vec![w.shape[1]], actual: vec![x.shape[1]] });
        }
        let (batch, in_dim, out_dim) = (x.shape[0], x.shape[1], w.shape[0]);
        if b.shape[0] != out_dim {
            return Err(CoreError::ShapeMismatch { expected: vec![out_dim], actual: b.shape.clone() });
        }

        let mut out = vec![0.0f32; batch * out_dim];

        #[cfg(feature = "matrixmultiply")]
        {
            use matrixmultiply::sgemm;
            // Compute: out = x @ w^T (w stored as [out_dim, in_dim]).
            // Treat w as B = w^T with strides: rsb=1, csb=in_dim.
            unsafe {
                sgemm(
                    batch,
                    in_dim,
                    out_dim,
                    1.0,
                    x.values.as_ptr(),
                    in_dim as isize,
                    1,
                    w.values.as_ptr(),
                    1,
                    in_dim as isize,
                    0.0,
                    out.as_mut_ptr(),
                    out_dim as isize,
                    1,
                );
            }
        }

        #[cfg(not(feature = "matrixmultiply"))]
        {
            let x_values = &x.values;
            let w_values = &w.values;
            for bb in 0..batch {
                for o in 0..out_dim {
                    let mut sum = 0.0;
                    for i in 0..in_dim {
                        sum += x_values[bb * in_dim + i] * w_values[o * in_dim + i];
                    }
                    out[bb * out_dim + o] = sum;
                }
            }
        }

        // Bias add in-place.
        if batch * out_dim > 4096 {
            use rayon::prelude::*;
            out.par_chunks_exact_mut(out_dim).for_each(|row| {
                for j in 0..out_dim {
                    row[j] += b.values[j];
                }
            });
        } else {
            for bb in 0..batch {
                let row = &mut out[bb * out_dim..(bb + 1) * out_dim];
                for j in 0..out_dim {
                    row[j] += b.values[j];
                }
            }
        }

        CpuTensor::new(out, &[batch, out_dim])
    }

    fn fused_linear_bias_relu(
        &self,
        input: &CpuTensor,
        weight: &Parameter<CpuBackend>,
        bias: &Parameter<CpuBackend>,
    ) -> Result<CpuTensor> {
        use wide::f32x8;
        let mut y = self.fused_linear_bias(input, weight, bias)?;

        let len = y.values.len();
        let zero = f32x8::splat(0.0);
        if len > 4096 {
            use rayon::prelude::*;
            y.values.par_chunks_mut(8).for_each(|chunk| {
                if chunk.len() == 8 {
                    let mut arr = [0.0f32; 8];
                    arr.copy_from_slice(chunk);
                    let v = f32x8::new(arr);
                    let r = v.max(zero);
                    let arr = r.to_array();
                    chunk.copy_from_slice(&arr);
                } else {
                    for v in chunk {
                        *v = (*v).max(0.0);
                    }
                }
            });
        } else {
            let mut chunks = y.values.chunks_exact_mut(8);
            for chunk in &mut chunks {
                let mut arr = [0.0f32; 8];
                arr.copy_from_slice(chunk);
                let v = f32x8::new(arr);
                let r = v.max(zero);
                let arr = r.to_array();
                chunk.copy_from_slice(&arr);
            }
            let remainder = chunks.into_remainder();
            for v in remainder {
                *v = (*v).max(0.0);
            }
        }

        Ok(y)
    }

    fn fused_linear_bias_gelu(
        &self,
        input: &CpuTensor,
        weight: &Parameter<CpuBackend>,
        bias: &Parameter<CpuBackend>,
    ) -> Result<CpuTensor> {
        let mut y = self.fused_linear_bias(input, weight, bias)?;

        // Same approximation used by CpuOps::gelu: 0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x^3))).
        let sqrt_2_pi = (2.0_f32 / std::f32::consts::PI).sqrt();

        for v in &mut y.values {
            let x = *v;
            let x_cubed = x * x * x;
            let x_plus = x + 0.044715 * x_cubed;
            let tanh_arg = sqrt_2_pi * x_plus;
            let tanh_val = tanh_arg.tanh();
            *v = 0.5 * x * (1.0 + tanh_val);
        }

        Ok(y)
    }
}

impl rustral_core::TensorView<CpuBackend> for CpuOps {
    fn as_slice_f32<'a>(&self, tensor: &'a CpuTensor) -> Result<&'a [f32]> {
        Ok(&tensor.values)
    }
}

impl rustral_core::TensorInPlaceOps<CpuBackend> for CpuOps {
    fn add_assign(&self, tensor: &mut CpuTensor, other: &CpuTensor) -> Result<()> {
        if tensor.shape != other.shape {
            return Err(CoreError::ShapeMismatch {
                expected: tensor.shape.clone(),
                actual: other.shape.clone(),
            });
        }
        use wide::f32x8;

        let len = tensor.values.len();
        if len > 4096 {
            use rayon::prelude::*;
            tensor.values.par_chunks_mut(8).zip(other.values.par_chunks(8)).for_each(|(t_chunk, o_chunk)| {
                if t_chunk.len() == 8 {
                    let mut t_arr = [0.0f32; 8];
                    let mut o_arr = [0.0f32; 8];
                    t_arr.copy_from_slice(t_chunk);
                    o_arr.copy_from_slice(o_chunk);
                    let t_v = f32x8::new(t_arr);
                    let o_v = f32x8::new(o_arr);
                    let sum = t_v + o_v;
                    let arr = sum.to_array();
                    t_chunk.copy_from_slice(&arr);
                } else {
                    for (t, &o) in t_chunk.iter_mut().zip(o_chunk.iter()) {
                        *t += o;
                    }
                }
            });
        } else {
            let mut t_chunks = tensor.values.chunks_exact_mut(8);
            let mut o_chunks = other.values.chunks_exact(8);

            for (t_chunk, o_chunk) in t_chunks.by_ref().zip(o_chunks.by_ref()) {
                let mut t_arr = [0.0f32; 8];
                let mut o_arr = [0.0f32; 8];
                t_arr.copy_from_slice(t_chunk);
                o_arr.copy_from_slice(o_chunk);
                let t_v = f32x8::new(t_arr);
                let o_v = f32x8::new(o_arr);
                let sum = t_v + o_v;
                let arr = sum.to_array();
                t_chunk.copy_from_slice(&arr);
            }

            let t_remainder = t_chunks.into_remainder();
            let o_remainder = o_chunks.remainder();
            for (t, &o) in t_remainder.iter_mut().zip(o_remainder.iter()) {
                *t += o;
            }
        }
        Ok(())
    }

    fn mul_assign(&self, tensor: &mut CpuTensor, scalar: f32) -> Result<()> {
        use wide::f32x8;

        let len = tensor.values.len();
        if len > 4096 {
            use rayon::prelude::*;
            let s = f32x8::splat(scalar);
            tensor.values.par_chunks_mut(8).for_each(|chunk| {
                if chunk.len() == 8 {
                    let mut arr = [0.0f32; 8];
                    arr.copy_from_slice(chunk);
                    let mut v = f32x8::new(arr);
                    v *= s;
                    let arr = v.to_array();
                    chunk.copy_from_slice(&arr);
                } else {
                    for t in chunk {
                        *t *= scalar;
                    }
                }
            });
        } else {
            let s = f32x8::splat(scalar);
            let mut chunks = tensor.values.chunks_exact_mut(8);
            for chunk in &mut chunks {
                let mut arr = [0.0f32; 8];
                arr.copy_from_slice(chunk);
                let mut v = f32x8::new(arr);
                v *= s;
                let arr = v.to_array();
                chunk.copy_from_slice(&arr);
            }
            let remainder = chunks.into_remainder();
            for t in remainder {
                *t *= scalar;
            }
        }
        Ok(())
    }

    fn axpy(&self, y: &mut CpuTensor, a: f32, x: &CpuTensor) -> Result<()> {
        if y.shape != x.shape {
            return Err(CoreError::ShapeMismatch { expected: y.shape.clone(), actual: x.shape.clone() });
        }
        use wide::f32x8;

        let len = y.values.len();
        if len > 4096 {
            use rayon::prelude::*;
            let a_v = f32x8::splat(a);
            y.values.par_chunks_mut(8).zip(x.values.par_chunks(8)).for_each(|(y_chunk, x_chunk)| {
                if y_chunk.len() == 8 {
                    let mut y_arr = [0.0f32; 8];
                    let mut x_arr = [0.0f32; 8];
                    y_arr.copy_from_slice(y_chunk);
                    x_arr.copy_from_slice(x_chunk);
                    let y_v = f32x8::new(y_arr);
                    let x_v = f32x8::new(x_arr);
                    let out = y_v + a_v * x_v;
                    let arr = out.to_array();
                    y_chunk.copy_from_slice(&arr);
                } else {
                    for (yi, &xi) in y_chunk.iter_mut().zip(x_chunk.iter()) {
                        *yi += a * xi;
                    }
                }
            });
        } else {
            let a_v = f32x8::splat(a);
            let mut y_chunks = y.values.chunks_exact_mut(8);
            let mut x_chunks = x.values.chunks_exact(8);

            for (y_chunk, x_chunk) in y_chunks.by_ref().zip(x_chunks.by_ref()) {
                let mut y_arr = [0.0f32; 8];
                let mut x_arr = [0.0f32; 8];
                y_arr.copy_from_slice(y_chunk);
                x_arr.copy_from_slice(x_chunk);
                let y_v = f32x8::new(y_arr);
                let x_v = f32x8::new(x_arr);
                let out = y_v + a_v * x_v;
                let arr = out.to_array();
                y_chunk.copy_from_slice(&arr);
            }

            let y_remainder = y_chunks.into_remainder();
            let x_remainder = x_chunks.remainder();
            for (yi, &xi) in y_remainder.iter_mut().zip(x_remainder.iter()) {
                *yi += a * xi;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod axis_ops_tests {
    use super::*;

    fn assert_close(a: &[f32], b: &[f32], tol: f32) {
        assert_eq!(a.len(), b.len());
        for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            let d = (x - y).abs();
            assert!(d <= tol, "idx={i} x={x} y={y} diff={d} tol={tol}");
        }
    }

    #[test]
    fn softmax_dim_row_sums_to_one() {
        let backend = CpuBackend::default();
        let ops = backend.ops;
        let x = ops.tensor_from_vec(vec![1.0, 2.0, 3.0, 0.0, -1.0, 2.0], &[2, 3]).unwrap();
        let y = ops.softmax_dim(&x, 1).unwrap();
        let v = y.values();
        for row in 0..2 {
            let s: f32 = v[row * 3..row * 3 + 3].iter().sum();
            assert!((s - 1.0).abs() <= 1e-6, "row {row} sum={s}");
        }
    }

    #[test]
    fn log_softmax_dim_matches_ln_softmax_dim() {
        let backend = CpuBackend::default();
        let ops = backend.ops;
        let x = ops.tensor_from_vec(vec![1.0, 2.0, 3.0, 0.0, -1.0, 2.0], &[2, 3]).unwrap();
        let ls = ops.log_softmax_dim(&x, 1).unwrap();
        let s = ops.softmax_dim(&x, 1).unwrap();
        let ln_s = ops.log(&s).unwrap();
        assert_close(ls.values(), ln_s.values(), 1e-5);
    }
}
mod tabular_tests;
