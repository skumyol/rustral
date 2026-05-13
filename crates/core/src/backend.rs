use crate::{CoreError, Parameter, Result};

/// Backend capabilities for device-agnostic optimization.
#[derive(Debug, Clone, PartialEq)]
pub struct BackendCapabilities {
    /// Whether the backend supports FP16 operations.
    pub supports_fp16: bool,
    /// Whether the backend supports BF16 operations.
    pub supports_bf16: bool,
    /// Whether the backend has tensor cores or similar hardware acceleration.
    pub tensor_cores: bool,
    /// Recommended batch size for optimal performance.
    pub optimal_batch_size: usize,
    /// Recommended chunk size for large operations.
    pub optimal_chunk_size: usize,
    /// Maximum memory allocation size in bytes.
    pub max_allocation_size: usize,
    /// Whether the backend prefers contiguous memory layouts.
    pub prefers_contiguous: bool,
    /// Whether the backend supports in-place operations.
    pub supports_in_place: bool,
    /// Whether the backend supports mixed precision training (FP16/BF16 compute, FP32 master weights).
    pub supports_mixed_precision: bool,
    /// Recommended dtype for training (FP16, BF16, or FP32).
    pub recommended_training_dtype: TrainingDtype,
    /// Whether the backend supports fast FP16 tensor cores.
    pub supports_fast_fp16_tensor_cores: bool,
    /// Preferred convolution memory layout (NHWC vs NCHW).
    pub preferred_conv_layout: ConvLayout,
    /// Whether the backend supports strided memory layouts.
    pub supports_strided_layouts: bool,
    /// Whether the backend supports packed/tiled layouts for tensor cores.
    pub supports_packed_layouts: bool,
}

impl BackendCapabilities {
    /// Clamp a requested batch size to a conservative range implied by this backend.
    ///
    /// Uses [`Self::optimal_batch_size`] as an upper hint (workloads may still use smaller
    /// batches for memory). This is a **soft** cap: backends report hints; callers decide policy.
    pub fn clamp_batch_size(&self, batch: usize) -> usize {
        batch.clamp(1, self.optimal_batch_size.max(1))
    }

    /// Determine if mixed precision training is recommended based on capabilities and current dtype.
    ///
    /// Returns true if the backend supports mixed precision and the recommended dtype
    /// is not FP32 (i.e., the backend has tensor cores and recommends FP16/BF16 training).
    /// This provides a concrete BackendCapabilities-driven decision for dtype selection.
    ///
    /// # Example
    ///
    /// ```
    /// use rustral_core::{BackendCapabilities, TrainingDtype};
    ///
    /// let caps = BackendCapabilities::default();
    /// if caps.recommends_mixed_precision() {
    ///     println!("Using mixed precision training with {:?}", caps.recommended_training_dtype);
    /// }
    /// ```
    pub fn recommends_mixed_precision(&self) -> bool {
        self.supports_mixed_precision && !matches!(self.recommended_training_dtype, TrainingDtype::F32)
    }

    /// Get the recommended dtype for a given operation based on backend capabilities.
    ///
    /// This is a BackendCapabilities-driven decision helper that returns the recommended
    /// dtype for different operations. For example, on tensor-core hardware, FP16 might be
    /// recommended for matrix multiplication while FP32 is used for reduction operations.
    pub fn recommended_dtype_for_operation(&self, operation: OperationType) -> TrainingDtype {
        match operation {
            OperationType::Matmul if self.tensor_cores && self.supports_fp16 => TrainingDtype::F16,
            OperationType::Matmul if self.tensor_cores && self.supports_bf16 => TrainingDtype::Bf16,
            OperationType::Convolution if self.supports_fast_fp16_tensor_cores => TrainingDtype::F16,
            _ => self.recommended_training_dtype,
        }
    }
}

/// Types of operations for dtype selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OperationType {
    /// Matrix multiplication operations.
    Matmul,
    /// Convolution operations.
    Convolution,
    /// Reduction operations.
    Reduction,
    /// Element-wise operations.
    Elementwise,
}

impl Default for BackendCapabilities {
    fn default() -> Self {
        Self {
            supports_fp16: false,
            supports_bf16: false,
            tensor_cores: false,
            optimal_batch_size: 8,
            optimal_chunk_size: 1024,
            max_allocation_size: usize::MAX,
            prefers_contiguous: true,
            supports_in_place: false,
            supports_mixed_precision: false,
            recommended_training_dtype: TrainingDtype::F32,
            supports_fast_fp16_tensor_cores: false,
            preferred_conv_layout: ConvLayout::NCHW,
            supports_strided_layouts: true,
            supports_packed_layouts: false,
        }
    }
}

/// Convolution memory layout preference.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConvLayout {
    /// NCHW layout: [batch, channels, height, width] - common in PyTorch/CUDA
    NCHW,
    /// NHWC layout: [batch, height, width, channels] - common in TensorFlow/Metal
    NHWC,
}

/// Training dtype selection for mixed precision training.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrainingDtype {
    /// FP32 training (full precision, no mixed precision).
    F32,
    /// FP16 mixed precision (FP16 compute, FP32 master weights).
    F16,
    /// BF16 mixed precision (BF16 compute, FP32 master weights, better for large models).
    Bf16,
}

/// Execution backend contract.
///
/// A backend owns the concrete tensor and device types for a runtime. The core
/// crates never assume CUDA, CPU, ndarray, Burn, Candle, or LibTorch directly;
/// they only require this trait. This keeps the model layer portable and makes
/// backend replacement explicit instead of hidden behind global state.
pub trait Backend: Clone + Send + Sync + 'static {
    /// Concrete tensor type used by this backend.
    type Tensor: Clone + Send + Sync + 'static;

    /// Concrete device handle used by this backend.
    type Device: Clone + Send + Sync + std::fmt::Debug + 'static;

    /// Return the device on which tensors and parameters are expected to live.
    fn device(&self) -> Self::Device;

    /// Return the operation table for this backend.
    fn ops(&self) -> &dyn TensorOps<Self>;

    /// Return optional fused operations for this backend.
    ///
    /// Returns None if the backend doesn't support fused operations.
    /// Layer implementations should try this first and fall back to
    /// individual operations if it returns None.
    fn fusion_ops(&self) -> Option<&dyn FusionOps<Self>> {
        None
    }

    /// Return optional attention operations for this backend.
    ///
    /// Returns None if the backend doesn't support optimized attention (e.g., Flash Attention).
    /// Attention implementations should try this first and fall back to
    /// standard attention if it returns None.
    fn attention_ops(&self) -> Option<&dyn AttentionOps<Self>> {
        None
    }

    /// Return optional quantization operations for this backend.
    ///
    /// Returns None if the backend doesn't support quantization (e.g., INT8 operations).
    /// Quantization-aware code should try this first and fall back to
    /// full precision if it returns None.
    fn quantization_ops(&self) -> Option<&dyn QuantizationOps<Self>> {
        None
    }

    /// Return the backend's capabilities for device-agnostic optimization.
    fn capabilities(&self) -> BackendCapabilities {
        BackendCapabilities::default()
    }

    /// Downcast this backend to `Any` for runtime type checking.
    ///
    /// This enables backend-specific optimizations that require knowing the concrete
    /// backend type at runtime. Use with caution - prefer trait-based polymorphism when possible.
    fn as_any(&self) -> &dyn std::any::Any;

    /// Create a parameter with deterministic uniform random initialization.
    ///
    /// `scale` controls the range `[-scale, scale)`. If `scale <= 0.0`, the
    /// parameter is filled with zeros.
    fn normal_parameter(&self, name: &str, shape: &[usize], seed: u64, scale: f32) -> Result<Parameter<Self>>
    where
        Self: Sized;

    /// Create a parameter from existing data.
    ///
    /// This is used for loading saved models.
    fn parameter_from_vec(&self, name: &str, values: Vec<f32>, shape: &[usize]) -> Result<Parameter<Self>>
    where
        Self: Sized;
}

/// Minimal tensor operation surface required by the higher-level modules.
///
/// This trait is intentionally small. Production backends can expose richer
/// APIs in their own crates, while shared modules depend only on these stable
/// operations.
pub trait TensorOps<B: Backend>: Send + Sync {
    /// Return the tensor shape as row-major dimensions.
    fn shape(&self, x: &B::Tensor) -> Vec<usize>;

    /// Build a tensor from a flat row-major vector and explicit shape.
    fn tensor_from_vec(&self, values: Vec<f32>, shape: &[usize]) -> Result<B::Tensor>;

    /// Allocate a tensor filled with zero values.
    fn zeros(&self, shape: &[usize]) -> Result<B::Tensor>;

    /// Matrix multiplication for rank-2 tensors: `[m, k] x [k, n] -> [m, n]`.
    fn matmul(&self, a: &B::Tensor, b: &B::Tensor) -> Result<B::Tensor>;

    /// Transpose a rank-2 tensor: `[m, n] -> [n, m]`.
    fn transpose(&self, x: &B::Tensor) -> Result<B::Tensor>;

    /// Element-wise addition for tensors with identical shape.
    fn add(&self, a: &B::Tensor, b: &B::Tensor) -> Result<B::Tensor>;

    /// Add a rank-1 row vector to every row of a rank-2 tensor.
    fn add_row_vector(&self, a: &B::Tensor, row: &B::Tensor) -> Result<B::Tensor>;

    /// Apply the ReLU activation element-wise.
    fn relu(&self, x: &B::Tensor) -> Result<B::Tensor>;

    /// Apply softmax over all tensor values.
    ///
    /// Backends may later add axis-aware softmax. This reference signature keeps
    /// the first implementation simple and deterministic.
    fn softmax(&self, x: &B::Tensor) -> Result<B::Tensor>;

    /// Apply log-softmax for numerical stability.
    ///
    /// Computes: x - log(sum(exp(x)))
    /// This is more stable than log(softmax(x)).
    fn log_softmax(&self, x: &B::Tensor) -> Result<B::Tensor>;

    // ---------------------------------------------------------------------
    // Axis-aware ops (preferred for modern models)
    // ---------------------------------------------------------------------

    /// Softmax along a single dimension.
    ///
    /// Default implementation returns an error so experimental backends can omit it.
    fn softmax_dim(&self, _x: &B::Tensor, _dim: usize) -> Result<B::Tensor> {
        Err(CoreError::Other("softmax_dim not supported by this backend".into()))
    }

    /// Log-softmax along a single dimension.
    ///
    /// Default implementation returns an error so experimental backends can omit it.
    fn log_softmax_dim(&self, _x: &B::Tensor, _dim: usize) -> Result<B::Tensor> {
        Err(CoreError::Other("log_softmax_dim not supported by this backend".into()))
    }

    /// Sum-reduction along a dimension.
    fn sum_dim(&self, _x: &B::Tensor, _dim: usize, _keepdim: bool) -> Result<B::Tensor> {
        Err(CoreError::Other("sum_dim not supported by this backend".into()))
    }

    /// Mean-reduction along a dimension.
    fn mean_dim(&self, _x: &B::Tensor, _dim: usize, _keepdim: bool) -> Result<B::Tensor> {
        Err(CoreError::Other("mean_dim not supported by this backend".into()))
    }

    /// Variance along a dimension.
    fn var_dim(&self, _x: &B::Tensor, _dim: usize, _unbiased: bool, _keepdim: bool) -> Result<B::Tensor> {
        Err(CoreError::Other("var_dim not supported by this backend".into()))
    }

    /// Layer normalization: `(x - mean) / sqrt(var + eps) * gamma + beta`
    ///
    /// This method allows backends to provide a highly optimized fused kernel
    /// for layer normalization, which is a critical operation in transformer models.
    /// Default implementation uses standard trait operations.
    fn layer_norm(
        &self,
        x: &B::Tensor,
        gamma: &B::Tensor,
        beta: &B::Tensor,
        eps: f32,
    ) -> Result<B::Tensor> {
        let shape = self.shape(x);
        let ndim = shape.len();
        if ndim == 0 {
            return Ok(x.clone());
        }

        // Default implementation using mean and variance
        let last_dim = ndim - 1;
        let mean = self.mean_dim(x, last_dim, true)?;
        let var = self.var_dim(x, last_dim, false, true)?;

        let x_centered = self.sub(x, &self.broadcast_to(&mean, &shape)?)?;
        let std = self.sqrt(&self.add_scalar(&var, eps)?)?;
        let x_hat = self.div(&x_centered, &self.broadcast_to(&std, &shape)?)?;

        let gamma_b = self.broadcast_to(gamma, &shape)?;
        let beta_b = self.broadcast_to(beta, &shape)?;

        let y = self.mul(&x_hat, &gamma_b)?;
        self.add(&y, &beta_b)
    }

    /// Broadcast to a target shape (numpy-style).
    fn broadcast_to(&self, _x: &B::Tensor, _shape: &[usize]) -> Result<B::Tensor> {
        Err(CoreError::Other("broadcast_to not supported by this backend".into()))
    }

    /// Return the flat index of the largest tensor value.
    fn argmax(&self, x: &B::Tensor) -> Result<usize>;

    /// Gather rows from a rank-2 parameter table using integer ids.
    fn gather_rows(&self, table: &Parameter<B>, ids: &[usize]) -> Result<B::Tensor>;

    /// Apply an affine projection: `input * weight^T + bias`.
    ///
    /// `input` may be rank-1 `[in_dim]` or rank-2 `[batch, in_dim]`; `weight`
    /// must be rank-2 `[out_dim, in_dim]`; `bias`, when present, must be rank-1
    /// `[out_dim]`.
    fn linear(
        &self,
        input: &B::Tensor,
        weight: &Parameter<B>,
        bias: Option<&Parameter<B>>,
    ) -> Result<B::Tensor>;

    /// Apply sigmoid element-wise: `1 / (1 + exp(-x))`.
    fn sigmoid(&self, x: &B::Tensor) -> Result<B::Tensor>;

    /// Apply tanh element-wise.
    fn tanh(&self, x: &B::Tensor) -> Result<B::Tensor>;

    /// Apply GELU activation element-wise.
    ///
    /// GELU (Gaussian Error Linear Unit) is a smooth, non-monotonic activation function
    /// that tends to work better than ReLU for transformer models.
    /// Approximation: `0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))`
    fn gelu(&self, x: &B::Tensor) -> Result<B::Tensor>;

    /// Element-wise multiplication (Hadamard product).
    fn mul(&self, a: &B::Tensor, b: &B::Tensor) -> Result<B::Tensor>;

    /// Apply dropout to tensor.
    ///
    /// During training: randomly zero elements with probability `p`,
    /// scaling remaining elements by `1/(1-p)`.
    /// During inference: return input unchanged.
    fn dropout(&self, x: &B::Tensor, p: f32, training: bool) -> Result<B::Tensor>;

    /// Apply dropout to tensor with an explicit seed.
    ///
    /// This is an **optional** determinism hook. Backends may ignore the seed by
    /// falling back to [`Self::dropout`].
    fn dropout_with_seed(&self, x: &B::Tensor, p: f32, seed: u64, training: bool) -> Result<B::Tensor> {
        let _ = seed;
        self.dropout(x, p, training)
    }

    /// Concatenate tensors along a dimension.
    fn concat(&self, tensors: &[&B::Tensor], dim: usize) -> Result<B::Tensor>;

    /// Slice a tensor along dimension 0.
    fn slice(&self, x: &B::Tensor, start: usize, end: usize) -> Result<B::Tensor>;

    /// Reshape a tensor to a new shape (total elements must match).
    fn reshape(&self, x: &B::Tensor, shape: &[usize]) -> Result<B::Tensor>;

    /// Element-wise addition.
    fn add_scalar(&self, x: &B::Tensor, scalar: f32) -> Result<B::Tensor>;

    /// Element-wise multiplication by a scalar.
    fn mul_scalar(&self, x: &B::Tensor, scalar: f32) -> Result<B::Tensor>;

    /// Broadcast a tensor to a new shape.
    fn broadcast(&self, x: &B::Tensor, shape: &[usize]) -> Result<B::Tensor>;

    /// Element-wise negation.
    fn neg(&self, x: &B::Tensor) -> Result<B::Tensor>;

    /// Element-wise subtraction: `a - b`.
    fn sub(&self, a: &B::Tensor, b: &B::Tensor) -> Result<B::Tensor>;

    /// Element-wise square root.
    fn sqrt(&self, x: &B::Tensor) -> Result<B::Tensor>;

    /// Element-wise division: `a / b`.
    fn div(&self, a: &B::Tensor, b: &B::Tensor) -> Result<B::Tensor>;

    /// Element-wise exponential: `exp(x)`.
    fn exp(&self, x: &B::Tensor) -> Result<B::Tensor>;

    /// Element-wise natural logarithm: `ln(x)`.
    fn log(&self, x: &B::Tensor) -> Result<B::Tensor>;

    /// Element-wise maximum: `max(a, b)`.
    fn maximum(&self, a: &B::Tensor, b: &B::Tensor) -> Result<B::Tensor>;

    /// Element-wise greater-than with a scalar: returns 1.0 if x > scalar else 0.0.
    fn gt_scalar(&self, x: &B::Tensor, scalar: f32) -> Result<B::Tensor>;

    /// Sum all elements of a tensor into a scalar.
    fn sum_all(&self, x: &B::Tensor) -> Result<B::Tensor>;

    /// Sum a rank-2 tensor over dimension 0: `[batch, features] -> [features]`.
    ///
    /// This is the common reduction needed for bias gradients and batch statistics.
    fn sum_dim0(&self, x: &B::Tensor) -> Result<B::Tensor>;

    /// Read all tensor values back into a flat row-major vector.
    ///
    /// This is the efficient counterpart to `tensor_element` for bulk access.
    /// CPU backends can usually provide a zero-copy view; GPU backends
    /// perform a single device-to-host transfer.
    fn tensor_to_vec(&self, x: &B::Tensor) -> Result<Vec<f32>>;

    /// Extract a single scalar element from a tensor at a flat index.
    ///
    /// This is useful for retrieving specific values from 1D tensors,
    /// such as probability scores or logits at particular indices.
    fn tensor_element(&self, x: &B::Tensor, index: usize) -> Result<f32>;
}

/// Extension trait for efficient tensor view operations.
///
/// This trait is optional. Backends that store tensors in CPU-accessible
/// memory (like the ndarray backend) can implement it to provide
/// zero-copy slice access. Backends that store tensors on GPU or other
/// devices should not implement this trait; callers will fall back to
/// `tensor_element` or explicit transfer operations.
///
/// # Design Note
///
/// This is kept as a separate extension trait (not part of `TensorOps`)
/// because not all backends can support it. The reference CPU backend
/// implements it for performance; GPU backends return errors.
pub trait TensorView<B: Backend> {
    /// Return a view of the tensor data as a slice.
    ///
    /// The slice contains values in row-major order. Returns an error
    /// if the backend cannot provide CPU-accessible memory (e.g., GPU tensors).
    ///
    /// The returned slice borrows from the tensor, not from `self`.
    fn as_slice_f32<'a>(&self, tensor: &'a B::Tensor) -> Result<&'a [f32]>;
}

/// Extension trait for in-place tensor operations.
///
/// This trait is optional. Backends that support efficient in-place mutation
/// (e.g., CPU backends, GPU backends with write-capable buffers) should implement
/// it. Optimizers will use these operations when available for better performance.
///
/// # Design Note
///
/// The optimizer implementations check for this trait at runtime and fall back
/// to allocate-and-replace operations if it's not available. This keeps the
/// optimizer code backend-agnostic while allowing performance optimization.
pub trait TensorInPlaceOps<B: Backend>: TensorOps<B> {
    /// Add `other` to `tensor` in-place: `tensor += other`.
    fn add_assign(&self, tensor: &mut B::Tensor, other: &B::Tensor) -> Result<()>;

    /// Multiply `tensor` by a scalar in-place: `tensor *= scalar`.
    fn mul_assign(&self, tensor: &mut B::Tensor, scalar: f32) -> Result<()>;

    /// Compute `y = a * x + y` (axpy) in-place on `y`.
    fn axpy(&self, y: &mut B::Tensor, a: f32, x: &B::Tensor) -> Result<()>;
}

/// Extension trait for fused operations.
///
/// This trait is optional. Backends that support efficient fused kernels
/// (e.g., CUDA backends with cuBLAS + custom kernels, GPU backends with fused shaders)
/// should implement it for high-value operation combinations.
///
/// Fused operations reduce memory traffic and kernel launch overhead by combining
/// multiple operations into a single kernel call. This is particularly valuable on GPUs
/// where memory bandwidth and kernel launch overhead dominate performance.
///
/// # Design Note
///
/// The layer implementations check for this trait at runtime and fall back
/// to the unfused operation sequence if it's not available. This keeps the
/// layer code backend-agnostic while allowing performance optimization on capable backends.
///
/// # Example
///
/// ```rust,ignore
/// // In a layer implementation
/// if let Some(fusion_ops) = ctx.backend().ops().downcast_ref::<FusionOps<_>>() {
///     fusion_ops.fused_linear_bias_relu(&x, &self.weight, &self.bias)
/// } else {
///     // Fallback to unfused sequence
///     let h = ops.matmul(&x, self.weight.tensor())?;
///     let h = ops.add(&h, self.bias.tensor())?;
///     ops.relu(&h)
/// }
/// ```
pub trait FusionOps<B: Backend>: TensorOps<B> {
    /// Fused linear + bias operation: `y = x @ w^T + b`
    ///
    /// Combines matrix multiplication and bias addition into a single operation.
    /// This eliminates one intermediate tensor and one kernel launch.
    ///
    /// # Arguments
    ///
    /// * `input` - Input tensor of shape `[batch, in_features]`
    /// * `weight` - Weight parameter of shape `[out_features, in_features]`
    /// * `bias` - Bias parameter of shape `[out_features]`
    ///
    /// # Returns
    ///
    /// Output tensor of shape `[batch, out_features]`
    fn fused_linear_bias(
        &self,
        input: &B::Tensor,
        weight: &Parameter<B>,
        bias: &Parameter<B>,
    ) -> Result<B::Tensor>;

    /// Fused linear + bias + ReLU operation: `y = relu(x @ w^T + b)`
    ///
    /// Combines matrix multiplication, bias addition, and ReLU activation into a single operation.
    /// This eliminates two intermediate tensors and two kernel launches.
    ///
    /// # Arguments
    ///
    /// * `input` - Input tensor of shape `[batch, in_features]`
    /// * `weight` - Weight parameter of shape `[out_features, in_features]`
    /// * `bias` - Bias parameter of shape `[out_features]`
    ///
    /// # Returns
    ///
    /// Output tensor of shape `[batch, out_features]`
    fn fused_linear_bias_relu(
        &self,
        input: &B::Tensor,
        weight: &Parameter<B>,
        bias: &Parameter<B>,
    ) -> Result<B::Tensor>;

    /// Fused linear + bias + GELU operation: `y = gelu(x @ w^T + b)`
    ///
    /// Combines matrix multiplication, bias addition, and GELU activation into a single operation.
    /// This eliminates two intermediate tensors and two kernel launches.
    ///
    /// # Arguments
    ///
    /// * `input` - Input tensor of shape `[batch, in_features]`
    /// * `weight` - Weight parameter of shape `[out_features, in_features]`
    /// * `bias` - Bias parameter of shape `[out_features]`
    ///
    /// # Returns
    ///
    /// Output tensor of shape `[batch, out_features]`
    fn fused_linear_bias_gelu(
        &self,
        input: &B::Tensor,
        weight: &Parameter<B>,
        bias: &Parameter<B>,
    ) -> Result<B::Tensor>;
}

/// Optional trait for attention optimizations (e.g., Flash Attention 2).
///
/// This trait provides optimized attention implementations that reduce memory usage
/// and improve performance by fusing operations and using memory-efficient algorithms.
/// Backends that don't implement this trait will fall back to standard attention.
///
/// # Design Principles
///
/// - **Optional**: Backends can implement this trait for hardware-specific optimizations
/// - **Fallback**: Model code should check `backend.attention_ops()` and fall back to standard attention
/// - **Backend-agnostic**: The trait abstracts backend-specific attention implementations
pub trait AttentionOps<B: Backend>: TensorOps<B> {
    /// Flash Attention 2: Memory-efficient scaled dot-product attention.
    ///
    /// Computes attention with O(N) memory complexity instead of O(N^2) by
    /// tiling the computation and using online softmax. This is the key
    /// optimization that enables training long-context models.
    ///
    /// # Arguments
    ///
    /// * `query` - Query tensor of shape `[batch, seq_len, num_heads, head_dim]`
    /// * `key` - Key tensor of shape `[batch, seq_len, num_heads, head_dim]`
    /// * `value` - Value tensor of shape `[batch, seq_len, num_heads, head_dim]`
    /// * `causal_mask` - Whether to apply causal masking (for autoregressive models)
    ///
    /// # Returns
    ///
    /// Output tensor of shape `[batch, seq_len, num_heads, head_dim]`
    ///
    /// # Memory Complexity
    ///
    /// - Standard attention: O(seq_len^2) for attention matrix
    /// - Flash Attention: O(seq_len) by tiling and online softmax
    fn flash_attention_2(
        &self,
        query: &B::Tensor,
        key: &B::Tensor,
        value: &B::Tensor,
        causal_mask: bool,
    ) -> Result<B::Tensor>;

    /// Memory-efficient attention with custom attention mask.
    ///
    /// Similar to `flash_attention_2` but supports arbitrary attention masks.
    ///
    /// # Arguments
    ///
    /// * `query` - Query tensor of shape `[batch, seq_len, num_heads, head_dim]`
    /// * `key` - Key tensor of shape `[batch, seq_len, num_heads, head_dim]`
    /// * `value` - Value tensor of shape `[batch, seq_len, num_heads, head_dim]`
    /// * `mask` - Attention mask of shape `[batch, seq_len, seq_len]` (1.0 = attend, 0.0 = mask)
    ///
    /// # Returns
    ///
    /// Output tensor of shape `[batch, seq_len, num_heads, head_dim]`
    fn flash_attention_2_with_mask(
        &self,
        query: &B::Tensor,
        key: &B::Tensor,
        value: &B::Tensor,
        mask: &B::Tensor,
    ) -> Result<B::Tensor>;
}

/// Optional trait for quantization operations (e.g., INT8 quantization).
///
/// This trait provides quantization operations that reduce memory usage and
/// improve inference speed by using lower precision representations.
/// Backends that don't implement this trait will fall back to full precision.
///
/// # Design Principles
///
/// - **Optional**: Backends can implement this trait for hardware-specific quantization support
/// - **Fallback**: Model code should check `backend.quantization_ops()` and fall back to FP32
/// - **Backend-agnostic**: The trait abstracts backend-specific quantization implementations
pub trait QuantizationOps<B: Backend>: TensorOps<B> {
    /// Quantize a tensor from FP32 to INT8.
    ///
    /// Uses symmetric quantization with scale and zero-point.
    /// Formula: `int8 = round(fp32 / scale)`
    ///
    /// # Arguments
    ///
    /// * `tensor` - FP32 tensor to quantize
    ///
    /// # Returns
    ///
    /// Tuple of (quantized INT8 tensor, scale factor)
    fn quantize_int8(&self, tensor: &B::Tensor) -> Result<(B::Tensor, f32)>;

    /// Dequantize a tensor from INT8 to FP32.
    ///
    /// Formula: `fp32 = int8 * scale`
    ///
    /// # Arguments
    ///
    /// * `tensor` - INT8 tensor to dequantize
    /// * `scale` - Scale factor used during quantization
    ///
    /// # Returns
    ///
    /// Dequantized FP32 tensor
    fn dequantize_int8(&self, tensor: &B::Tensor, scale: f32) -> Result<B::Tensor>;

    /// Quantize a parameter from FP32 to INT8 for inference.
    ///
    /// This is a convenience method that quantizes a parameter and stores
    /// the scale factor for later dequantization.
    ///
    /// # Arguments
    ///
    /// * `parameter` - Parameter to quantize
    ///
    /// # Returns
    ///
    /// Tuple of (quantized parameter, scale factor)
    fn quantize_parameter_int8(&self, parameter: &Parameter<B>) -> Result<(B::Tensor, f32)>;

    /// INT8 matrix multiplication (quantized matmul).
    ///
    /// Performs matrix multiplication on INT8 tensors with accumulation in INT32,
    /// then dequantizes the result to FP32. This is much faster than FP32 matmul
    /// on hardware with INT8 support (e.g., NVIDIA tensor cores, ARM NEON).
    ///
    /// # Arguments
    ///
    /// * `a` - First INT8 tensor
    /// * `b` - Second INT8 tensor
    /// * `scale_a` - Scale factor for tensor A
    /// * `scale_b` - Scale factor for tensor B
    ///
    /// # Returns
    ///
    /// Dequantized FP32 result of matmul
    fn int8_matmul(&self, a: &B::Tensor, b: &B::Tensor, scale_a: f32, scale_b: f32) -> Result<B::Tensor>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_backend_capabilities_mixed_precision_recommendation() {
        // Test with FP32 only
        let caps_fp32 = BackendCapabilities {
            supports_fp16: false,
            supports_bf16: false,
            tensor_cores: false,
            optimal_batch_size: 32,
            optimal_chunk_size: 1024,
            max_allocation_size: 1024 * 1024 * 1024,
            prefers_contiguous: true,
            supports_in_place: true,
            supports_mixed_precision: false,
            recommended_training_dtype: TrainingDtype::F32,
            supports_fast_fp16_tensor_cores: false,
            preferred_conv_layout: ConvLayout::NCHW,
            supports_strided_layouts: true,
            supports_packed_layouts: false,
        };
        assert!(!caps_fp32.recommends_mixed_precision());

        // Test with FP16 and tensor cores
        let caps_fp16 = BackendCapabilities {
            supports_fp16: true,
            supports_bf16: false,
            tensor_cores: true,
            optimal_batch_size: 32,
            optimal_chunk_size: 1024,
            max_allocation_size: 1024 * 1024 * 1024,
            prefers_contiguous: true,
            supports_in_place: true,
            supports_mixed_precision: true,
            recommended_training_dtype: TrainingDtype::F16,
            supports_fast_fp16_tensor_cores: true,
            preferred_conv_layout: ConvLayout::NCHW,
            supports_strided_layouts: true,
            supports_packed_layouts: true,
        };
        assert!(caps_fp16.recommends_mixed_precision());

        // Test with BF16 and tensor cores
        let caps_bf16 = BackendCapabilities {
            supports_fp16: false,
            supports_bf16: true,
            tensor_cores: true,
            optimal_batch_size: 32,
            optimal_chunk_size: 1024,
            max_allocation_size: 1024 * 1024 * 1024,
            prefers_contiguous: true,
            supports_in_place: true,
            supports_mixed_precision: true,
            recommended_training_dtype: TrainingDtype::Bf16,
            supports_fast_fp16_tensor_cores: false,
            preferred_conv_layout: ConvLayout::NCHW,
            supports_strided_layouts: true,
            supports_packed_layouts: true,
        };
        assert!(caps_bf16.recommends_mixed_precision());
    }

    #[test]
    fn test_backend_capabilities_recommended_dtype() {
        // Test with FP32 only
        let caps_fp32 = BackendCapabilities {
            supports_fp16: false,
            supports_bf16: false,
            tensor_cores: false,
            optimal_batch_size: 32,
            optimal_chunk_size: 1024,
            max_allocation_size: 1024 * 1024 * 1024,
            prefers_contiguous: true,
            supports_in_place: true,
            supports_mixed_precision: false,
            recommended_training_dtype: TrainingDtype::F32,
            supports_fast_fp16_tensor_cores: false,
            preferred_conv_layout: ConvLayout::NCHW,
            supports_strided_layouts: true,
            supports_packed_layouts: false,
        };
        assert_eq!(
            caps_fp32.recommended_dtype_for_operation(OperationType::Matmul),
            TrainingDtype::F32
        );
        assert_eq!(
            caps_fp32.recommended_dtype_for_operation(OperationType::Convolution),
            TrainingDtype::F32
        );

        // Test with BF16 and tensor cores
        let caps_bf16 = BackendCapabilities {
            supports_fp16: false,
            supports_bf16: true,
            tensor_cores: true,
            optimal_batch_size: 32,
            optimal_chunk_size: 1024,
            max_allocation_size: 1024 * 1024 * 1024,
            prefers_contiguous: true,
            supports_in_place: true,
            supports_mixed_precision: true,
            recommended_training_dtype: TrainingDtype::Bf16,
            supports_fast_fp16_tensor_cores: false,
            preferred_conv_layout: ConvLayout::NCHW,
            supports_strided_layouts: true,
            supports_packed_layouts: true,
        };
        assert_eq!(
            caps_bf16.recommended_dtype_for_operation(OperationType::Matmul),
            TrainingDtype::Bf16
        );
        assert_eq!(
            caps_bf16.recommended_dtype_for_operation(OperationType::Convolution),
            TrainingDtype::Bf16
        );

        // Test with FP16 and tensor cores
        let caps_fp16 = BackendCapabilities {
            supports_fp16: true,
            supports_bf16: false,
            tensor_cores: true,
            optimal_batch_size: 32,
            optimal_chunk_size: 1024,
            max_allocation_size: 1024 * 1024 * 1024,
            prefers_contiguous: true,
            supports_in_place: true,
            supports_mixed_precision: true,
            recommended_training_dtype: TrainingDtype::F16,
            supports_fast_fp16_tensor_cores: true,
            preferred_conv_layout: ConvLayout::NCHW,
            supports_strided_layouts: true,
            supports_packed_layouts: true,
        };
        assert_eq!(
            caps_fp16.recommended_dtype_for_operation(OperationType::Matmul),
            TrainingDtype::F16
        );
        assert_eq!(
            caps_fp16.recommended_dtype_for_operation(OperationType::Convolution),
            TrainingDtype::F16
        );
    }
}
