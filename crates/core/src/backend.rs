use crate::{Parameter, Result};

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

    /// Create a parameter with deterministic uniform random initialization.
    ///
    /// `scale` controls the range `[-scale, scale)`. If `scale <= 0.0`, the
    /// parameter is filled with zeros.
    fn normal_parameter(
        &self,
        name: &str,
        shape: &[usize],
        seed: u64,
        scale: f32,
    ) -> Result<Parameter<Self>>
    where
        Self: Sized;

    /// Create a parameter from existing data.
    ///
    /// This is used for loading saved models.
    fn parameter_from_vec(
        &self,
        name: &str,
        values: Vec<f32>,
        shape: &[usize],
    ) -> Result<Parameter<Self>>
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

    /// Element-wise multiplication (Hadamard product).
    fn mul(&self, a: &B::Tensor, b: &B::Tensor) -> Result<B::Tensor>;

    /// Apply dropout to tensor.
    ///
    /// During training: randomly zero elements with probability `p`,
    /// scaling remaining elements by `1/(1-p)`.
    /// During inference: return input unchanged.
    fn dropout(&self, x: &B::Tensor, p: f32, training: bool) -> Result<B::Tensor>;

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
