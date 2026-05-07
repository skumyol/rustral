//! Automatic differentiation for Rustral using reverse-mode autodiff.
#![allow(dead_code)]
//!
//! Provides a `Tape` for recording operations and computing gradients
//! via backpropagation. Works with any `Backend` implementation.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};

use rustral_core::{Backend, ForwardCtx, Parameter, ParameterId, Result};

pub mod checkpoint;

pub use checkpoint::{checkpoint_segment, CheckpointConfig, CheckpointManager, MemoryStats};

/// Unique identifier for a tensor in the computation graph.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TensorId(u64);

impl TensorId {
    /// Allocate a fresh tensor id.
    fn fresh() -> Self {
        Self(NEXT_TENSOR_ID.fetch_add(1, Ordering::Relaxed))
    }
}

static NEXT_TENSOR_ID: AtomicU64 = AtomicU64::new(1);

/// A recorded operation in the computation graph.
struct Op<B: Backend> {
    /// Input tensor ids.
    inputs: Vec<TensorId>,
    /// Output tensor id.
    output: TensorId,
    /// Backward function that computes gradients given upstream grad and tensor ops.
    backward: Box<
        dyn Fn(&B::Tensor, &mut GradientStore<B>, &dyn rustral_core::TensorOps<B>) -> Result<Vec<B::Tensor>>
            + Send
            + Sync,
    >,
}

/// Gradient accumulator for each tensor.
type GradientStore<B> = HashMap<TensorId, <B as Backend>::Tensor>;

/// Mapping from ParameterId to TensorId for gradient extraction.
pub type ParameterMap = HashMap<ParameterId, TensorId>;

/// Reverse-mode autodiff tape that records operations for backprop.
///
/// This is the main entry point for automatic differentiation. Usage:
///
/// ```ignore
/// let mut tape = Tape::<CpuBackend>::new();
/// let x = tape.watch(backend.tensor_from_vec(vec![2.0], &[1])?);
/// let y = tape.square(x)?; // y = x^2
/// let grad = tape.backward(y)?; // dy/dx = 2x = 4.0
/// ```
pub struct Tape<B: Backend> {
    /// Recorded operations in forward order.
    ops: Vec<Op<B>>,
    /// Tensor values (id -> tensor mapping).
    values: HashMap<TensorId, B::Tensor>,
    /// Gradients accumulated during backward pass.
    grads: GradientStore<B>,
    /// Parameter to tensor mapping for gradient extraction.
    param_map: ParameterMap,
}

impl<B: Backend> Default for Tape<B> {
    fn default() -> Self {
        Self::new()
    }
}

impl<B: Backend> Tape<B> {
    /// Create a new empty tape.
    pub fn new() -> Self {
        Self { ops: Vec::new(), values: HashMap::new(), grads: HashMap::new(), param_map: HashMap::new() }
    }

    /// Watch a tensor (make it a leaf node requiring gradients).
    pub fn watch(&mut self, tensor: B::Tensor) -> TensorId {
        let id = TensorId::fresh();
        self.values.insert(id, tensor);
        id
    }

    /// Watch a parameter's tensor and register it for gradient extraction.
    pub fn watch_parameter(&mut self, param: &Parameter<B>) -> TensorId
    where
        B::Tensor: Clone,
    {
        let id = self.watch(param.tensor().clone());
        self.param_map.insert(param.id(), id);
        id
    }

    /// Get the tensor id associated with a parameter (if registered).
    pub fn param_tensor_id(&self, param_id: ParameterId) -> Option<TensorId> {
        self.param_map.get(&param_id).copied()
    }

    /// Get the parameter map for gradient extraction.
    pub fn param_map(&self) -> &ParameterMap {
        &self.param_map
    }

    /// Get the tensor value for an id.
    pub fn value(&self, id: TensorId) -> Option<&B::Tensor> {
        self.values.get(&id)
    }

    /// Record an operation and return the output tensor id.
    ///
    /// The backward function takes the upstream gradient, the gradient
    /// store, and the tensor ops, and returns gradients for each input.
    pub fn record<F>(&mut self, inputs: &[TensorId], output_tensor: B::Tensor, backward: F) -> TensorId
    where
        F: Fn(&B::Tensor, &mut GradientStore<B>, &dyn rustral_core::TensorOps<B>) -> Result<Vec<B::Tensor>>
            + Send
            + Sync
            + 'static,
    {
        let output = TensorId::fresh();
        self.values.insert(output, output_tensor);

        self.ops.push(Op { inputs: inputs.to_vec(), output, backward: Box::new(backward) });

        output
    }

    /// Get a tensor value by id, returning an error if not found.
    fn get_value(&self, id: TensorId) -> Result<&B::Tensor> {
        self.values.get(&id).ok_or_else(|| {
            rustral_core::CoreError::InvalidArgument(format!("TensorId {:?} not found in tape values", id.0))
        })
    }

    /// Element-wise multiplication (records gradient).
    pub fn mul(&mut self, a: TensorId, b: TensorId, ctx: &mut ForwardCtx<B>) -> Result<TensorId>
    where
        B::Tensor: Clone,
    {
        let a_val = self.get_value(a)?.clone();
        let b_val = self.get_value(b)?.clone();
        let out = ctx.backend().ops().mul(&a_val, &b_val)?;

        // Store values for backward pass
        Ok(self.record(&[a, b], out, move |grad_out, _store, ops| {
            // d(a*b)/da = b * grad_out, d(a*b)/db = a * grad_out
            let grad_a = ops.mul(grad_out, &b_val)?;
            let grad_b = ops.mul(grad_out, &a_val)?;
            Ok(vec![grad_a, grad_b])
        }))
    }

    /// Element-wise addition (distributes gradient).
    pub fn add(&mut self, a: TensorId, b: TensorId, ctx: &mut ForwardCtx<B>) -> Result<TensorId>
    where
        B::Tensor: Clone,
    {
        let a_val = self.get_value(a)?.clone();
        let b_val = self.get_value(b)?.clone();
        let out = ctx.backend().ops().add(&a_val, &b_val)?;

        Ok(self.record(&[a, b], out, move |grad_out, _store, _ops| {
            // d(a+b)/da = 1, d(a+b)/db = 1
            Ok(vec![grad_out.clone(), grad_out.clone()])
        }))
    }

    /// Multiply a tensor by a scalar (records gradient).
    pub fn mul_scalar(&mut self, a: TensorId, scalar: f32, ctx: &mut ForwardCtx<B>) -> Result<TensorId>
    where
        B::Tensor: Clone,
    {
        let a_val = self.get_value(a)?.clone();
        let out = ctx.backend().ops().mul_scalar(&a_val, scalar)?;
        Ok(self.record(&[a], out, move |grad_out, _store, ops| {
            let grad_a = ops.mul_scalar(grad_out, scalar)?;
            Ok(vec![grad_a])
        }))
    }

    /// ReLU activation.
    pub fn relu(&mut self, a: TensorId, ctx: &mut ForwardCtx<B>) -> Result<TensorId>
    where
        B::Tensor: Clone,
    {
        let a_val = self.get_value(a)?.clone();
        let out = ctx.backend().ops().relu(&a_val)?;

        // Store input for backward (to check which elements were > 0)
        Ok(self.record(&[a], out.clone(), move |grad_out, _store, ops| {
            // d(relu)/dx = 1 if x > 0 else 0
            // We compute: grad_out * (output > 0)
            // Since relu(x) = max(0, x), output > 0 iff x > 0
            let mask = ops.gt_scalar(&out, 0.0)?;
            let grad = ops.mul(grad_out, &mask)?;
            Ok(vec![grad])
        }))
    }

    /// Matrix multiplication.
    ///
    /// Computes `a @ b` where `a` is [m, k] and `b` is [k, n].
    /// Returns a tensor of shape [m, n].
    pub fn matmul(&mut self, a: TensorId, b: TensorId, ctx: &mut ForwardCtx<B>) -> Result<TensorId>
    where
        B::Tensor: Clone,
    {
        let a_val = self.get_value(a)?.clone();
        let b_val = self.get_value(b)?.clone();
        let out = ctx.backend().ops().matmul(&a_val, &b_val)?;

        // Move values into the closure so they're owned and 'static
        Ok(self.record(&[a, b], out, move |grad_out, _store, ops| {
            // dL/dA = grad_out @ B^T
            // dL/dB = A^T @ grad_out
            let b_t = ops.transpose(&b_val)?;
            let grad_a = ops.matmul(grad_out, &b_t)?;

            let a_t = ops.transpose(&a_val)?;
            let grad_b = ops.matmul(&a_t, grad_out)?;

            Ok(vec![grad_a, grad_b])
        }))
    }

    /// Transpose a rank-2 tensor.
    ///
    /// Forward: `out = x^T`
    /// Backward: `dL/dx = (dL/dout)^T`
    pub fn transpose_tape(&mut self, x: TensorId, ctx: &mut ForwardCtx<B>) -> Result<TensorId>
    where
        B::Tensor: Clone,
    {
        let x_val = self.get_value(x)?.clone();
        let out = ctx.backend().ops().transpose(&x_val)?;
        Ok(self.record(&[x], out, move |grad_out, _store, ops| {
            let grad_x = ops.transpose(grad_out)?;
            Ok(vec![grad_x])
        }))
    }

    /// Reduce a tensor into a scalar by summing all elements.
    ///
    /// Backward: broadcast the upstream scalar gradient back to the input shape.
    pub fn sum_all_tape(&mut self, x: TensorId, ctx: &mut ForwardCtx<B>) -> Result<TensorId>
    where
        B::Tensor: Clone,
    {
        let x_val = self.get_value(x)?.clone();
        let ops = ctx.backend().ops();
        let shape = ops.shape(&x_val);
        let out = ops.sum_all(&x_val)?;
        Ok(self.record(&[x], out, move |grad_out, _store, ops| {
            let grad_x = ops.broadcast(grad_out, &shape)?;
            Ok(vec![grad_x])
        }))
    }

    /// Add a row vector to every row of a rank-2 tensor.
    ///
    /// Forward: `out = a + row` (row broadcast across batch dimension).
    ///
    /// Backward:
    /// - `dL/da = grad_out`
    /// - `dL/drow = sum_rows(grad_out)` (sum across batch dimension)
    pub fn add_row_vector_tape(
        &mut self,
        a: TensorId,
        row: TensorId,
        ctx: &mut ForwardCtx<B>,
    ) -> Result<TensorId>
    where
        B::Tensor: Clone,
    {
        let a_val = self.get_value(a)?.clone();
        let row_val = self.get_value(row)?.clone();
        let ops = ctx.backend().ops();

        let a_shape = ops.shape(&a_val);
        let row_shape = ops.shape(&row_val);
        if a_shape.len() != 2 || row_shape.len() != 1 || row_shape[0] != a_shape[1] {
            return Err(rustral_core::CoreError::InvalidArgument(format!(
                "add_row_vector_tape expects a:[batch, features] and row:[features], got a:{:?}, row:{:?}",
                a_shape, row_shape
            )));
        }
        let _batch = a_shape[0];
        let _features = a_shape[1];

        let out = ops.add_row_vector(&a_val, &row_val)?;
        Ok(self.record(&[a, row], out, move |grad_out, _store, ops| {
            // dL/da = grad_out
            let grad_a = grad_out.clone();

            // dL/drow = sum over batch dimension.
            let grad_row = ops.sum_dim0(grad_out)?;

            Ok(vec![grad_a, grad_row])
        }))
    }

    /// Linear layer forward pass using the tape.
    ///
    /// Note: For now, this delegates to matmul + add_row_vector.
    /// Full implementation would handle parameter gradients separately.
    pub fn linear_tape(
        &mut self,
        input: TensorId,
        weight_tensor: B::Tensor,
        bias_tensor: Option<B::Tensor>,
        ctx: &mut ForwardCtx<B>,
    ) -> Result<TensorId>
    where
        B::Tensor: Clone,
    {
        let input_val = self.get_value(input)?.clone();
        let ops = ctx.backend().ops();

        // For linear layer: input @ weight^T + bias
        // Transpose weight: [out_dim, in_dim] -> [in_dim, out_dim]
        let w_t = ops.transpose(&weight_tensor)?;

        // matmul: [batch, in_dim] @ [in_dim, out_dim] -> [batch, out_dim]
        let output = ops.matmul(&input_val, &w_t)?;

        // Add bias if present
        let output = if let Some(bias) = bias_tensor { ops.add_row_vector(&output, &bias)? } else { output };

        // Record the operation with gradient computation
        // Clone weight_tensor for the closure
        let weight_for_grad = weight_tensor.clone();
        Ok(self.record(&[input], output, move |grad_out, _store, ops| {
            // dL/d_input = grad_out @ weight
            let grad_input = ops.matmul(grad_out, &weight_for_grad)?;
            Ok(vec![grad_input])
        }))
    }

    /// Softmax activation along the last dimension.
    ///
    /// Accepts rank-1 `[features]` (treated as a single row) or rank-2 `[batch, features]`.
    /// Returns the same shape with each row summing to 1.
    ///
    /// Backward implements the full Jacobian:
    ///
    /// ```text
    /// dL/dx[i,j] = y[i,j] * (dL/dy[i,j] - sum_k(dL/dy[i,k] * y[i,k]))
    /// ```
    ///
    /// For rank-2 input we use axis-aware `sum_dim` + `broadcast_to`. For rank-1 input we
    /// fall back to the simpler `sum_all` + `broadcast` path which is exactly the same
    /// formula because the row dimension is implicit.
    pub fn softmax(&mut self, a: TensorId, ctx: &mut ForwardCtx<B>) -> Result<TensorId>
    where
        B::Tensor: Clone,
    {
        let a_val = self.get_value(a)?.clone();
        let ops_for_shape = ctx.backend().ops();
        let in_shape = ops_for_shape.shape(&a_val);
        let is_rank1 = in_shape.len() == 1;
        if !is_rank1 && in_shape.len() != 2 {
            return Err(rustral_core::CoreError::InvalidArgument(format!(
                "tape softmax expects rank-1 [features] or rank-2 [batch, features], got shape {:?}",
                in_shape
            )));
        }
        // Prefer axis-aware softmax when supported; fall back to row-wise default softmax.
        let last_dim = in_shape.len() - 1;
        let out = match ops_for_shape.softmax_dim(&a_val, last_dim) {
            Ok(t) => t,
            Err(_) => ops_for_shape.softmax(&a_val)?,
        };
        let out_shape = in_shape.clone();

        Ok(self.record(&[a], out.clone(), move |grad_out, _store, ops| {
            let y = &out;
            let grad_times_y = ops.mul(grad_out, y)?;
            let row_sum_broadcast = if is_rank1 {
                let scalar = ops.sum_all(&grad_times_y)?;
                ops.broadcast(&scalar, &out_shape)?
            } else {
                let row_sum = ops.sum_dim(&grad_times_y, 1, true)?;
                ops.broadcast_to(&row_sum, &out_shape)?
            };
            let inner = ops.sub(grad_out, &row_sum_broadcast)?;
            let grad = ops.mul(y, &inner)?;
            Ok(vec![grad])
        }))
    }

    /// Log-softmax activation (row-wise for 2D tensors).
    ///
    /// More numerically stable than log(softmax(x)).
    pub fn log_softmax(&mut self, a: TensorId, ctx: &mut ForwardCtx<B>) -> Result<TensorId>
    where
        B::Tensor: Clone,
    {
        let a_val = self.get_value(a)?.clone();
        let out = ctx.backend().ops().log_softmax(&a_val)?;

        // Store output value for backward computation
        Ok(self.record(&[a], out.clone(), move |grad_out, _store, ops| {
            // Log-softmax backward: grad_out - exp(log_softmax(x)) * sum(grad_out)
            // = grad_out - softmax(x) * sum(grad_out)
            let log_softmax_out = &out;

            // exp(log_softmax(x)) = softmax(x)
            let softmax_out = ops.exp(log_softmax_out)?;

            // sum(grad_out) - simplified to sum_all
            let _sum_grad = ops.sum_all(grad_out)?;

            // softmax_out * sum_grad - simplified, needs broadcasting for proper implementation
            // For now: return grad_out - softmax_out (simplified)
            let grad = ops.sub(grad_out, &softmax_out)?;

            Ok(vec![grad])
        }))
    }

    /// Cross-entropy loss with built-in log-softmax.
    ///
    /// Supports two target formats:
    ///
    /// - **Index targets**: `target` has shape `[batch]` and contains class indices as `f32`.
    /// - **One-hot targets**: `target` has shape `[batch, classes]`.
    ///
    /// Computes mean cross-entropy over batch:
    ///
    /// \[
    /// \mathrm{loss} = -\frac{1}{B}\sum_{i=0}^{B-1}\log p_{i,y_i}
    /// \]
    ///
    /// where \(p = \mathrm{softmax}(\mathrm{logits})\).
    ///
    /// Returns a scalar loss tensor and records gradients.
    pub fn cross_entropy_loss(
        &mut self,
        logits: TensorId,
        target: TensorId,
        ctx: &mut ForwardCtx<B>,
    ) -> Result<TensorId>
    where
        B::Tensor: Clone,
    {
        let logits_val = self.get_value(logits)?.clone();
        let target_val = self.get_value(target)?.clone();
        let ops = ctx.backend().ops();
        let logits_shape = ops.shape(&logits_val);
        if logits_shape.len() != 2 {
            return Err(rustral_core::CoreError::InvalidArgument(format!(
                "cross_entropy_loss expects logits rank-2 [batch, classes], got shape {:?}",
                logits_shape
            )));
        }
        let batch = logits_shape[0].max(1);
        let classes = logits_shape[1].max(1);
        let inv_batch = 1.0 / batch as f32;

        // Prefer axis-aware log-softmax when available (ndarray + candle implement it).
        // Fall back to a manual row-wise implementation otherwise.
        let log_probs = match ops.log_softmax_dim(&logits_val, 1) {
            Ok(t) => t,
            Err(_) => {
                let logits_vec = ops.tensor_to_vec(&logits_val)?;
                if logits_vec.len() != batch * classes {
                    return Err(rustral_core::CoreError::InvalidArgument(format!(
                        "cross_entropy_loss: logits had {} elements, expected {}",
                        logits_vec.len(),
                        batch * classes
                    )));
                }
                let mut log_probs_vec = vec![0.0f32; batch * classes];
                for i in 0..batch {
                    let row = &logits_vec[i * classes..i * classes + classes];
                    let mut row_max = f32::NEG_INFINITY;
                    for &v in row {
                        if v > row_max {
                            row_max = v;
                        }
                    }
                    let mut sum_exp = 0.0f32;
                    for &v in row {
                        sum_exp += (v - row_max).exp();
                    }
                    let log_denom = row_max + sum_exp.ln();
                    for j in 0..classes {
                        log_probs_vec[i * classes + j] = row[j] - log_denom;
                    }
                }
                ops.tensor_from_vec(log_probs_vec, &[batch, classes])?
            }
        };
        let target_shape = ops.shape(&target_val);

        // Build a one-hot tensor for the backward formula and compute the forward loss.
        // We use one-hot in all cases to keep the gradient path uniform.
        let (target_one_hot, loss) = if target_shape == logits_shape {
            // One-hot targets: loss = -sum(target * log_probs) / batch
            let neg_log_probs = ops.neg(&log_probs)?;
            let loss_per_elem = ops.mul(&target_val, &neg_log_probs)?;
            let loss_sum = ops.sum_all(&loss_per_elem)?;
            let loss = ops.mul_scalar(&loss_sum, inv_batch)?;
            (target_val.clone(), loss)
        } else if target_shape.len() == 1 && target_shape[0] == batch {
            // Index targets: compute -mean(log_probs[i, y_i]) and also build one-hot.
            let target_idx_f32 = ops.tensor_to_vec(&target_val)?;
            if target_idx_f32.len() != batch {
                return Err(rustral_core::CoreError::InvalidArgument(format!(
                    "cross_entropy_loss: target had {} elements, expected {}",
                    target_idx_f32.len(),
                    batch
                )));
            }

            let mut one_hot = vec![0.0f32; batch * classes];
            let log_probs_vec = ops.tensor_to_vec(&log_probs)?;
            let mut nll_sum = 0.0f32;
            for i in 0..batch {
                let yi = target_idx_f32[i] as isize;
                if yi < 0 || yi as usize >= classes {
                    return Err(rustral_core::CoreError::InvalidArgument(format!(
                        "cross_entropy_loss: target index {} out of range [0, {}) at batch {}",
                        yi, classes, i
                    )));
                }
                let yi = yi as usize;
                one_hot[i * classes + yi] = 1.0;
                nll_sum += -log_probs_vec[i * classes + yi];
            }

            let loss = ops.tensor_from_vec(vec![nll_sum * inv_batch], &[1])?;
            let one_hot_t = ops.tensor_from_vec(one_hot, &[batch, classes])?;
            (one_hot_t, loss)
        } else {
            return Err(rustral_core::CoreError::InvalidArgument(format!(
                "cross_entropy_loss target must be [batch] indices or [batch, classes] one-hot; got {:?} with logits {:?}",
                target_shape, logits_shape
            )));
        };

        // Record with backward pass: gradient is softmax(logits) - target
        let logits_for_grad = logits_val.clone();
        let target_one_hot_for_grad = target_one_hot.clone();
        let scale_for_grad = inv_batch;
        let batch_for_grad = batch;
        let classes_for_grad = classes;

        Ok(self.record(&[logits, target], loss, move |grad_out, _store, ops| {
            // grad_out is a scalar for this loss; propagate scaling for composed losses.
            let upstream = ops.tensor_to_vec(grad_out)?.first().copied().unwrap_or(1.0);
            let grad_scale = upstream * scale_for_grad;

            // Gradient w.r.t. logits: softmax(logits) - target
            let log_probs_grad = match ops.log_softmax_dim(&logits_for_grad, 1) {
                Ok(t) => t,
                Err(_) => {
                    // Manual row-wise fallback.
                    let logits_vec = ops.tensor_to_vec(&logits_for_grad)?;
                    if logits_vec.len() != batch_for_grad * classes_for_grad {
                        return Err(rustral_core::CoreError::InvalidArgument(format!(
                            "cross_entropy_loss backward: logits had {} elements, expected {}",
                            logits_vec.len(),
                            batch_for_grad * classes_for_grad
                        )));
                    }
                    let mut log_probs_vec = vec![0.0f32; batch_for_grad * classes_for_grad];
                    for i in 0..batch_for_grad {
                        let row =
                            &logits_vec[i * classes_for_grad..i * classes_for_grad + classes_for_grad];
                        let mut row_max = f32::NEG_INFINITY;
                        for &v in row {
                            if v > row_max {
                                row_max = v;
                            }
                        }
                        let mut sum_exp = 0.0f32;
                        for &v in row {
                            sum_exp += (v - row_max).exp();
                        }
                        let log_denom = row_max + sum_exp.ln();
                        for j in 0..classes_for_grad {
                            log_probs_vec[i * classes_for_grad + j] = row[j] - log_denom;
                        }
                    }
                    ops.tensor_from_vec(log_probs_vec, &[batch_for_grad, classes_for_grad])?
                }
            };
            let softmax_out = ops.exp(&log_probs_grad)?;
            let grad_logits_unscaled = ops.sub(&softmax_out, &target_one_hot_for_grad)?;
            let grad_logits = ops.mul_scalar(&grad_logits_unscaled, grad_scale)?;

            // Targets are treated as constants (especially for index targets).
            // Provide a zero gradient with the same shape as the provided target tensor.
            let grad_target = ops.zeros(&ops.shape(&target_one_hot_for_grad))?;

            Ok(vec![grad_logits, grad_target])
        }))
    }

    /// Mean-squared error loss.
    ///
    /// Computes: \(\mathrm{mean}((pred - target)^2)\) over all elements.
    /// Returns a scalar loss tensor and records gradients.
    pub fn mse_loss(&mut self, pred: TensorId, target: TensorId, ctx: &mut ForwardCtx<B>) -> Result<TensorId>
    where
        B::Tensor: Clone,
    {
        let pred_val = self.get_value(pred)?.clone();
        let target_val = self.get_value(target)?.clone();
        let ops = ctx.backend().ops();
        let shape = ops.shape(&pred_val);
        let elem_count: usize = shape.iter().product();
        if elem_count == 0 {
            return Err(rustral_core::CoreError::InvalidArgument("mse_loss: empty tensor".into()));
        }
        let inv_n = 1.0 / elem_count as f32;

        let diff = ops.sub(&pred_val, &target_val)?;
        let sq = ops.mul(&diff, &diff)?;
        let sum = ops.sum_all(&sq)?;
        let loss = ops.mul_scalar(&sum, inv_n)?;

        let pred_for_grad = pred_val.clone();
        let target_for_grad = target_val.clone();
        Ok(self.record(&[pred, target], loss, move |grad_out, _store, ops| {
            let upstream = ops.tensor_to_vec(grad_out)?.first().copied().unwrap_or(1.0);
            let scale = upstream * (2.0 * inv_n);
            let diff = ops.sub(&pred_for_grad, &target_for_grad)?;
            let grad_pred = ops.mul_scalar(&diff, scale)?;
            let grad_target = ops.mul_scalar(&diff, -scale)?;
            Ok(vec![grad_pred, grad_target])
        }))
    }

    /// Gather rows from a parameter table using indices.
    ///
    /// Forward: `output[i] = table[ids[i]]`
    /// Backward: gradients accumulate into table rows
    pub fn gather_rows_tape(
        &mut self,
        table: &rustral_core::Parameter<B>,
        ids: TensorId,
        ctx: &mut ForwardCtx<B>,
    ) -> Result<TensorId>
    where
        B::Tensor: Clone,
    {
        let table_id = self.watch_parameter(table);
        let ids_val = self.get_value(ids)?.clone();
        let ops = ctx.backend().ops();

        // Extract indices from tensor
        let ids_shape = ops.shape(&ids_val);
        let num_indices = ids_shape.iter().product::<usize>();

        // Read indices in a single bulk transfer.
        // Indices are stored as f32 and converted to usize.
        let ids_f32 = ops.tensor_to_vec(&ids_val)?;
        if ids_f32.len() != num_indices {
            return Err(rustral_core::CoreError::InvalidArgument(format!(
                "ids tensor had {} elements but shape implies {}",
                ids_f32.len(),
                num_indices
            )));
        }

        let id_vec: Vec<usize> = ids_f32.iter().map(|v| *v as usize).collect();

        // Forward: gather rows from table
        let output = ops.gather_rows(table, &id_vec)?;

        // Clone for closure
        let ids_shape_for_grad = ids_shape.clone();
        let table_shape_for_grad = ops.shape(table.tensor());
        let id_vec_for_grad = id_vec.clone();

        Ok(self.record(&[table_id, ids], output, move |grad_out, _store, ops| {
            // Backward:
            // - ids are non-differentiable => grad_ids = 0
            // - table grad accumulates per used row: grad_table[row] += grad_out[i] for each i with ids[i] == row
            //
            // NOTE: This uses a single bulk readback of `grad_out` to build a dense table gradient.
            // It is correct, but GPU backends will pay for that sync. Later we can add a backend scatter op.
            let table_shape = table_shape_for_grad.clone();
            let out_shape = ops.shape(grad_out);
            if out_shape.len() != 2 || out_shape[0] != id_vec_for_grad.len() {
                return Err(rustral_core::CoreError::InvalidArgument(format!(
                    "gather_rows_tape grad_out had shape {:?}, expected [{}, dim]",
                    out_shape,
                    id_vec_for_grad.len()
                )));
            }
            let dim = out_shape[1];
            if table_shape.len() != 2 || table_shape[1] != dim {
                return Err(rustral_core::CoreError::InvalidArgument(format!(
                    "gather_rows_tape table had shape {:?}, grad_out had shape {:?}",
                    table_shape, out_shape
                )));
            }

            let grad_out_vals = ops.tensor_to_vec(grad_out)?;
            let mut grad_table_vals = vec![0.0f32; table_shape[0] * table_shape[1]];

            for (i, &row) in id_vec_for_grad.iter().enumerate() {
                if row >= table_shape[0] {
                    return Err(rustral_core::CoreError::InvalidArgument(format!(
                        "gather_rows_tape id {} out of range for vocab_size {}",
                        row, table_shape[0]
                    )));
                }
                let src_off = i * dim;
                let dst_off = row * dim;
                for j in 0..dim {
                    grad_table_vals[dst_off + j] += grad_out_vals[src_off + j];
                }
            }

            let grad_table = ops.tensor_from_vec(grad_table_vals, &table_shape)?;
            let grad_ids = ops.zeros(&ids_shape_for_grad)?;
            Ok(vec![grad_table, grad_ids])
        }))
    }

    /// Concatenate tensors along a dimension.
    ///
    /// Forward: output = concat([t1, t2, ...], dim)
    /// Backward: split grad_out into pieces matching input shapes
    pub fn concat_tape(
        &mut self,
        inputs: &[TensorId],
        dim: usize,
        ctx: &mut ForwardCtx<B>,
    ) -> Result<TensorId>
    where
        B::Tensor: Clone,
    {
        let ops = ctx.backend().ops();

        // Collect input tensors and their shapes
        let input_tensors: Vec<B::Tensor> = inputs
            .iter()
            .map(|&id| self.get_value(id))
            .collect::<Result<Vec<_>>>()?
            .into_iter()
            .cloned()
            .collect();
        let input_shapes: Vec<Vec<usize>> = input_tensors.iter().map(|t| ops.shape(t)).collect();

        // Forward: concatenate
        let input_refs: Vec<&B::Tensor> = input_tensors.iter().collect();
        let output = ops.concat(&input_refs, dim)?;

        // Store shapes for backward
        let input_shapes_for_grad = input_shapes.clone();

        Ok(self.record(inputs, output, move |grad_out, _store, ops| {
            // Backward: split the gradient along the concatenation dimension
            // For each input, extract the corresponding slice of grad_out
            let mut grads = Vec::with_capacity(input_shapes_for_grad.len());
            let mut offset = 0;

            for shape in &input_shapes_for_grad {
                let size = shape[dim];
                // Simplified: just slice the grad_out
                // Full implementation needs to handle multi-dimensional slicing
                let grad_slice = ops.slice(grad_out, offset, offset + size)?;
                grads.push(grad_slice);
                offset += size;
            }
            Ok(grads)
        }))
    }

    /// Slice a tensor along dimension 0.
    ///
    /// Forward: `output = input[start..end]` along dim 0.
    /// Backward: routes `grad_out` back into the correct slice of an otherwise-zero tensor
    /// of the input shape, using `concat([zeros_before, grad_out, zeros_after], dim=0)`.
    pub fn slice_tape(
        &mut self,
        input: TensorId,
        start: usize,
        end: usize,
        ctx: &mut ForwardCtx<B>,
    ) -> Result<TensorId>
    where
        B::Tensor: Clone,
    {
        let input_val = self.get_value(input)?.clone();
        let ops = ctx.backend().ops();

        let output = ops.slice(&input_val, start, end)?;
        let input_shape = ops.shape(&input_val);

        Ok(self.record(&[input], output, move |grad_out, _store, ops| {
            let dim0 = input_shape[0];
            // Fast paths.
            if start == 0 && end == dim0 {
                return Ok(vec![grad_out.clone()]);
            }
            // Build [zeros_before? , grad_out, zeros_after?] then concat along dim 0.
            let mut before_shape = input_shape.clone();
            before_shape[0] = start;
            let mut after_shape = input_shape.clone();
            after_shape[0] = dim0.saturating_sub(end);

            let zeros_before = if start > 0 { Some(ops.zeros(&before_shape)?) } else { None };
            let zeros_after = if dim0 > end { Some(ops.zeros(&after_shape)?) } else { None };
            let mut owned: Vec<B::Tensor> = Vec::new();
            if let Some(z) = zeros_before {
                owned.push(z);
            }
            owned.push(grad_out.clone());
            if let Some(z) = zeros_after {
                owned.push(z);
            }
            let refs: Vec<&B::Tensor> = owned.iter().collect();
            let grad_input = ops.concat(&refs, 0)?;
            Ok(vec![grad_input])
        }))
    }

    /// Reshape a tensor.
    ///
    /// Forward: output = reshape(input, new_shape)
    /// Backward: reshape grad_out back to input shape
    pub fn reshape_tape(
        &mut self,
        input: TensorId,
        new_shape: &[usize],
        ctx: &mut ForwardCtx<B>,
    ) -> Result<TensorId>
    where
        B::Tensor: Clone,
    {
        let input_val = self.get_value(input)?.clone();
        let ops = ctx.backend().ops();

        // Forward: reshape
        let output = ops.reshape(&input_val, new_shape)?;

        // Store input shape for backward
        let input_shape = ops.shape(&input_val);

        Ok(self.record(&[input], output, move |grad_out, _store, ops| {
            // Backward: reshape gradient back to input shape
            let grad_input = ops.reshape(grad_out, &input_shape)?;
            Ok(vec![grad_input])
        }))
    }

    /// Layer normalization with gradient tracking.
    ///
    /// Forward: y = (x - mean) / sqrt(var + eps) * gamma + beta
    ///
    /// Normalizes over the last `feature_count` elements of the flattened input (row-major),
    /// matching `LayerNorm` when `normalized_shape` is a single dimension or its product equals
    /// `feature_count` and the input is contiguous in that layout.
    ///
    /// Uses bulk `tensor_to_vec` for forward/backward (no per-element reads in the hot path).
    ///
    /// Backward: gradients w.r.t. input, gamma, and beta (gamma/beta must be registered via
    /// `watch_parameter` inside this call).
    pub fn layer_norm_tape(
        &mut self,
        input: TensorId,
        gamma: &rustral_core::Parameter<B>,
        beta: &rustral_core::Parameter<B>,
        eps: f32,
        feature_count: usize,
        ctx: &mut ForwardCtx<B>,
    ) -> Result<TensorId>
    where
        B::Tensor: Clone,
    {
        let gamma_id = self.watch_parameter(gamma);
        let beta_id = self.watch_parameter(beta);

        let input_val = self.get_value(input)?.clone();
        let ops = ctx.backend().ops();
        let input_shape = ops.shape(&input_val);
        let total_elems: usize = input_shape.iter().product();

        if feature_count == 0 {
            return Err(rustral_core::CoreError::InvalidArgument(
                "layer_norm_tape: feature_count must be > 0".into(),
            ));
        }
        if total_elems % feature_count != 0 {
            return Err(rustral_core::CoreError::InvalidShape {
                shape: input_shape.clone(),
                reason: format!(
                    "layer_norm_tape: input element count {} not divisible by feature_count {}",
                    total_elems, feature_count
                ),
            });
        }
        let num_groups = total_elems / feature_count;

        let gamma_tensor = gamma.tensor();
        let beta_tensor = beta.tensor();
        let gamma_shape = ops.shape(gamma_tensor);
        let beta_shape = ops.shape(beta_tensor);
        let gamma_elems: usize = gamma_shape.iter().product();
        let beta_elems: usize = beta_shape.iter().product();
        if gamma_elems != feature_count || beta_elems != feature_count {
            return Err(rustral_core::CoreError::InvalidArgument(format!(
                "layer_norm_tape: gamma shape {:?}, beta shape {:?}, expected {} elements each",
                gamma_shape, beta_shape, feature_count
            )));
        }

        let input_values = ops.tensor_to_vec(&input_val)?;
        if input_values.len() != total_elems {
            return Err(rustral_core::CoreError::InvalidArgument(
                "layer_norm_tape: input tensor_to_vec length mismatch".into(),
            ));
        }
        let gamma_values = ops.tensor_to_vec(gamma_tensor)?;
        let beta_values = ops.tensor_to_vec(beta_tensor)?;

        let mut output_values = vec![0.0f32; total_elems];
        let mut stds = vec![0.0f32; num_groups];
        let mut normalized_values = vec![0.0f32; total_elems];

        for g in 0..num_groups {
            let group_start = g * feature_count;

            let sum: f32 = input_values[group_start..group_start + feature_count].iter().sum();
            let mean = sum / feature_count as f32;

            let var_sum: f32 = input_values[group_start..group_start + feature_count]
                .iter()
                .map(|&x| (x - mean).powi(2))
                .sum();
            let var = var_sum / feature_count as f32;
            let std = (var + eps).sqrt();
            stds[g] = std;

            for i in 0..feature_count {
                let idx = group_start + i;
                let normalized = (input_values[idx] - mean) / std;
                normalized_values[idx] = normalized;
                let gv = gamma_values[i];
                let bv = beta_values[i];
                output_values[idx] = normalized * gv + bv;
            }
        }

        let output = ops.tensor_from_vec(output_values, &input_shape)?;

        let input_shape_for_grad = input_shape.clone();
        let gamma_values_for_grad = gamma_values.clone();
        let stds_for_grad = stds.clone();
        let normalized_values_for_grad = normalized_values.clone();
        let feature_count_for_grad = feature_count;
        let num_groups_for_grad = num_groups;

        Ok(self.record(&[input, gamma_id, beta_id], output, move |grad_out, _store, ops| {
            let total_elems = input_shape_for_grad.iter().product();
            let grad_out_values = ops.tensor_to_vec(grad_out)?;
            if grad_out_values.len() != total_elems {
                return Err(rustral_core::CoreError::InvalidArgument(
                    "layer_norm_tape backward: grad_out length mismatch".into(),
                ));
            }

            let mut grad_input = vec![0.0f32; total_elems];
            let mut grad_gamma = vec![0.0f32; feature_count_for_grad];
            let mut grad_beta = vec![0.0f32; feature_count_for_grad];

            for g in 0..num_groups_for_grad {
                let group_start = g * feature_count_for_grad;
                let std = stds_for_grad[g];
                let inv_std = 1.0 / std;

                let mut grad_normalized = vec![0.0f32; feature_count_for_grad];
                for i in 0..feature_count_for_grad {
                    let gamma = gamma_values_for_grad[i];
                    grad_normalized[i] = grad_out_values[group_start + i] * gamma;
                    grad_gamma[i] +=
                        grad_out_values[group_start + i] * normalized_values_for_grad[group_start + i];
                    grad_beta[i] += grad_out_values[group_start + i];
                }

                let grad_normalized_sum: f32 = grad_normalized.iter().sum();
                let grad_normalized_dot_n: f32 = (0..feature_count_for_grad)
                    .map(|i| grad_normalized[i] * normalized_values_for_grad[group_start + i])
                    .sum();

                let n = feature_count_for_grad as f32;
                for i in 0..feature_count_for_grad {
                    let normalized = normalized_values_for_grad[group_start + i];
                    let grad = inv_std
                        * (grad_normalized[i]
                            - grad_normalized_sum / n
                            - normalized * grad_normalized_dot_n / n);
                    grad_input[group_start + i] = grad;
                }
            }

            let grad_input_tensor = ops.tensor_from_vec(grad_input, &input_shape_for_grad)?;
            let grad_gamma_tensor = ops.tensor_from_vec(grad_gamma, &[feature_count_for_grad])?;
            let grad_beta_tensor = ops.tensor_from_vec(grad_beta, &[feature_count_for_grad])?;
            Ok(vec![grad_input_tensor, grad_gamma_tensor, grad_beta_tensor])
        }))
    }

    /// Compute gradients from an output tensor.
    ///
    /// For a scalar loss, this computes gradients w.r.t. all inputs.
    /// For a tensor output, this computes gradients assuming a scalar loss
    /// that is the sum of all output elements (i.e., seed gradient is all ones).
    ///
    /// Returns a map from input tensor ids to their gradients.
    ///
    /// # Type Parameters
    /// - `F`: A function to create a tensor from data (backend-specific).
    pub fn backward<F>(
        mut self,
        output: TensorId,
        make_ones: F,
        ops: &dyn rustral_core::TensorOps<B>,
    ) -> Result<GradientStore<B>>
    where
        B::Tensor: Clone,
        F: FnOnce(Vec<f32>, &[usize]) -> Result<B::Tensor>,
    {
        // Seed gradient at output: d(output)/d(output) = 1.0 for each element
        // This assumes the loss is the sum of all output elements
        let out_val = self.values.get(&output).ok_or_else(|| {
            rustral_core::CoreError::InvalidArgument(format!(
                "Output tensor {:?} not found for backward",
                output.0
            ))
        })?;
        let out_shape = ops.shape(out_val);
        let ones = vec![1.0f32; out_shape.iter().product()];
        let seed = make_ones(ones, &out_shape)?;
        self.grads.insert(output, seed);

        // Reverse through operations
        for op in self.ops.iter().rev() {
            let out_grad = match self.grads.get(&op.output) {
                Some(g) => g.clone(),
                None => continue, // No gradient flowing to this output
            };

            let in_grads = (op.backward)(&out_grad, &mut self.grads, ops)?;

            for (input_id, grad) in op.inputs.iter().zip(in_grads.into_iter()) {
                if let Some(existing) = self.grads.get_mut(input_id) {
                    // Accumulate gradient: grad_new = grad_old + grad
                    let sum = ops.add(existing, &grad)?;
                    *existing = sum;
                } else {
                    self.grads.insert(*input_id, grad);
                }
            }
        }

        Ok(self.grads)
    }

    /// Get the gradient for a specific tensor id after backward.
    pub fn grad(&self, id: TensorId) -> Option<&B::Tensor> {
        self.grads.get(&id)
    }
}

/// Extension trait for `Parameter` to extract gradients.
pub trait GradExt<B: Backend> {
    /// Get the gradient for this parameter from the tape.
    fn gradient<'a>(&self, tape: &'a Tape<B>) -> Option<&'a B::Tensor>;
}

impl<B: Backend> GradExt<B> for Parameter<B> {
    fn gradient<'a>(&self, tape: &'a Tape<B>) -> Option<&'a B::Tensor> {
        // Use the parameter map to find the corresponding tensor id
        tape.param_tensor_id(self.id()).and_then(|tensor_id| tape.grad(tensor_id))
    }
}

/// Extension trait for `Parameter` to extract gradients after backward.
pub trait GradExtFromStore<B: Backend> {
    /// Get the gradient for this parameter from the gradient store using the param map.
    fn gradient_from_store<'a>(
        &self,
        grads: &'a GradientStore<B>,
        param_map: &ParameterMap,
    ) -> Option<&'a B::Tensor>;
}

impl<B: Backend> GradExtFromStore<B> for Parameter<B> {
    fn gradient_from_store<'a>(
        &self,
        grads: &'a GradientStore<B>,
        param_map: &ParameterMap,
    ) -> Option<&'a B::Tensor> {
        param_map.get(&self.id()).and_then(|&tensor_id| grads.get(&tensor_id))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustral_core::{ForwardCtx, Mode};
    use rustral_ndarray_backend::CpuBackend;

    #[test]
    fn test_tape_creation() {
        let mut tape = Tape::<CpuBackend>::new();
        let backend = CpuBackend::default();
        let tensor = backend.tensor_from_vec(vec![2.0], &[1]).unwrap();
        let id = tape.watch(tensor);

        assert!(tape.value(id).is_some());
    }

    #[test]
    fn test_simple_add() {
        let backend = CpuBackend::default();
        let mut ctx = ForwardCtx::new(&backend, Mode::Train);
        let mut tape = Tape::<CpuBackend>::new();

        let x = tape.watch(backend.tensor_from_vec(vec![2.0], &[1]).unwrap());
        let y = tape.watch(backend.tensor_from_vec(vec![3.0], &[1]).unwrap());
        let z = tape.add(x, y, &mut ctx).unwrap();

        // z = x + y = 5.0
        let z_val: Vec<f32> = tape.value(z).unwrap().values().to_vec();
        assert!((z_val[0] - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_matmul_backward() {
        let backend = CpuBackend::default();
        let mut ctx = ForwardCtx::new(&backend, Mode::Train);
        let mut tape = Tape::<CpuBackend>::new();

        // A: [2, 3] = [[1, 2, 3], [4, 5, 6]]
        let a = tape.watch(backend.tensor_from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap());
        // B: [3, 2] = [[1, 2], [3, 4], [5, 6]]
        let b = tape.watch(backend.tensor_from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]).unwrap());

        // C = A @ B: [2, 2]
        // [[1*1+2*3+3*5, 1*2+2*4+3*6], [4*1+5*3+6*5, 4*2+5*4+6*6]]
        // = [[22, 28], [49, 64]]
        let c = tape.matmul(a, b, &mut ctx).unwrap();

        let c_vals: Vec<f32> = tape.value(c).unwrap().values().to_vec();
        assert_eq!(c_vals.len(), 4);
        assert!((c_vals[0] - 22.0).abs() < 1e-5, "C[0,0] expected 22, got {}", c_vals[0]);
        assert!((c_vals[1] - 28.0).abs() < 1e-5, "C[0,1] expected 28, got {}", c_vals[1]);
        assert!((c_vals[2] - 49.0).abs() < 1e-5, "C[1,0] expected 49, got {}", c_vals[2]);
        assert!((c_vals[3] - 64.0).abs() < 1e-5, "C[1,1] expected 64, got {}", c_vals[3]);

        // Backward: dL/dA = dL/dC @ B^T
        // For scalar loss = sum(C), dL/dC = ones [2, 2]
        let grads =
            tape.backward(c, |data, shape| backend.tensor_from_vec(data, shape), backend.ops()).unwrap();

        // Check gradients exist
        assert!(grads.contains_key(&a));
        assert!(grads.contains_key(&b));
    }

    #[test]
    fn test_watch_parameter_and_param_map() {
        let backend = CpuBackend::default();
        let mut tape = Tape::<CpuBackend>::new();
        let tensor = backend.tensor_from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();
        let param = Parameter::new("test_param", tensor.clone());

        let id = tape.watch_parameter(&param);
        assert_eq!(tape.param_tensor_id(param.id()), Some(id));
        assert_eq!(tape.param_map().len(), 1);
        assert!(tape.param_map().contains_key(&param.id()));
    }

    #[test]
    fn test_mul_and_backward() {
        let backend = CpuBackend::default();
        let mut ctx = ForwardCtx::new(&backend, Mode::Train);
        let mut tape = Tape::<CpuBackend>::new();

        let a = tape.watch(backend.tensor_from_vec(vec![2.0], &[1]).unwrap());
        let b = tape.watch(backend.tensor_from_vec(vec![3.0], &[1]).unwrap());
        let c = tape.mul(a, b, &mut ctx).unwrap();

        let c_vals: Vec<f32> = tape.value(c).unwrap().values().to_vec();
        assert!((c_vals[0] - 6.0).abs() < 1e-6);

        let grads =
            tape.backward(c, |data, shape| backend.tensor_from_vec(data, shape), backend.ops()).unwrap();
        assert!(grads.contains_key(&a));
        assert!(grads.contains_key(&b));
    }

    #[test]
    fn test_relu_and_backward() {
        let backend = CpuBackend::default();
        let mut ctx = ForwardCtx::new(&backend, Mode::Train);
        let mut tape = Tape::<CpuBackend>::new();

        let a = tape.watch(backend.tensor_from_vec(vec![-1.0, 2.0, -3.0, 4.0], &[4]).unwrap());
        let b = tape.relu(a, &mut ctx).unwrap();

        let b_vals: Vec<f32> = tape.value(b).unwrap().values().to_vec();
        assert!((b_vals[0] - 0.0).abs() < 1e-6);
        assert!((b_vals[1] - 2.0).abs() < 1e-6);
        assert!((b_vals[2] - 0.0).abs() < 1e-6);
        assert!((b_vals[3] - 4.0).abs() < 1e-6);

        let grads =
            tape.backward(b, |data, shape| backend.tensor_from_vec(data, shape), backend.ops()).unwrap();
        assert!(grads.contains_key(&a));
    }

    #[test]
    fn test_linear_tape() {
        let backend = CpuBackend::default();
        let mut ctx = ForwardCtx::new(&backend, Mode::Train);
        let mut tape = Tape::<CpuBackend>::new();

        let input = tape.watch(backend.tensor_from_vec(vec![1.0, 2.0], &[1, 2]).unwrap());
        let weight = backend.tensor_from_vec(vec![1.0, 0.0, 0.0, 1.0], &[2, 2]).unwrap();
        let bias = Some(backend.tensor_from_vec(vec![0.5, 0.5], &[2]).unwrap());

        let output = tape.linear_tape(input, weight, bias, &mut ctx).unwrap();
        let out_vals: Vec<f32> = tape.value(output).unwrap().values().to_vec();
        assert_eq!(out_vals.len(), 2);

        let grads =
            tape.backward(output, |data, shape| backend.tensor_from_vec(data, shape), backend.ops()).unwrap();
        assert!(grads.contains_key(&input));
    }

    #[test]
    fn test_softmax_and_backward() {
        let backend = CpuBackend::default();
        let mut ctx = ForwardCtx::new(&backend, Mode::Train);
        let mut tape = Tape::<CpuBackend>::new();

        let a = tape.watch(backend.tensor_from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap());
        let b = tape.softmax(a, &mut ctx).unwrap();

        let b_vals: Vec<f32> = tape.value(b).unwrap().values().to_vec();
        assert!((b_vals.iter().sum::<f32>() - 1.0).abs() < 1e-5);

        let grads =
            tape.backward(b, |data, shape| backend.tensor_from_vec(data, shape), backend.ops()).unwrap();
        assert!(grads.contains_key(&a));
    }

    #[test]
    fn test_log_softmax_and_backward() {
        let backend = CpuBackend::default();
        let mut ctx = ForwardCtx::new(&backend, Mode::Train);
        let mut tape = Tape::<CpuBackend>::new();

        let a = tape.watch(backend.tensor_from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap());
        let b = tape.log_softmax(a, &mut ctx).unwrap();

        let b_vals: Vec<f32> = tape.value(b).unwrap().values().to_vec();
        assert!(b_vals.iter().all(|&v| v <= 0.0));

        let grads =
            tape.backward(b, |data, shape| backend.tensor_from_vec(data, shape), backend.ops()).unwrap();
        assert!(grads.contains_key(&a));
    }

    #[test]
    fn test_cross_entropy_loss() {
        let backend = CpuBackend::default();
        let mut ctx = ForwardCtx::new(&backend, Mode::Train);
        let mut tape = Tape::<CpuBackend>::new();

            let logits = tape.watch(backend.tensor_from_vec(vec![1.0, 2.0, 3.0, 0.0, -1.0, 2.0], &[2, 3]).unwrap());

            // Index targets (encoded as f32 ids): [batch]
            let target = tape.watch(backend.tensor_from_vec(vec![2.0, 0.0], &[2]).unwrap());
        let loss = tape.cross_entropy_loss(logits, target, &mut ctx).unwrap();

            let loss_val: Vec<f32> = tape.value(loss).unwrap().values().to_vec();
        assert_eq!(loss_val.len(), 1);
        assert!(loss_val[0] > 0.0);

        let grads =
            tape.backward(loss, |data, shape| backend.tensor_from_vec(data, shape), backend.ops()).unwrap();
        assert!(grads.contains_key(&logits));
        assert!(grads.contains_key(&target));

            // For cross-entropy, per-row gradient sums to ~0 (softmax - one_hot).
            let grad_logits = grads.get(&logits).unwrap();
            let g = backend.ops().tensor_to_vec(grad_logits).unwrap();
            assert_eq!(g.len(), 2 * 3);
            for row in 0..2 {
                let s: f32 = g[row * 3..row * 3 + 3].iter().sum();
                assert!(s.abs() <= 1e-5, "row {row} grad sum expected ~0, got {s}");
            }
    }

    #[test]
    fn test_mse_loss() {
        let backend = CpuBackend::default();
        let mut ctx = ForwardCtx::new(&backend, Mode::Train);
        let mut tape = Tape::<CpuBackend>::new();

        let pred = tape.watch(backend.tensor_from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap());
        let target = tape.watch(backend.tensor_from_vec(vec![0.0, 0.0, 0.0, 0.0], &[2, 2]).unwrap());
        let loss = tape.mse_loss(pred, target, &mut ctx).unwrap();

        let loss_val: Vec<f32> = tape.value(loss).unwrap().values().to_vec();
        assert_eq!(loss_val.len(), 1);
        // mean([1,4,9,16]) = 7.5
        assert!((loss_val[0] - 7.5).abs() < 1e-4);

        let grads =
            tape.backward(loss, |data, shape| backend.tensor_from_vec(data, shape), backend.ops()).unwrap();
        assert!(grads.contains_key(&pred));
        assert!(grads.contains_key(&target));
    }

    #[test]
    fn test_gather_rows_tape() {
        let backend = CpuBackend::default();
        let mut ctx = ForwardCtx::new(&backend, Mode::Train);
        let mut tape = Tape::<CpuBackend>::new();

        let table_data = backend.tensor_from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
        let table = Parameter::new("table", table_data);
        let ids = tape.watch(backend.tensor_from_vec(vec![0.0, 1.0], &[2]).unwrap());

        let output = tape.gather_rows_tape(&table, ids, &mut ctx).unwrap();
        let out_vals: Vec<f32> = tape.value(output).unwrap().values().to_vec();
        assert_eq!(out_vals.len(), 6);

        let grads =
            tape.backward(output, |data, shape| backend.tensor_from_vec(data, shape), backend.ops()).unwrap();
        assert!(grads.contains_key(&ids));
    }

    #[test]
    fn test_concat_tape() {
        let backend = CpuBackend::default();
        let mut ctx = ForwardCtx::new(&backend, Mode::Train);
        let mut tape = Tape::<CpuBackend>::new();

        let a = tape.watch(backend.tensor_from_vec(vec![1.0, 2.0], &[2]).unwrap());
        let b = tape.watch(backend.tensor_from_vec(vec![3.0, 4.0], &[2]).unwrap());
        let c = tape.concat_tape(&[a, b], 0, &mut ctx).unwrap();

        let c_vals: Vec<f32> = tape.value(c).unwrap().values().to_vec();
        assert_eq!(c_vals.len(), 4);

        let grads =
            tape.backward(c, |data, shape| backend.tensor_from_vec(data, shape), backend.ops()).unwrap();
        assert!(grads.contains_key(&a));
        assert!(grads.contains_key(&b));
    }

    #[test]
    fn test_slice_tape() {
        let backend = CpuBackend::default();
        let mut ctx = ForwardCtx::new(&backend, Mode::Train);
        let mut tape = Tape::<CpuBackend>::new();

        let a = tape.watch(backend.tensor_from_vec(vec![1.0, 2.0, 3.0, 4.0], &[4]).unwrap());
        let b = tape.slice_tape(a, 1, 3, &mut ctx).unwrap();

        let b_vals: Vec<f32> = tape.value(b).unwrap().values().to_vec();
        assert_eq!(b_vals.len(), 2);

        let grads =
            tape.backward(b, |data, shape| backend.tensor_from_vec(data, shape), backend.ops()).unwrap();
        assert!(grads.contains_key(&a));
    }

    #[test]
    fn test_reshape_tape() {
        let backend = CpuBackend::default();
        let mut ctx = ForwardCtx::new(&backend, Mode::Train);
        let mut tape = Tape::<CpuBackend>::new();

        let a = tape.watch(backend.tensor_from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap());
        let b = tape.reshape_tape(a, &[4], &mut ctx).unwrap();

        let b_shape = backend.ops().shape(tape.value(b).unwrap());
        assert_eq!(b_shape, vec![4]);

        let grads =
            tape.backward(b, |data, shape| backend.tensor_from_vec(data, shape), backend.ops()).unwrap();
        assert!(grads.contains_key(&a));
    }

    #[test]
    fn test_layer_norm_tape() {
        let backend = CpuBackend::default();
        let mut ctx = ForwardCtx::new(&backend, Mode::Train);
        let mut tape = Tape::<CpuBackend>::new();

        let input = tape.watch(backend.tensor_from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap());
        let gamma = Parameter::new("gamma", backend.tensor_from_vec(vec![1.0, 1.0], &[2]).unwrap());
        let beta = Parameter::new("beta", backend.tensor_from_vec(vec![0.0, 0.0], &[2]).unwrap());

        let output = tape.layer_norm_tape(input, &gamma, &beta, 1e-5, 2, &mut ctx).unwrap();
        let out_vals: Vec<f32> = tape.value(output).unwrap().values().to_vec();
        assert_eq!(out_vals.len(), 4);

        let param_map = tape.param_map().clone();
        let grads =
            tape.backward(output, |data, shape| backend.tensor_from_vec(data, shape), backend.ops()).unwrap();
        assert!(grads.contains_key(&input));
        assert!(gamma.gradient_from_store(&grads, &param_map).is_some());
        assert!(beta.gradient_from_store(&grads, &param_map).is_some());
    }

    #[test]
    fn test_grad_and_grad_ext() {
        let backend = CpuBackend::default();
        let mut ctx = ForwardCtx::new(&backend, Mode::Train);
        let mut tape = Tape::<CpuBackend>::new();

        let param_tensor = backend.tensor_from_vec(vec![2.0], &[1]).unwrap();
        let param = Parameter::new("p", param_tensor.clone());
        let a = tape.watch_parameter(&param);
        let b = tape.watch(backend.tensor_from_vec(vec![3.0], &[1]).unwrap());
        let c = tape.add(a, b, &mut ctx).unwrap();

        let param_map = tape.param_map().clone();
        let grads =
            tape.backward(c, |data, shape| backend.tensor_from_vec(data, shape), backend.ops()).unwrap();

        // Test GradExtFromStore
        assert!(param.gradient_from_store(&grads, &param_map).is_some());
    }

    #[test]
    fn test_tape_value_missing() {
        let tape = Tape::<CpuBackend>::new();
        let fake_id = TensorId(999);
        assert!(tape.value(fake_id).is_none());
    }

    #[test]
    fn test_get_value_error() {
        let tape = Tape::<CpuBackend>::new();
        let fake_id = TensorId(999);
        assert!(tape.get_value(fake_id).is_err());
    }
}
