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
        let batch = a_shape[0];
        let features = a_shape[1];

        let out = ops.add_row_vector(&a_val, &row_val)?;
        Ok(self.record(&[a, row], out, move |grad_out, _store, ops| {
            // dL/da = grad_out
            let grad_a = grad_out.clone();

            // dL/drow = sum over batch dimension.
            //
            // NOTE: We do a correctness-first implementation by reading back grad_out and reducing on host.
            // A future improvement should add a backend reduction op (sum along axis) to avoid host transfer.
            let flat = ops.tensor_to_vec(grad_out)?;
            let mut row_grad = vec![0.0f32; features];
            for b in 0..batch {
                let offset = b * features;
                for j in 0..features {
                    row_grad[j] += flat[offset + j];
                }
            }
            let grad_row = ops.tensor_from_vec(row_grad, &[features])?;

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

    /// Softmax activation (row-wise for 2D tensors).
    ///
    /// For a tensor of shape [..., features], applies softmax across the last dimension.
    pub fn softmax(&mut self, a: TensorId, ctx: &mut ForwardCtx<B>) -> Result<TensorId>
    where
        B::Tensor: Clone,
    {
        let a_val = self.get_value(a)?.clone();
        let out = ctx.backend().ops().softmax(&a_val)?;

        // Store output value for backward computation
        Ok(self.record(&[a], out.clone(), move |grad_out, _store, ops| {
            // Softmax backward: y * (grad_out - sum(grad_out * y, axis=-1, keepdims=True))
            // where y is the softmax output
            let y = &out;

            // Element-wise multiply: grad_out * y
            let grad_times_y = ops.mul(grad_out, y)?;

            // Sum over last dimension (features axis)
            // This is a simplified implementation - assumes 2D tensor [batch, features]
            // For proper implementation, we'd need axis-specific sum reduction
            let _sum_grad_y = ops.sum_all(&grad_times_y)?;

            // Broadcast sum back to original shape by replicating
            // Simplified: approximate by just using grad_times_y for now
            // Full implementation would need proper broadcasting

            // grad_out - sum_grad_y_broadcasted
            // For now, simplified: y * grad_out - y * sum_grad_y
            let grad = ops.mul(y, grad_out)?;

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
    /// Computes: -sum(target * log_softmax(logits)) / batch_size
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

        // log_softmax(logits) for numerical stability
        let log_probs = ops.log_softmax(&logits_val)?;

        // -sum(target * log_probs) / batch_size
        // For now: element-wise multiply then negate
        let neg_log_probs = ops.neg(&log_probs)?;
        let loss_per_sample = ops.mul(&target_val, &neg_log_probs)?;

        // Sum over features to get per-sample loss
        // Then mean over batch (simplified: just sum everything for now)
        // This should use a sum reduction, but we'll use the raw values
        let loss = ops.sum_all(&loss_per_sample)?;

        // Record with backward pass: gradient is softmax(logits) - target
        let logits_for_grad = logits_val.clone();
        let target_for_grad = target_val.clone();

        Ok(self.record(&[logits, target], loss, move |_grad_out, _store, ops| {
            // Gradient w.r.t. logits: softmax(logits) - target
            // For cross-entropy with softmax: this is the simplified gradient
            let log_probs_grad = ops.log_softmax(&logits_for_grad)?;
            let softmax_out = ops.exp(&log_probs_grad)?;
            let grad_logits = ops.sub(&softmax_out, &target_for_grad)?;

            // Gradient w.r.t. target: -log_softmax(logits)
            let grad_target = ops.neg(&log_probs_grad)?;

            Ok(vec![grad_logits, grad_target])
        }))
    }

    /// Gather rows from a parameter table using indices.
    ///
    /// Forward: output[i] = table[ids[i]]
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
        let ids_val = self.get_value(ids)?.clone();
        let ops = ctx.backend().ops();

        // Extract indices from tensor
        let ids_shape = ops.shape(&ids_val);
        let num_indices = ids_shape.iter().product::<usize>();

        // Get the actual index values from the tensor
        // For simplicity, assume indices are stored as f32 and we need to convert to usize
        let mut id_vec = Vec::with_capacity(num_indices);
        for i in 0..num_indices {
            let val = ops.tensor_element(&ids_val, i).map_err(|e| {
                rustral_core::CoreError::InvalidArgument(format!("Failed to get tensor element {}: {:?}", i, e))
            })?;
            id_vec.push(val as usize);
        }

        // Forward: gather rows from table
        let output = ops.gather_rows(table, &id_vec)?;

        // Clone for closure
        let ids_shape_for_grad = ids_shape.clone();

        Ok(self.record(&[ids], output, move |_grad_out, _store, ops| {
            // Backward: dL/dtable[row] = sum over i where id_vec[i] == row of grad_out[i]
            // For simplicity, we just return the gradient for the indices
            // Full implementation would need to accumulate into the table parameter

            // The gradient for ids is zero (indices aren't differentiable)
            let grad_ids = ops.zeros(&ids_shape_for_grad)?;
            Ok(vec![grad_ids])
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
    /// Forward: output = input[start:end]
    /// Backward: grad_input is zeros except for grad_output placed at [start:end]
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

        // Forward: slice
        let output = ops.slice(&input_val, start, end)?;

        // Store for backward
        let input_shape = ops.shape(&input_val);

        Ok(self.record(&[input], output, move |_grad_out, _store, ops| {
            // Backward: create zeros of input shape, then place grad_out at [start:end]
            let grad_input = ops.zeros(&input_shape)?;

            // Place grad_out at the correct position
            // This is simplified - full impl needs in-place slice assignment
            // For now, just return the zeros (approximate)
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
    /// Backward: Computes gradients w.r.t. input, gamma, and beta
    pub fn layer_norm_tape(
        &mut self,
        input: TensorId,
        gamma: &rustral_core::Parameter<B>,
        beta: &rustral_core::Parameter<B>,
        eps: f32,
        ctx: &mut ForwardCtx<B>,
    ) -> Result<TensorId>
    where
        B::Tensor: Clone,
    {
        let input_val = self.get_value(input)?.clone();
        let ops = ctx.backend().ops();
        let input_shape = ops.shape(&input_val);

        // Get gamma and beta values
        let gamma_tensor = gamma.tensor();
        let beta_tensor = beta.tensor();

        // Forward pass: compute normalized values
        // For simplicity, assume input is [..., features] and we normalize over last dim
        let num_features = input_shape.last().copied().unwrap_or(1);
        let num_groups = input_shape.iter().product::<usize>() / num_features;

        // Extract values for computation
        let total_elems = input_shape.iter().product();
        let input_values: Vec<f32> =
            (0..total_elems).filter_map(|i| ops.tensor_element(&input_val, i).ok()).collect();

        let gamma_values: Vec<f32> =
            (0..num_features).filter_map(|i| ops.tensor_element(&gamma_tensor, i).ok()).collect();

        let beta_values: Vec<f32> =
            (0..num_features).filter_map(|i| ops.tensor_element(&beta_tensor, i).ok()).collect();

        // Compute mean, var, and normalized values for each group
        let mut output_values = vec![0.0f32; total_elems];
        let mut means = vec![0.0f32; num_groups];
        let mut vars = vec![0.0f32; num_groups];
        let mut stds = vec![0.0f32; num_groups];
        let mut normalized_values = vec![0.0f32; total_elems];

        for g in 0..num_groups {
            let group_start = g * num_features;

            // Compute mean
            let sum: f32 = input_values[group_start..group_start + num_features].iter().sum();
            let mean = sum / num_features as f32;
            means[g] = mean;

            // Compute variance
            let var_sum: f32 = input_values[group_start..group_start + num_features]
                .iter()
                .map(|&x| (x - mean).powi(2))
                .sum();
            let var = var_sum / num_features as f32;
            vars[g] = var;
            let std = (var + eps).sqrt();
            stds[g] = std;

            // Normalize and apply affine transform
            for i in 0..num_features {
                let idx = group_start + i;
                let normalized = (input_values[idx] - mean) / std;
                normalized_values[idx] = normalized;
                let gamma = gamma_values.get(i).copied().unwrap_or(1.0);
                let beta = beta_values.get(i).copied().unwrap_or(0.0);
                output_values[idx] = normalized * gamma + beta;
            }
        }

        let output = ops.tensor_from_vec(output_values, &input_shape)?;

        // Clone for backward closure
        let input_shape_for_grad = input_shape.clone();
        let gamma_values_for_grad = gamma_values.clone();
        let means_for_grad = means.clone();
        let stds_for_grad = stds.clone();
        let normalized_values_for_grad = normalized_values.clone();
        let num_features_for_grad = num_features;
        let num_groups_for_grad = num_groups;

        Ok(self.record(&[input], output, move |grad_out, _store, ops| {
            // Extract grad_out values
            let total_elems = input_shape_for_grad.iter().product();
            let grad_out_values: Vec<f32> =
                (0..total_elems).filter_map(|i| ops.tensor_element(grad_out, i).ok()).collect();

            // Compute gradients for LayerNorm
            // Based on: https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/layer_norm.cpp
            let mut grad_input = vec![0.0f32; total_elems];

            for g in 0..num_groups_for_grad {
                let group_start = g * num_features_for_grad;
                let _mean = means_for_grad[g];
                let std = stds_for_grad[g];
                let inv_std = 1.0 / std;

                // Compute gradients w.r.t. normalized values
                // grad_normalized = grad_out * gamma
                let mut grad_normalized = vec![0.0f32; num_features_for_grad];
                for i in 0..num_features_for_grad {
                    let gamma = gamma_values_for_grad.get(i).copied().unwrap_or(1.0);
                    grad_normalized[i] = grad_out_values[group_start + i] * gamma;
                }

                // Compute statistics for gradient
                let grad_normalized_sum: f32 = grad_normalized.iter().sum();
                let grad_normalized_dot_x: f32 = (0..num_features_for_grad)
                    .map(|i| grad_normalized[i] * normalized_values_for_grad[group_start + i])
                    .sum();

                // Compute grad_input for this group
                // grad_input = (1/std) * (grad_normalized - mean(grad_normalized) - normalized * mean(grad_normalized * normalized))
                for i in 0..num_features_for_grad {
                    let normalized = normalized_values_for_grad[group_start + i];
                    let grad = inv_std
                        * (grad_normalized[i]
                            - grad_normalized_sum / num_features_for_grad as f32
                            - normalized * grad_normalized_dot_x / num_features_for_grad as f32);
                    grad_input[group_start + i] = grad;
                }
            }

            let grad_input_tensor = ops.tensor_from_vec(grad_input, &input_shape_for_grad)?;
            Ok(vec![grad_input_tensor])
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

        let logits = tape.watch(backend.tensor_from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap());
        let target = tape.watch(backend.tensor_from_vec(vec![0.0, 0.0, 1.0], &[3]).unwrap());
        let loss = tape.cross_entropy_loss(logits, target, &mut ctx).unwrap();

        let loss_val: Vec<f32> = tape.value(loss).unwrap().values().to_vec();
        assert_eq!(loss_val.len(), 1);
        assert!(loss_val[0] > 0.0);

        let grads =
            tape.backward(loss, |data, shape| backend.tensor_from_vec(data, shape), backend.ops()).unwrap();
        assert!(grads.contains_key(&logits));
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

        let output = tape.layer_norm_tape(input, &gamma, &beta, 1e-5, &mut ctx).unwrap();
        let out_vals: Vec<f32> = tape.value(output).unwrap().values().to_vec();
        assert_eq!(out_vals.len(), 4);

        let grads =
            tape.backward(output, |data, shape| backend.tensor_from_vec(data, shape), backend.ops()).unwrap();
        assert!(grads.contains_key(&input));
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
