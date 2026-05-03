//! Loss functions for neural network training.
//!
//! Provides common loss functions like MSE and Cross-Entropy
//! that work with any Backend implementation.

use mnr_core::{Backend, ForwardCtx, Result};

/// Mean Squared Error loss.
///
/// Computes: (1/n) * sum((prediction - target)^2)
///
/// # Example
/// ```
/// use mnr_core::{Backend, ForwardCtx, Mode};
/// use mnr_nn::MSELoss;
///
/// // In practice, use a concrete backend like CpuBackend
/// ```
pub struct MSELoss;

impl MSELoss {
    /// Create a new MSE loss instance.
    pub fn new() -> Self {
        Self
    }

    /// Compute the loss between prediction and target.
    ///
    /// # Arguments
    /// * `prediction` - The model's output tensor
    /// * `target` - The ground truth tensor
    /// * `ctx` - Forward context for operations
    ///
    /// # Returns
    /// Scalar loss value as a tensor
    pub fn forward<B: Backend>(
        &self,
        prediction: &B::Tensor,
        target: &B::Tensor,
        ctx: &mut ForwardCtx<B>,
    ) -> Result<B::Tensor> {
        let ops = ctx.backend().ops();

        // diff = prediction - target
        let diff = ops.sub(prediction, target)?;

        // squared = diff^2
        let squared = ops.mul(&diff, &diff)?;

        // loss = mean(squared) = sum(squared) / n
        let sum = ops.sum_all(&squared)?;
        let shape = ops.shape(prediction);
        let n = shape.iter().product::<usize>() as f32;
        let mean = ops.mul(&sum, &ops.tensor_from_vec(vec![1.0 / n], &[1])?)?;

        Ok(mean)
    }
}

impl Default for MSELoss {
    fn default() -> Self {
        Self::new()
    }
}

/// Cross-Entropy loss for classification.
///
/// Computes: -sum(target * log_softmax(prediction)) / batch_size
///
/// This combines log_softmax and negative log-likelihood for numerical stability.
pub struct CrossEntropyLoss;

impl CrossEntropyLoss {
    /// Create a new Cross-Entropy loss instance.
    pub fn new() -> Self {
        Self
    }

    /// Compute the loss between logits and target class indices.
    ///
    /// # Arguments
    /// * `logits` - The model's raw output (before softmax) [batch_size, num_classes]
    /// * `target` - One-hot encoded target [batch_size, num_classes]
    /// * `ctx` - Forward context for operations
    ///
    /// # Returns
    /// Scalar loss value as a tensor
    pub fn forward<B: Backend>(
        &self,
        logits: &B::Tensor,
        target: &B::Tensor,
        ctx: &mut ForwardCtx<B>,
    ) -> Result<B::Tensor> {
        let ops = ctx.backend().ops();

        // log_probs = log_softmax(logits) for numerical stability
        let log_probs = ops.log_softmax(logits)?;

        // nll = -sum(target * log_probs) / batch_size
        // For one-hot targets: this picks out the log_prob of the correct class
        let target_log_probs = ops.mul(target, &log_probs)?;

        // Sum over all elements (target is one-hot, so this is sum of correct class log_probs)
        let sum = ops.sum_all(&target_log_probs)?;

        // Negate to get positive loss
        let neg_sum = ops.neg(&sum)?;

        // Divide by batch size
        let shape = ops.shape(logits);
        let batch_size = shape[0] as f32;
        let mean = ops.mul(&neg_sum, &ops.tensor_from_vec(vec![1.0 / batch_size], &[1])?)?;

        Ok(mean)
    }

    /// Compute loss from class indices (not one-hot).
    ///
    /// # Arguments
    /// * `logits` - The model's raw output [batch_size, num_classes]
    /// * `target_indices` - Class indices [batch_size]
    /// * `ctx` - Forward context for operations
    ///
    /// # Returns
    /// Scalar loss value as a tensor
    pub fn forward_indices<B: Backend>(
        &self,
        logits: &B::Tensor,
        target_indices: &[usize],
        ctx: &mut ForwardCtx<B>,
    ) -> Result<B::Tensor>
    where
        B::Tensor: Clone,
    {
        let ops = ctx.backend().ops();

        // Create one-hot target from indices
        let shape = ops.shape(logits);
        let num_classes = shape[1];
        let batch_size = target_indices.len();
        let mut target_values = vec![0.0f32; batch_size * num_classes];

        for (i, &class_idx) in target_indices.iter().enumerate() {
            if class_idx < num_classes {
                target_values[i * num_classes + class_idx] = 1.0;
            }
        }

        let target = ops.tensor_from_vec(target_values, &[batch_size, num_classes])?;
        self.forward(logits, &target, ctx)
    }
}

impl Default for CrossEntropyLoss {
    fn default() -> Self {
        Self::new()
    }
}

/// Binary Cross-Entropy with Logits loss.
///
/// Computes: -[y * log(sigmoid(x)) + (1-y) * log(1 - sigmoid(x))]
///
/// More stable than using sigmoid followed by BCE.
pub struct BCEWithLogitsLoss;

impl BCEWithLogitsLoss {
    /// Create a new BCE with logits loss instance.
    pub fn new() -> Self {
        Self
    }

    /// Compute the loss between logits and binary targets.
    ///
    /// # Arguments
    /// * `logits` - The model's raw output (before sigmoid)
    /// * `target` - Binary targets (0.0 or 1.0)
    /// * `ctx` - Forward context for operations
    pub fn forward<B: Backend>(
        &self,
        logits: &B::Tensor,
        target: &B::Tensor,
        ctx: &mut ForwardCtx<B>,
    ) -> Result<B::Tensor> {
        let ops = ctx.backend().ops();

        // max_val = max(logits, 0) for numerical stability
        let zeros = ops.zeros(&ops.shape(logits))?;
        let max_val = ops.maximum(logits, &zeros)?;

        // loss = max_val - logits * target + log(1 + exp(-max_val))
        // This is the numerically stable form of BCE with logits

        // term1 = max_val
        let term1 = max_val.clone();

        // term2 = -logits * target
        let neg_logits = ops.neg(logits)?;
        let term2 = ops.mul(&neg_logits, target)?;

        // term3 = log(1 + exp(-max_val))
        let neg_max = ops.neg(&max_val)?;
        let exp_neg_max = ops.exp(&neg_max)?;
        let one_plus_exp = ops.add_scalar(&exp_neg_max, 1.0)?;
        let term3 = ops.log(&one_plus_exp)?;

        // Combine: term1 + term2 + term3
        let temp = ops.add(&term1, &term2)?;
        let loss_per_element = ops.add(&temp, &term3)?;

        // Mean over all elements
        let sum = ops.sum_all(&loss_per_element)?;
        let shape = ops.shape(logits);
        let n = shape.iter().product::<usize>() as f32;
        let mean = ops.mul(&sum, &ops.tensor_from_vec(vec![1.0 / n], &[1])?)?;

        Ok(mean)
    }
}

impl Default for BCEWithLogitsLoss {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mnr_core::{Mode, ForwardCtx};
    use mnr_ndarray_backend::CpuBackend;

    #[test]
    fn test_mse_loss_forward() {
        let backend = CpuBackend::default();
        let mut ctx = ForwardCtx::new(&backend, Mode::Train);

        // prediction = [1.0, 2.0, 3.0], target = [1.0, 2.0, 3.0]
        // MSE should be 0
        let pred = backend.tensor_from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();
        let target = backend.tensor_from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();

        let loss = MSELoss::new().forward(&pred, &target, &mut ctx).unwrap();
        let loss_val: Vec<f32> = loss.values().to_vec();

        assert!((loss_val[0]).abs() < 1e-6, "MSE should be 0 for identical tensors");
    }

    #[test]
    fn test_mse_loss_nonzero() {
        let backend = CpuBackend::default();
        let mut ctx = ForwardCtx::new(&backend, Mode::Train);

        // prediction = [0.0, 0.0, 0.0], target = [1.0, 2.0, 3.0]
        // diff = [-1, -2, -3], squared = [1, 4, 9], mean = 14/3 = 4.667
        let pred = backend.tensor_from_vec(vec![0.0, 0.0, 0.0], &[3]).unwrap();
        let target = backend.tensor_from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();

        let loss = MSELoss::new().forward(&pred, &target, &mut ctx).unwrap();
        let loss_val: Vec<f32> = loss.values().to_vec();

        let expected = (1.0 + 4.0 + 9.0) / 3.0;
        assert!((loss_val[0] - expected).abs() < 1e-5,
            "MSE expected {}, got {}", expected, loss_val[0]);
    }

    #[test]
    fn test_cross_entropy_loss() {
        let backend = CpuBackend::default();
        let mut ctx = ForwardCtx::new(&backend, Mode::Train);

        // Simple 2-class classification
        // logits = [[1.0, 0.0]] -> class 1 is "more likely"
        // target = [[1.0, 0.0]] -> correct class is 0
        let logits = backend.tensor_from_vec(vec![1.0, 0.0], &[1, 2]).unwrap();
        let target = backend.tensor_from_vec(vec![1.0, 0.0], &[1, 2]).unwrap();

        let loss = CrossEntropyLoss::new().forward(&logits, &target, &mut ctx).unwrap();
        let loss_val: Vec<f32> = loss.values().to_vec();

        // Loss should be positive
        assert!(loss_val[0] > 0.0, "Cross-entropy loss should be positive");
    }

    #[test]
    fn test_cross_entropy_loss_indices() {
        let backend = CpuBackend::default();
        let mut ctx = ForwardCtx::new(&backend, Mode::Train);

        // logits for 2 samples, 3 classes
        let logits = backend.tensor_from_vec(
            vec![2.0, 1.0, 0.0,  // Sample 0: class 0 is highest
                 0.0, 2.0, 1.0], // Sample 1: class 1 is highest
            &[2, 3]
        ).unwrap();

        // Target: sample 0 -> class 1, sample 1 -> class 1
        let loss = CrossEntropyLoss::new()
            .forward_indices(&logits, &[1, 1], &mut ctx)
            .unwrap();
        let loss_val: Vec<f32> = loss.values().to_vec();

        assert!(loss_val[0] > 0.0, "Cross-entropy loss should be positive");
        assert!(loss_val[0].is_finite(), "Loss should be finite");
    }

    #[test]
    fn test_bce_with_logits_loss() {
        let backend = CpuBackend::default();
        let mut ctx = ForwardCtx::new(&backend, Mode::Train);

        // Binary classification
        // logits = [0.0, 0.0], target = [0.0, 1.0]
        let logits = backend.tensor_from_vec(vec![0.0, 0.0], &[2]).unwrap();
        let target = backend.tensor_from_vec(vec![0.0, 1.0], &[2]).unwrap();

        let loss = BCEWithLogitsLoss::new().forward(&logits, &target, &mut ctx).unwrap();
        let loss_val: Vec<f32> = loss.values().to_vec();

        // Loss should be positive and finite
        assert!(loss_val[0] > 0.0, "BCE loss should be positive");
        assert!(loss_val[0].is_finite(), "Loss should be finite");
    }
}
