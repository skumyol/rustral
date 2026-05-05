//! Multi-label classification and binary readout heads.

use std::sync::Arc;

use mnr_core::{Backend, CoreError, ForwardCtx, Module, ParameterRef, Result, Trainable};
use mnr_symbolic::Vocabulary;
use serde::{Deserialize, Serialize};

use crate::Linear;

/// Multi-readout model for multi-label classification.
///
/// Equivalent to having a binary classifier for each possible label.
/// Each label is predicted independently (not mutually exclusive).
pub struct MultiReadout<B: Backend> {
    labels: Arc<Vocabulary>,
    /// One linear projection per label (binary classification).
    projections: Vec<Linear<B>>,
}

/// Prediction for a single label in multi-label classification.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LabelBinaryPrediction {
    /// Predicted label string.
    pub label: String,
    /// Label id.
    pub id: usize,
    /// Logit score (before sigmoid).
    pub logit: f32,
    /// Probability after sigmoid.
    pub probability: f32,
    /// Whether the label is predicted (probability > threshold).
    pub predicted: bool,
}

impl<B: Backend> MultiReadout<B> {
    /// Create a multi-readout model.
    pub fn new(labels: Arc<Vocabulary>, projections: Vec<Linear<B>>) -> Self {
        assert_eq!(labels.len(), projections.len(), "Number of labels must match number of projections");
        Self { labels, projections }
    }

    /// Compute logits for all labels.
    pub fn logits(&self, hidden: &B::Tensor, ctx: &mut ForwardCtx<B>) -> Result<Vec<B::Tensor>> {
        self.projections
            .iter()
            .map(|proj| proj.forward(hidden.clone(), ctx))
            .collect()
    }

    /// Predict labels with a threshold.
    pub fn predict(&self, hidden: B::Tensor, ctx: &mut ForwardCtx<B>, threshold: f32) -> Result<Vec<LabelBinaryPrediction>> {
        let logits = self.logits(&hidden, ctx)?;
        let ops = ctx.backend().ops();
        let mut predictions = Vec::new();

        for (id, logit_tensor) in logits.iter().enumerate() {
            // Extract logit value from the tensor (assumes 1D tensor with single element)
            let logit = ops.tensor_element(logit_tensor, 0)?;

            let label = self.labels.token(id).map_err(|e| CoreError::InvalidArgument(e.to_string()))?;

            // Apply sigmoid to get probability
            let prob_tensor = ops.sigmoid(logit_tensor)?;
            let prob = ops.tensor_element(&prob_tensor, 0)?;

            predictions.push(LabelBinaryPrediction {
                label: label.to_string(),
                id,
                logit,
                probability: prob,
                predicted: prob > threshold,
            });
        }

        Ok(predictions)
    }

    /// Get top-k predicted labels by probability.
    pub fn top_k(&self, hidden: B::Tensor, ctx: &mut ForwardCtx<B>, k: usize) -> Result<Vec<LabelBinaryPrediction>> {
        let mut predictions = self.predict(hidden, ctx, 0.0)?;

        // Sort by probability in descending order
        predictions.sort_by(|a, b| b.probability.partial_cmp(&a.probability).unwrap_or(std::cmp::Ordering::Equal));

        // Take top k (or fewer if not enough predictions)
        let k = k.min(predictions.len());
        predictions.truncate(k);

        Ok(predictions)
    }
}

impl<B: Backend> Trainable<B> for MultiReadout<B> {
    fn parameters(&self) -> Vec<ParameterRef> {
        self.projections.iter().flat_map(|p| p.parameters()).collect()
    }
}

/// Binary readout model for single binary classification.
pub struct BinaryReadout<B: Backend> {
    projection: Linear<B>,
}

/// Binary classification result.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BinaryPrediction {
    /// Logit score (before sigmoid).
    pub logit: f32,
    /// Probability of positive class.
    pub probability: f32,
    /// Predicted class (true if probability > 0.5).
    pub predicted: bool,
}

impl<B: Backend> BinaryReadout<B> {
    /// Create a binary readout model.
    pub fn new(projection: Linear<B>) -> Self {
        Self { projection }
    }

    /// Compute logit for binary classification.
    pub fn logit(&self, hidden: B::Tensor, ctx: &mut ForwardCtx<B>) -> Result<B::Tensor> {
        self.projection.forward(hidden, ctx)
    }

    /// Predict binary class.
    pub fn predict(&self, hidden: B::Tensor, ctx: &mut ForwardCtx<B>) -> Result<BinaryPrediction> {
        let logit_tensor = self.logit(hidden, ctx)?;
        let ops = ctx.backend().ops();

        // Extract logit value (assumes 1D tensor with single element)
        let logit = ops.tensor_element(&logit_tensor, 0)?;

        // Apply sigmoid to get probability
        let prob_tensor = ops.sigmoid(&logit_tensor)?;
        let prob = ops.tensor_element(&prob_tensor, 0)?;

        Ok(BinaryPrediction {
            logit,
            probability: prob,
            predicted: prob > 0.5,
        })
    }
}

impl<B: Backend> Module<B> for BinaryReadout<B> {
    type Input = B::Tensor;
    type Output = B::Tensor;

    fn forward(&self, input: Self::Input, ctx: &mut ForwardCtx<B>) -> Result<Self::Output> {
        self.projection.forward(input, ctx)
    }
}

impl<B: Backend> Trainable<B> for BinaryReadout<B> {
    fn parameters(&self) -> Vec<ParameterRef> {
        self.projection.parameters()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mnr_core::{ForwardCtx, Mode};
    use mnr_ndarray_backend::CpuBackend;

    fn create_mock_multi_readout(num_labels: usize, hidden_dim: usize) -> MultiReadout<CpuBackend> {
        let backend = CpuBackend::default();

        // Use minimal vocab to ensure exact label count matching
        let mut vocab = Vocabulary::with_specials("<unk>");
        for i in 0..num_labels {
            let _ = vocab.insert(format!("label_{}", i));
        }
        let vocab = Arc::new(vocab);

        // vocab includes <unk> + num_labels labels, so we need num_labels + 1 projections
        let projections: Vec<_> = (0..num_labels + 1)
            .map(|i| {
                let w = backend.normal_parameter(&format!("W_{}", i), &[1, hidden_dim], 42 + i as u64, 0.01).unwrap();
                crate::Linear::from_parameters(
                    crate::LinearConfig { in_dim: hidden_dim, out_dim: 1, bias: false },
                    w,
                    None,
                )
            })
            .collect();

        MultiReadout::new(vocab, projections)
    }

    fn create_mock_binary_readout(hidden_dim: usize) -> BinaryReadout<CpuBackend> {
        let backend = CpuBackend::default();
        let w = backend.normal_parameter("W", &[1, hidden_dim], 42, 0.01).unwrap();
        let projection = crate::Linear::from_parameters(
            crate::LinearConfig { in_dim: hidden_dim, out_dim: 1, bias: false },
            w,
            None,
        );
        BinaryReadout::new(projection)
    }

    #[test]
    fn test_multi_readout_parameters() {
        let mr = create_mock_multi_readout(5, 10);
        // 5 labels + 1 <unk> special token = 6 projections
        assert_eq!(mr.parameters().len(), 6);
    }

    #[test]
    fn test_multi_readout_logit_shapes() {
        let backend = CpuBackend::default();
        let mut ctx = ForwardCtx::new(&backend, Mode::Inference);
        let mr = create_mock_multi_readout(3, 8);

        let hidden = backend.tensor_from_vec(vec![0.1; 8], &[8]).unwrap();
        let logits = mr.logits(&hidden, &mut ctx).unwrap();

        // 3 labels + 1 <unk> = 4 logits
        assert_eq!(logits.len(), 4);
    }

    #[test]
    fn test_binary_readout_forward_shape() {
        let backend = CpuBackend::default();
        let mut ctx = ForwardCtx::new(&backend, Mode::Inference);
        let br = create_mock_binary_readout(10);

        let hidden = backend.tensor_from_vec(vec![0.1; 10], &[10]).unwrap();
        let logit = br.logit(hidden, &mut ctx).unwrap();

        assert_eq!(logit.shape(), &[1, 1]);
    }

    #[test]
    fn test_binary_readout_parameters() {
        let br = create_mock_binary_readout(10);
        assert_eq!(br.parameters().len(), 1);  // Just weight
    }

    #[test]
    fn test_multi_readout_predict() {
        let backend = CpuBackend::default();
        let mut ctx = ForwardCtx::new(&backend, Mode::Inference);
        let mr = create_mock_multi_readout(3, 4);

        let hidden = backend.tensor_from_vec(vec![0.1, 0.2, 0.3, 0.4], &[4]).unwrap();
        let predictions = mr.predict(hidden, &mut ctx, 0.5).unwrap();

        // 3 labels + 1 <unk> = 4 predictions
        assert_eq!(predictions.len(), 4);

        // Each prediction should have a valid probability between 0 and 1
        for pred in &predictions {
            assert!(pred.probability >= 0.0 && pred.probability <= 1.0,
                "probability should be in [0,1], got {} for {}", pred.probability, pred.label);
        }
    }

    #[test]
    fn test_multi_readout_top_k() {
        let backend = CpuBackend::default();
        let mut ctx = ForwardCtx::new(&backend, Mode::Inference);
        let mr = create_mock_multi_readout(5, 4);

        let hidden = backend.tensor_from_vec(vec![0.1, 0.2, 0.3, 0.4], &[4]).unwrap();
        let top_k = mr.top_k(hidden, &mut ctx, 3).unwrap();

        // Should return at most 3 predictions
        assert_eq!(top_k.len(), 3);

        // Probabilities should be in descending order
        for i in 1..top_k.len() {
            assert!(top_k[i-1].probability >= top_k[i].probability,
                "top_k should be sorted by probability");
        }
    }

    #[test]
    fn test_binary_readout_predict() {
        let backend = CpuBackend::default();
        let mut ctx = ForwardCtx::new(&backend, Mode::Inference);
        let br = create_mock_binary_readout(4);

        let hidden = backend.tensor_from_vec(vec![0.1, 0.2, 0.3, 0.4], &[4]).unwrap();
        let prediction = br.predict(hidden, &mut ctx).unwrap();

        // Probability should be between 0 and 1
        assert!(prediction.probability >= 0.0 && prediction.probability <= 1.0,
            "probability should be in [0,1], got {}", prediction.probability);

        // Predicted should be based on threshold 0.5
        assert_eq!(prediction.predicted, prediction.probability > 0.5);
    }

    #[test]
    fn test_binary_readout_module_forward() {
        let backend = CpuBackend::default();
        let mut ctx = ForwardCtx::new(&backend, Mode::Inference);
        let br = create_mock_binary_readout(4);

        let hidden = backend.tensor_from_vec(vec![0.1, 0.2, 0.3, 0.4], &[4]).unwrap();
        let output = br.forward(hidden, &mut ctx).unwrap();
        assert_eq!(output.shape(), &[1, 1]);
    }
}
