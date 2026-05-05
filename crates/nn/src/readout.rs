use std::sync::Arc;

use mnr_core::{Backend, CoreError, ForwardCtx, Module, ParameterRef, Result, Trainable};
use mnr_symbolic::{LabelPrediction, Vocabulary};
use serde::{Deserialize, Serialize};

use crate::Linear;

/// Configuration for a label readout head.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ReadoutConfig {
    /// Hidden-state dimension consumed by the readout.
    pub hidden_dim: usize,

    /// Number of output labels.
    pub labels: usize,
}

/// Linear classification head paired with a label vocabulary.
pub struct Readout<B: Backend> {
    labels: Arc<Vocabulary>,
    projection: Linear<B>,
}

impl<B: Backend> Readout<B> {
    /// Create a readout module from a label vocabulary and projection layer.
    pub fn new(labels: Arc<Vocabulary>, projection: Linear<B>) -> Self {
        Self { labels, projection }
    }

    /// Compute raw label logits from hidden state.
    pub fn logits(&self, hidden: B::Tensor, ctx: &mut ForwardCtx<B>) -> Result<B::Tensor> {
        self.projection.forward(hidden, ctx)
    }

    /// Compute normalized label probabilities from hidden state.
    pub fn probabilities(&self, hidden: B::Tensor, ctx: &mut ForwardCtx<B>) -> Result<B::Tensor> {
        let logits = self.logits(hidden, ctx)?;
        ctx.backend().ops().softmax(&logits)
    }

    /// Predict the most likely label for hidden state.
    pub fn predict(&self, hidden: B::Tensor, ctx: &mut ForwardCtx<B>) -> Result<LabelPrediction> {
        let probs = self.probabilities(hidden, ctx)?;
        let id = ctx.backend().ops().argmax(&probs)?;
        let label = self.labels.token(id).map_err(|e| CoreError::InvalidArgument(e.to_string()))?.to_string();
        // Extract the actual probability score at the predicted index
        let score = ctx.backend().ops().tensor_element(&probs, id)?;
        Ok(LabelPrediction { label, id, score })
    }
}

impl<B: Backend> Trainable<B> for Readout<B> {
    fn parameters(&self) -> Vec<ParameterRef> {
        self.projection.parameters()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mnr_core::{ForwardCtx, Mode, Parameter};
    use mnr_ndarray_backend::CpuBackend;

    fn create_mock_readout(num_labels: usize, hidden_dim: usize) -> (Readout<CpuBackend>, Arc<Vocabulary>) {
        let backend = CpuBackend::default();

        // Create label vocabulary
        let mut vocab = Vocabulary::with_specials("<unk>");
        for i in 0..num_labels {
            let _ = vocab.insert(format!("label_{}", i));
        }
        let vocab = Arc::new(vocab);

        // Create projection layer: hidden_dim -> num_labels
        let weight_values: Vec<f32> = (0..num_labels * hidden_dim).map(|i| (i as f32) * 0.01).collect();
        let weight =
            Parameter::new("W", backend.tensor_from_vec(weight_values, &[num_labels, hidden_dim]).unwrap());
        let projection = crate::Linear::from_parameters(
            crate::LinearConfig { in_dim: hidden_dim, out_dim: num_labels, bias: false },
            weight,
            None,
        );

        (Readout::new(vocab.clone(), projection), vocab)
    }

    #[test]
    fn test_readout_logit_shape() {
        let backend = CpuBackend::default();
        let mut ctx = ForwardCtx::new(&backend, Mode::Inference);
        let (readout, _vocab) = create_mock_readout(5, 10);

        let hidden = backend.tensor_from_vec(vec![0.5; 10], &[10]).unwrap();
        let logits = readout.logits(hidden, &mut ctx).unwrap();

        assert_eq!(logits.shape(), &[1, 5]);
    }

    #[test]
    fn test_readout_probabilities_sum_to_one() {
        let backend = CpuBackend::default();
        let mut ctx = ForwardCtx::new(&backend, Mode::Inference);
        let (readout, _vocab) = create_mock_readout(4, 8);

        let hidden = backend.tensor_from_vec(vec![0.1; 8], &[8]).unwrap();
        let probs = readout.probabilities(hidden, &mut ctx).unwrap();

        let sum: f32 = probs.values().iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "probabilities should sum to 1, got {}", sum);
    }

    #[test]
    fn test_readout_predict() {
        let backend = CpuBackend::default();
        let mut ctx = ForwardCtx::new(&backend, Mode::Inference);
        let (readout, vocab) = create_mock_readout(3, 6);

        let hidden = backend.tensor_from_vec(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6], &[6]).unwrap();
        let prediction = readout.predict(hidden, &mut ctx).unwrap();

        assert!(prediction.id < vocab.len());
        assert!(!prediction.label.is_empty());
        // Score should be a valid probability (after softmax, all probs sum to 1)
        assert!(
            prediction.score >= 0.0 && prediction.score <= 1.0,
            "score should be in [0,1], got {}",
            prediction.score
        );
    }

    #[test]
    fn test_readout_trainable() {
        let (readout, _vocab) = create_mock_readout(5, 10);
        assert_eq!(readout.parameters().len(), 1); // Just the projection weight
    }
}
