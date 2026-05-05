use std::sync::Arc;

use rustral_core::{
    Backend, ForwardCtx, Module, NamedParameters, Parameter, ParameterRef, Result, Trainable,
};
use rustral_symbolic::Vocabulary;
use serde::{Deserialize, Serialize};

/// Configuration for an embedding table.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EmbeddingConfig {
    /// Vocabulary size (number of embeddings).
    pub vocab_size: usize,
    /// Number of features in each embedding vector.
    pub dim: usize,
}

impl EmbeddingConfig {
    /// Create a new embedding configuration.
    pub fn new(vocab_size: usize, dim: usize) -> Self {
        Self { vocab_size, dim }
    }
}

/// Embedding lookup backed by a rank-2 parameter table.
pub struct Embedding<B: Backend> {
    config: EmbeddingConfig,
    table: Parameter<B>,
    vocab: Arc<Vocabulary>,
}

impl<B: Backend> Embedding<B> {
    /// Create an embedding module from explicit table and vocabulary values.
    pub fn from_parts(config: EmbeddingConfig, table: Parameter<B>, vocab: Arc<Vocabulary>) -> Self {
        Self { config, table, vocab }
    }

    /// Create an embedding module with randomly initialized weights.
    pub fn new(backend: &B, config: EmbeddingConfig, seed: u64) -> Result<Self> {
        let table = backend.normal_parameter("embed", &[config.vocab_size, config.dim], seed, 0.02)?;
        let vocab = Arc::new(Vocabulary::with_specials("<unk>"));
        Ok(Self { config, table, vocab })
    }

    /// Borrow the vocabulary used by this embedding module.
    pub fn vocab(&self) -> &Vocabulary {
        &self.vocab
    }

    /// Borrow the immutable embedding configuration.
    pub fn config(&self) -> &EmbeddingConfig {
        &self.config
    }

    /// Borrow the embedding table parameter.
    pub fn table(&self) -> &Parameter<B> {
        &self.table
    }
}

impl<B: Backend> NamedParameters<B> for Embedding<B> {
    fn visit_parameters(&self, f: &mut dyn FnMut(&str, &Parameter<B>)) {
        f("embed", &self.table);
    }

    fn visit_parameters_mut(&mut self, f: &mut dyn FnMut(&str, &mut Parameter<B>)) {
        f("embed", &mut self.table);
    }
}

impl<B: Backend> Module<B> for Embedding<B> {
    type Input = Vec<usize>;
    type Output = B::Tensor;

    /// Gather one embedding row per input id.
    fn forward(&self, ids: Self::Input, ctx: &mut ForwardCtx<B>) -> Result<Self::Output> {
        ctx.backend().ops().gather_rows(&self.table, &ids)
    }
}

impl<B: Backend> Trainable<B> for Embedding<B> {
    /// Return the table parameter reference.
    fn parameters(&self) -> Vec<ParameterRef> {
        vec![ParameterRef { id: self.table.id() }]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustral_core::{ForwardCtx, Mode, Parameter};
    use rustral_ndarray_backend::CpuBackend;

    fn create_mock_embedding(vocab_size: usize, dim: usize) -> (Embedding<CpuBackend>, Arc<Vocabulary>) {
        let _backend = CpuBackend::default();
        let mut vocab = Vocabulary::with_specials("<unk>");
        for i in 0..vocab_size {
            let _ = vocab.insert(format!("token_{}", i));
        }
        let vocab = Arc::new(vocab);

        // Create embedding table: vocab_size x dim
        let values: Vec<f32> = (0..vocab_size * dim).map(|i| (i as f32) * 0.01).collect();
        let table = CpuBackend::default().tensor_from_vec(values, &[vocab_size, dim]).unwrap();
        let table_param = Parameter::new("embed", table);

        let config = EmbeddingConfig { vocab_size, dim };
        let embedding = Embedding::from_parts(config, table_param, vocab.clone());
        (embedding, vocab)
    }

    #[test]
    fn test_embedding_lookup() {
        let backend = CpuBackend::default();
        let mut ctx = ForwardCtx::new(&backend, Mode::Inference);
        let (embedding, _vocab) = create_mock_embedding(10, 8);

        let ids = vec![0, 1, 2];
        let output = embedding.forward(ids, &mut ctx).unwrap();

        assert_eq!(output.shape(), &[3, 8]);
    }

    #[test]
    fn test_embedding_single_lookup() {
        let backend = CpuBackend::default();
        let mut ctx = ForwardCtx::new(&backend, Mode::Inference);
        let (embedding, _vocab) = create_mock_embedding(10, 8);

        let ids = vec![5];
        let output = embedding.forward(ids, &mut ctx).unwrap();

        assert_eq!(output.shape(), &[1, 8]);
    }

    #[test]
    fn test_embedding_trainable() {
        let (embedding, _vocab) = create_mock_embedding(10, 8);
        assert_eq!(embedding.parameters().len(), 1);
    }

    #[test]
    fn test_embedding_vocab_accessor() {
        let (embedding, vocab) = create_mock_embedding(10, 8);
        assert_eq!(embedding.vocab().len(), vocab.len());
    }

    #[test]
    fn test_embedding_config_accessor() {
        let (embedding, _vocab) = create_mock_embedding(10, 8);
        assert_eq!(embedding.config().vocab_size, 10);
        assert_eq!(embedding.config().dim, 8);
    }

    #[test]
    fn test_embedding_new() {
        let backend = CpuBackend::default();
        let config = EmbeddingConfig::new(100, 64);
        let embedding = Embedding::new(&backend, config, 42).unwrap();
        assert_eq!(embedding.config().vocab_size, 100);
        assert_eq!(embedding.config().dim, 64);
        assert_eq!(embedding.parameters().len(), 1);
    }
}
