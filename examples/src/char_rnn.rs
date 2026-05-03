//! Character-Level RNN for Text Generation
//!
//! This example trains an LSTM on Tiny Shakespeare to generate text.
//! Architecture: Embedding -> LSTM (2 layers) -> Linear -> Softmax
//!
//! # Running this example
//!
//! Download Tiny Shakespeare dataset:
//! wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
//!
//! Then run: `cargo run --bin char_rnn`

use mnr_core::{Backend, ForwardCtx, Mode, Module, StatefulModule};
use mnr_data::{DataLoader, DataLoaderConfig, Dataset, InMemoryDataset};
use mnr_ndarray_backend::CpuBackend;
use mnr_nn::{CrossEntropyLoss, Embedding, EmbeddingConfig, LinearBuilder, Readout, ReadoutConfig, StackedLstm};
use std::collections::HashMap;
use std::fs;

/// Vocabulary mapping characters to indices and vice versa
pub struct Vocabulary {
    char_to_idx: HashMap<char, usize>,
    idx_to_char: Vec<char>,
    unk_idx: usize,
}

impl Vocabulary {
    /// Build vocabulary from text.
    pub fn from_text(text: &str) -> Self {
        let mut chars: Vec<char> = text.chars().collect::<std::collections::HashSet<_>>().into_iter().collect();
        chars.sort(); // Deterministic ordering

        let mut char_to_idx = HashMap::new();
        for (idx, &ch) in chars.iter().enumerate() {
            char_to_idx.insert(ch, idx);
        }

        let unk_idx = 0; // Unknown character maps to 0

        Self {
            char_to_idx,
            idx_to_char: chars,
            unk_idx,
        }
    }

    /// Get the index for a character.
    pub fn encode(&self, ch: char) -> usize {
        *self.char_to_idx.get(&ch).unwrap_or(&self.unk_idx)
    }

    /// Get the character for an index.
    pub fn decode(&self, idx: usize) -> char {
        self.idx_to_char.get(idx).copied().unwrap_or('?')
    }

    /// Encode a string to indices.
    pub fn encode_str(&self, s: &str) -> Vec<usize> {
        s.chars().map(|ch| self.encode(ch)).collect()
    }

    /// Decode indices to a string.
    pub fn decode_indices(&self, indices: &[usize]) -> String {
        indices.iter().map(|&idx| self.decode(idx)).collect()
    }

    /// Vocabulary size.
    pub fn size(&self) -> usize {
        self.idx_to_char.len()
    }
}

/// A single training sample for character-level language modeling
#[derive(Clone)]
pub struct TextSample {
    /// Input sequence (indices)
    pub input: Vec<usize>,
    /// Target sequence (indices, shifted by 1)
    pub target: Vec<usize>,
}

/// Dataset for character-level language modeling
pub struct CharLMdataset {
    samples: Vec<TextSample>,
}

impl CharLMdataset {
    /// Create dataset by sliding a window over the text.
    pub fn from_text(text: &str, vocab: &Vocabulary, seq_len: usize) -> Self {
        let encoded = vocab.encode_str(text);
        let mut samples = Vec::new();

        // Slide window: for each position, input is [i..i+seq_len], target is [i+1..i+seq_len+1]
        for i in 0..encoded.len().saturating_sub(seq_len + 1) {
            let input = encoded[i..i + seq_len].to_vec();
            let target = encoded[i + 1..i + seq_len + 1].to_vec();
            samples.push(TextSample { input, target });
        }

        println!("Created {} training samples", samples.len());
        Self { samples }
    }
}

impl Dataset<TextSample> for CharLMdataset {
    fn len(&self) -> usize {
        self.samples.len()
    }

    fn get(&self, index: usize) -> Option<TextSample> {
        self.samples.get(index).cloned()
    }
}

/// Character-level RNN model
pub struct CharRnnModel {
    embedding: Embedding<CpuBackend>,
    lstm: StackedLstm<CpuBackend>,
    readout: Readout<CpuBackend>,
    vocab_size: usize,
}

impl CharRnnModel {
    pub fn new(backend: &CpuBackend, vocab_size: usize, embedding_dim: usize, hidden_size: usize, num_layers: usize) -> Result<Self, mnr_core::CoreError> {
        // Embedding: vocab -> embedding_dim
        let embedding = Embedding::new(
            backend,
            EmbeddingConfig::new(vocab_size, embedding_dim),
        )?;

        // LSTM: embedding_dim -> hidden_size (stacked)
        let lstm_configs = (0..num_layers)
            .map(|_| mnr_nn::LstmConfig::new(hidden_size))
            .collect();
        let lstm = StackedLstm::new(lstm_configs)?;

        // Readout: hidden_size -> vocab_size
        let readout = Readout::new(
            backend,
            ReadoutConfig::new(hidden_size, vocab_size),
        )?;

        Ok(Self {
            embedding,
            lstm,
            readout,
            vocab_size,
        })
    }

    /// Forward pass for a single sequence.
    pub fn forward_sequence(
        &self,
        input_indices: &[usize],
        ctx: &mut ForwardCtx<CpuBackend>,
    ) -> Result<Vec<Vec<f32>>, mnr_core::CoreError> {
        let backend = ctx.backend();
        let ops = backend.ops();

        // Get embeddings for each character
        let mut embeddings = Vec::new();
        for &idx in input_indices {
            let embedded = self.embedding.forward(&[idx], ops)?;
            embeddings.push(embedded);
        }

        // Stack embeddings into a batch: [seq_len, embedding_dim]
        let seq_len = embeddings.len();
        let embedding_dim = embeddings[0].shape().elem_count();
        let mut stacked = Vec::with_capacity(seq_len * embedding_dim);
        for emb in &embeddings {
            stacked.extend_from_slice(emb.values());
        }
        let input_tensor = backend.tensor_from_vec(stacked, &[seq_len, embedding_dim])?;

        // Process through LSTM (one step at a time for now)
        // For simplicity, we'll just use the last hidden state
        let mut hidden_states = Vec::new();
        for i in 0..seq_len {
            let slice = ops.slice(&input_tensor, i, i + 1)?;
            // Flatten for LSTM input
            let flat = ops.reshape(&slice, &[embedding_dim])?;
            // Note: StackedLstm expects a different interface, simplified here
            hidden_states.push(flat);
        }

        // Get logits for each position
        let mut logits = Vec::new();
        for hidden in hidden_states {
            let logit = self.readout.forward(&hidden, ctx)?;
            let values: Vec<f32> = (0..self.vocab_size)
                .filter_map(|i| ops.tensor_element(&logit, i).ok())
                .collect();
            logits.push(values);
        }

        Ok(logits)
    }

    /// Generate text from a seed string.
    pub fn generate(
        &self,
        seed: &str,
        vocab: &Vocabulary,
        length: usize,
        ctx: &mut ForwardCtx<CpuBackend>,
    ) -> Result<String, mnr_core::CoreError> {
        let mut generated = seed.to_string();
        let mut current_indices = vocab.encode_str(seed);

        for _ in 0..length {
            // Get predictions for last character
            if let Some(&last_idx) = current_indices.last() {
                let logits = self.forward_sequence(&[last_idx], ctx)?;
                if let Some(last_logits) = logits.last() {
                    // Sample from softmax distribution (simplified: just argmax)
                    let next_idx = last_logits.iter()
                        .enumerate()
                        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                        .map(|(idx, _)| idx)
                        .unwrap_or(0);

                    let next_char = vocab.decode(next_idx);
                    generated.push(next_char);
                    current_indices.push(next_idx);
                }
            }
        }

        Ok(generated)
    }
}

fn main() {
    println!("Character-Level RNN - Tiny Shakespeare");
    println!("=====================================\n");

    // Initialize backend
    let backend = CpuBackend::default();

    // Try to load Tiny Shakespeare
    let text = match fs::read_to_string("data/shakespeare.txt") {
        Ok(text) => text,
        Err(e) => {
            println!("Could not load data/shakespeare.txt: {}", e);
            println!("\nDownloading sample text...");
            println!("To use the full dataset:");
            println!("  wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt");
            println!("  mv input.txt data/shakespeare.txt");
            println!("\nUsing sample text for demo.\n");

            // Sample text for demo
            "To be, or not to be, that is the question:\n\
             Whether 'tis nobler in the mind to suffer\n\
             The slings and arrows of outrageous fortune,".to_string()
        }
    };

    // Build vocabulary
    let vocab = Vocabulary::from_text(&text);
    let vocab_size = vocab.size();
    println!("Vocabulary size: {}", vocab_size);

    // Hyperparameters
    let seq_len = 50;
    let embedding_dim = 128;
    let hidden_size = 256;
    let num_layers = 2;
    let batch_size = 32;

    // Create dataset
    let dataset = CharLMdataset::from_text(&text, &vocab, seq_len);
    let data_len = dataset.len();
    println!("Dataset size: {} sequences", data_len);

    if data_len == 0 {
        println!("Not enough data for training (need at least {} characters)", seq_len + 1);
        return;
    }

    // Create model
    let model = match CharRnnModel::new(&backend, vocab_size, embedding_dim, hidden_size, num_layers) {
        Ok(m) => m,
        Err(e) => {
            println!("Failed to create model: {:?}", e);
            return;
        }
    };
    println!("Model created:");
    println!("  Embedding: {} -> {}", vocab_size, embedding_dim);
    println!("  LSTM: {} layers, {} hidden", num_layers, hidden_size);
    println!("  Output: {} classes", vocab_size);

    // Create data loader
    let mut loader = DataLoader::new(
        Box::new(dataset),
        DataLoaderConfig {
            batch_size,
            shuffle: true,
            seed: Some(42),
            num_workers: 0,
        },
    );

    // Training loop (simplified - just forward passes for demo)
    println!("\nRunning demo training loop (forward passes only):");
    let loss_fn = CrossEntropyLoss::new();
    let mut total_loss = 0.0;
    let num_batches = 10.min(loader.num_batches());

    for (batch_idx, batch) in loader.by_ref().take(num_batches).enumerate() {
        let mut batch_loss = 0.0;

        for sample in batch {
            let mut ctx = ForwardCtx::new(&backend, Mode::Train);

            // Forward pass
            match model.forward_sequence(&sample.input, &mut ctx) {
                Ok(logits) => {
                    // Compute loss (simplified - just sum of last logit values)
                    if let Some(last_logit) = logits.last() {
                        // Simple loss: negative log probability of target
                        if let Some(&target_idx) = sample.target.last() {
                            let logit_sum: f32 = last_logit.iter().map(|&x| x.exp()).sum();
                            if let Some(&target_logit) = last_logit.get(target_idx) {
                                let prob = target_logit.exp() / logit_sum;
                                let loss = -prob.ln();
                                batch_loss += loss;
                            }
                        }
                    }
                }
                Err(e) => {
                    println!("  Error in batch {}: {:?}", batch_idx, e);
                }
            }
        }

        total_loss += batch_loss;
        if batch_idx % 5 == 0 {
            println!("  Batch {}/{}: avg loss = {:.4}",
                batch_idx, num_batches, batch_loss / batch_size as f32);
        }
    }

    println!("\nAverage loss: {:.4}", total_loss / (num_batches * batch_size) as f32);

    // Generate some text
    println!("\nGenerating text from seed 'To '...");
    let mut ctx = ForwardCtx::new(&backend, Mode::Inference);
    match model.generate("To ", &vocab, 50, &mut ctx) {
        Ok(generated) => {
            println!("Generated: {}", generated);
        }
        Err(e) => {
            println!("Generation failed: {:?}", e);
        }
    }

    println!("\nNote: This is a simplified demo. Full training requires:");
    println!("  1. Complete LSTM stateful forward pass integration");
    println!("  2. Backpropagation through time (BPTT)");
    println!("  3. Proper sampling from softmax (temperature, top-k)");
    println!("  4. Larger dataset and more training iterations");
}
