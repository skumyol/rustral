//! NanoGPT - Character-Level Language Model on GPU (via Candle)
//!
//! A minimal GPT-style language model trained with manual backpropagation.
//! This example demonstrates how to train a small neural network on GPU
//! using the candle backend with MNR's tensor operations.
//!
//! ## Architecture
//! - Character-level tokenization (each character = one token)
//! - Token embedding via one-hot projection
//! - 2-layer MLP with ReLU activation
//! - Log-softmax output over vocabulary
//!
//! ## Training
//! - Manual backpropagation (computes gradients explicitly with tensor ops)
//! - SGD optimizer with learning rate decay
//! - Next-character prediction on Shakespeare text
//!
//! ## Run
//! ```bash
//! # CPU training (default)
//! cargo run --bin nanogpt
//!
//! # GPU training (requires CUDA feature)
//! cargo run --bin nanogpt --features cuda
//! ```
//!
//! ## Note on Autodiff
//! This example uses manual backprop for educational clarity and to avoid
//! limitations in MNR's tape-based autodiff for parameter gradients.
//! Future versions will integrate full automatic differentiation.

use mnr_candle_backend::CandleBackend;
use mnr_core::{Backend, Parameter, Result, TensorOps};
use rand::Rng;

/// Tiny Shakespeare excerpt for training.
const DATA: &str = "To be, or not to be, that is the question:\n\
Whether 'tis nobler in the mind to suffer\n\
The slings and arrows of outrageous fortune,\n\
Or to take Arms against a Sea of troubles,\n\
And by opposing end them: to die, to sleep;\n\
No more; and by a sleep, to say we end\n\
The heart-ache, and the thousand natural shocks\n\
That Flesh is heir to? 'Tis a consummation\n\
Devoutly to be wished. To die, to sleep;\n\
To sleep, perchance to Dream; aye, there's the rub.";

/// Model hyperparameters.
const D_MODEL: usize = 64;
const HIDDEN_DIM: usize = 256;
const BLOCK_SIZE: usize = 8;
const LEARNING_RATE: f32 = 0.01;
const EPOCHS: usize = 2000;
const EVAL_INTERVAL: usize = 200;

fn main() -> Result<()> {
    println!("NanoGPT - Character-Level Language Model");
    println!("=========================================\n");

    // Initialize candle backend (auto-detects CUDA if compiled with --features cuda)
    let backend = CandleBackend::new();
    let ops = backend.ops();

    // Build vocabulary from characters
    let vocab = build_vocab(DATA);
    let vocab_size = vocab.len();
    println!("Vocabulary size: {} (characters)", vocab_size);
    println!("Dataset length: {} characters", DATA.len());
    println!("Block size: {} (predict next char from {} previous)", BLOCK_SIZE, BLOCK_SIZE);
    println!("Model: {} -> {} -> {} -> {}", vocab_size, D_MODEL, HIDDEN_DIM, vocab_size);
    println!();

    // Encode data
    let encoded: Vec<usize> = DATA.chars().map(|c| vocab.encode(c)).collect();

    // Initialize parameters
    let mut w_emb = backend.normal_parameter("w_emb", &[vocab_size, D_MODEL], 42, 0.1)?;
    let mut w1 = backend.normal_parameter("w1", &[D_MODEL, HIDDEN_DIM], 43, 0.1)?;
    let mut b1 = backend.normal_parameter("b1", &[HIDDEN_DIM], 44, 0.0)?;
    let mut w2 = backend.normal_parameter("w2", &[HIDDEN_DIM, vocab_size], 45, 0.1)?;
    let mut b2 = backend.normal_parameter("b2", &[vocab_size], 46, 0.0)?;

    println!("Training for {} epochs...\n", EPOCHS);

    // Training loop
    let mut rng = rand::thread_rng();
    let mut lr = LEARNING_RATE;

    for epoch in 0..EPOCHS {
        // Sample a random block
        let max_start = encoded.len().saturating_sub(BLOCK_SIZE + 1);
        let start = rng.gen_range(0..max_start.max(1));
        let input_tokens = &encoded[start..start + BLOCK_SIZE];
        let target_token = encoded[start + BLOCK_SIZE];

        // Forward pass
        let (loss, logits, h1) = forward(
            ops, input_tokens, target_token,
            &w_emb, &w1, &b1, &w2, &b2,
        )?;

        // Backward pass - compute gradients manually
        let grads = backward(
            ops, input_tokens, target_token,
            &w_emb, &w1, &b1, &w2, &b2,
            &logits, &h1,
        )?;

        // SGD update
        update_param(&mut w_emb, &grads.dw_emb, lr, ops)?;
        update_param(&mut w1, &grads.dw1, lr, ops)?;
        update_param(&mut b1, &grads.db1, lr, ops)?;
        update_param(&mut w2, &grads.dw2, lr, ops)?;
        update_param(&mut b2, &grads.db2, lr, ops)?;

        // Learning rate decay
        if epoch > 0 && epoch % 500 == 0 {
            lr *= 0.5;
        }

        // Evaluation
        if epoch % EVAL_INTERVAL == 0 {
            println!("Epoch {:4} | lr={:.5} | loss={:.4}", epoch, lr, loss);

            if epoch % (EVAL_INTERVAL * 2) == 0 {
                let prompt = "To be";
                let generated = generate(
                    ops, prompt, 80, &vocab,
                    &w_emb, &w1, &b1, &w2, &b2,
                )?;
                println!("  Prompt: '{}' -> Generated: '{}'", prompt, generated);
            }
        }
    }

    // Final generation
    println!("\nFinal generation samples:");
    for prompt in &["To be", "Whether", "And by"] {
        let generated = generate(
            ops, prompt, 120, &vocab,
            &w_emb, &w1, &b1, &w2, &b2,
        )?;
        println!("  '{}' -> '{}'", prompt, generated);
    }

    println!("\nTraining complete!");
    Ok(())
}

// =============================================================================
// Forward Pass
// =============================================================================

/// Forward pass: returns (loss_value, logits, h1_after_relu)
fn forward(
    ops: &dyn TensorOps<CandleBackend>,
    input_tokens: &[usize],
    target_token: usize,
    w_emb: &Parameter<CandleBackend>,
    w1: &Parameter<CandleBackend>,
    b1: &Parameter<CandleBackend>,
    w2: &Parameter<CandleBackend>,
    b2: &Parameter<CandleBackend>,
) -> Result<(f32, candle_core::Tensor, candle_core::Tensor)> {
    let seq_len = input_tokens.len();
    let vocab_size = ops.shape(w_emb.tensor())[0];

    // One-hot encoding: [seq_len, vocab_size]
    let mut one_hot = vec![0.0f32; seq_len * vocab_size];
    for (i, &t) in input_tokens.iter().enumerate() {
        one_hot[i * vocab_size + t] = 1.0;
    }
    let x = ops.tensor_from_vec(one_hot, &[seq_len, vocab_size])?;

    // Embedding: [seq_len, vocab_size] @ [vocab_size, d_model] -> [seq_len, d_model]
    let emb = ops.matmul(&x, w_emb.tensor())?;
    let _emb_data = to_vec(ops, &emb)?; // force eval

    // Pool over sequence: [seq_len, d_model] -> [1, d_model] (mean pooling)
    let pooled = mean_pool(ops, &emb, seq_len)?;

    // Hidden layer: [1, d_model] @ [d_model, hidden] -> [1, hidden]
    let h1 = ops.matmul(&pooled, w1.tensor())?;
    let h1_b = ops.add_row_vector(&h1, b1.tensor())?;
    let _h1_b_data = to_vec(ops, &h1_b)?; // force eval
    let h1_relu = ops.relu(&h1_b)?;
    let _h1_relu_data = to_vec(ops, &h1_relu)?; // force eval

    // Output layer: [1, hidden] @ [hidden, vocab] -> [1, vocab]
    let logits = ops.matmul(&h1_relu, w2.tensor())?;
    let logits_b = ops.add_row_vector(&logits, b2.tensor())?;
    let _logits_b_data = to_vec(ops, &logits_b)?; // force eval

    // Log-softmax over vocab
    let log_probs = ops.log_softmax(&logits_b)?;

    // Extract target log-probability
    let log_probs_data = to_vec(ops, &log_probs)?;
    let target_logprob = log_probs_data[target_token];
    let loss = -target_logprob;

    Ok((loss, logits_b, h1_relu))
}

/// Read a flat Vec<f32> from any tensor.
fn to_vec(ops: &dyn TensorOps<CandleBackend>, tensor: &candle_core::Tensor) -> Result<Vec<f32>> {
    ops.tensor_to_vec(tensor)
}

/// Mean pool over sequence dimension: [seq_len, d_model] -> [1, d_model]
fn mean_pool(
    ops: &dyn TensorOps<CandleBackend>,
    tensor: &candle_core::Tensor,
    seq_len: usize,
) -> Result<candle_core::Tensor> {
    let shape = ops.shape(tensor);
    let d_model = shape[1];
    let data = to_vec(ops, tensor)?;

    let mut pooled = vec![0.0f32; d_model];
    for i in 0..data.len() {
        let col = i % d_model;
        pooled[col] += data[i];
    }
    for v in &mut pooled {
        *v /= seq_len as f32;
    }
    ops.tensor_from_vec(pooled, &[1, d_model])
}

// =============================================================================
// Backward Pass
// =============================================================================

/// Gradient struct for all parameters.
struct Gradients {
    dw_emb: candle_core::Tensor,
    dw1: candle_core::Tensor,
    db1: candle_core::Tensor,
    dw2: candle_core::Tensor,
    db2: candle_core::Tensor,
}

/// Manual backward pass for cross-entropy + MLP.
fn backward(
    ops: &dyn TensorOps<CandleBackend>,
    input_tokens: &[usize],
    target_token: usize,
    w_emb: &Parameter<CandleBackend>,
    w1: &Parameter<CandleBackend>,
    _b1: &Parameter<CandleBackend>,
    w2: &Parameter<CandleBackend>,
    _b2: &Parameter<CandleBackend>,
    logits: &candle_core::Tensor,
    h1_relu: &candle_core::Tensor,
) -> Result<Gradients> {
    let seq_len = input_tokens.len();
    let vocab_size = ops.shape(w_emb.tensor())[0];

    // dL/dlogits = softmax(logits) - one_hot(target)
    // Compute softmax manually to avoid candle shape issues
    let logits_data = to_vec(ops, logits)?;
    let max_logit = logits_data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let exps: Vec<f32> = logits_data.iter().map(|&v| (v - max_logit).exp()).collect();
    let sum_exp: f32 = exps.iter().sum();
    let probs_data: Vec<f32> = exps.iter().map(|&v| v / sum_exp).collect();

    let mut dlogits_data: Vec<f32> = vec![0.0; vocab_size];
    for i in 0..vocab_size {
        dlogits_data[i] = probs_data[i];
    }
    dlogits_data[target_token] -= 1.0;
    let _dlogits = ops.tensor_from_vec(dlogits_data.clone(), &[1, vocab_size])?;

    // dL/dW2 = h1_relu^T @ dlogits  (outer product)
    let h1_relu_data = to_vec(ops, h1_relu)?;
    let mut dw2 = vec![0.0f32; HIDDEN_DIM * vocab_size];
    for i in 0..HIDDEN_DIM {
        for j in 0..vocab_size {
            dw2[i * vocab_size + j] = h1_relu_data[i] * dlogits_data[j];
        }
    }
    let dw2_tensor = ops.tensor_from_vec(dw2, &[HIDDEN_DIM, vocab_size])?;

    // dL/db2 = dlogits
    let db2_tensor = ops.tensor_from_vec(dlogits_data.clone(), &[vocab_size])?;

    // dL/dh1 = dlogits @ W2^T
    let w2_data = to_vec(ops, w2.tensor())?;
    let mut dh1_pre_data = vec![0.0f32; HIDDEN_DIM];
    for j in 0..HIDDEN_DIM {
        let mut sum = 0.0;
        for i in 0..vocab_size {
            sum += dlogits_data[i] * w2_data[j * vocab_size + i];
        }
        dh1_pre_data[j] = sum;
    }
    let mut dh1_data = vec![0.0f32; HIDDEN_DIM];
    for i in 0..HIDDEN_DIM {
        if h1_relu_data[i] > 0.0 {
            dh1_data[i] = dh1_pre_data[i];
        }
    }
    let dh1 = ops.tensor_from_vec(dh1_data.clone(), &[HIDDEN_DIM])?;

    // dL/dW1 = pooled^T @ dh1
    let pooled = compute_pooled(ops, input_tokens, w_emb)?;
    let pooled_data = to_vec(ops, &pooled)?;
    let mut dw1 = vec![0.0f32; D_MODEL * HIDDEN_DIM];
    for i in 0..D_MODEL {
        for j in 0..HIDDEN_DIM {
            dw1[i * HIDDEN_DIM + j] = pooled_data[i] * dh1_data[j];
        }
    }
    let dw1_tensor = ops.tensor_from_vec(dw1, &[D_MODEL, HIDDEN_DIM])?;

    // dL/db1 = dh1
    let db1_tensor = dh1;

    // dL/dW_emb: chain through embedding
    let w1_t = ops.transpose(w1.tensor())?;
    let dh1_2d = ops.reshape(&db1_tensor, &[1, HIDDEN_DIM])?;
    let dpooled = ops.matmul(&dh1_2d, &w1_t)?;
    let dpooled_flat = ops.reshape(&dpooled, &[D_MODEL])?;
    let dpooled_data = to_vec(ops, &dpooled_flat)?;

    let mut dw_emb = vec![0.0f32; vocab_size * D_MODEL];
    for (_idx, &t) in input_tokens.iter().enumerate() {
        for j in 0..D_MODEL {
            let grad_j = dpooled_data[j] / seq_len as f32;
            dw_emb[t * D_MODEL + j] += grad_j;
        }
    }
    let dw_emb_tensor = ops.tensor_from_vec(dw_emb, &[vocab_size, D_MODEL])?;

    Ok(Gradients {
        dw_emb: dw_emb_tensor,
        dw1: dw1_tensor,
        db1: db1_tensor,
        dw2: dw2_tensor,
        db2: db2_tensor,
    })
}

/// Compute pooled embedding for backward.
fn compute_pooled(
    ops: &dyn TensorOps<CandleBackend>,
    input_tokens: &[usize],
    w_emb: &Parameter<CandleBackend>,
) -> Result<candle_core::Tensor> {
    let seq_len = input_tokens.len();
    let vocab_size = ops.shape(w_emb.tensor())[0];
    let mut one_hot = vec![0.0f32; seq_len * vocab_size];
    for (i, &t) in input_tokens.iter().enumerate() {
        one_hot[i * vocab_size + t] = 1.0;
    }
    let x = ops.tensor_from_vec(one_hot, &[seq_len, vocab_size])?;
    let emb = ops.matmul(&x, w_emb.tensor())?;
    mean_pool(ops, &emb, seq_len)
}

// =============================================================================
// SGD Update
// =============================================================================

/// SGD parameter update: param = param - lr * grad
fn update_param(
    param: &mut Parameter<CandleBackend>,
    grad: &candle_core::Tensor,
    lr: f32,
    ops: &dyn TensorOps<CandleBackend>,
) -> Result<()> {
    let current = param.tensor();
    let scaled_grad = ops.mul_scalar(grad, lr)?;
    let new_val = ops.sub(current, &scaled_grad)?;
    *param = mnr_core::Parameter::new(param.name(), new_val);
    Ok(())
}

// =============================================================================
// Generation
// =============================================================================

/// Greedy text generation.
fn generate(
    ops: &dyn TensorOps<CandleBackend>,
    prompt: &str,
    max_new_tokens: usize,
    vocab: &Vocab,
    w_emb: &Parameter<CandleBackend>,
    w1: &Parameter<CandleBackend>,
    b1: &Parameter<CandleBackend>,
    w2: &Parameter<CandleBackend>,
    b2: &Parameter<CandleBackend>,
) -> Result<String> {
    let mut tokens: Vec<usize> = prompt.chars().map(|c| vocab.encode(c)).collect();

    for _ in 0..max_new_tokens {
        let start = tokens.len().saturating_sub(BLOCK_SIZE);
        let context = &tokens[start..];

        let (_, logits, _) = forward(
            ops, context, 0, w_emb, w1, b1, w2, b2,
        )?;

        let logits_data = to_vec(ops, &logits)?;
        let vocab_size = logits_data.len();

        let mut max_idx = 0;
        let mut max_val = f32::NEG_INFINITY;
        for i in 0..vocab_size {
            if logits_data[i] > max_val {
                max_val = logits_data[i];
                max_idx = i;
            }
        }

        tokens.push(max_idx);
    }

    let generated: String = tokens[prompt.len()..].iter().map(|&t| vocab.decode(t)).collect();
    Ok(generated)
}

// =============================================================================
// Vocabulary
// =============================================================================

/// Character vocabulary.
struct Vocab {
    chars: Vec<char>,
    char_to_idx: std::collections::HashMap<char, usize>,
}

impl Vocab {
    fn build(text: &str) -> Self {
        let mut unique: Vec<char> = text.chars().collect();
        unique.sort_unstable();
        unique.dedup();

        let mut map = std::collections::HashMap::new();
        for (i, c) in unique.iter().enumerate() {
            map.insert(*c, i);
        }
        Self { chars: unique, char_to_idx: map }
    }

    fn len(&self) -> usize {
        self.chars.len()
    }

    fn encode(&self, c: char) -> usize {
        *self.char_to_idx.get(&c).unwrap_or(&0)
    }

    fn decode(&self, idx: usize) -> char {
        self.chars.get(idx).copied().unwrap_or('?')
    }
}

fn build_vocab(text: &str) -> Vocab {
    Vocab::build(text)
}
