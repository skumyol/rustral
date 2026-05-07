# Rustral Examples Gallery

This directory contains runnable examples for learning Rustral. The examples start small, then move into vision, NLP, and advanced model patterns.

Some newer examples live inside their owning crates instead of this `examples/` workspace. The runtime NLP examples are in `crates/runtime/examples/` because they use the trainer, dataset cache, and manifest writer from `rustral-runtime`.

## Directory Structure

```
examples/
├── basics/          # Fundamental concepts (start here!)
│   ├── xor.rs       # Simplest neural network
│   ├── train_demo.rs# Complete training loop
│   ├── mnist.rs     # Image classification
│   └── char_rnn.rs  # Text generation
├── vision/          # Computer vision examples
│   ├── building_blocks.rs
│   ├── resnet_image_classification.rs
│   └── diffusion_model.rs
├── nlp/             # Natural language processing
│   ├── bert_fine_tuning.rs
│   └── gpt_training.rs
└── advanced/        # Complex architectures
    ├── moe_training.rs
    └── custom_layer.rs
```

Runtime examples worth knowing about:

```text
crates/runtime/examples/
├── emnlp_char_lm.rs       # tiny char LM, determinism check, save/load/infer
├── sst2_classifier.rs     # SST-2 dev accuracy, writes manifest.json
└── wikitext2_lm.rs        # WikiText-2 dev perplexity, writes manifest.json
```

## Getting Started

All examples can be run with:

```bash
# Basic examples (learning fundamentals)
cargo run --bin xor          # Simplest neural network
cargo run --bin train_demo   # Complete training loop
cargo run --bin mnist        # Image classification
cargo run --bin char_rnn     # Text generation

# Vision examples (computer vision)
cargo run --example building_blocks
cargo run --example resnet_image_classification
cargo run --example diffusion_model

# NLP examples (natural language processing)
cargo run --example bert_fine_tuning
cargo run --example gpt_training

# EMNLP-ready char-level next-token LM (lives in the runtime crate)
# - one command for: train -> eval -> save -> load -> infer -> generate
# - `--determinism-check` runs the same training 3 times and asserts bitwise equality
cargo run -p rustral-runtime --features training --example emnlp_char_lm
cargo run -p rustral-runtime --features training --example emnlp_char_lm -- --determinism-check

# Real-corpus NLP examples
# - SST-2 sentiment classifier with reproducibility manifest
# - WikiText-2 small word-level LM with dev perplexity
# Use --quick for a fast smoke run. See EVALUATION.md for the methodology.
cargo run --release -p rustral-runtime --features training --example sst2_classifier
cargo run --release -p rustral-runtime --features training --example wikitext2_lm

# CI / offline-only runs, datasets pre-staged in $RUSTRAL_CACHE_DIR/datasets/
RUSTRAL_DATASET_OFFLINE=1 RUSTRAL_DATASET_SKIP_CHECKSUM=1 \
    cargo run --release -p rustral-runtime --features training --example sst2_classifier -- --quick

# Advanced examples (complex architectures)
cargo run --example moe_training
cargo run --example custom_layer
```

---

## Learning Path

### Level 1: Beginner (Understanding Basics)

**Goal:** Understand tensors, forward pass, and what a neural network actually does.

#### 1. XOR (`src/xor.rs`)
**Concepts:** Forward pass, activation functions, layers, inference

```bash
cargo run --bin xor
```

**What it teaches:**
- How a neural network transforms input through layers
- What ReLU and Sigmoid do to numbers
- Why we need a hidden layer (XOR isn't linearly separable)
- How to read predictions and compare to targets

**Key abstractions shown:**
- `Backend`: creates tensors
- `Linear` layer: matrix multiplication + bias
- `ForwardCtx`: explicit context for computation
- `Mode::Inference`: we're not training, just predicting

**After this example, you should understand:**
> A neural network is a series of mathematical transformations. Each layer takes numbers, does matrix multiplication, adds a bias, and applies an activation function. The output is the prediction.

---

#### 2. Linear Regression (`src/train_demo.rs`)
**Concepts:** Training loop, autodiff, gradients, optimizer

```bash
cargo run --bin train_demo
```

**What it teaches:**
- How the training loop works (forward → loss → backward → update)
- What a "Tape" is (records operations for gradient computation)
- How Adam optimizer adjusts weights
- What checkpointing means (saving/resuming training)

**Key abstractions shown:**
- `Tape`: records operations for automatic differentiation
- `Tape::watch_parameter()`: mark weights for gradient tracking
- `Tape::backward()`: compute gradients via chain rule
- `Adam` optimizer: updates weights using gradients
- `Optimizer::save_checkpoint()`: persist training state

**The training loop pattern:**
```rust
for epoch in 0..epochs {
    for (input, target) in data {
        // 1. FORWARD: make prediction
        let prediction = model.forward(input, &mut ctx)?;

        // 2. LOSS: how wrong were we?
        let loss = mse(&prediction, &target)?;

        // 3. BACKWARD: what caused the error?
        let grads = tape.backward(loss, ...)?;

        // 4. UPDATE: fix the weights
        optimizer.step(&mut params, &grads, &mut ctx)?;
    }
}
```

**After this example, you should understand:**
> Training is a loop: predict → measure error → find which weights caused it → adjust them slightly. The "magic" is the chain rule from calculus, which tells us exactly how much each weight contributed to the error.

---

#### 3. MNIST (`src/mnist.rs`)
**Concepts:** Convolution, pooling, flattening, data loading, batching

```bash
# Download data first (or run without for synthetic demo)
mkdir -p data/mnist
curl -o data/mnist/train-images-idx3-ubyte.gz http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
curl -o data/mnist/train-labels-idx1-ubyte.gz http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
gunzip data/mnist/*.gz

cargo run --bin mnist --release
```

**What it teaches:**
- What a Convolutional layer does (sliding filters over images)
- Why we need pooling (reduces size, keeps important features)
- How to load real datasets from files
- How batching works (process multiple images at once)
- How to count parameters in a model

**Key abstractions shown:**
- `Conv2d`: detects patterns in images (edges, textures, shapes)
- `max_pool2d`: reduces image size by taking maximum values
- `Dataset` trait: standard interface for any data source
- `DataLoader`: batches, shuffles, and loads data efficiently
- `Mode::Inference` vs `Mode::Train`

**Architecture flow:**
```
Input: [1, 1, 28, 28]  (1 image, 1 channel, 28x28 pixels)
  ↓ Conv2d(1→6, 3x3)
[1, 6, 26, 26]           (6 feature maps)
  ↓ ReLU
  ↓ MaxPool(2x2)
[1, 6, 13, 13]           (smaller, but more features)
  ↓ Conv2d(6→16, 3x3)
[1, 16, 11, 11]
  ↓ ReLU
  ↓ MaxPool(2x2)
[1, 16, 5, 5]            (16 feature maps, 5x5 each)
  ↓ Flatten
[1, 400]                 (16 * 5 * 5 = 400)
  ↓ Linear(400→128)
[1, 128]
  ↓ ReLU
  ↓ Linear(128→10)
[1, 10]                  (10 class probabilities)
```

**After this example, you should understand:**
> CNNs work by sliding small filters over images to detect features. Early layers detect edges, middle layers detect shapes, later layers detect objects. Each layer makes the image smaller but with more feature channels.

---

### Level 2: Intermediate (Working with Sequences and Modern Architectures)

**Goal:** Understand how models work with text, sequences, and attention.

#### 4. Character RNN (`src/char_rnn.rs`)
**Concepts:** Embeddings, LSTM/GRU, sequence processing, vocabulary, text generation

```bash
# Optional: download real dataset
curl -o data/shakespeare.txt https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

cargo run --bin char_rnn --release
```

**What it teaches:**
- What an embedding is (converting characters to vectors)
- How LSTM remembers information across time steps
- Why we need a vocabulary (map characters to numbers)
- How autoregressive generation works (predict next character)
- What a DataLoader configuration looks like

**Key abstractions shown:**
- `Embedding`: converts discrete tokens to dense vectors
- `StackedLstm`: memory cells for sequence modeling
- `Readout`: converts hidden state to vocabulary predictions
- `Vocabulary`: maps characters to indices and back
- `Dataset` with window sliding: creates training samples from text

**The text generation process:**
```
"T" -> embedding -> LSTM -> predict "h"
"h" -> embedding -> LSTM -> predict "e"
"e" -> embedding -> LSTM -> predict " "
" " -> embedding -> LSTM -> predict "q"
...
"To be, or not to be..."
```

**After this example, you should understand:**
> Language models predict one character at a time, using everything they've seen so far. An LSTM is a type of memory cell that can remember relevant information (like whether we're inside parentheses) and forget irrelevant details.

---

#### 5. ResNet Image Classification (`resnet_image_classification.rs`)
**Concepts:** Residual connections, batch normalization, deep networks, modern CNNs

```bash
cargo run --example resnet_image_classification
```

**What it teaches:**
- Why deep networks are hard to train (vanishing gradients)
- How residual connections solve this (skip connections)
- What batch normalization does (stabilize training)
- How to build complex architectures from simple blocks

**Key abstractions shown:**
- `ResidualBlock`: adds input to output (skip connection)
- `BatchNorm2d`: normalizes activations per batch
- `GlobalAvgPool2d`: replaces large FC layers
- Composing blocks into stages

**The residual trick:**
```
Standard: output = layer(input)           # Must learn identity for small changes
Residual: output = input + layer(input)   # Can easily learn identity (set layer=0)
```

**After this example, you should understand:**
> ResNet can be 100+ layers deep because skip connections provide a "highway" for gradients. Even if a layer learns nothing, the identity connection ensures information still flows through.

---

### Level 3: Advanced (Transformers, Large Models, Distributed Training)

**Goal:** Understand modern transformer architectures and distributed training.

#### 6. BERT Fine-tuning (`bert_fine_tuning.rs`)
**Concepts:** Transformers, self-attention, pre-training vs fine-tuning, tokenization

```bash
cargo run --example bert_fine_tuning
```

**What it teaches:**
- How BERT works (bidirectional encoding)
- What attention is (each word looks at every other word)
- How to fine-tune a pre-trained model
- Why tokenization matters (converting text to numbers)

**Key abstractions shown:**
- `TransformerEncoder`: bidirectional transformer layers
- `LayerNorm`: stabilizes transformer training
- `Dropout`: prevents overfitting
- Tokenization with [CLS] and [SEP] special tokens

**The attention mechanism:**
```
For each word, compute three vectors:
  - Query: "What am I looking for?"
  - Key: "What do I contain?"
  - Value: "What information do I have?"

Score = Query · Key  (how relevant is word B to word A?)
Output = weighted sum of Values (collect relevant information)
```

**After this example, you should understand:**
> Transformers don't process text left-to-right. Every word can immediately see every other word. This is why BERT understands context so well (e.g., "bank" as river bank vs financial bank).

---

#### 7. GPT Training (`gpt_training.rs`)
**Concepts:** Causal/decoding-only transformers, next-token prediction, autoregressive generation

```bash
cargo run --example gpt_training
```

**What it teaches:**
- How GPT differs from BERT (causal vs bidirectional)
- Why we need causal masking (can only see past, not future)
- How temperature and top-k sampling work for generation
- Memory requirements for large language models

**Key abstractions shown:**
- `TransformerDecoder`: causal (autoregressive) transformer
- `TransformerDecoderConfig`: sets causal=True
- Text generation with top-k sampling
- Memory estimation for different model sizes

**Causal vs Bidirectional:**
```
BERT (bidirectional): The cat sat on the [MASK]
  "sat" can see "The", "cat", "on", "the", "mat"

GPT (causal): The cat sat
  "sat" can only see "The", "cat" (not future words)
```

**After this example, you should understand:**
> GPT predicts the next word given all previous words. This is why it can generate text: you just keep asking "what comes next?" The causal mask ensures it doesn't "cheat" by looking at future words during training.

---

#### 8. Diffusion Model (`diffusion_model.rs`)
**Concepts:** Diffusion process, U-Net architecture, VAE, noise scheduling, classifier-free guidance

```bash
cargo run --example diffusion_model
```

**What it teaches:**
- How diffusion models generate images (reverse noise process)
- What a U-Net is (encoder-decoder with skip connections)
- How VAE compresses images to latent space
- Why noise scheduling matters (how fast to add/remove noise)

**Key abstractions shown:**
- `UNet2D`: noise-prediction network
- `DDPMScheduler`: controls noise schedule
- `VAEEncoder`/`VAEDecoder`: image to latent space and back
- `ClassifierFreeGuidance`: controls generation quality/diversity

**The diffusion process:**
```
Training:
  1. Take image
  2. Add random noise (controlled by timestep t)
  3. Train U-Net to predict the noise

Generation:
  1. Start with pure noise
  2. Predict noise with U-Net
  3. Subtract predicted noise (get slightly less noise)
  4. Repeat 1000 times → final image
```

**After this example, you should understand:**
> Diffusion models learn to undo noise. They start with random static and gradually refine it into a coherent image, like a sculptor revealing a statue from a block of marble.

---

#### 9. Mixture of Experts (`moe_training.rs`)
**Concepts:** Sparse activation, routing, all-to-all communication, distributed training, load balancing

```bash
cargo run --example moe_training
```

**What it teaches:**
- How MoE scales models without proportionally scaling compute
- What top-k routing means (each token picks k experts)
- Why load balancing is crucial (don't want all tokens on one GPU)
- How all-to-all communication works in distributed systems

**Key abstractions shown:**
- `ExpertLayer`: container with multiple expert networks
- `SwitchGating` / `TopKGating`: routing networks
- `ExpertParallelConfig`: configures DP + EP + TP parallelism
- Load balancing loss (auxiliary loss)

**The MoE idea:**
```
Standard model: Every token goes through ALL parameters
  (e.g., 7B parameters, 7B active)

MoE model: Each token goes through a SUBSET of parameters
  (e.g., 64B total parameters, only 2B active per token)
  
Result: More knowledge capacity, same compute cost!
```

**After this example, you should understand:**
> MoE is like having 64 specialists instead of one generalist. Each problem (token) is routed to the most relevant 2 specialists. This lets you build models with trillions of parameters that only use a fraction per input.

---

## Example Organization

### By Difficulty

| Level | Example | Lines | Time to Understand | Location |
|-------|---------|-------|-------------------|----------|
| Beginner | XOR | 139 | 15 min | `basics/xor.rs` |
| Beginner | Linear Regression | 136 | 20 min | `basics/train_demo.rs` |
| Beginner | MNIST | 370 | 30 min | `basics/mnist.rs` |
| Beginner | Character RNN | 365 | 45 min | `basics/char_rnn.rs` |
| Intermediate | ResNet | ~200 | 30 min | `vision/resnet_image_classification.rs` |
| Advanced | BERT | ~200 | 45 min | `nlp/bert_fine_tuning.rs` |
| Advanced | GPT | ~250 | 45 min | `nlp/gpt_training.rs` |
| Advanced | Diffusion | ~400 | 60 min | `vision/diffusion_model.rs` |
| Advanced | MoE | ~300 | 60 min | `advanced/moe_training.rs` |

### By Concept

| Concept | Best Example | File |
|---------|-------------|------|
| Forward pass | XOR | `basics/xor.rs` |
| Training loop (full) | Linear Regression | `basics/train_demo.rs` |
| Autodiff/Gradients | Linear Regression | `basics/train_demo.rs` |
| Convolution | MNIST | `basics/mnist.rs` |
| Data loading | MNIST | `basics/mnist.rs` |
| Embeddings | Character RNN | `basics/char_rnn.rs` |
| LSTM/RNN | Character RNN | `basics/char_rnn.rs` |
| Residual connections | ResNet | `vision/resnet_image_classification.rs` |
| BatchNorm | ResNet | `vision/resnet_image_classification.rs` |
| Transformers (encoder) | BERT | `nlp/bert_fine_tuning.rs` |
| Transformers (decoder) | GPT | `nlp/gpt_training.rs` |
| Attention | BERT, GPT | `nlp/bert_fine_tuning.rs` |
| Transfer learning | BERT | `nlp/bert_fine_tuning.rs` |
| Text generation | GPT, Character RNN | `nlp/gpt_training.rs` |
| Diffusion | Diffusion | `vision/diffusion_model.rs` |
| U-Net | Diffusion | `vision/diffusion_model.rs` |
| VAE | Diffusion | `vision/diffusion_model.rs` |
| MoE | MoE | `advanced/moe_training.rs` |
| Distributed training | MoE | `advanced/moe_training.rs` |
| All-to-all communication | MoE | `advanced/moe_training.rs` |
| Custom Layers | Custom Layer | `advanced/custom_layer.rs` |

---

## Running Examples

### Requirements

**Minimal (CPU only):**
- Rust 1.75+
- 4GB RAM
- No GPU needed

**Recommended (with GPU):**
- Rust 1.75+
- NVIDIA GPU with CUDA support (optional, for WGPU backend)
- 8GB+ RAM

### Quick Test

```bash
# Verify everything works
cargo run --bin xor          # Should complete in <1s
cargo run --bin train_demo  # Should complete in <1s
```

### With Real Data

```bash
# MNIST
mkdir -p data/mnist
curl -O http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
curl -O http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
gunzip *.gz
mv *.ubyte data/mnist/

cargo run --bin mnist --release

# Shakespeare (for char_rnn)
mkdir -p data
curl -o data/shakespeare.txt https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

cargo run --bin char_rnn --release
```

---

## Writing Your Own Example

### Template

```rust
//! My Custom Example
//!
//! What this demonstrates: [one sentence]
//!
//! # Usage
//!
//! ```bash
//! cargo run --example my_example
//! ```

use rustral_core::{Backend, ForwardCtx, Mode, Module};
use rustral_ndarray_backend::CpuBackend;
use rustral_nn::{Linear, LinearBuilder, MSELoss};
use rustral_optim::Adam;

fn main() {
    println!("My Custom Example");
    println!("==================\n");

    // 1. Initialize backend
    let backend = CpuBackend::default();

    // 2. Create model
    let model = MyModel::new(&backend);

    // 3. Create data
    let data = generate_data();

    // 4. Train/evaluate
    // ... your code here

    // 5. Print results
    println!("Done!");
}
```

### Guidelines

1. **Start with a clear comment** explaining what the example teaches
2. **Print the architecture** so users can see the model structure
3. **Show intermediate outputs** (not just final results)
4. **Include error handling** with helpful messages
5. **Add a synthetic data fallback** if real data isn't available
6. **Comment every non-obvious line**, remember, beginners will read this

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| "Could not load data" | Example runs with synthetic data automatically |
| "Out of memory" | Reduce batch size or model size |
| "Compilation error" | Ensure Rust 1.75+ (`rustc --version`) |
| "Slow execution" | Add `--release` flag: `cargo run --bin mnist --release` |

---

## Contributing Examples

We welcome new examples! Good additions:

- Techniques not yet covered (e.g., GANs, reinforcement learning)
- Domain-specific examples (audio, time series, graph networks)
- Simplified versions of complex architectures
- Benchmarks comparing approaches

See `CONTRIBUTING.md` for submission guidelines.
