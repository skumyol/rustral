# MNR Concepts Guide

> **A beginner-friendly explanation of every concept in the Modular Neural Runtime**

This document explains every concept you'll encounter when building neural networks with MNR. No prior Rust knowledge is required—just basic familiarity with how neural networks work.

---

## Table of Contents

1. [The Big Picture](#the-big-picture)
2. [Tensors](#tensors)
3. [Backends](#backends)
4. [The Module Trait](#the-module-trait)
5. [Forward Context](#forward-context)
6. [Parameters](#parameters)
7. [Autodiff and Gradients](#autodiff-and-gradients)
8. [Optimizers](#optimizers)
9. [Loss Functions](#loss-functions)
10. [Data Loading](#data-loading)
11. [Distributed Training](#distributed-training)
12. [Common Patterns](#common-patterns)

---

## The Big Picture

A neural network training pipeline has these steps:

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Data      │ -> │   Model     │ -> │    Loss     │ -> │  Optimizer  │
│  (images,   │    │  (layers    │    │  (how wrong │    │ (update     │
│   text)     │    │   of math)  │    │   are we?)  │    │  weights)   │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
       │                  │                  │                  │
       v                  v                  v                  v
   DataLoader        Forward pass        Compare to        Backward pass
   (batching)         (predictions)       targets            (gradients)
                                                              then step()
```

MNR provides tools for each step.

---

## Tensors

### What is a Tensor?

A **tensor** is a container for numbers with a specific shape. Think of it like a spreadsheet with dimensions:

| Dimension | Example | Shape | Size |
|-----------|---------|-------|------|
| 0D (scalar) | Single number `5.0` | `[]` | 1 |
| 1D (vector) | List of numbers `[0.1, 0.5, 0.9]` | `[3]` | 3 |
| 2D (matrix) | Grid `[[1, 2], [3, 4]]` | `[2, 2]` | 4 |
| 3D | Batch of 2D images `[batch, height, width]` | `[32, 28, 28]` | 25,088 |
| 4D | Batch of color images `[batch, H, W, channels]` | `[32, 28, 28, 3]` | 75,264 |

### Creating Tensors

```rust
use mnr_core::Backend;
use mnr_ndarray_backend::CpuBackend;

let backend = CpuBackend::default();
let ops = backend.ops();

// From a vector: 2 samples, 3 features each
let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
let tensor = backend.tensor_from_vec(data, &[2, 3]).unwrap();

// Zeros: 10 elements
let zeros = ops.zeros(&[10]).unwrap();

// Random normal distribution: mean=0, std=1
let random = ops.randn(&[100], 42).unwrap();  // 42 is the seed
```

### Tensor Operations

```rust
// Add two tensors element-wise
let c = ops.add(&a, &b).unwrap();

// Matrix multiplication: [M, K] @ [K, N] -> [M, N]
let product = ops.matmul(&a, &b).unwrap();

// Apply ReLU (set negative values to 0)
let activated = ops.relu(&x).unwrap();

// Reshape: change dimensions without changing data
let flat = ops.reshape(&image, &[1, 784]).unwrap();
```

### Understanding Shapes

Shape is **critical**. Mismatched shapes cause errors:

```rust
// OK: [2, 3] + [2, 3] -> [2, 3]
let a = backend.tensor_from_vec(vec![1.0; 6], &[2, 3]).unwrap();
let b = backend.tensor_from_vec(vec![2.0; 6], &[2, 3]).unwrap();
let c = ops.add(&a, &b).unwrap();  // Works!

// ERROR: [2, 3] + [3, 2] -> shape mismatch!
let d = backend.tensor_from_vec(vec![2.0; 6], &[3, 2]).unwrap();
// let e = ops.add(&a, &d).unwrap();  // Would panic!
```

**Rule:** For element-wise operations, shapes must match exactly. For matrix multiplication, inner dimensions must match: `[M, K] @ [K, N]`.

---

## Backends

### What is a Backend?

The **Backend** is where computation happens. MNR supports multiple backends:

| Backend | Where Math Runs | Use Case |
|---------|----------------|----------|
| `CpuBackend` | CPU (your processor) | Testing, small models, debugging |
| `WgpuBackend` | GPU (graphics card) | Production, large models |
| *Your own* | Custom hardware | Specialized deployments |

### Why Abstract the Backend?

Your model code looks the same regardless of backend:

```rust
// This works on CPU OR GPU—same code!
let output = linear.forward(input, &mut ctx)?;
let activated = ops.relu(&output)?;
```

The backend handles:
- Where tensors live (RAM vs GPU memory)
- How operations execute (CPU loops vs GPU shaders)
- Memory allocation and cleanup

### Switching Backends

```rust
use mnr_ndarray_backend::CpuBackend;
// use mnr_wgpu_backend::WgpuBackend;  // Uncomment for GPU

// Same code, different speed
let backend = CpuBackend::default();
// let backend = WgpuBackend::new()?;  // GPU version
```

---

## The Module Trait

### What is a Module?

A **Module** is anything that takes input, transforms it, and produces output. This includes:

- `Linear` layer (matrix multiplication + bias)
- `Conv2d` layer (sliding window filters)
- `LSTM` (memory cells for sequences)
- `Transformer` (attention mechanism)
- **Your custom layers**

### The Contract

Every module implements:

```rust
trait Module<B: Backend> {
    type Input;   // What goes in (e.g., a tensor)
    type Output;  // What comes out (e.g., another tensor)

    fn forward(&self, input: Self::Input, ctx: &mut ForwardCtx<B>) -> Result<Self::Output>;
}
```

**Key insight:** The `forward` method is pure. No hidden state. No global variables. This makes debugging and testing trivial.

### Chaining Modules

```rust
// A simple network: Input -> Linear -> ReLU -> Linear -> Output
let hidden = linear1.forward(input, &mut ctx)?;
let activated = ops.relu(&hidden)?;
let output = linear2.forward(activated, &mut ctx)?;
```

### Building Your Own Module

```rust
use mnr_core::{Backend, ForwardCtx, Module, Mode};
use mnr_nn::{Linear, LinearBuilder};

// A custom residual block: output = input + f(input)
struct ResidualBlock<B: Backend> {
    linear: Linear<B>,
}

impl<B: Backend> Module<B> for ResidualBlock<B> {
    type Input = B::Tensor;
    type Output = B::Tensor;

    fn forward(&self, input: Self::Input, ctx: &mut ForwardCtx<B>) -> Result<Self::Output> {
        let ops = ctx.backend().ops();

        // f(input)
        let transformed = self.linear.forward(input.clone(), ctx)?;
        let activated = ops.relu(&transformed)?;

        // input + f(input)
        ops.add(&input, &activated)
    }
}
```

---

## Forward Context

### Why Explicit Context?

PyTorch hides a global "mode" (training vs inference). MNR makes it explicit:

```rust
// Training mode: Dropout is active, BatchNorm updates running stats
let mut train_ctx = ForwardCtx::new(&backend, Mode::Train);

// Inference mode: Dropout disabled, BatchNorm uses frozen stats
let mut eval_ctx = ForwardCtx::new(&backend, Mode::Inference);
```

**Benefits:**
- No accidentally leaving a model in "train mode" during evaluation
- Multiple models can run simultaneously with different modes
- Thread-safe (no shared mutable state)

### What's in the Context?

| Field | Purpose |
|-------|---------|
| `backend` | Where computation happens |
| `mode` | Train or inference |
| `run_id` | Unique identifier for this forward pass |

---

## Parameters

### What is a Parameter?

A **parameter** is a tensor that the model **learns** during training. Examples:

- Weights in a Linear layer (`W` in `y = xW^T + b`)
- Convolution filters
- Embedding vectors

### Creating Parameters

```rust
// Random initialization: shape [out, in], seed=42, std=0.1
let weight = backend.normal_parameter("linear.weight", &[10, 5], 42, 0.1)?;

// Zero bias: shape [out]
let bias = backend.zeros_parameter("linear.bias", &[10])?;
```

### Parameters vs Tensors

| | `Parameter` | Regular `Tensor` |
|--|------------|-----------------|
| Learnable? | Yes | No |
| Has ID? | Yes (for gradient tracking) | No |
| Has name? | Yes (for saving/loading) | No |

---

## Autodiff and Gradients

### What is Backpropagation?

Backpropagation is how neural networks learn:

1. **Forward pass:** Make a prediction
2. **Compare:** How wrong was the prediction? (loss)
3. **Backward pass:** Compute how much each weight contributed to the error (gradient)
4. **Update:** Adjust weights to reduce error (optimizer step)

### The Chain Rule (High School Version)

If `y = f(g(x))`, then:

```
dy/dx = dy/dg * dg/dx
```

**Example:** If a weight change of `+0.01` increases loss by `0.05`, the gradient is `5.0`. We should decrease the weight.

### Using the Tape

MNR records operations on a **Tape** for automatic differentiation:

```rust
use mnr_autodiff::Tape;

// 1. Create tape
let mut tape = Tape::new();

// 2. Watch inputs (tensors we want gradients for)
let input_id = tape.watch(input_tensor);
let weight_id = tape.watch_parameter(&weight);

// 3. Record operations
let hidden = tape.matmul(weight_id, input_id, &mut ctx)?;
let output = tape.add(hidden, bias_id, &mut ctx)?;

// 4. Compute loss (separately, using tape values)
let predictions = tape.value(output)?;
let loss = loss_fn.forward(&predictions, &targets, &mut ctx)?;

// 5. Backward pass: compute gradients
let grads = tape.backward(output, make_ones_fn, ops)?;

// 6. Get gradient for a specific parameter
let weight_grad = weight.gradient_from_store(&grads, tape.param_map())?;
```

### Gradient Accumulation

For large batch sizes that don't fit in memory, accumulate gradients over multiple mini-batches:

```rust
let mut accumulated_grads = HashMap::new();

for micro_batch in batches {
    let grads = compute_gradients(micro_batch)?;
    for (param_id, grad) in grads {
        *accumulated_grads.entry(param_id).or_insert(zero) += grad;
    }
}

// Now step with accumulated gradients
optimizer.step(&mut params, &accumulated_grads, &mut ctx)?;
```

---

## Optimizers

### What Does an Optimizer Do?

An optimizer updates weights based on gradients. Different optimizers have different strategies:

### SGD (Stochastic Gradient Descent)

```rust
use mnr_optim::Sgd;

// Simple: weight = weight - learning_rate * gradient
let mut optimizer = Sgd::new(0.01);  // lr = 0.01
```

**Intuition:** Walk downhill on the loss surface. Simple, but can get stuck in ravines.

### Adam (Adaptive Moment Estimation)

```rust
use mnr_optim::Adam;

// Smart: adapts learning rate per parameter
let mut optimizer = Adam::new(0.001)
    .with_betas(0.9, 0.999);  // Momentum coefficients
```

**Intuition:** Like SGD, but remembers:
- **First moment:** Average gradient direction (momentum)
- **Second moment:** Average gradient magnitude (adaptive scaling)

Faster convergence, works well for most problems.

### AdamW

```rust
use mnr_optim::AdamW;

// Adam + proper weight decay (regularization)
let mut optimizer = AdamW::new(0.001)
    .with_weight_decay(0.01);
```

**When to use:** Modern default for training transformers. Weight decay prevents overfitting.

### Learning Rate Schedules

```rust
use mnr_optim::{LinearWarmup, CosineAnnealingLR, LRScheduler};

// Warm up: lr goes from 0 to 0.001 over 1000 steps
let warmup = LinearWarmup::new(0.001, 1000);

// Then decay: lr follows cosine curve from 0.001 to 0 over remaining steps
let decay = CosineAnnealingLR::new(0.001, total_steps - 1000);

// Combine: warmup then decay
let scheduler = warmup.chain(decay);
```

**Why schedules matter:**
- **Warmup:** Prevents early instability (large gradients from random init)
- **Decay:** Helps fine-tune as training progresses

---

## Loss Functions

### What is a Loss Function?

A **loss function** measures how wrong your predictions are. Lower = better.

### Mean Squared Error (MSE)

For regression (predicting continuous values):

```rust
use mnr_nn::MSELoss;

// loss = average((prediction - target)²)
let loss_fn = MSELoss::new();
let loss = loss_fn.forward(&predictions, &targets, &mut ctx)?;
```

**Use when:** Predicting prices, temperatures, coordinates.

### Cross-Entropy Loss

For classification (predicting categories):

```rust
use mnr_nn::CrossEntropyLoss;

// loss = -log(probability assigned to correct class)
let loss_fn = CrossEntropyLoss::new();
let loss = loss_fn.forward(&logits, &targets, &mut ctx)?;
```

**Use when:** Classifying images (cat/dog), text sentiment (positive/negative).

**Why logits, not probabilities?** The loss function applies softmax internally for numerical stability.

### Binary Cross-Entropy

For binary classification:

```rust
use mnr_nn::BCEWithLogitsLoss;

let loss_fn = BCEWithLogitsLoss::new();
```

**Use when:** Spam detection, medical diagnosis (sick/healthy).

---

## Data Loading

### The Dataset Trait

Your data source implements a simple trait:

```rust
use mnr_data::Dataset;

struct MyDataset {
    samples: Vec<Sample>,
}

impl Dataset<Sample> for MyDataset {
    fn len(&self) -> usize {
        self.samples.len()
    }

    fn get(&self, index: usize) -> Option<Sample> {
        self.samples.get(index).cloned()
    }
}
```

### The DataLoader

```rust
use mnr_data::{DataLoader, DataLoaderConfig};

let loader = DataLoader::new(
    Box::new(dataset),
    DataLoaderConfig {
        batch_size: 32,      // 32 samples per batch
        shuffle: true,       // Randomize order each epoch
        seed: Some(42),      // Reproducible shuffling
        num_workers: 4,      // 4 threads loading data
    },
);

// Iterate over batches
for batch in loader {
    // batch has 32 samples
    train_on_batch(batch)?;
}
```

### Data Augmentation (Images)

```rust
// Common transforms
let augmented = ops.random_crop(&image, 224, 224, seed)?;  // Random 224x224 crop
let flipped = ops.random_horizontal_flip(&augmented, 0.5, seed)?;  // 50% chance
let normalized = ops.normalize(&flipped, mean, std)?;  // Zero mean, unit variance
```

---

## Distributed Training

### When Do You Need Multiple GPUs?

| Scenario | Solution |
|----------|----------|
| Model fits on 1 GPU, but training is slow | **Data Parallel** — same model, split batch |
| Model too big for 1 GPU | **Tensor Parallel** — split layers across GPUs |
| Model + batch too big | **Pipeline Parallel** — different layers on different GPUs |
| Need all of the above | **3D Parallelism** — combine strategies |

### Data Parallel (Simplest)

```rust
use mnr_distributed::{ProcessGroup, DataParallelTrainer};

// 8 GPUs, I'm GPU #rank
let pg = ProcessGroup::new_threaded(8, rank)?;

// Batch is automatically split: 256 -> 32 per GPU
let mut trainer = DataParallelTrainer::new(pg, Adam::new(0.001));

// Gradients are automatically synchronized after each step
let loss = trainer.step(&mut params, &batch, &mut loss_fn, &mut ctx)?;
```

### ZeRO (Memory Optimization)

```rust
use mnr_distributed::{ZeROOptimizer, Zero2Optimizer};

// Standard: each GPU stores all optimizer states
// Memory = 4x model size (params + grads + Adam m + Adam v)

// ZeRO-2: Shard optimizer states across GPUs
// Memory = model_size + (4x model_size / num_gpus)
// With 8 GPUs: ~50% memory reduction!

let optimizer = Zero2Optimizer::new(Adam::new(0.001), pg, total_params);
```

### Pipeline Parallel (Model Too Big)

```rust
use mnr_distributed::pipeline_parallel::{PipelineParallel, PipelineConfig};

// Layer 0-3 on GPU 0, Layer 4-7 on GPU 1, etc.
let config = PipelineConfig::new()
    .with_micro_batches(8);  // Pipeline 8 micro-batches

let pipeline = PipelineParallel::new(stages, pg, config)?;
```

---

## Common Patterns

### Pattern 1: Training Loop Template

```rust
fn train_model<B: Backend>(
    model: &impl Module<B>,
    dataset: impl Dataset<Sample>,
    epochs: usize,
    backend: &B,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut optimizer = Adam::new(0.001);
    let loss_fn = CrossEntropyLoss::new();
    let mut loader = DataLoader::new(Box::new(dataset), DataLoaderConfig::default());

    for epoch in 0..epochs {
        let mut total_loss = 0.0;
        let mut num_batches = 0;

        for batch in &mut loader {
            let mut ctx = ForwardCtx::new(backend, Mode::Train);
            let mut tape = Tape::new();

            // Forward
            let output = model.forward(batch.data, &mut ctx)?;
            let loss = loss_fn.forward(&output, &batch.targets, &mut ctx)?;

            // Backward
            let grads = tape.backward(loss_node, make_ones, backend.ops())?;

            // Update
            optimizer.step(&mut params, &grads, &mut ctx)?;

            total_loss += loss_value;
            num_batches += 1;
        }

        println!("Epoch {}: avg_loss = {:.4}", epoch, total_loss / num_batches as f32);
    }

    Ok(())
}
```

### Pattern 2: Inference/Evaluation

```rust
fn evaluate<B: Backend>(
    model: &impl Module<B>,
    dataset: impl Dataset<Sample>,
    backend: &B,
) -> Result<f32, Box<dyn std::error::Error>> {
    let mut correct = 0;
    let mut total = 0;

    for sample in dataset {
        let mut ctx = ForwardCtx::new(backend, Mode::Inference);
        let output = model.forward(sample.data, &mut ctx)?;

        let predicted = argmax(&output)?;
        if predicted == sample.label {
            correct += 1;
        }
        total += 1;
    }

    Ok(correct as f32 / total as f32)
}
```

### Pattern 3: Model Checkpointing

```rust
use mnr_io::{save, load};

// Save
save(&params, "model_checkpoint.safetensors")?;

// Load (resumes training from saved state)
let loaded_params = load("model_checkpoint.safetensors", backend)?;
```

### Pattern 4: Mixed Precision

```rust
use mnr_optim::{MixedPrecisionOptimizer, DType};

let optimizer = MixedPrecisionOptimizer::new(Adam::new(0.001))
    .with_dtype(DType::Float16);  // 16-bit instead of 32-bit

// 50% memory reduction, 2-3x faster on modern GPUs
```

---

## Glossary

| Term | Simple Explanation |
|------|-------------------|
| **Tensor** | A multi-dimensional array of numbers |
| **Forward pass** | Making predictions (input → output) |
| **Backward pass** | Computing how much each weight contributed to the error |
| **Gradient** | The direction and magnitude to change a weight |
| **Loss** | How wrong the prediction is (lower = better) |
| **Epoch** | One complete pass through the entire dataset |
| **Batch** | A group of samples processed together (e.g., 32 images) |
| **Learning rate** | How big of a step to take when updating weights |
| **Overfitting** | Memorizing training data instead of learning patterns |
| **Regularization** | Techniques to prevent overfitting (e.g., weight decay) |
| **Dropout** | Randomly disabling neurons during training (prevents overfitting) |
| **Activation function** | Non-linear transform (ReLU, Sigmoid, Tanh) |
| **Embedding** | Converting discrete items (words) to vectors |
| **Attention** | Letting the model focus on relevant parts of input |
| **Transformer** | Architecture using attention (GPT, BERT) |
| **Kernel** | A small matrix used in convolution (detects features) |
| **Stride** | How far the kernel moves between operations |
| **Padding** | Adding zeros around the edge to preserve size |
| **Pooling** | Reducing size by taking max/average of regions |
| **BatchNorm** | Normalizing layer outputs to stabilize training |
| **LayerNorm** | Normalizing within each sample (used in transformers) |

---

## Next Steps

1. **Run examples:** `cargo run --example xor` then `cargo run --example mnist`
2. **Read the code:** Each example has extensive comments explaining every step
3. **Modify an example:** Change the learning rate, add a layer, see what happens
4. **Build your own:** Start with a simple dataset and Linear layers
5. **Read API docs:** `cargo doc --open` for full function signatures

Happy learning! 🦀🧠
