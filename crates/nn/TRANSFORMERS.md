# Transformer Architectures

This module provides complete transformer implementations for various NLP tasks,
following the architectures from BERT, GPT, T5, and BART.

## Overview

| Architecture | Model | Best For |
|--------------|-------|----------|
| **Encoder** | BERT, RoBERTa | Classification, embeddings, masked LM |
| **Decoder** | GPT, LLaMA | Text generation, completion, chat |
| **Encoder-Decoder** | T5, BART | Translation, summarization, QA |

## Quick Start

### BERT-Style Encoder

```rust
use rustral_nn::{TransformerEncoder, TransformerEncoderConfig};
use rustral_ndarray_backend::CpuBackend;

let backend = CpuBackend::default();

// Create BERT-base configuration
let config = TransformerEncoderConfig::new(768, 12, 12, 3072)
    .with_dropout(0.1)
    .with_max_seq_len(512);

// Build encoder (vocab_size = 30,000)
let encoder = TransformerEncoder::new(&backend, config, 30000, 42)?;

// Forward pass: [batch, seq] → [batch, seq, d_model]
let output = encoder.forward(input_tokens, &mut ctx)?;

// Extract CLS token for classification
let cls = encoder.cls_token(&output, backend.ops())?; // [batch, d_model]
```

### GPT-Style Decoder

```rust
use rustral_nn::{TransformerDecoder, TransformerDecoderConfig};

let config = TransformerDecoderConfig::new(768, 12, 12, 3072)
    .with_max_seq_len(1024);

// Build decoder (vocab_size = 50,000)
let decoder = TransformerDecoder::new(&backend, config, 50000, 42)?;

// Training: [batch, seq] → [batch, seq, vocab]
let logits = decoder.forward(input_tokens, &mut ctx)?;

// Generation: greedy decode next token
let next_token = decoder.generate_token(&prefix, &mut ctx)?;
```

### T5-Style Encoder-Decoder

```rust
use rustral_nn::{
    TransformerEncoderDecoder,
    EncoderDecoderConfig,
    TransformerEncoderConfig,
};

// Symmetric config: same d_model, heads, layers for encoder/decoder
let config = EncoderDecoderConfig::symmetric(512, 8, 6, 2048);

let model = TransformerEncoderDecoder::new(
    &backend,
    config,
    src_vocab_size,
    tgt_vocab_size,
    42
)?;

// Training: source → target
let logits = model.forward(src_tokens, tgt_tokens, &mut ctx)?;

// Inference: autoregressive generation
let generated = model.generate(src, max_len, bos, eos, &mut ctx)?;
```

## Configuration

### TransformerEncoderConfig

| Parameter | Default | Description |
|-----------|---------|-------------|
| `d_model` | - | Model dimension (embeddings) |
| `num_heads` | - | Attention heads (must divide d_model) |
| `num_layers` | - | Number of transformer layers |
| `ff_dim` | - | Feed-forward dimension |
| `dropout` | 0.1 | Dropout probability |
| `max_seq_len` | 512 | Maximum sequence length |
| `pre_norm` | true | Use pre-normalization |

### TransformerDecoderConfig

Same parameters as encoder, plus causal masking for autoregressive generation.

## Examples

Run the examples with:

```bash
cargo run --example transformer_bert_encoder
cargo run --example transformer_gpt_decoder
cargo run --example transformer_seq2seq
cargo run --example transformer_composition
```

### BERT Encoder Example

Demonstrates:
- Sentence encoding with positional embeddings
- CLS token extraction for classification
- Model scaling (Tiny → Base → Large)

### GPT Decoder Example

Demonstrates:
- Autoregressive text generation
- Greedy decoding
- Temperature sampling strategies
- GPT scaling series (Small → XL)

### Seq2Seq Example

Demonstrates:
- Machine translation workflow
- Teacher forcing during training
- Greedy decoding for inference
- T5/BART architecture comparison

### Composition Example

Demonstrates:
- Multi-task learning with shared encoder
- Adapter layers for parameter-efficient tuning
- Architecture patterns (single-task, multi-task, adapter tuning)

## Architecture Details

### Encoder Layer

```
Input → LayerNorm → Self-Attention → Add
  ↓
LayerNorm → FeedForward → Add → Output
```

**Components:**
- Multi-head self-attention (bidirectional)
- Position-wise feed-forward network
- Residual connections
- Layer normalization

### Decoder Layer

```
Input → LayerNorm → Masked Self-Attention → Add
  ↓
LayerNorm → FeedForward → Add → Output
```

**Components:**
- Masked causal self-attention
- Position-wise feed-forward network
- Causal masking (can't attend to future tokens)

### Encoder-Decoder

```
Source → [Encoder × N] → Memory
Target + Memory → [Decoder × N] → Output
```

**Components:**
- Bidirectional encoder
- Autoregressive decoder with cross-attention
- Shared embeddings (optional)

## Model Sizes

### BERT

| Size | d_model | Heads | Layers | Parameters |
|------|---------|-------|--------|------------|
| Tiny | 128 | 2 | 2 | ~4M |
| Small | 256 | 4 | 4 | ~14M |
| Base | 768 | 12 | 12 | ~110M |
| Large | 1024 | 16 | 24 | ~340M |

### GPT

| Size | d_model | Heads | Layers | Parameters |
|------|---------|-------|--------|------------|
| Small | 768 | 12 | 12 | ~124M |
| Medium | 1024 | 16 | 24 | ~350M |
| Large | 1280 | 20 | 36 | ~774M |
| XL | 1600 | 25 | 48 | ~1.5B |

### T5

| Size | d_model | Heads | Layers | Parameters |
|------|---------|-------|--------|------------|
| Small | 512 | 8 | 6 | ~60M |
| Base | 768 | 12 | 12 | ~220M |
| Large | 1024 | 16 | 24 | ~770M |

## Advanced Usage

### Multi-Task Learning

Share a single encoder across multiple task-specific heads:

```rust
let encoder = TransformerEncoder::new(&backend, config, vocab, 42)?;

// Task A: Sentiment (3 classes)
let sentiment_head = Linear::new(&backend, LinearConfig::new(d_model, 3))?;

// Task B: NER (17 tags)
let ner_head = Linear::new(&backend, LinearConfig::new(d_model, 17))?;

// Both use same encoder representation
```

### Adapter Tuning

Add small bottleneck layers for parameter-efficient fine-tuning:

```rust
// Only train ~0.5% additional parameters
let adapter = Adapter::new(&backend, d_model, 64, 42)?;
```

### Generation Strategies

```rust
// Greedy decoding
let next_token = logits.argmax();

// Temperature sampling
let scaled = logits / temperature;
let probs = softmax(scaled);
let next_token = sample(probs);

// Top-K sampling
let top_k_logits = logits.topk(k);
let probs = softmax(top_k_logits);
```

## Performance Tips

1. **Batch size**: Larger batches = better GPU utilization
2. **Sequence length**: Use `max_seq_len` to pre-allocate
3. **Mixed precision**: Use FP16 for 2x speedup
4. **KV caching**: Essential for autoregressive generation
5. **Gradient checkpointing**: Trade compute for memory

## References

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805)
- [Language Models are Unsupervised Multitask Learners (GPT-2)](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- [Exploring the Limits of Transfer Learning with T5](https://arxiv.org/abs/1910.10683)
