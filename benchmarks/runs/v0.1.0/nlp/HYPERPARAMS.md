# NLP hyperparameters (v0.1.0)

This file records the exact hyperparameters used for the canonical **real-data** runs captured in:

- `benchmarks/runs/v0.1.0/nlp/sst2.json`
- `benchmarks/runs/v0.1.0/nlp/wikitext2.json`

## SST-2 (`sst2_classifier`)

- **model**: transformer encoder
- **seq_len**: 32
- **d_model**: 64
- **num_heads**: 4
- **ffn_dim**: 128
- **num_layers**: 2
- **pooling**: mean over sequence
- **head**: linear \(64 \rightarrow 2\)
- **optimizer**: Adam
- **learning_rate**: 5e-4
- **batch_size**: 32
- **epochs**: 3
- **tokenizer**: `WordLevelTokenizer` (lowercased, whitespace), `max_vocab=8192`
- **seeds**: 0, 1, 2

## WikiText-2 (`wikitext2_lm`)

- **model**: causal transformer LM (encoder layers + causal additive mask)
- **block_size**: 32
- **d_model**: 64
- **num_heads**: 4
- **ffn_dim**: 128
- **num_layers**: 2
- **prediction**: next-token logits from the last position hidden state
- **optimizer**: Adam
- **learning_rate**: 5e-4
- **batch_size**: 32
- **epochs**: 1
- **tokenizer**: `WordLevelTokenizer` (lowercased, whitespace), `max_vocab=16384`
- **train_tokens_used**: 50_000 (token cap before windowing)
- **train_windows_used**: 2_000 (cap applied after windowing; keeps runtime reasonable)
- **eval_windows_used**: 20_000 (cap applied on validation windows for perplexity)
- **seeds**: 0, 1, 2

