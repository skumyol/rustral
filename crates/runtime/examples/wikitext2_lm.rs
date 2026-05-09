//! WikiText-2 small word-level language model (Phase 2 P1 H3).
//!
//! Reproducible end-to-end run on the WikiText-2 raw corpus. The model is a small Rustral
//! transformer LM with causal self-attention: word embedding + learned positional embedding
//! plus N transformer encoder layers with a `[block_size, block_size]` causal additive mask,
//! pooled by taking the last position's hidden state and projected to vocabulary logits.
//! Training reports per-epoch loss and final dev perplexity; the manifest captures full
//! provenance.
//!
//! Output artifacts (`--out-dir`, default `./out/wikitext2`):
//! - `manifest.json` — git SHA, seed, hyperparameters, vocab size, dev perplexity,
//!   throughput, dataset stats, model parameter count.
//! - `vocab.txt` — one token per line.
//!
//! Online run (downloads + extracts the WikiText-2 raw zip; extraction uses the system
//! `unzip` helper inside `rustral-data` when not in offline mode):
//!
//! ```bash
//! cargo run --release -p rustral-runtime --features training --example wikitext2_lm
//! ```
//!
//! Model size and context length are configurable, for example a fast CPU benchmark:
//!
//! ```bash
//! cargo run --release -p rustral-runtime --features training --example wikitext2_lm -- \
//!   --quick --block-size 16 --d-model 32 --num-heads 2 --ffn-dim 64 --num-layers 1
//! ```
//!
//! Offline / CI run (pre-staged train/valid/test text files in
//! `~/.cache/rustral/datasets/wikitext-2/{train,valid,test}.txt`):
//!
//! ```bash
//! RUSTRAL_DATASET_OFFLINE=1 RUSTRAL_DATASET_SKIP_CHECKSUM=1 \
//! cargo run --release -p rustral-runtime --features training --example wikitext2_lm
//! ```

#[cfg(feature = "training")]
fn main() -> anyhow::Result<()> {
    runner::run()
}

#[cfg(not(feature = "training"))]
fn main() {
    eprintln!("This example requires `--features training`.");
}

#[cfg(feature = "training")]
mod runner {
    use std::fs;
    use std::path::PathBuf;
    use std::time::Instant;

    use rustral_autodiff::{Tape, TensorId};
    #[cfg(feature = "cuda")]
    use rustral_candle_backend::CandleBackend;
    use rustral_core::{Backend, ForwardCtx, Mode, NamedParameters, Parameter, PoolStrategy, Result, TensorPool};
    use rustral_data::datasets::wikitext2::load_wikitext2;
    use rustral_data::tokenizer::{WordLevelConfig, WordLevelTokenizer};
    #[cfg(not(feature = "cuda"))]
    use rustral_ndarray_backend::CpuBackend;
    use rustral_nn::tape::TapeModule;
    use rustral_nn::tape_transformer::{
        causal_mask_tape, TapeTransformerEncoderConfig, TapeTransformerEncoderLayer,
    };
    use rustral_nn::{Embedding, EmbeddingConfig, Linear, LinearBuilder};
    use rustral_optim::Adam;
    use rustral_runtime::{SupervisedTapeModel, TapeTrainer, TapeTrainerConfig, TrainingReport};

    const DEFAULT_SEED: u64 = 0xC0FFEE;
    const DEFAULT_BLOCK_SIZE: usize = 32;
    const DEFAULT_D_MODEL: usize = 64;
    const DEFAULT_NUM_HEADS: usize = 4;
    const DEFAULT_FFN_DIM: usize = 128;
    const DEFAULT_NUM_LAYERS: usize = 2;

    #[derive(Clone, Copy)]
    pub struct WikiLmDims {
        block: usize,
        d_model: usize,
        num_heads: usize,
        ffn_dim: usize,
        num_layers: usize,
    }
    const MAX_VOCAB: usize = 16_384;
    const DEFAULT_EPOCHS: usize = 1;
    const DEFAULT_BATCH: usize = 32;
    const DEFAULT_LR: f32 = 5e-4;
    const DEFAULT_TRAIN_TOKENS_QUICK: usize = 4_000;
    const DEFAULT_TRAIN_TOKENS: usize = 50_000;
    /// 0 means "evaluate on all windows".
    const DEFAULT_EVAL_WINDOWS: usize = 0;
    /// 0 means "train on all windows".
    const DEFAULT_TRAIN_WINDOWS: usize = 0;

    #[cfg(not(feature = "cuda"))]
    type DefaultBackend = CpuBackend;
    #[cfg(feature = "cuda")]
    type DefaultBackend = CandleBackend;

    fn make_backend() -> Result<DefaultBackend> {
        #[cfg(feature = "cuda")]
        {
            Ok(CandleBackend::cuda(0).unwrap_or_else(|_| CandleBackend::cpu()))
        }
        #[cfg(not(feature = "cuda"))]
        {
            Ok(CpuBackend::default())
        }
    }

    /// Small Rustral-native causal-attention LM for WikiText-2.
    pub struct WikiTextLm<B: Backend> {
        pub tok_embed: Embedding<B>,
        pub pos_embed: Embedding<B>,
        pub layers: Vec<TapeTransformerEncoderLayer<B>>,
        pub head: Linear<B>,
        pub block: usize,
    }

    impl<B: Backend> NamedParameters<B> for WikiTextLm<B> {
        fn visit_parameters(&self, f: &mut dyn FnMut(&str, &Parameter<B>)) {
            self.tok_embed.visit_parameters(&mut |n, p| f(&format!("tok_embed.{n}"), p));
            self.pos_embed.visit_parameters(&mut |n, p| f(&format!("pos_embed.{n}"), p));
            for (i, layer) in self.layers.iter().enumerate() {
                layer.visit_parameters(&mut |n, p| f(&format!("layers.{i}.{n}"), p));
            }
            self.head.visit_parameters(&mut |n, p| f(&format!("head.{n}"), p));
        }
        fn visit_parameters_mut(&mut self, f: &mut dyn FnMut(&str, &mut Parameter<B>)) {
            self.tok_embed.visit_parameters_mut(&mut |n, p| f(&format!("tok_embed.{n}"), p));
            self.pos_embed.visit_parameters_mut(&mut |n, p| f(&format!("pos_embed.{n}"), p));
            for (i, layer) in self.layers.iter_mut().enumerate() {
                layer.visit_parameters_mut(&mut |n, p| f(&format!("layers.{i}.{n}"), p));
            }
            self.head.visit_parameters_mut(&mut |n, p| f(&format!("head.{n}"), p));
        }
    }

    impl<B: Backend> WikiTextLm<B>
    where
        B::Tensor: Clone,
    {
        pub fn new(backend: &B, vocab: usize, seed: u64, dims: WikiLmDims) -> Result<Self> {
            let tok_embed =
                Embedding::new(backend, EmbeddingConfig::new(vocab, dims.d_model), seed.wrapping_add(1))?;
            let pos_embed = Embedding::new(
                backend,
                EmbeddingConfig::new(dims.block, dims.d_model),
                seed.wrapping_add(2),
            )?;
            let mut layers = Vec::with_capacity(dims.num_layers);
            for i in 0..dims.num_layers {
                let cfg = TapeTransformerEncoderConfig::new(dims.d_model, dims.num_heads, dims.ffn_dim);
                let layer =
                    TapeTransformerEncoderLayer::new(backend, cfg, seed.wrapping_add(100 + i as u64))?;
                layers.push(layer);
            }
            let head = LinearBuilder::new(dims.d_model, vocab)
                .with_bias(true)
                .seed(seed.wrapping_add(999))
                .build(backend)?;
            Ok(Self { tok_embed, pos_embed, layers, head, block: dims.block })
        }

        /// Single-window inference: returns `[vocab_size]` logits for the next-token prediction.
        pub fn logits(&self, backend: &B, ids: &[usize]) -> Result<Vec<f32>> {
            let mut ctx = ForwardCtx::new(backend, Mode::Inference);
            let mut tape = Tape::<B>::new();
            let logits_id = self.forward_tape_internal(backend, ids, &mut tape, &mut ctx)?;
            let t = tape.value(logits_id).expect("forward produced a value");
            backend.ops().tensor_to_vec(t)
        }

        fn forward_tape_internal(
            &self,
            backend: &B,
            ids: &[usize],
            tape: &mut Tape<B>,
            ctx: &mut ForwardCtx<B>,
        ) -> Result<TensorId> {
            assert_eq!(ids.len(), self.block, "block size mismatch in forward");
            let ops = backend.ops();
            let tok_ids_f32: Vec<f32> = ids.iter().map(|&i| i as f32).collect();
            let tok_ids_t = ops.tensor_from_vec(tok_ids_f32, &[self.block])?;
            let tok_ids_id = tape.watch(tok_ids_t);
            let tok_emb = self.tok_embed.forward_tape(tok_ids_id, tape, ctx)?;

            let pos_ids_f32: Vec<f32> = (0..self.block).map(|i| i as f32).collect();
            let pos_ids_t = ops.tensor_from_vec(pos_ids_f32, &[self.block])?;
            let pos_ids_id = tape.watch(pos_ids_t);
            let pos_emb = self.pos_embed.forward_tape(pos_ids_id, tape, ctx)?;

            let mut x = tape.add(tok_emb, pos_emb, ctx)?;

            // Causal mask once for the whole stack.
            let mask = causal_mask_tape::<B>(self.block, tape, ctx)?;
            for layer in &self.layers {
                x = layer.forward_tape_with_mask(x, Some(mask), tape, ctx)?;
            }

            // Take the last position [seq=block-1, :] -> shape [1, d_model] via slice_tape (dim 0).
            let last_hidden = tape.slice_tape(x, self.block - 1, self.block, ctx)?;
            // Project to vocab logits.
            self.head.forward_tape(last_hidden, tape, ctx)
        }
    }

    impl<B: Backend> SupervisedTapeModel<B, Vec<usize>, usize> for WikiTextLm<B>
    where
        B::Tensor: Clone,
    {
        fn forward_tape(
            &mut self,
            input: Vec<usize>,
            tape: &mut Tape<B>,
            ctx: &mut ForwardCtx<B>,
        ) -> Result<TensorId> {
            let backend_ptr = ctx.backend() as *const B;
            // SAFETY: see comment in sst2_classifier::TransformerSst2::forward_tape.
            let backend = unsafe { &*backend_ptr };
            self.forward_tape_internal(backend, &input, tape, ctx)
        }

        fn loss_tape(
            &mut self,
            logits: TensorId,
            target: usize,
            tape: &mut Tape<B>,
            ctx: &mut ForwardCtx<B>,
        ) -> Result<TensorId> {
            let t = ctx.backend().ops().tensor_from_vec(vec![target as f32], &[1])?;
            let t = tape.watch(t);
            tape.cross_entropy_loss(logits, t, ctx)
        }
    }

    /// Build (X, Y) windows from a stream of token ids.
    fn build_windows(ids: &[usize], block: usize) -> Vec<(Vec<usize>, usize)> {
        if ids.len() <= block {
            return Vec::new();
        }
        let mut out = Vec::with_capacity(ids.len() - block);
        for i in 0..(ids.len() - block) {
            let x = ids[i..i + block].to_vec();
            let y = ids[i + block];
            out.push((x, y));
        }
        out
    }

    /// Mean CE / perplexity / top-1 / top-5 accuracy on next-token prediction.
    fn evaluate<B: Backend>(
        backend: &B,
        model: &WikiTextLm<B>,
        examples: &[(Vec<usize>, usize)],
    ) -> Result<(f32, f32, f32, f32)>
    where
        B::Tensor: Clone,
    {
        if examples.is_empty() {
            return Ok((0.0, 1.0, 0.0, 0.0));
        }
        let mut total_loss = 0.0f64;
        let mut top1_ok = 0u64;
        let mut top5_ok = 0u64;
        for (ids, label) in examples {
            let logits = model.logits(backend, ids)?;
            let mut max_v = f32::NEG_INFINITY;
            for v in &logits {
                if *v > max_v {
                    max_v = *v;
                }
            }
            let mut sum_exp = 0.0f64;
            for v in &logits {
                sum_exp += ((*v - max_v) as f64).exp();
            }
            let log_denom = (max_v as f64) + sum_exp.ln();
            total_loss += log_denom - logits[*label] as f64;

            let lab = *label;
            let mut order: Vec<usize> = (0..logits.len()).collect();
            order.sort_by(|&a, &b| logits[b].partial_cmp(&logits[a]).unwrap_or(std::cmp::Ordering::Equal));
            if order.first().copied() == Some(lab) {
                top1_ok += 1;
            }
            if order.iter().take(5).any(|&i| i == lab) {
                top5_ok += 1;
            }
        }
        let n = examples.len() as f32;
        let mean = (total_loss / examples.len() as f64) as f32;
        let ppl = (mean as f64).exp() as f32;
        Ok((mean, ppl, top1_ok as f32 / n, top5_ok as f32 / n))
    }

    fn fmt_f32_list(xs: &[f32]) -> String {
        let mut s = String::from("[");
        for (i, v) in xs.iter().enumerate() {
            if i > 0 {
                s.push(',');
            }
            s.push_str(&format!("{:.6}", v));
        }
        s.push(']');
        s
    }

    fn parameter_l2_by_prefix_json<B: Backend, M: NamedParameters<B>>(backend: &B, model: &M) -> String {
        use std::collections::HashMap;
        let ops = backend.ops();
        let mut sumsq: HashMap<String, f64> = HashMap::new();
        model.visit_parameters(&mut |name, p| {
            let prefix = name.split('.').take(2).collect::<Vec<_>>().join(".");
            if let Ok(v) = ops.tensor_to_vec(p.tensor()) {
                let s: f64 = v.iter().map(|x| (*x as f64).powi(2)).sum();
                *sumsq.entry(prefix).or_insert(0.0) += s;
            }
        });
        let mut parts: Vec<String> = sumsq
            .into_iter()
            .map(|(k, s)| {
                let esc = k.replace('\\', "\\\\").replace('"', "\\\"");
                format!("\"{esc}\": {:.6}", s.sqrt())
            })
            .collect();
        parts.sort();
        format!("{{{}}}", parts.join(", "))
    }

    fn fnv1a_hex(data: &[u8]) -> String {
        let mut h: u64 = 0xcbf29ce484222325;
        for &b in data {
            h ^= b as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
        format!("{:016x}", h)
    }

    fn parse_arg<T: std::str::FromStr>(args: &[String], name: &str, default: T) -> T {
        for w in args.windows(2) {
            if w[0] == name {
                if let Ok(v) = w[1].parse::<T>() {
                    return v;
                }
            }
        }
        default
    }

    fn parse_flag(args: &[String], name: &str) -> bool {
        args.iter().any(|a| a == name)
    }

    fn detect_git_sha() -> String {
        std::process::Command::new("git")
            .args(["rev-parse", "HEAD"])
            .output()
            .ok()
            .and_then(|o| String::from_utf8(o.stdout).ok())
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| "unknown".into())
    }

    fn count_total_params<B: Backend, M: NamedParameters<B>>(backend: &B, model: &M) -> u64 {
        let mut total: u64 = 0;
        let ops = backend.ops();
        model.visit_parameters(&mut |_, p| {
            let shape = ops.shape(p.tensor());
            total = total.saturating_add(shape.iter().product::<usize>() as u64);
        });
        total
    }

    pub fn run() -> anyhow::Result<()> {
        let args: Vec<String> = std::env::args().collect();
        let paper = parse_flag(&args, "--paper");
        let overfit_tiny = parse_flag(&args, "--overfit-tiny");
        let quick = parse_flag(&args, "--quick") && !paper;
        let seed: u64 = parse_arg(&args, "--seed", DEFAULT_SEED);
        let epochs: usize = parse_arg(
            &args,
            "--epochs",
            if overfit_tiny {
                200
            } else if paper {
                5
            } else if quick {
                1
            } else {
                DEFAULT_EPOCHS
            },
        );
        let batch: usize = parse_arg(
            &args,
            "--batch",
            if overfit_tiny {
                8
            } else if paper {
                32
            } else {
                DEFAULT_BATCH
            },
        );
        let lr: f32 = parse_arg(
            &args,
            "--lr",
            if paper {
                3e-4
            } else if overfit_tiny {
                1e-3
            } else {
                DEFAULT_LR
            },
        );
        let eval_windows_cap: usize =
            parse_arg(&args, "--eval-windows", if overfit_tiny { 2_048 } else { DEFAULT_EVAL_WINDOWS });
        let train_windows_cap: usize = parse_arg(&args, "--train-windows", DEFAULT_TRAIN_WINDOWS);
        let train_token_cap: usize = parse_arg(
            &args,
            "--train-tokens",
            if overfit_tiny {
                64
            } else if paper {
                200_000
            } else if quick {
                DEFAULT_TRAIN_TOKENS_QUICK
            } else {
                DEFAULT_TRAIN_TOKENS
            },
        );
        let block_size: usize = parse_arg(
            &args,
            "--block-size",
            if overfit_tiny {
                8
            } else if paper {
                64
            } else {
                DEFAULT_BLOCK_SIZE
            },
        );
        let d_model: usize = parse_arg(&args, "--d-model", if paper { 256 } else { DEFAULT_D_MODEL });
        let num_heads: usize = parse_arg(&args, "--num-heads", if paper { 4 } else { DEFAULT_NUM_HEADS });
        let ffn_dim: usize = parse_arg(&args, "--ffn-dim", if paper { 1024 } else { DEFAULT_FFN_DIM });
        let num_layers: usize = parse_arg(&args, "--num-layers", if paper { 4 } else { DEFAULT_NUM_LAYERS });
        let out_dir = PathBuf::from(parse_arg::<String>(&args, "--out-dir", "out/wikitext2".into()));
        fs::create_dir_all(&out_dir)?;

        if block_size == 0 || d_model == 0 || num_heads == 0 || ffn_dim == 0 || num_layers == 0 {
            anyhow::bail!("--block-size, --d-model, --num-heads, --ffn-dim, --num-layers must be > 0");
        }
        if d_model % num_heads != 0 {
            anyhow::bail!("--d-model ({d_model}) must be divisible by --num-heads ({num_heads})");
        }

        println!("WikiText-2 transformer LM (rustral)");
        println!("===================================");
        println!("seed         : {seed}");
        println!("block_size   : {block_size}");
        println!("d_model      : {d_model}");
        println!("num_heads    : {num_heads}");
        println!("ffn_dim      : {ffn_dim}");
        println!("num_layers   : {num_layers}");
        println!("max_vocab    : {MAX_VOCAB}");
        println!("epochs       : {epochs}");
        println!("batch_size   : {batch}");
        println!("lr           : {lr}");
        println!("paper        : {paper}");
        println!("overfit_tiny : {overfit_tiny}");
        println!("train_tokens : {train_token_cap}");
        println!(
            "train_windows: {}",
            if train_windows_cap == 0 { "all".into() } else { train_windows_cap.to_string() }
        );
        println!(
            "eval_windows : {}",
            if eval_windows_cap == 0 { "all".into() } else { eval_windows_cap.to_string() }
        );
        println!("offline      : {}", std::env::var("RUSTRAL_DATASET_OFFLINE").is_ok());
        println!("out_dir      : {}", out_dir.display());
        println!();

        let load_t0 = Instant::now();
        let splits = load_wikitext2()?;
        println!(
            "loaded train={} bytes valid={} bytes test={} bytes in {:?}",
            splits.train.len(),
            splits.valid.len(),
            splits.test.len(),
            load_t0.elapsed()
        );

        fn parse_arg_opt(args: &[String], name: &str) -> Option<String> {
            for w in args.windows(2) {
                if w[0] == name {
                    return Some(w[1].clone());
                }
            }
            None
        }

        let tok = if let Some(vp) = parse_arg_opt(&args, "--vocab") {
            WordLevelTokenizer::from_vocab_file(
                WordLevelConfig { lowercase: true, max_vocab: Some(MAX_VOCAB), min_freq: 1 },
                std::path::Path::new(&vp),
            )?
        } else {
            let fit_on = if overfit_tiny {
                // Fit only on the tiny prefix we train on so vocab (and LM head) stay small.
                splits.train.split_whitespace().take(train_token_cap.max(1)).collect::<Vec<_>>().join(" ")
            } else {
                splits.train.clone()
            };
            WordLevelTokenizer::fit_from_iter(
                WordLevelConfig { lowercase: true, max_vocab: Some(MAX_VOCAB), min_freq: 1 },
                std::iter::once(fit_on.as_str()),
            )
        };
        println!("vocab_size   : {} (capped at {})", tok.vocab_size(), MAX_VOCAB);

        let mut train_ids = tok.encode(&splits.train);
        if train_token_cap > 0 && train_ids.len() > train_token_cap {
            train_ids.truncate(train_token_cap);
        }
        let valid_ids = tok.encode(&splits.valid);

        let mut train = build_windows(&train_ids, block_size);
        let valid = build_windows(&valid_ids, block_size);
        println!("train_windows: {}  valid_windows: {}", train.len(), valid.len());
        let train_windows_used: usize =
            if train_windows_cap == 0 { train.len() } else { train.len().min(train_windows_cap) };
        if train_windows_used != train.len() {
            println!("train_windows_used: {} (cap {})", train_windows_used, train_windows_cap);
            train.truncate(train_windows_used);
        }
        let eval_windows_used: usize =
            if eval_windows_cap == 0 { valid.len() } else { valid.len().min(eval_windows_cap) };
        if eval_windows_used != valid.len() {
            println!("eval_windows_used: {} (cap {})", eval_windows_used, eval_windows_cap);
        }

        let dataset_hash = fnv1a_hex(splits.train.as_bytes());

        let backend = make_backend()?;
        let mut model = WikiTextLm::<DefaultBackend>::new(
            &backend,
            tok.vocab_size(),
            seed,
            WikiLmDims { block: block_size, d_model, num_heads, ffn_dim, num_layers },
        )?;
        let total_params = count_total_params::<DefaultBackend, _>(&backend, &model);
        println!("total parameters: {}", total_params);

        let cfg =
            TapeTrainerConfig { epochs, batch_size: batch, shuffle: !overfit_tiny, seed, learning_rate: lr };
        let mut trainer = TapeTrainer::<DefaultBackend, _>::new(cfg, Adam::new(lr));

        // Optional tensor pooling (high visibility) — opt-in via env var.
        // - RUSTRAL_POOL=1 enables pooling
        // - RUSTRAL_POOL_STRATEGY=training_arena|standard (default: training_arena)
        if std::env::var("RUSTRAL_POOL").as_deref() == Ok("1") {
            let strategy = match std::env::var("RUSTRAL_POOL_STRATEGY").as_deref() {
                Ok("standard") => PoolStrategy::Standard,
                _ => PoolStrategy::TrainingArena,
            };
            let pool = TensorPool::with_strategy(strategy);
            trainer = trainer.with_tensor_pool(pool);
        }

        let train_t0 = Instant::now();
        let report: TrainingReport = trainer.fit_classification(&backend, &mut model, &train)?;
        let train_elapsed = train_t0.elapsed();
        let throughput = (train.len() * epochs) as f32 / train_elapsed.as_secs_f32().max(1e-9);
        for e in &report.epochs {
            println!("epoch {:>3}: train_loss={:.4} elapsed={:?}", e.epoch, e.mean_loss, e.elapsed);
        }

        let (val_loss, val_ppl, val_top1, val_top5) =
            evaluate(&backend, &model, &valid[..eval_windows_used])?;
        let train_metrics_windows = train.len().min(2048);
        let (train_loss, train_ppl, train_top1, train_top5) =
            evaluate(&backend, &model, &train[..train_metrics_windows])?;
        println!("dev: loss={:.4} ppl={:.2} top1={:.3} top5={:.3}", val_loss, val_ppl, val_top1, val_top5);
        println!("training throughput: {:.1} windows/sec", throughput);

        let vocab_path = out_dir.join("vocab.txt");
        fs::write(&vocab_path, tok.vocab.tokens.join("\n"))?;
        println!("wrote {}", vocab_path.display());

        let epoch_losses: Vec<f32> = report.epochs.iter().map(|e| e.mean_loss).collect();
        let param_l2 = parameter_l2_by_prefix_json(&backend, &model);
        let diagnostics_json = format!(
            "{{\n\
  \"epoch_mean_losses\": {},\n\
  \"epoch_gradient_l2_norms\": [],\n\
  \"train_eval_subset_windows\": {},\n\
  \"train_loss_nats\": {:.6},\n\
  \"train_perplexity\": {:.6},\n\
  \"train_top1_acc\": {:.6},\n\
  \"train_top5_acc\": {:.6},\n\
  \"dev_top1_acc\": {:.6},\n\
  \"dev_top5_acc\": {:.6},\n\
  \"parameter_l2_by_prefix\": {}\n\
}}",
            fmt_f32_list(&epoch_losses),
            train_metrics_windows,
            train_loss,
            train_ppl,
            train_top1,
            train_top5,
            val_top1,
            val_top5,
            param_l2,
        );

        let manifest = serde_json_minimal::Object::new()
            .insert_str("task", "wikitext2_word_lm")
            .insert_str("model_type", "transformer_lm")
            .insert_str("git_sha", &detect_git_sha())
            .insert_u64("seed", seed)
            .insert_u64("block_size", block_size as u64)
            .insert_u64("d_model", d_model as u64)
            .insert_u64("num_heads", num_heads as u64)
            .insert_u64("ffn_dim", ffn_dim as u64)
            .insert_u64("num_layers", num_layers as u64)
            .insert_u64("total_params", total_params)
            .insert_u64("max_vocab", MAX_VOCAB as u64)
            .insert_u64("vocab_size", tok.vocab_size() as u64)
            .insert_u64("epochs", epochs as u64)
            .insert_u64("batch_size", batch as u64)
            .insert_f32("learning_rate", lr)
            .insert_u64("train_tokens_used", train_ids.len() as u64)
            .insert_u64("train_windows", train.len() as u64)
            .insert_u64("train_windows_used", train_windows_used as u64)
            .insert_u64("valid_windows", valid.len() as u64)
            .insert_u64("eval_windows_used", eval_windows_used as u64)
            .insert_str("dataset_hash_fnv1a", &dataset_hash)
            .insert_f32("dev_loss_nats", val_loss)
            .insert_f32("dev_perplexity", val_ppl)
            .insert_f32("windows_per_sec", throughput)
            .insert_f32("train_elapsed_sec", train_elapsed.as_secs_f32())
            .insert_str("dataset", "WikiText-2 raw v1 (smerity.com mirror)")
            .insert_str("tokenizer", "rustral-data WordLevelTokenizer (whitespace, lowercased)")
            .insert_bool("quick_mode", quick)
            .insert_bool("paper_mode", paper)
            .insert_bool("overfit_tiny", overfit_tiny)
            .insert_str(
                "gradient_clip_note",
                if paper { "clip_1.0_not_wired_in_tape_trainer_yet" } else { "none" },
            )
            .insert_raw("diagnostics", &diagnostics_json);
        let manifest_path = out_dir.join("manifest.json");
        fs::write(&manifest_path, manifest.to_pretty_json())?;
        println!("wrote {}", manifest_path.display());

        Ok(())
    }

    /// Tiny dependency-free JSON object writer; mirror of the SST-2 helper. Duplicated on
    /// purpose so each example is self-contained and editable in isolation.
    mod serde_json_minimal {
        pub struct Object {
            entries: Vec<(String, String)>,
        }

        impl Object {
            pub fn new() -> Self {
                Self { entries: Vec::new() }
            }

            fn escape(s: &str) -> String {
                let mut o = String::with_capacity(s.len() + 2);
                for c in s.chars() {
                    match c {
                        '"' => o.push_str("\\\""),
                        '\\' => o.push_str("\\\\"),
                        '\n' => o.push_str("\\n"),
                        '\r' => o.push_str("\\r"),
                        '\t' => o.push_str("\\t"),
                        c if (c as u32) < 0x20 => o.push_str(&format!("\\u{:04x}", c as u32)),
                        c => o.push(c),
                    }
                }
                o
            }

            pub fn insert_str(mut self, k: &str, v: &str) -> Self {
                self.entries.push((k.into(), format!("\"{}\"", Self::escape(v))));
                self
            }
            pub fn insert_u64(mut self, k: &str, v: u64) -> Self {
                self.entries.push((k.into(), v.to_string()));
                self
            }
            pub fn insert_f32(mut self, k: &str, v: f32) -> Self {
                self.entries.push((k.into(), format!("{:.6}", v)));
                self
            }
            pub fn insert_bool(mut self, k: &str, v: bool) -> Self {
                self.entries.push((k.into(), v.to_string()));
                self
            }

            pub fn insert_raw(mut self, k: &str, raw_json: &str) -> Self {
                self.entries.push((k.into(), raw_json.to_string()));
                self
            }

            pub fn to_pretty_json(&self) -> String {
                let mut s = String::from("{\n");
                for (i, (k, v)) in self.entries.iter().enumerate() {
                    s.push_str(&format!("  \"{}\": {}", Self::escape(k), v));
                    if i + 1 < self.entries.len() {
                        s.push(',');
                    }
                    s.push('\n');
                }
                s.push_str("}\n");
                s
            }
        }
    }
}
