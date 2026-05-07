//! SST-2 sentiment classifier (Phase 2 P1 H2).
//!
//! Reproducible end-to-end run on the Stanford Sentiment Treebank-2 binary task with a
//! small Rustral-native transformer encoder trained through the autodiff `Tape`.
//!
//! Architecture: `Embedding + learned positional Embedding -> N x TapeTransformerEncoderLayer
//! -> mean-pool over sequence -> Linear(d_model -> 2)`. This is a real transformer baseline
//! that exercises multi-head self-attention, layer norm, and feed-forward through tape
//! backward, replacing the previous "bag of positions" linear classifier.
//!
//! Output artifacts (written to `--out-dir`, default `./out/sst2`):
//!
//! - `manifest.json` — full run metadata: git SHA, seed, hyperparameters, dataset hash
//!   and size, vocab size, dev accuracy, dev loss, model parameter count, throughput.
//! - `vocab.txt` — one token per line; line index = token id.
//!
//! Usage (typical, online):
//!
//! ```bash
//! cargo run --release -p rustral-runtime --features training --example sst2_classifier
//! ```
//!
//! Offline / CI-friendly (datasets pre-staged in `~/.cache/rustral/datasets/sst2/`):
//!
//! ```bash
//! RUSTRAL_DATASET_OFFLINE=1 \
//! cargo run --release -p rustral-runtime --features training --example sst2_classifier
//! ```
//!
//! Quick smoke (used by the smoke test): `--quick` caps the training set to 256 examples
//! and trains one epoch so the whole run finishes in seconds.

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
    use rustral_core::{Backend, ForwardCtx, Mode, NamedParameters, Parameter, Result};
    use rustral_data::datasets::sst2::{load_sst2, Sst2Example};
    use rustral_data::tokenizer::{WordLevelConfig, WordLevelTokenizer};
    use rustral_ndarray_backend::CpuBackend;
    use rustral_nn::tape::TapeModule;
    use rustral_nn::tape_transformer::{TapeTransformerEncoderConfig, TapeTransformerEncoderLayer};
    use rustral_nn::{Embedding, EmbeddingConfig, Linear, LinearBuilder};
    use rustral_optim::Adam;
    use rustral_runtime::{SupervisedTapeModel, TapeTrainer, TapeTrainerConfig, TrainingReport};

    const DEFAULT_SEED: u64 = 0xC0FFEE;
    const SEQ_LEN: usize = 32;
    const D_MODEL: usize = 64;
    const NUM_HEADS: usize = 4;
    const FFN_DIM: usize = 128;
    const NUM_LAYERS: usize = 2;
    const NUM_CLASSES: usize = 2;
    const MAX_VOCAB: usize = 8_192;
    const DEFAULT_EPOCHS: usize = 3;
    const DEFAULT_BATCH: usize = 32;
    const DEFAULT_LR: f32 = 5e-4;

    /// Small Rustral-native transformer for SST-2.
    pub struct TransformerSst2<B: Backend> {
        pub tok_embed: Embedding<B>,
        pub pos_embed: Embedding<B>,
        pub layers: Vec<TapeTransformerEncoderLayer<B>>,
        pub head: Linear<B>,
        pub seq_len: usize,
    }

    impl<B: Backend> NamedParameters<B> for TransformerSst2<B> {
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

    impl<B: Backend> TransformerSst2<B>
    where
        B::Tensor: Clone,
    {
        pub fn new(backend: &B, vocab_size: usize, seed: u64) -> Result<Self> {
            let tok_embed = Embedding::new(
                backend,
                EmbeddingConfig::new(vocab_size, D_MODEL),
                seed.wrapping_add(1),
            )?;
            let pos_embed = Embedding::new(
                backend,
                EmbeddingConfig::new(SEQ_LEN, D_MODEL),
                seed.wrapping_add(2),
            )?;
            let mut layers = Vec::with_capacity(NUM_LAYERS);
            for i in 0..NUM_LAYERS {
                let cfg = TapeTransformerEncoderConfig::new(D_MODEL, NUM_HEADS, FFN_DIM);
                let layer =
                    TapeTransformerEncoderLayer::new(backend, cfg, seed.wrapping_add(100 + i as u64))?;
                layers.push(layer);
            }
            let head = LinearBuilder::new(D_MODEL, NUM_CLASSES)
                .with_bias(true)
                .seed(seed.wrapping_add(999))
                .build(backend)?;
            Ok(Self { tok_embed, pos_embed, layers, head, seq_len: SEQ_LEN })
        }

        /// Single-sample inference: returns flat `[NUM_CLASSES]` logits.
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
            assert_eq!(ids.len(), self.seq_len, "seq len mismatch in forward");
            let ops = backend.ops();
            // Token embeddings: [seq_len, d_model].
            let tok_ids_f32: Vec<f32> = ids.iter().map(|&i| i as f32).collect();
            let tok_ids_t = ops.tensor_from_vec(tok_ids_f32, &[self.seq_len])?;
            let tok_ids_id = tape.watch(tok_ids_t);
            let tok_emb = self.tok_embed.forward_tape(tok_ids_id, tape, ctx)?;

            // Positional embeddings: just gather rows 0..seq_len.
            let pos_ids_f32: Vec<f32> = (0..self.seq_len).map(|i| i as f32).collect();
            let pos_ids_t = ops.tensor_from_vec(pos_ids_f32, &[self.seq_len])?;
            let pos_ids_id = tape.watch(pos_ids_t);
            let pos_emb = self.pos_embed.forward_tape(pos_ids_id, tape, ctx)?;

            // x = tok + pos.
            let mut x = tape.add(tok_emb, pos_emb, ctx)?;

            // Transformer stack.
            for layer in &self.layers {
                x = layer.forward_tape(x, tape, ctx)?;
            }

            // Mean pool over sequence dim: [1, seq_len] @ [seq_len, d_model] -> [1, d_model],
            // then divide by seq_len.
            let ones_row = ops.tensor_from_vec(vec![1.0_f32; self.seq_len], &[1, self.seq_len])?;
            let ones_id = tape.watch(ones_row);
            let pooled_sum = tape.matmul(ones_id, x, ctx)?;
            let pooled = tape.mul_scalar(pooled_sum, 1.0 / self.seq_len as f32, ctx)?;

            // Final classification head: [1, d_model] -> [1, NUM_CLASSES].
            self.head.forward_tape(pooled, tape, ctx)
        }
    }

    impl<B: Backend> SupervisedTapeModel<B, Vec<usize>, u8> for TransformerSst2<B>
    where
        B::Tensor: Clone,
    {
        fn forward_tape(
            &mut self,
            input: Vec<usize>,
            tape: &mut Tape<B>,
            ctx: &mut ForwardCtx<B>,
        ) -> Result<TensorId> {
            // We need access to backend through ctx.
            let backend_ptr = ctx.backend() as *const B;
            // SAFETY: we don't keep this pointer past the function and the borrow rules in
            // `forward_tape_internal` only require an immutable backend reference (the same
            // one ctx holds). Rust's strict borrow checker doesn't know that &mut ctx and a
            // shared &B can coexist, but they do at runtime.
            let backend = unsafe { &*backend_ptr };
            self.forward_tape_internal(backend, &input, tape, ctx)
        }

        fn loss_tape(
            &mut self,
            logits: TensorId,
            target: u8,
            tape: &mut Tape<B>,
            ctx: &mut ForwardCtx<B>,
        ) -> Result<TensorId> {
            let t = ctx.backend().ops().tensor_from_vec(vec![target as f32], &[1])?;
            let t = tape.watch(t);
            tape.cross_entropy_loss(logits, t, ctx)
        }
    }

    /// Encode a sentence into `[seq_len]` ids, padding with `<pad>` and truncating to len.
    fn encode_padded(tok: &WordLevelTokenizer, sentence: &str, seq_len: usize) -> Vec<usize> {
        let mut ids = tok.encode(sentence);
        if ids.len() > seq_len {
            ids.truncate(seq_len);
        } else {
            ids.resize(seq_len, tok.vocab.pad_id);
        }
        ids
    }

    /// Compute dev accuracy + mean cross-entropy loss.
    fn evaluate<B: Backend>(
        backend: &B,
        model: &TransformerSst2<B>,
        examples: &[(Vec<usize>, u8)],
    ) -> Result<(f32, f32)>
    where
        B::Tensor: Clone,
    {
        if examples.is_empty() {
            return Ok((0.0, 0.0));
        }
        let mut total_loss = 0.0f64;
        let mut correct = 0usize;
        for (ids, label) in examples {
            let logits = model.logits(backend, ids)?;
            let mut argmax = 0usize;
            let mut max_v = f32::NEG_INFINITY;
            for (i, v) in logits.iter().enumerate() {
                if *v > max_v {
                    max_v = *v;
                    argmax = i;
                }
            }
            let mut sum_exp = 0.0f64;
            for v in &logits {
                sum_exp += ((*v - max_v) as f64).exp();
            }
            let log_denom = (max_v as f64) + sum_exp.ln();
            total_loss += log_denom - logits[*label as usize] as f64;
            if argmax == *label as usize {
                correct += 1;
            }
        }
        Ok((
            (total_loss / examples.len() as f64) as f32,
            correct as f32 / examples.len() as f32,
        ))
    }

    /// Hash the concatenation of the train + dev sentences with a stable Fnv1a so the
    /// manifest can record a "dataset_hash" column without an extra dep.
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
        let quick = parse_flag(&args, "--quick");
        let seed: u64 = parse_arg(&args, "--seed", DEFAULT_SEED);
        let epochs: usize = parse_arg(&args, "--epochs", if quick { 1 } else { DEFAULT_EPOCHS });
        let batch: usize = parse_arg(&args, "--batch", DEFAULT_BATCH);
        let lr: f32 = parse_arg(&args, "--lr", DEFAULT_LR);
        let out_dir =
            PathBuf::from(parse_arg::<String>(&args, "--out-dir", "out/sst2".into()));
        fs::create_dir_all(&out_dir)?;

        println!("SST-2 transformer classifier (rustral)");
        println!("======================================");
        println!("seed       : {seed}");
        println!("seq_len    : {SEQ_LEN}");
        println!("d_model    : {D_MODEL}");
        println!("num_heads  : {NUM_HEADS}");
        println!("ffn_dim    : {FFN_DIM}");
        println!("num_layers : {NUM_LAYERS}");
        println!("max_vocab  : {MAX_VOCAB}");
        println!("epochs     : {epochs}");
        println!("batch_size : {batch}");
        println!("lr         : {lr}");
        println!("offline    : {}", std::env::var("RUSTRAL_DATASET_OFFLINE").is_ok());
        println!("out_dir    : {}", out_dir.display());
        println!();

        let load_t0 = Instant::now();
        let (mut train_raw, dev_raw) = load_sst2()?;
        let load_elapsed = load_t0.elapsed();
        println!(
            "loaded {} train / {} dev examples in {:?}",
            train_raw.len(),
            dev_raw.len(),
            load_elapsed
        );

        if quick {
            train_raw.truncate(256);
        }

        let tok = WordLevelTokenizer::fit_from_iter(
            WordLevelConfig { lowercase: true, max_vocab: Some(MAX_VOCAB), min_freq: 1 },
            train_raw.iter().map(|e| e.sentence.as_str()),
        );
        println!("vocab_size : {} (capped at {})", tok.vocab_size(), MAX_VOCAB);

        let train: Vec<(Vec<usize>, u8)> = train_raw
            .iter()
            .map(|e| (encode_padded(&tok, &e.sentence, SEQ_LEN), e.label))
            .collect();
        let dev: Vec<(Vec<usize>, u8)> = dev_raw
            .iter()
            .map(|e| (encode_padded(&tok, &e.sentence, SEQ_LEN), e.label))
            .collect();

        let dataset_hash = {
            let mut buf = Vec::new();
            for Sst2Example { sentence, label } in train_raw.iter().chain(dev_raw.iter()) {
                buf.extend(sentence.as_bytes());
                buf.push(*label);
            }
            fnv1a_hex(&buf)
        };

        let backend = CpuBackend::default();
        let mut model = TransformerSst2::<CpuBackend>::new(&backend, tok.vocab_size(), seed)?;
        let total_params = count_total_params::<CpuBackend, _>(&backend, &model);
        println!("total parameters: {}", total_params);

        let cfg = TapeTrainerConfig {
            epochs,
            batch_size: batch,
            shuffle: true,
            seed,
            learning_rate: lr,
        };
        let optimizer = Adam::new(lr);
        let mut trainer = TapeTrainer::<CpuBackend, _>::new(cfg, optimizer);

        let train_t0 = Instant::now();
        let report: TrainingReport = trainer.fit_classification(&backend, &mut model, &train)?;
        let train_elapsed = train_t0.elapsed();
        let throughput =
            (train.len() * epochs) as f32 / train_elapsed.as_secs_f32().max(1e-9);
        for e in &report.epochs {
            println!(
                "epoch {:>3}: train_loss={:.4} elapsed={:?}",
                e.epoch, e.mean_loss, e.elapsed
            );
        }
        if let Some(acc) = report.accuracy.as_ref().and_then(|v| v.last()) {
            println!("final train acc: {:.3}", acc);
        }

        let (dev_loss, dev_acc) = evaluate(&backend, &model, &dev)?;
        println!("dev: loss={:.4} acc={:.3}", dev_loss, dev_acc);
        println!("training throughput: {:.1} samples/sec", throughput);

        let vocab_path = out_dir.join("vocab.txt");
        fs::write(&vocab_path, tok.vocab.tokens.join("\n"))?;
        println!("wrote {}", vocab_path.display());

        let manifest = serde_json_minimal::Object::new()
            .insert_str("task", "sst2_classifier")
            .insert_str("model_type", "transformer_encoder")
            .insert_str("git_sha", &detect_git_sha())
            .insert_u64("seed", seed)
            .insert_u64("seq_len", SEQ_LEN as u64)
            .insert_u64("d_model", D_MODEL as u64)
            .insert_u64("num_heads", NUM_HEADS as u64)
            .insert_u64("ffn_dim", FFN_DIM as u64)
            .insert_u64("num_layers", NUM_LAYERS as u64)
            .insert_u64("total_params", total_params)
            .insert_u64("max_vocab", MAX_VOCAB as u64)
            .insert_u64("vocab_size", tok.vocab_size() as u64)
            .insert_u64("epochs", epochs as u64)
            .insert_u64("batch_size", batch as u64)
            .insert_f32("learning_rate", lr)
            .insert_u64("train_examples", train.len() as u64)
            .insert_u64("dev_examples", dev.len() as u64)
            .insert_str("dataset_hash_fnv1a", &dataset_hash)
            .insert_f32("dev_loss", dev_loss)
            .insert_f32("dev_accuracy", dev_acc)
            .insert_f32("samples_per_sec", throughput)
            .insert_f32("train_elapsed_sec", train_elapsed.as_secs_f32())
            .insert_str("dataset", "SST-2 (binary, HuggingFace SetFit/sst2 mirror)")
            .insert_str("tokenizer", "rustral-data WordLevelTokenizer (whitespace, lowercased)")
            .insert_bool("quick_mode", quick);
        let manifest_path = out_dir.join("manifest.json");
        fs::write(&manifest_path, manifest.to_pretty_json())?;
        println!("wrote {}", manifest_path.display());

        Ok(())
    }

    /// Tiny dependency-free JSON object writer so the manifest does not grow a `serde_json`
    /// requirement on every example. Output is ASCII, indented, key-order preserved.
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
