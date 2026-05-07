//! SST-2 sentiment classifier (Phase 2 P1 H2).
//!
//! Reproducible end-to-end run on the Stanford Sentiment Treebank-2 binary task. Designed
//! as a small honest baseline: word-level whitespace tokenization, mean-of-positions
//! embedding pooling via a flat Linear head, Adam optimizer, fixed seed, deterministic
//! train/val split. The goal is a *credible* number with full provenance, not SOTA.
//!
//! Output artifacts (written to `--out-dir`, default `./out/sst2`):
//!
//! - `manifest.json`  — full run metadata (git SHA, seed, hyperparameters, dataset
//!   hash + size, vocab size, dev accuracy, dev loss, throughput).
//! - `vocab.txt`      — one token per line; line index = token id (used to debug
//!   decoding mismatches between training and downstream tools).
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
    use rustral_core::{Backend, ForwardCtx, Mode, NamedParameters, Result};
    use rustral_data::datasets::sst2::{load_sst2, Sst2Example};
    use rustral_data::tokenizer::{WordLevelConfig, WordLevelTokenizer};
    use rustral_ndarray_backend::CpuBackend;
    use rustral_nn::tape::TapeModule;
    use rustral_nn::{Embedding, EmbeddingConfig, Linear, LinearBuilder};
    use rustral_optim::Adam;
    use rustral_runtime::{SupervisedTapeModel, TapeTrainer, TapeTrainerConfig, TrainingReport};

    const DEFAULT_SEED: u64 = 0xC0FFEE;
    const SEQ_LEN: usize = 32;
    const EMBED_DIM: usize = 32;
    const NUM_CLASSES: usize = 2;
    const MAX_VOCAB: usize = 8_192;
    const DEFAULT_EPOCHS: usize = 3;
    const DEFAULT_BATCH: usize = 32;
    const DEFAULT_LR: f32 = 3e-3;

    /// Bag-of-positions classifier: `Embedding -> reshape([1, seq*emb]) -> Linear -> 2`.
    pub struct BagOfPositions<B: Backend> {
        pub embed: Embedding<B>,
        pub head: Linear<B>,
        pub seq_len: usize,
        pub embed_dim: usize,
    }

    impl<B: Backend> NamedParameters<B> for BagOfPositions<B> {
        fn visit_parameters(&self, f: &mut dyn FnMut(&str, &rustral_core::Parameter<B>)) {
            self.embed.visit_parameters(&mut |n, p| f(&format!("embed.{n}"), p));
            self.head.visit_parameters(&mut |n, p| f(&format!("head.{n}"), p));
        }

        fn visit_parameters_mut(
            &mut self,
            f: &mut dyn FnMut(&str, &mut rustral_core::Parameter<B>),
        ) {
            self.embed.visit_parameters_mut(&mut |n, p| f(&format!("embed.{n}"), p));
            self.head.visit_parameters_mut(&mut |n, p| f(&format!("head.{n}"), p));
        }
    }

    impl<B: Backend> SupervisedTapeModel<B, Vec<usize>, u8> for BagOfPositions<B>
    where
        B::Tensor: Clone,
    {
        fn forward_tape(
            &mut self,
            input: Vec<usize>,
            tape: &mut Tape<B>,
            ctx: &mut ForwardCtx<B>,
        ) -> Result<TensorId> {
            assert_eq!(input.len(), self.seq_len, "seq len mismatch in forward_tape");
            let ids_f32: Vec<f32> = input.iter().map(|&i| i as f32).collect();
            let ids_t = ctx.backend().ops().tensor_from_vec(ids_f32, &[self.seq_len])?;
            let ids_id = tape.watch(ids_t);
            let emb = self.embed.forward_tape(ids_id, tape, ctx)?;
            let flat = tape.reshape_tape(emb, &[1, self.seq_len * self.embed_dim], ctx)?;
            self.head.forward_tape(flat, tape, ctx)
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

    impl<B: Backend> BagOfPositions<B>
    where
        B::Tensor: Clone,
    {
        pub fn new(backend: &B, vocab_size: usize, seq_len: usize, embed_dim: usize, seed: u64) -> Result<Self> {
            let embed = Embedding::new(
                backend,
                EmbeddingConfig::new(vocab_size, embed_dim),
                seed,
            )?;
            let head = LinearBuilder::new(seq_len * embed_dim, NUM_CLASSES)
                .with_bias(true)
                .seed(seed.wrapping_add(101))
                .build(backend)?;
            Ok(Self { embed, head, seq_len, embed_dim })
        }

        /// Single-sample inference logits.
        pub fn logits(&self, backend: &B, ids: &[usize]) -> Result<Vec<f32>> {
            use rustral_core::Module;
            let mut ctx = ForwardCtx::new(backend, Mode::Inference);
            let emb = self.embed.forward(ids.to_vec(), &mut ctx)?;
            let flat = backend.ops().reshape(&emb, &[1, self.seq_len * self.embed_dim])?;
            let logits = self.head.forward(flat, &mut ctx)?;
            backend.ops().tensor_to_vec(&logits)
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
        model: &BagOfPositions<B>,
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

        println!("SST-2 classifier (rustral)");
        println!("==========================");
        println!("seed       : {seed}");
        println!("seq_len    : {SEQ_LEN}");
        println!("embed_dim  : {EMBED_DIM}");
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

        // Tokeniser fit on training set only.
        let tok = WordLevelTokenizer::fit_from_iter(
            WordLevelConfig { lowercase: true, max_vocab: Some(MAX_VOCAB), min_freq: 1 },
            train_raw.iter().map(|e| e.sentence.as_str()),
        );
        println!("vocab_size : {} (capped at {})", tok.vocab_size(), MAX_VOCAB);

        // Encode + pad.
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

        // Train.
        let backend = CpuBackend::default();
        let mut model =
            BagOfPositions::<CpuBackend>::new(&backend, tok.vocab_size(), SEQ_LEN, EMBED_DIM, seed)?;

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

        // Persist vocab + manifest.
        let vocab_path = out_dir.join("vocab.txt");
        fs::write(&vocab_path, tok.vocab.tokens.join("\n"))?;
        println!("wrote {}", vocab_path.display());

        let manifest = serde_json_minimal::Object::new()
            .insert_str("task", "sst2_classifier")
            .insert_str("git_sha", &detect_git_sha())
            .insert_u64("seed", seed)
            .insert_u64("seq_len", SEQ_LEN as u64)
            .insert_u64("embed_dim", EMBED_DIM as u64)
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
            .insert_str("dataset", "SST-2 (binary, GLUE mirror)")
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
