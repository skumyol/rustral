//! WikiText-2 small word-level language model (Phase 2 P1 H3).
//!
//! Reproducible end-to-end run on the WikiText-2 raw corpus. Architecture is intentionally
//! tiny so a `--quick` smoke run finishes in seconds: an embedding table + a flat Linear
//! head over a fixed-size context window. Training reports per-epoch loss and final dev
//! perplexity; manifest captures full provenance.
//!
//! Output artifacts (`--out-dir`, default `./out/wikitext2`):
//! - `manifest.json`  — git SHA, seed, hyperparameters, vocab size, dev perplexity,
//!   throughput, dataset stats.
//! - `vocab.txt`      — one token per line.
//!
//! Online run (downloads + extracts the WikiText-2 raw zip via `unzip`):
//!
//! ```bash
//! cargo run --release -p rustral-runtime --features training --example wikitext2_lm
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
    use rustral_core::{Backend, ForwardCtx, Mode, Module, NamedParameters, Result};
    use rustral_data::datasets::wikitext2::load_wikitext2;
    use rustral_data::tokenizer::{WordLevelConfig, WordLevelTokenizer};
    use rustral_ndarray_backend::CpuBackend;
    use rustral_nn::tape::TapeModule;
    use rustral_nn::{Embedding, EmbeddingConfig, Linear, LinearBuilder};
    use rustral_optim::Adam;
    use rustral_runtime::{SupervisedTapeModel, TapeTrainer, TapeTrainerConfig, TrainingReport};

    const DEFAULT_SEED: u64 = 0xC0FFEE;
    const BLOCK_SIZE: usize = 16;
    const EMBED_DIM: usize = 32;
    const MAX_VOCAB: usize = 16_384;
    const DEFAULT_EPOCHS: usize = 1;
    const DEFAULT_BATCH: usize = 32;
    const DEFAULT_LR: f32 = 5e-3;
    const DEFAULT_TRAIN_TOKENS_QUICK: usize = 4_000;
    const DEFAULT_TRAIN_TOKENS: usize = 50_000;

    /// Tiny LM: `Embedding(V, D) -> reshape([1, block*D]) -> Linear(block*D, V)`.
    pub struct WordLm<B: Backend> {
        pub embed: Embedding<B>,
        pub head: Linear<B>,
        pub block: usize,
        pub embed_dim: usize,
    }

    impl<B: Backend> NamedParameters<B> for WordLm<B> {
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

    impl<B: Backend> SupervisedTapeModel<B, Vec<usize>, usize> for WordLm<B>
    where
        B::Tensor: Clone,
    {
        fn forward_tape(
            &mut self,
            input: Vec<usize>,
            tape: &mut Tape<B>,
            ctx: &mut ForwardCtx<B>,
        ) -> Result<TensorId> {
            assert_eq!(input.len(), self.block);
            let ids_f32: Vec<f32> = input.iter().map(|&i| i as f32).collect();
            let ids_t = ctx.backend().ops().tensor_from_vec(ids_f32, &[self.block])?;
            let ids_id = tape.watch(ids_t);
            let emb = self.embed.forward_tape(ids_id, tape, ctx)?;
            let flat = tape.reshape_tape(emb, &[1, self.block * self.embed_dim], ctx)?;
            self.head.forward_tape(flat, tape, ctx)
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

    impl<B: Backend> WordLm<B>
    where
        B::Tensor: Clone,
    {
        pub fn new(backend: &B, vocab: usize, block: usize, embed_dim: usize, seed: u64) -> Result<Self> {
            let embed = Embedding::new(backend, EmbeddingConfig::new(vocab, embed_dim), seed)?;
            let head = LinearBuilder::new(block * embed_dim, vocab)
                .with_bias(true)
                .seed(seed.wrapping_add(101))
                .build(backend)?;
            Ok(Self { embed, head, block, embed_dim })
        }

        pub fn logits(&self, backend: &B, ids: &[usize]) -> Result<Vec<f32>> {
            let mut ctx = ForwardCtx::new(backend, Mode::Inference);
            let emb = self.embed.forward(ids.to_vec(), &mut ctx)?;
            let flat = backend.ops().reshape(&emb, &[1, self.block * self.embed_dim])?;
            let logits = self.head.forward(flat, &mut ctx)?;
            backend.ops().tensor_to_vec(&logits)
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

    /// Compute mean cross-entropy over `examples`, returning (loss_nats, perplexity).
    fn evaluate<B: Backend>(
        backend: &B,
        model: &WordLm<B>,
        examples: &[(Vec<usize>, usize)],
    ) -> Result<(f32, f32)>
    where
        B::Tensor: Clone,
    {
        if examples.is_empty() {
            return Ok((0.0, 1.0));
        }
        let mut total_loss = 0.0f64;
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
        }
        let mean = (total_loss / examples.len() as f64) as f32;
        let ppl = (mean as f64).exp() as f32;
        Ok((mean, ppl))
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

    pub fn run() -> anyhow::Result<()> {
        let args: Vec<String> = std::env::args().collect();
        let quick = parse_flag(&args, "--quick");
        let seed: u64 = parse_arg(&args, "--seed", DEFAULT_SEED);
        let epochs: usize = parse_arg(&args, "--epochs", DEFAULT_EPOCHS);
        let batch: usize = parse_arg(&args, "--batch", DEFAULT_BATCH);
        let lr: f32 = parse_arg(&args, "--lr", DEFAULT_LR);
        let train_token_cap: usize = parse_arg(
            &args,
            "--train-tokens",
            if quick { DEFAULT_TRAIN_TOKENS_QUICK } else { DEFAULT_TRAIN_TOKENS },
        );
        let out_dir =
            PathBuf::from(parse_arg::<String>(&args, "--out-dir", "out/wikitext2".into()));
        fs::create_dir_all(&out_dir)?;

        println!("WikiText-2 small LM (rustral)");
        println!("=============================");
        println!("seed         : {seed}");
        println!("block_size   : {BLOCK_SIZE}");
        println!("embed_dim    : {EMBED_DIM}");
        println!("max_vocab    : {MAX_VOCAB}");
        println!("epochs       : {epochs}");
        println!("batch_size   : {batch}");
        println!("lr           : {lr}");
        println!("train_tokens : {train_token_cap}");
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

        let tok = WordLevelTokenizer::fit_from_iter(
            WordLevelConfig { lowercase: true, max_vocab: Some(MAX_VOCAB), min_freq: 1 },
            std::iter::once(splits.train.as_str()),
        );
        println!("vocab_size   : {} (capped at {})", tok.vocab_size(), MAX_VOCAB);

        let mut train_ids = tok.encode(&splits.train);
        if train_ids.len() > train_token_cap {
            train_ids.truncate(train_token_cap);
        }
        let valid_ids = tok.encode(&splits.valid);

        let train = build_windows(&train_ids, BLOCK_SIZE);
        let valid = build_windows(&valid_ids, BLOCK_SIZE);
        println!("train_windows: {}  valid_windows: {}", train.len(), valid.len());

        let dataset_hash = fnv1a_hex(splits.train.as_bytes());

        let backend = CpuBackend::default();
        let mut model =
            WordLm::<CpuBackend>::new(&backend, tok.vocab_size(), BLOCK_SIZE, EMBED_DIM, seed)?;

        let cfg = TapeTrainerConfig {
            epochs,
            batch_size: batch,
            shuffle: true,
            seed,
            learning_rate: lr,
        };
        let mut trainer = TapeTrainer::<CpuBackend, _>::new(cfg, Adam::new(lr));

        let train_t0 = Instant::now();
        let report: TrainingReport = trainer.fit_classification(&backend, &mut model, &train)?;
        let train_elapsed = train_t0.elapsed();
        let throughput = (train.len() * epochs) as f32 / train_elapsed.as_secs_f32().max(1e-9);
        for e in &report.epochs {
            println!(
                "epoch {:>3}: train_loss={:.4} elapsed={:?}",
                e.epoch, e.mean_loss, e.elapsed
            );
        }

        let (val_loss, val_ppl) = evaluate(&backend, &model, &valid)?;
        println!("dev: loss={:.4} ppl={:.2}", val_loss, val_ppl);
        println!("training throughput: {:.1} windows/sec", throughput);

        let vocab_path = out_dir.join("vocab.txt");
        fs::write(&vocab_path, tok.vocab.tokens.join("\n"))?;
        println!("wrote {}", vocab_path.display());

        let manifest = serde_json_minimal::Object::new()
            .insert_str("task", "wikitext2_word_lm")
            .insert_str("git_sha", &detect_git_sha())
            .insert_u64("seed", seed)
            .insert_u64("block_size", BLOCK_SIZE as u64)
            .insert_u64("embed_dim", EMBED_DIM as u64)
            .insert_u64("max_vocab", MAX_VOCAB as u64)
            .insert_u64("vocab_size", tok.vocab_size() as u64)
            .insert_u64("epochs", epochs as u64)
            .insert_u64("batch_size", batch as u64)
            .insert_f32("learning_rate", lr)
            .insert_u64("train_tokens_used", train_ids.len() as u64)
            .insert_u64("train_windows", train.len() as u64)
            .insert_u64("valid_windows", valid.len() as u64)
            .insert_str("dataset_hash_fnv1a", &dataset_hash)
            .insert_f32("dev_loss_nats", val_loss)
            .insert_f32("dev_perplexity", val_ppl)
            .insert_f32("windows_per_sec", throughput)
            .insert_f32("train_elapsed_sec", train_elapsed.as_secs_f32())
            .insert_str("dataset", "WikiText-2 raw v1")
            .insert_str("tokenizer", "rustral-data WordLevelTokenizer (whitespace, lowercased)")
            .insert_bool("quick_mode", quick);
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
