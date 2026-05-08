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
    use rustral_optim::{Adam, Optimizer, Sgd};
    use rustral_runtime::{SupervisedTapeModel, TapeTrainer, TapeTrainerConfig, TrainingReport};

    const DEFAULT_SEED: u64 = 0xC0FFEE;
    const DEFAULT_SEQ_LEN: usize = 32;
    const DEFAULT_D_MODEL: usize = 64;
    const DEFAULT_NUM_HEADS: usize = 4;
    const DEFAULT_FFN_DIM: usize = 128;
    const DEFAULT_NUM_LAYERS: usize = 2;
    const NUM_CLASSES: usize = 2;

    #[derive(Clone, Copy)]
    pub struct Sst2Dims {
        seq_len: usize,
        d_model: usize,
        num_heads: usize,
        ffn_dim: usize,
    }
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
        pub fn new(backend: &B, vocab_size: usize, num_layers: usize, seed: u64, dims: Sst2Dims) -> Result<Self> {
            let tok_embed = Embedding::new(
                backend,
                EmbeddingConfig::new(vocab_size, dims.d_model),
                seed.wrapping_add(1),
            )?;
            let pos_embed = Embedding::new(
                backend,
                EmbeddingConfig::new(dims.seq_len, dims.d_model),
                seed.wrapping_add(2),
            )?;
            let mut layers = Vec::with_capacity(num_layers);
            for i in 0..num_layers {
                let cfg = TapeTransformerEncoderConfig::new(dims.d_model, dims.num_heads, dims.ffn_dim);
                let layer =
                    TapeTransformerEncoderLayer::new(backend, cfg, seed.wrapping_add(100 + i as u64))?;
                layers.push(layer);
            }
            let head = LinearBuilder::new(dims.d_model, NUM_CLASSES)
                .with_bias(true)
                .seed(seed.wrapping_add(999))
                .build(backend)?;
            Ok(Self { tok_embed, pos_embed, layers, head, seq_len: dims.seq_len })
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

    #[derive(Clone, Copy, Debug)]
    struct EvalDiagnostics {
        loss: f32,
        accuracy: f32,
        /// confusion[true_label][pred_label]
        confusion: [[u64; 2]; 2],
        /// predicted_counts[pred_label]
        predicted_counts: [u64; 2],
        /// Histogram of predicted probability for class 1 (positive).
        /// counts[i] corresponds to bin [i/bins, (i+1)/bins).
        pos_prob_hist_bins: u64,
        pos_prob_hist_counts: [u64; 10],
    }

    fn softmax_2(logits: [f32; 2]) -> [f32; 2] {
        let m = logits[0].max(logits[1]);
        let e0 = (logits[0] - m).exp();
        let e1 = (logits[1] - m).exp();
        let z = (e0 + e1).max(1e-20);
        [e0 / z, e1 / z]
    }

    fn evaluate_with_diagnostics<B: Backend>(
        backend: &B,
        model: &TransformerSst2<B>,
        examples: &[(Vec<usize>, u8)],
    ) -> Result<EvalDiagnostics>
    where
        B::Tensor: Clone,
    {
        if examples.is_empty() {
            return Ok(EvalDiagnostics {
                loss: 0.0,
                accuracy: 0.0,
                confusion: [[0, 0], [0, 0]],
                predicted_counts: [0, 0],
                pos_prob_hist_bins: 10,
                pos_prob_hist_counts: [0; 10],
            });
        }

        let mut total_loss = 0.0f64;
        let mut correct = 0u64;
        let mut confusion = [[0u64; 2]; 2];
        let mut predicted_counts = [0u64; 2];
        let mut pos_prob_hist_counts = [0u64; 10];

        for (ids, label) in examples {
            let logits_vec = model.logits(backend, ids)?;
            let logits = [
                *logits_vec.first().unwrap_or(&0.0),
                *logits_vec.get(1).unwrap_or(&0.0),
            ];

            let pred = if logits[1] > logits[0] { 1usize } else { 0usize };
            let true_label = (*label as usize).min(1);
            predicted_counts[pred] += 1;
            confusion[true_label][pred] += 1;
            if pred == true_label {
                correct += 1;
            }

            // Cross-entropy contribution (stable log-softmax).
            let max_v = logits[0].max(logits[1]);
            let sum_exp = ((logits[0] - max_v) as f64).exp() + ((logits[1] - max_v) as f64).exp();
            let log_denom = (max_v as f64) + sum_exp.ln();
            total_loss += log_denom - logits[true_label] as f64;

            // Histogram of P(class=1).
            let p = softmax_2(logits)[1];
            let mut bin = (p * 10.0).floor() as usize;
            if bin >= 10 {
                bin = 9;
            }
            pos_prob_hist_counts[bin] += 1;
        }

        Ok(EvalDiagnostics {
            loss: (total_loss / examples.len() as f64) as f32,
            accuracy: correct as f32 / examples.len() as f32,
            confusion,
            predicted_counts,
            pos_prob_hist_bins: 10,
            pos_prob_hist_counts,
        })
    }

    fn label_counts(examples: &[Sst2Example]) -> [u64; 2] {
        let mut c = [0u64; 2];
        for e in examples {
            let idx = (e.label as usize).min(1);
            c[idx] += 1;
        }
        c
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

    fn parse_arg_opt(args: &[String], name: &str) -> Option<String> {
        for w in args.windows(2) {
            if w[0] == name {
                return Some(w[1].clone());
            }
        }
        None
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
        let overfit_32 = parse_flag(&args, "--overfit-32");
        let use_sgd = parse_flag(&args, "--sgd");
        let parity_dump_path = parse_arg_opt(&args, "--parity-dump");
        let parity_one_step = parse_flag(&args, "--parity-one-step");
        let parity_batch: usize = parse_arg(&args, "--parity-batch", 8usize);
        let seed: u64 = parse_arg(&args, "--seed", DEFAULT_SEED);
        let num_layers: usize = parse_arg(&args, "--num-layers", DEFAULT_NUM_LAYERS);
        let epochs: usize = parse_arg(
            &args,
            "--epochs",
            if quick {
                1
            } else if overfit_32 {
                50
            } else {
                DEFAULT_EPOCHS
            },
        );
        let batch: usize = parse_arg(&args, "--batch", DEFAULT_BATCH);
        let lr: f32 = parse_arg(&args, "--lr", DEFAULT_LR);
        let seq_len: usize = parse_arg(&args, "--seq-len", DEFAULT_SEQ_LEN);
        let d_model: usize = parse_arg(&args, "--d-model", DEFAULT_D_MODEL);
        let num_heads: usize = parse_arg(&args, "--num-heads", DEFAULT_NUM_HEADS);
        let ffn_dim: usize = parse_arg(&args, "--ffn-dim", DEFAULT_FFN_DIM);
        let out_dir =
            PathBuf::from(parse_arg::<String>(&args, "--out-dir", "out/sst2".into()));
        fs::create_dir_all(&out_dir)?;

        if seq_len == 0 {
            anyhow::bail!("--seq-len must be > 0");
        }
        if d_model == 0 || num_heads == 0 || ffn_dim == 0 {
            anyhow::bail!("--d-model, --num-heads, and --ffn-dim must be > 0");
        }
        if d_model % num_heads != 0 {
            anyhow::bail!("--d-model ({d_model}) must be divisible by --num-heads ({num_heads})");
        }

        println!("SST-2 transformer classifier (rustral)");
        println!("======================================");
        println!("seed       : {seed}");
        println!("seq_len    : {seq_len}");
        println!("d_model    : {d_model}");
        println!("num_heads  : {num_heads}");
        println!("ffn_dim    : {ffn_dim}");
        println!("num_layers : {num_layers}");
        println!("max_vocab  : {MAX_VOCAB}");
        println!("epochs     : {epochs}");
        println!("batch_size : {batch}");
        println!("lr         : {lr}");
        println!("overfit_32 : {overfit_32}");
        println!("optimizer  : {}", if use_sgd { "sgd" } else { "adam" });
        if parity_one_step {
            println!("parity_one_step: true (batch={})", parity_batch);
        }
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

        let train_label_counts = label_counts(&train_raw);
        let dev_label_counts = label_counts(&dev_raw);

        if quick {
            train_raw.truncate(256);
        }
        if overfit_32 {
            train_raw.truncate(32);
        }

        let tok = WordLevelTokenizer::fit_from_iter(
            WordLevelConfig { lowercase: true, max_vocab: Some(MAX_VOCAB), min_freq: 1 },
            train_raw.iter().map(|e| e.sentence.as_str()),
        );
        println!("vocab_size : {} (capped at {})", tok.vocab_size(), MAX_VOCAB);

        let train: Vec<(Vec<usize>, u8)> = train_raw
            .iter()
            .map(|e| (encode_padded(&tok, &e.sentence, seq_len), e.label))
            .collect();
        let dev: Vec<(Vec<usize>, u8)> = dev_raw
            .iter()
            .map(|e| (encode_padded(&tok, &e.sentence, seq_len), e.label))
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
        let mut model = TransformerSst2::<CpuBackend>::new(
            &backend,
            tok.vocab_size(),
            num_layers,
            seed,
            Sst2Dims { seq_len, d_model, num_heads, ffn_dim },
        )?;
        let total_params = count_total_params::<CpuBackend, _>(&backend, &model);
        println!("total parameters: {}", total_params);

        if parity_one_step {
            let dump_path = parity_dump_path
                .as_ref()
                .map(|s| PathBuf::from(s))
                .unwrap_or_else(|| out_dir.join("parity_dump.json"));
            let parity = ParityOneStepArgs {
                batch_size: parity_batch,
                lr,
                seed,
                num_layers,
                seq_len,
                d_model,
                num_heads,
                ffn_dim,
                dump_path: dump_path.clone(),
            };
            run_parity_one_step(&backend, &tok, &mut model, &train, &train_raw, &dev_raw, &parity)?;
            println!("wrote {}", dump_path.display());
            return Ok(());
        }

        let cfg = TapeTrainerConfig {
            epochs,
            batch_size: batch,
            // For an overfit test we want stable ordering and maximum repeatability.
            shuffle: !overfit_32,
            seed,
            learning_rate: lr,
        };
        if use_sgd {
            let optimizer = Sgd::new(lr);
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

            let eval = if overfit_32 {
                evaluate_with_diagnostics(&backend, &model, &train)?
            } else {
                evaluate_with_diagnostics(&backend, &model, &dev)?
            };
            println!("dev: loss={:.4} acc={:.3}", eval.loss, eval.accuracy);
            println!("training throughput: {:.1} samples/sec", throughput);

            let vocab_path = out_dir.join("vocab.txt");
            fs::write(&vocab_path, tok.vocab.tokens.join("\n"))?;
            println!("wrote {}", vocab_path.display());

            let diagnostics_json = format!(
                "{{\n\
  \"train_label_counts\": [{}, {}],\n\
  \"dev_label_counts\": [{}, {}],\n\
  \"dev_confusion_matrix\": [[{}, {}], [{}, {}]],\n\
  \"dev_predicted_counts\": [{}, {}],\n\
  \"dev_positive_prob_hist\": {{\"bins\": {}, \"counts\": [{}, {}, {}, {}, {}, {}, {}, {}, {}, {}]}}\n\
}}",
                train_label_counts[0],
                train_label_counts[1],
                dev_label_counts[0],
                dev_label_counts[1],
                eval.confusion[0][0],
                eval.confusion[0][1],
                eval.confusion[1][0],
                eval.confusion[1][1],
                eval.predicted_counts[0],
                eval.predicted_counts[1],
                eval.pos_prob_hist_bins,
                eval.pos_prob_hist_counts[0],
                eval.pos_prob_hist_counts[1],
                eval.pos_prob_hist_counts[2],
                eval.pos_prob_hist_counts[3],
                eval.pos_prob_hist_counts[4],
                eval.pos_prob_hist_counts[5],
                eval.pos_prob_hist_counts[6],
                eval.pos_prob_hist_counts[7],
                eval.pos_prob_hist_counts[8],
                eval.pos_prob_hist_counts[9],
            );

            let manifest = serde_json_minimal::Object::new()
                .insert_str("task", "sst2_classifier")
                .insert_str("model_type", "transformer_encoder")
                .insert_str("git_sha", &detect_git_sha())
                .insert_u64("seed", seed)
                .insert_u64("seq_len", seq_len as u64)
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
                .insert_u64("train_examples", train.len() as u64)
                .insert_u64("dev_examples", dev.len() as u64)
                .insert_str("dataset_hash_fnv1a", &dataset_hash)
                .insert_f32("dev_loss", eval.loss)
                .insert_f32("dev_accuracy", eval.accuracy)
                .insert_f32("samples_per_sec", throughput)
                .insert_f32("train_elapsed_sec", train_elapsed.as_secs_f32())
                .insert_str("dataset", "SST-2 (binary, HuggingFace SetFit/sst2 mirror)")
                .insert_str("tokenizer", "rustral-data WordLevelTokenizer (whitespace, lowercased)")
                .insert_bool("quick_mode", quick)
                .insert_bool("overfit_32", overfit_32)
                .insert_raw("diagnostics", &diagnostics_json);
            let manifest_path = out_dir.join("manifest.json");
            fs::write(&manifest_path, manifest.to_pretty_json())?;
            println!("wrote {}", manifest_path.display());

            return Ok(());
        }

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

        let eval = if overfit_32 {
            evaluate_with_diagnostics(&backend, &model, &train)?
        } else {
            evaluate_with_diagnostics(&backend, &model, &dev)?
        };
        println!("dev: loss={:.4} acc={:.3}", eval.loss, eval.accuracy);
        println!("training throughput: {:.1} samples/sec", throughput);

        let vocab_path = out_dir.join("vocab.txt");
        fs::write(&vocab_path, tok.vocab.tokens.join("\n"))?;
        println!("wrote {}", vocab_path.display());

        let diagnostics_json = format!(
            "{{\n\
  \"train_label_counts\": [{}, {}],\n\
  \"dev_label_counts\": [{}, {}],\n\
  \"dev_confusion_matrix\": [[{}, {}], [{}, {}]],\n\
  \"dev_predicted_counts\": [{}, {}],\n\
  \"dev_positive_prob_hist\": {{\"bins\": {}, \"counts\": [{}, {}, {}, {}, {}, {}, {}, {}, {}, {}]}}\n\
}}",
            train_label_counts[0],
            train_label_counts[1],
            dev_label_counts[0],
            dev_label_counts[1],
            eval.confusion[0][0],
            eval.confusion[0][1],
            eval.confusion[1][0],
            eval.confusion[1][1],
            eval.predicted_counts[0],
            eval.predicted_counts[1],
            eval.pos_prob_hist_bins,
            eval.pos_prob_hist_counts[0],
            eval.pos_prob_hist_counts[1],
            eval.pos_prob_hist_counts[2],
            eval.pos_prob_hist_counts[3],
            eval.pos_prob_hist_counts[4],
            eval.pos_prob_hist_counts[5],
            eval.pos_prob_hist_counts[6],
            eval.pos_prob_hist_counts[7],
            eval.pos_prob_hist_counts[8],
            eval.pos_prob_hist_counts[9],
        );

        let manifest = serde_json_minimal::Object::new()
            .insert_str("task", "sst2_classifier")
            .insert_str("model_type", "transformer_encoder")
            .insert_str("git_sha", &detect_git_sha())
            .insert_u64("seed", seed)
            .insert_u64("seq_len", seq_len as u64)
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
            .insert_u64("train_examples", train.len() as u64)
            .insert_u64("dev_examples", dev.len() as u64)
            .insert_str("dataset_hash_fnv1a", &dataset_hash)
            .insert_f32("dev_loss", eval.loss)
            .insert_f32("dev_accuracy", eval.accuracy)
            .insert_f32("samples_per_sec", throughput)
            .insert_f32("train_elapsed_sec", train_elapsed.as_secs_f32())
            .insert_str("dataset", "SST-2 (binary, HuggingFace SetFit/sst2 mirror)")
            .insert_str("tokenizer", "rustral-data WordLevelTokenizer (whitespace, lowercased)")
            .insert_bool("quick_mode", quick)
            .insert_bool("overfit_32", overfit_32)
            .insert_raw("diagnostics", &diagnostics_json);
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

            /// Insert a value that is already valid JSON (object/array/number/etc).
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

    fn json_escape(s: &str) -> String {
        let mut o = String::with_capacity(s.len() + 8);
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

    fn write_f32_array(out: &mut String, xs: &[f32]) {
        out.push('[');
        for (i, v) in xs.iter().enumerate() {
            if i > 0 {
                out.push(',');
            }
            // Use a stable-ish formatting (enough for parity diffs; not for exact reproduction).
            out.push_str(&format!("{:.8}", v));
        }
        out.push(']');
    }

    fn write_usize_2d(out: &mut String, rows: &[Vec<usize>]) {
        out.push('[');
        for (i, r) in rows.iter().enumerate() {
            if i > 0 {
                out.push(',');
            }
            out.push('[');
            for (j, v) in r.iter().enumerate() {
                if j > 0 {
                    out.push(',');
                }
                out.push_str(&v.to_string());
            }
            out.push(']');
        }
        out.push(']');
    }

    struct ParityOneStepArgs {
        batch_size: usize,
        lr: f32,
        seed: u64,
        num_layers: usize,
        seq_len: usize,
        d_model: usize,
        num_heads: usize,
        ffn_dim: usize,
        dump_path: PathBuf,
    }

    fn run_parity_one_step(
        backend: &CpuBackend,
        tok: &WordLevelTokenizer,
        model: &mut TransformerSst2<CpuBackend>,
        train_encoded: &[(Vec<usize>, u8)],
        train_raw: &[Sst2Example],
        dev_raw: &[Sst2Example],
        args: &ParityOneStepArgs,
    ) -> anyhow::Result<()> {
        use rustral_autodiff::GradExtFromStore;

        let bsz = args.batch_size.max(1).min(train_encoded.len());
        let batch_x: Vec<Vec<usize>> = train_encoded.iter().take(bsz).map(|(x, _)| x.clone()).collect();
        let batch_y: Vec<u8> = train_encoded.iter().take(bsz).map(|(_, y)| *y).collect();

        let ops = backend.ops();
        let mut ctx = ForwardCtx::new(backend, Mode::Train);
        let mut tape = Tape::<CpuBackend>::new();

        // Watch parameters for grads.
        model.visit_parameters(&mut |_name, p| {
            tape.watch_parameter(p);
        });

        // Forward + loss over the batch (mean of per-sample CE).
        let mut loss_acc: Option<TensorId> = None;
        let mut logits_vec: Vec<f32> = Vec::with_capacity(bsz * 2);
        let mut logits_ids: Vec<TensorId> = Vec::with_capacity(bsz);
        for (x, y) in batch_x.iter().zip(batch_y.iter()) {
            let logits = model.forward_tape(x.clone(), &mut tape, &mut ctx)?;
            logits_ids.push(logits);
            if let Some(t) = tape.value(logits) {
                logits_vec.extend_from_slice(&ops.tensor_to_vec(t)?);
            }
            let l = model.loss_tape(logits, *y, &mut tape, &mut ctx)?;
            loss_acc = Some(match loss_acc {
                Some(acc) => tape.add(acc, l, &mut ctx)?,
                None => l,
            });
        }
        let loss_acc = loss_acc.ok_or_else(|| anyhow::anyhow!("empty batch"))?;
        let loss_mean = tape.mul_scalar(loss_acc, 1.0 / bsz as f32, &mut ctx)?;
        let loss_val = tape
            .value(loss_mean)
            .and_then(|t| ops.tensor_to_vec(t).ok())
            .and_then(|v| v.first().copied())
            .unwrap_or(0.0);

        let param_map = tape.param_map().clone();
        let make_ones = |data: Vec<f32>, shape: &[usize]| ops.tensor_from_vec(data, shape);
        let grads_store = tape.backward(loss_mean, make_ones, ops)?;

        // Extract dL/dlogits for each sample (flattened).
        let mut dlogits_flat: Vec<f32> = Vec::with_capacity(bsz * 2);
        for lid in &logits_ids {
            let g = grads_store.get(lid).ok_or_else(|| anyhow::anyhow!("missing grad for logits"))?;
            dlogits_flat.extend_from_slice(&ops.tensor_to_vec(g)?);
        }

        // Extract a few key tensors.
        let mut tok_id = None;
        let mut pos_id = None;
        let mut head_w_id = None;
        let mut head_b_id = None;
        let mut head_w: Option<Vec<f32>> = None;
        let mut head_b: Option<Vec<f32>> = None;
        let mut head_w_g: Option<Vec<f32>> = None;
        let mut head_b_g: Option<Vec<f32>> = None;
        let mut tok_table: Option<Vec<f32>> = None;
        let mut tok_table_g: Option<Vec<f32>> = None;
        let mut pos_table: Option<Vec<f32>> = None;
        let mut pos_table_g: Option<Vec<f32>> = None;

        model.visit_parameters(&mut |name, p| {
            let t = ops.tensor_to_vec(p.tensor()).ok();
            let g = p.gradient_from_store(&grads_store, &param_map).and_then(|gt| ops.tensor_to_vec(gt).ok());
            match name {
                "tok_embed.embed" => {
                    tok_id = Some(p.id());
                    tok_table = t;
                    tok_table_g = g;
                }
                "pos_embed.embed" => {
                    pos_id = Some(p.id());
                    pos_table = t;
                    pos_table_g = g;
                }
                "head.weight" => {
                    head_w_id = Some(p.id());
                    head_w = t;
                    head_w_g = g;
                }
                "head.bias" => {
                    head_b_id = Some(p.id());
                    head_b = t;
                    head_b_g = g;
                }
                _ => {}
            }
        });

        // One Adam step (in-place via cloned params + writeback like trainer does).
        let mut params_vec: Vec<Parameter<CpuBackend>> = Vec::new();
        model.visit_parameters(&mut |_name, p| params_vec.push(p.clone()));

        let mut grads = Vec::new();
        model.visit_parameters(&mut |_name, p| {
            if let Some(g) = p.gradient_from_store(&grads_store, &param_map) {
                grads.push(rustral_optim::Gradient { param_id: p.id(), tensor: g.clone() });
            }
        });

        let mut opt = Adam::new(args.lr);
        opt.step(&mut params_vec, &grads, &mut ctx)
            .map_err(|e| anyhow::anyhow!("{e:?}"))?;

        // Capture updated params by ParameterId (stable and unambiguous).
        let mut tok_table_after: Option<Vec<f32>> = None;
        let mut pos_table_after: Option<Vec<f32>> = None;
        let mut head_w_after: Option<Vec<f32>> = None;
        let mut head_b_after: Option<Vec<f32>> = None;
        for p in &params_vec {
            if let Ok(v) = ops.tensor_to_vec(p.tensor()) {
                if Some(p.id()) == tok_id {
                    tok_table_after = Some(v);
                } else if Some(p.id()) == pos_id {
                    pos_table_after = Some(v);
                } else if Some(p.id()) == head_w_id {
                    head_w_after = Some(v);
                } else if Some(p.id()) == head_b_id {
                    head_b_after = Some(v);
                }
            }
        }

        // Build JSON dump.
        let mut s = String::new();
        s.push_str("{\n");
        s.push_str("  \"task\": \"sst2_parity_one_step\",\n");
        s.push_str(&format!("  \"git_sha\": \"{}\",\n", json_escape(&detect_git_sha())));
        s.push_str(&format!("  \"seed\": {},\n", args.seed));
        s.push_str(&format!("  \"seq_len\": {},\n", args.seq_len));
        s.push_str(&format!("  \"d_model\": {},\n", args.d_model));
        s.push_str(&format!("  \"num_heads\": {},\n", args.num_heads));
        s.push_str(&format!("  \"ffn_dim\": {},\n", args.ffn_dim));
        s.push_str(&format!("  \"num_layers\": {},\n", args.num_layers));
        s.push_str(&format!("  \"lr\": {:.8},\n", args.lr));
        s.push_str(&format!("  \"batch_size\": {},\n", bsz));
        s.push_str(&format!("  \"loss\": {:.8},\n", loss_val));

        // vocab
        s.push_str("  \"vocab\": [");
        for (i, t) in tok.vocab.tokens.iter().enumerate() {
            if i > 0 {
                s.push(',');
            }
            s.push('"');
            s.push_str(&json_escape(t));
            s.push('"');
        }
        s.push_str("],\n");

        // batch
        s.push_str("  \"batch_ids\": ");
        write_usize_2d(&mut s, &batch_x);
        s.push_str(",\n");
        s.push_str("  \"batch_labels\": [");
        for (i, y) in batch_y.iter().enumerate() {
            if i > 0 {
                s.push(',');
            }
            s.push_str(&y.to_string());
        }
        s.push_str("],\n");

        s.push_str("  \"logits_flat\": ");
        write_f32_array(&mut s, &logits_vec);
        s.push_str(",\n");
        s.push_str("  \"dlogits_flat\": ");
        write_f32_array(&mut s, &dlogits_flat);
        s.push_str(",\n");

        // params + grads (flat)
        macro_rules! emit_opt_arr {
            ($k:expr, $v:expr) => {{
                s.push_str(&format!("  \"{}\": ", $k));
                if let Some(ref xs) = $v {
                    write_f32_array(&mut s, xs);
                } else {
                    s.push_str("null");
                }
                s.push_str(",\n");
            }};
        }
        emit_opt_arr!("tok_embed_table", tok_table);
        emit_opt_arr!("tok_embed_grad", tok_table_g);
        emit_opt_arr!("pos_embed_table", pos_table);
        emit_opt_arr!("pos_embed_grad", pos_table_g);
        emit_opt_arr!("head_weight", head_w);
        emit_opt_arr!("head_weight_grad", head_w_g);
        emit_opt_arr!("head_bias", head_b);
        emit_opt_arr!("head_bias_grad", head_b_g);
        emit_opt_arr!("tok_embed_table_after", tok_table_after);
        emit_opt_arr!("pos_embed_table_after", pos_table_after);
        emit_opt_arr!("head_weight_after", head_w_after);
        emit_opt_arr!("head_bias_after", head_b_after);

        // Dataset label counts for sanity.
        let tr = label_counts(train_raw);
        let dv = label_counts(dev_raw);
        s.push_str(&format!(
            "  \"train_label_counts\": [{}, {}],\n",
            tr[0], tr[1]
        ));
        s.push_str(&format!("  \"dev_label_counts\": [{}, {}]\n", dv[0], dv[1]));
        s.push_str("}\n");

        fs::write(&args.dump_path, s)?;
        Ok(())
    }
}
