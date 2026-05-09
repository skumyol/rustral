//! EMNLP-ready char-level next-character LM demo.
//!
//! One-command demo that exercises the full Rustral training spine:
//! - dataset prep (in-repo TinyShakespeare excerpt, no downloads)
//! - char-level vocab + deterministic train/val split
//! - tape-based supervised training via [`TapeTrainer::fit_classification`]
//! - per-epoch metrics (train loss, train accuracy)
//! - validation loss + accuracy
//! - strict save -> load roundtrip with logit equality assertion
//! - greedy deterministic generation from a fixed prompt
//! - optional 3-run determinism evidence (`--determinism-check`)
//!
//! ## Run
//!
//! Default training + roundtrip + generation:
//!
//! ```bash
//! cargo run -p rustral-runtime --features training --example emnlp_char_lm
//! ```
//!
//! 3-run determinism evidence (writes JSON report next to the binary):
//!
//! ```bash
//! cargo run -p rustral-runtime --features training --example emnlp_char_lm -- --determinism-check
//! ```

#[cfg(feature = "training")]
fn main() -> anyhow::Result<()> {
    demo::run()
}

#[cfg(not(feature = "training"))]
fn main() {
    eprintln!("This example requires `--features training`.");
}

#[cfg(feature = "training")]
mod demo {
    use std::time::Instant;

    use rustral_autodiff::{Tape, TensorId};
    use rustral_core::{Backend, ForwardCtx, Mode, Module, NamedParameters, Result};
    use rustral_ndarray_backend::CpuBackend;
    use rustral_nn::tape::TapeModule;
    use rustral_nn::{Embedding, EmbeddingConfig, Linear, LinearBuilder};
    use rustral_optim::Adam;
    use rustral_runtime::{
        load_model, save_model, SupervisedTapeModel, TapeTrainer, TapeTrainerConfig, TrainingReport,
    };

    /// Tiny in-repo character corpus (Shakespeare excerpt). Kept short so the demo finishes fast.
    const CORPUS: &str = "To be, or not to be, that is the question:\n\
                          Whether 'tis nobler in the mind to suffer\n\
                          The slings and arrows of outrageous fortune,\n\
                          Or to take arms against a sea of troubles,\n\
                          And by opposing end them. To die: to sleep;\n\
                          No more; and by a sleep to say we end\n\
                          The heart-ache and the thousand natural shocks\n\
                          That flesh is heir to.\n";

    const BLOCK_SIZE: usize = 8;
    const EMBED_DIM: usize = 16;
    const EPOCHS: usize = 12;
    const BATCH_SIZE: usize = 16;
    const LEARNING_RATE: f32 = 5e-3;
    const VAL_FRACTION: f32 = 0.15;
    const GENERATE_LEN: usize = 80;
    const PROMPT: &str = "To be";

    /// Char-level vocabulary built deterministically from the corpus characters.
    pub struct CharVocab {
        chars: Vec<char>,
        id_of: std::collections::HashMap<char, usize>,
    }

    impl CharVocab {
        pub fn from_corpus(text: &str) -> Self {
            let mut chars: Vec<char> =
                text.chars().collect::<std::collections::BTreeSet<_>>().into_iter().collect();
            chars.sort();
            let id_of: std::collections::HashMap<char, usize> =
                chars.iter().enumerate().map(|(i, c)| (*c, i)).collect();
            Self { chars, id_of }
        }

        pub fn size(&self) -> usize {
            self.chars.len()
        }

        pub fn encode(&self, c: char) -> usize {
            *self.id_of.get(&c).expect("character not in corpus vocab")
        }

        pub fn decode(&self, id: usize) -> char {
            self.chars[id]
        }
    }

    /// Build (X, Y) pairs from a string of token ids.
    /// X is `block_size` consecutive ids; Y is the next id.
    pub fn build_samples(ids: &[usize], block_size: usize) -> Vec<(Vec<usize>, usize)> {
        if ids.len() <= block_size {
            return Vec::new();
        }
        let mut out = Vec::with_capacity(ids.len() - block_size);
        for i in 0..ids.len() - block_size {
            let x = ids[i..i + block_size].to_vec();
            let y = ids[i + block_size];
            out.push((x, y));
        }
        out
    }

    /// A minimal char-level LM: Embedding -> reshape -> Linear over vocab.
    /// Kept tiny so the demo is fast and deterministic on CPU.
    pub struct CharLm<B: Backend> {
        pub embed: Embedding<B>,
        pub head: Linear<B>,
        pub block_size: usize,
        pub embed_dim: usize,
    }

    impl<B: Backend> NamedParameters<B> for CharLm<B> {
        fn visit_parameters(&self, f: &mut dyn FnMut(&str, &rustral_core::Parameter<B>)) {
            self.embed.visit_parameters(&mut |n, p| f(&format!("embed.{n}"), p));
            self.head.visit_parameters(&mut |n, p| f(&format!("head.{n}"), p));
        }

        fn visit_parameters_mut(&mut self, f: &mut dyn FnMut(&str, &mut rustral_core::Parameter<B>)) {
            self.embed.visit_parameters_mut(&mut |n, p| f(&format!("embed.{n}"), p));
            self.head.visit_parameters_mut(&mut |n, p| f(&format!("head.{n}"), p));
        }
    }

    impl<B: Backend> SupervisedTapeModel<B, Vec<usize>, usize> for CharLm<B>
    where
        B::Tensor: Clone,
    {
        fn forward_tape(
            &mut self,
            input: Vec<usize>,
            tape: &mut Tape<B>,
            ctx: &mut ForwardCtx<B>,
        ) -> Result<TensorId> {
            assert_eq!(input.len(), self.block_size, "block size mismatch in forward_tape");
            let ids_f32: Vec<f32> = input.iter().map(|&i| i as f32).collect();
            let ids_t = ctx.backend().ops().tensor_from_vec(ids_f32, &[self.block_size])?;
            let ids_id = tape.watch(ids_t);
            let emb = self.embed.forward_tape(ids_id, tape, ctx)?;
            let flat = tape.reshape_tape(emb, &[1, self.block_size * self.embed_dim], ctx)?;
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

    impl<B: Backend> CharLm<B>
    where
        B::Tensor: Clone,
    {
        pub fn new(
            backend: &B,
            vocab_size: usize,
            block_size: usize,
            embed_dim: usize,
            seed: u64,
        ) -> Result<Self> {
            let embed = Embedding::new(backend, EmbeddingConfig::new(vocab_size, embed_dim), seed)?;
            let head = LinearBuilder::new(block_size * embed_dim, vocab_size)
                .with_bias(true)
                .seed(seed.wrapping_add(101))
                .build(backend)?;
            let _ = vocab_size;
            Ok(Self { embed, head, block_size, embed_dim })
        }

        /// Compute logits for a single token-id window in inference mode.
        pub fn logits(&self, backend: &B, ids: &[usize]) -> Result<Vec<f32>> {
            let mut ctx = ForwardCtx::new(backend, Mode::Inference);
            let emb = self.embed.forward(ids.to_vec(), &mut ctx)?;
            let flat = backend.ops().reshape(&emb, &[1, self.block_size * self.embed_dim])?;
            let logits = self.head.forward(flat, &mut ctx)?;
            backend.ops().tensor_to_vec(&logits)
        }
    }

    /// Greedy deterministic generation: argmax of logits, sliding window.
    pub fn generate_greedy<B: Backend>(
        backend: &B,
        model: &CharLm<B>,
        vocab: &CharVocab,
        prompt: &str,
        n_new: usize,
    ) -> Result<String>
    where
        B::Tensor: Clone,
    {
        let mut window: Vec<usize> = prompt.chars().map(|c| vocab.encode(c)).collect();
        if window.len() < model.block_size {
            let pad = model.block_size - window.len();
            let pad_id = vocab.encode(' ');
            window = std::iter::repeat(pad_id).take(pad).chain(window.into_iter()).collect();
        } else if window.len() > model.block_size {
            window = window[window.len() - model.block_size..].to_vec();
        }

        let mut out = String::from(prompt);
        for _ in 0..n_new {
            let logits = model.logits(backend, &window)?;
            let mut best = 0usize;
            let mut best_v = f32::NEG_INFINITY;
            for (i, v) in logits.iter().enumerate() {
                if *v > best_v {
                    best_v = *v;
                    best = i;
                }
            }
            out.push(vocab.decode(best));
            window.remove(0);
            window.push(best);
        }
        Ok(out)
    }

    /// Validation pass: returns (mean_loss, accuracy).
    pub fn validate<B: Backend>(
        backend: &B,
        model: &CharLm<B>,
        val: &[(Vec<usize>, usize)],
    ) -> Result<(f32, f32)>
    where
        B::Tensor: Clone,
    {
        if val.is_empty() {
            return Ok((0.0, 0.0));
        }
        let mut total_loss = 0.0f64;
        let mut correct = 0usize;
        for (x, y) in val {
            let logits = model.logits(backend, x)?;
            let mut max_v = f32::NEG_INFINITY;
            let mut argmax = 0usize;
            for (i, v) in logits.iter().enumerate() {
                if *v > max_v {
                    max_v = *v;
                    argmax = i;
                }
            }
            // numerically stable cross-entropy on a single sample
            let mut sum_exp = 0.0f64;
            for v in &logits {
                sum_exp += ((*v - max_v) as f64).exp();
            }
            let log_denom = (max_v as f64) + sum_exp.ln();
            let loss = log_denom - logits[*y] as f64;
            total_loss += loss;
            if argmax == *y {
                correct += 1;
            }
        }
        Ok(((total_loss / val.len() as f64) as f32, correct as f32 / val.len() as f32))
    }

    /// One full training run with a given seed; returns a structured result.
    pub fn train_once(seed: u64, log: bool) -> anyhow::Result<RunReport> {
        let backend = CpuBackend::default();
        let vocab = CharVocab::from_corpus(CORPUS);
        let ids: Vec<usize> = CORPUS.chars().map(|c| vocab.encode(c)).collect();
        let samples = build_samples(&ids, BLOCK_SIZE);

        // Deterministic train/val split: take the last VAL_FRACTION of samples (no random shuffle of the split).
        let split = ((samples.len() as f32) * (1.0 - VAL_FRACTION)) as usize;
        let train: Vec<(Vec<usize>, usize)> = samples[..split].to_vec();
        let val: Vec<(Vec<usize>, usize)> = samples[split..].to_vec();

        if log {
            println!("EMNLP char-LM demo");
            println!("==================");
            println!("seed         : {seed}");
            println!("corpus_chars : {}", CORPUS.len());
            println!("vocab_size   : {}", vocab.size());
            println!("block_size   : {BLOCK_SIZE}");
            println!("embed_dim    : {EMBED_DIM}");
            println!("epochs       : {EPOCHS}");
            println!("batch_size   : {BATCH_SIZE}");
            println!("learning_rate: {LEARNING_RATE}");
            println!("train_samples: {}", train.len());
            println!("val_samples  : {}", val.len());
            println!();
        }

        let mut model = CharLm::<CpuBackend>::new(&backend, vocab.size(), BLOCK_SIZE, EMBED_DIM, seed)?;

        let cfg = TapeTrainerConfig {
            epochs: EPOCHS,
            batch_size: BATCH_SIZE,
            shuffle: true,
            seed,
            learning_rate: LEARNING_RATE,
        };
        let optimizer = Adam::new(LEARNING_RATE);
        let mut trainer = TapeTrainer::<CpuBackend, _>::new(cfg, optimizer);

        let t0 = Instant::now();
        let report: TrainingReport = trainer.fit_classification(&backend, &mut model, &train)?;
        let elapsed = t0.elapsed();
        let total_examples = (train.len() * EPOCHS) as f32;
        let throughput = total_examples / elapsed.as_secs_f32().max(1e-9);

        if log {
            for epoch in &report.epochs {
                println!(
                    "epoch {:>3}: train_loss={:.4}  elapsed={:?}",
                    epoch.epoch, epoch.mean_loss, epoch.elapsed
                );
            }
            if let Some(acc) = report.accuracy.as_ref() {
                if let Some(last) = acc.last() {
                    println!("final train acc: {:.3}", last);
                }
            }
            println!("training throughput: {:.1} samples/sec", throughput);
        }

        let (val_loss, val_acc) = validate(&backend, &model, &val)?;
        if log {
            println!("validation: loss={:.4} acc={:.3}", val_loss, val_acc);
        }

        // Save -> load -> infer roundtrip on a fixed input.
        let bytes = save_model(&model, &backend)?;
        let mut model2 =
            CharLm::<CpuBackend>::new(&backend, vocab.size(), BLOCK_SIZE, EMBED_DIM, seed.wrapping_add(999))?;
        load_model(&mut model2, &backend, &bytes)?;

        let probe: Vec<usize> = CORPUS.chars().take(BLOCK_SIZE).map(|c| vocab.encode(c)).collect();
        let l1 = model.logits(&backend, &probe)?;
        let l2 = model2.logits(&backend, &probe)?;
        anyhow::ensure!(l1 == l2, "save/load roundtrip changed logits: {:?} vs {:?}", l1, l2);

        // Deterministic greedy generation.
        let generated = generate_greedy(&backend, &model, &vocab, PROMPT, GENERATE_LEN)?;
        if log {
            println!("\ngenerated ({} chars from prompt {:?}):", GENERATE_LEN, PROMPT);
            println!("{generated}");
        }

        let final_train_loss = report.epochs.last().map(|e| e.mean_loss).unwrap_or(f32::NAN);
        let final_train_acc = report.accuracy.as_ref().and_then(|v| v.last().copied()).unwrap_or(f32::NAN);

        Ok(RunReport {
            seed,
            vocab_size: vocab.size(),
            train_samples: train.len(),
            val_samples: val.len(),
            epochs: EPOCHS,
            final_train_loss,
            final_train_acc,
            val_loss,
            val_acc,
            samples_per_sec: throughput,
            generated,
        })
    }

    #[derive(Clone, Debug)]
    pub struct RunReport {
        pub seed: u64,
        pub vocab_size: usize,
        pub train_samples: usize,
        pub val_samples: usize,
        pub epochs: usize,
        pub final_train_loss: f32,
        pub final_train_acc: f32,
        pub val_loss: f32,
        pub val_acc: f32,
        pub samples_per_sec: f32,
        pub generated: String,
    }

    impl RunReport {
        pub fn to_json_line(&self) -> String {
            let escaped = self
                .generated
                .replace('\\', "\\\\")
                .replace('"', "\\\"")
                .replace('\n', "\\n")
                .replace('\r', "\\r")
                .replace('\t', "\\t");
            format!(
                "{{\"seed\":{},\"vocab_size\":{},\"train_samples\":{},\"val_samples\":{},\"epochs\":{},\"final_train_loss\":{:.6},\"final_train_acc\":{:.6},\"val_loss\":{:.6},\"val_acc\":{:.6},\"samples_per_sec\":{:.3},\"generated\":\"{}\"}}",
                self.seed,
                self.vocab_size,
                self.train_samples,
                self.val_samples,
                self.epochs,
                self.final_train_loss,
                self.final_train_acc,
                self.val_loss,
                self.val_acc,
                self.samples_per_sec,
                escaped,
            )
        }
    }

    pub fn run() -> anyhow::Result<()> {
        let args: Vec<String> = std::env::args().collect();
        let determinism = args.iter().any(|a| a == "--determinism-check");

        if !determinism {
            train_once(0xC0FFEE, true)?;
            return Ok(());
        }

        // Determinism mode: 3 fresh runs with the same seed; report exact equality.
        println!("Determinism check: 3 runs with the same seed (CPU should be exact).");
        let mut runs: Vec<RunReport> = Vec::with_capacity(3);
        for i in 0..3 {
            println!("\n--- run {} ---", i + 1);
            let r = train_once(0xC0FFEE, true)?;
            runs.push(r);
        }

        let mut all_equal = true;
        for w in runs.windows(2) {
            let a = &w[0];
            let b = &w[1];
            if a.final_train_loss != b.final_train_loss
                || a.val_loss != b.val_loss
                || a.val_acc != b.val_acc
                || a.generated != b.generated
            {
                all_equal = false;
                break;
            }
        }

        let mut json = String::from("[\n");
        for (i, r) in runs.iter().enumerate() {
            json.push_str("  ");
            json.push_str(&r.to_json_line());
            if i + 1 < runs.len() {
                json.push(',');
            }
            json.push('\n');
        }
        json.push(']');
        let report_path = std::path::PathBuf::from("emnlp_determinism_report.json");
        std::fs::write(&report_path, &json)?;
        println!("\nWrote report: {}", report_path.display());

        anyhow::ensure!(all_equal, "determinism check failed: runs diverged on CPU");
        println!("Determinism check: OK (all 3 runs identical).");
        Ok(())
    }
}
