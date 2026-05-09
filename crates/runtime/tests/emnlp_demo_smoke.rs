//! Smoke test for the EMNLP-ready char-level LM demo.
//!
//! Exercises the same spine as `examples/emnlp_char_lm.rs` but with a tiny budget so it
//! can run in the regular test suite:
//! - dataset prep (in-test corpus + char vocab)
//! - tape supervised training via [`TapeTrainer::fit_classification`]
//! - save -> load -> infer logit equality
//! - 3-run determinism on CPU (exact equality)

#![cfg(feature = "training")]

use rustral_autodiff::{Tape, TensorId};
use rustral_core::{Backend, ForwardCtx, Mode, Module, NamedParameters, Result};
use rustral_ndarray_backend::CpuBackend;
use rustral_nn::tape::TapeModule;
use rustral_nn::{Embedding, EmbeddingConfig, Linear, LinearBuilder};
use rustral_optim::Adam;
use rustral_runtime::{load_model, save_model, SupervisedTapeModel, TapeTrainer, TapeTrainerConfig};

const CORPUS: &str = "abcabcabcabcabcabcabcabcabcabcabcabcabc";
const BLOCK_SIZE: usize = 4;
const EMBED_DIM: usize = 8;
const EPOCHS: usize = 3;
const BATCH_SIZE: usize = 8;
const LEARNING_RATE: f32 = 1e-2;

struct CharVocab {
    chars: Vec<char>,
    id_of: std::collections::HashMap<char, usize>,
}

impl CharVocab {
    fn from_corpus(text: &str) -> Self {
        let mut chars: Vec<char> =
            text.chars().collect::<std::collections::BTreeSet<_>>().into_iter().collect();
        chars.sort();
        let id_of: std::collections::HashMap<char, usize> =
            chars.iter().enumerate().map(|(i, c)| (*c, i)).collect();
        Self { chars, id_of }
    }

    fn size(&self) -> usize {
        self.chars.len()
    }

    fn encode(&self, c: char) -> usize {
        *self.id_of.get(&c).expect("character not in corpus vocab")
    }
}

fn build_samples(ids: &[usize], block: usize) -> Vec<(Vec<usize>, usize)> {
    if ids.len() <= block {
        return Vec::new();
    }
    (0..ids.len() - block).map(|i| (ids[i..i + block].to_vec(), ids[i + block])).collect()
}

struct CharLm<B: Backend> {
    embed: Embedding<B>,
    head: Linear<B>,
    block_size: usize,
    embed_dim: usize,
}

impl<B: Backend> CharLm<B>
where
    B::Tensor: Clone,
{
    fn new(backend: &B, vocab_size: usize, block: usize, dim: usize, seed: u64) -> Result<Self> {
        Ok(Self {
            embed: Embedding::new(backend, EmbeddingConfig::new(vocab_size, dim), seed)?,
            head: LinearBuilder::new(block * dim, vocab_size)
                .with_bias(true)
                .seed(seed.wrapping_add(101))
                .build(backend)?,
            block_size: block,
            embed_dim: dim,
        })
    }

    fn logits(&self, backend: &B, ids: &[usize]) -> Result<Vec<f32>> {
        let mut ctx = ForwardCtx::new(backend, Mode::Inference);
        let emb = self.embed.forward(ids.to_vec(), &mut ctx)?;
        let flat = backend.ops().reshape(&emb, &[1, self.block_size * self.embed_dim])?;
        let logits = self.head.forward(flat, &mut ctx)?;
        backend.ops().tensor_to_vec(&logits)
    }
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

fn train_once(seed: u64) -> anyhow::Result<(Vec<f32>, f32, Vec<u8>)> {
    let backend = CpuBackend::default();
    let vocab = CharVocab::from_corpus(CORPUS);
    let ids: Vec<usize> = CORPUS.chars().map(|c| vocab.encode(c)).collect();
    let samples = build_samples(&ids, BLOCK_SIZE);

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

    let report = trainer.fit_classification(&backend, &mut model, &samples)?;
    let final_loss = report.epochs.last().map(|e| e.mean_loss).unwrap_or(f32::NAN);

    let probe: Vec<usize> = ids.iter().take(BLOCK_SIZE).copied().collect();
    let logits = model.logits(&backend, &probe)?;
    let bytes = save_model(&model, &backend)?;

    Ok((logits, final_loss, bytes))
}

#[test]
fn emnlp_char_lm_trains_save_load_roundtrip_and_is_deterministic() {
    let seed = 0xCAFEu64;

    // Three runs with identical config must produce identical logits + final loss on CPU.
    let (logits_a, loss_a, bytes_a) = train_once(seed).expect("run a");
    let (logits_b, loss_b, _) = train_once(seed).expect("run b");
    let (logits_c, loss_c, _) = train_once(seed).expect("run c");

    assert_eq!(logits_a, logits_b, "determinism failure between runs A and B");
    assert_eq!(logits_b, logits_c, "determinism failure between runs B and C");
    assert!(loss_a == loss_b && loss_b == loss_c, "loss diverged across deterministic runs");

    // Save -> load -> infer roundtrip preserves logits exactly.
    let backend = CpuBackend::default();
    let vocab = CharVocab::from_corpus(CORPUS);
    let mut fresh =
        CharLm::<CpuBackend>::new(&backend, vocab.size(), BLOCK_SIZE, EMBED_DIM, seed.wrapping_add(7777))
            .expect("fresh model");
    load_model(&mut fresh, &backend, &bytes_a).expect("load_model");
    let probe: Vec<usize> = CORPUS.chars().take(BLOCK_SIZE).map(|c| vocab.encode(c)).collect();
    let logits_loaded = fresh.logits(&backend, &probe).expect("logits");
    assert_eq!(logits_a, logits_loaded, "save/load roundtrip changed logits");
}
