//! Backend-agnostic causal language modeling surface for [`crate::gpt2::Gpt2Decoder`] and future families.
//!
//! Generation runs through an explicit [`ForwardCtx`] so callers control inference vs train mode,
//! attach profilers, and reuse context across steps.

use rustral_core::{Backend, ForwardCtx};

use crate::LlmError;

/// Autoregressive LM with greedy decoding over integer token ids.
///
/// Implementations should run the full sequence forward each step (no KV cache yet); [`ForwardCtx`]
/// carries [`rustral_core::Mode::Inference`] for deterministic dropout/norm behavior.
pub trait CausalLm<B: Backend> {
    /// Append up to `max_new_tokens` greedy tokens to `input_ids` and return the full id sequence.
    fn generate_greedy(
        &self,
        ctx: &mut ForwardCtx<'_, B>,
        input_ids: Vec<usize>,
        max_new_tokens: usize,
    ) -> Result<Vec<usize>, LlmError>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustral_core::Mode;

    use crate::gpt2::{Gpt2Decoder, HfGpt2Config};

    #[test]
    fn causal_lm_trait_uses_explicit_forward_ctx() {
        let cfg = HfGpt2Config {
            vocab_size: 64,
            n_positions: 32,
            n_embd: 16,
            n_layer: 1,
            n_head: 2,
            resid_pdrop: Some(0.0),
        };
        let model = Gpt2Decoder::new_random(&cfg, 1).expect("decoder");
        let mut ctx = ForwardCtx::new(model.backend(), Mode::Inference);
        let run_id = ctx.run_id();

        let out = CausalLm::generate_greedy(&model, &mut ctx, vec![0usize], 2).expect("greedy");
        assert_eq!(out.len(), 3);
        assert_eq!(ctx.run_id(), run_id);
        assert!(!ctx.is_training());
    }
}
