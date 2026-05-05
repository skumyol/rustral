//! Shared Expert MoE Architecture
//!
//! Combines always-active shared experts with routed specialized experts.
//! The shared expert provides consistent computation for all tokens,
//! while routed experts handle specialized cases.
//!
//! # Architecture
//!
//! ```text
//! Input → [Shared Expert (always active)] ─┐
//!       → [Router] → [Routed Experts (k selected)] ─┤
//!                                              ↓
//!                                        [Weighted Sum] → Output
//! ```
//!
//! # Benefits
//!
//! - **Stability**: Shared expert provides consistent base computation
//! - **Specialization**: Routed experts handle diverse patterns
//! - **Efficiency**: Only k/N experts active per token
//! - **Load Balancing**: Expert choice routing variant available
//!
//! # Example
//!
//! ```rust,ignore
//! use mnr_nn::shared_expert::{SharedExpertLayer, SharedExpertConfig};
//!
//! let config = SharedExpertConfig::new(512, 8, 2048)
//!     .with_shared_experts(1)
//!     .with_routed_experts(63)
//!     .with_top_k(2);
//!
//! let layer = SharedExpertLayer::new(&backend, config, 42)?;
//! let output = layer.forward(input, &mut ctx)?;
//! ```

use mnr_core::{Backend, CoreError, ForwardCtx, Module, Result, TensorOps};
use serde::{Deserialize, Serialize};

use crate::{Linear, LinearConfig};

/// Configuration for shared expert architecture.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SharedExpertConfig {
    /// Input/output dimension
    pub d_model: usize,
    /// Number of shared experts (always active)
    pub num_shared: usize,
    /// Number of routed experts (selected via gating)
    pub num_routed: usize,
    /// Hidden dimension per expert
    pub expert_hidden_dim: usize,
    /// Top-k experts to activate per token
    pub top_k: usize,
    /// Weight for shared vs routed contributions
    pub shared_weight: f32,
    /// Weight for routed contributions
    pub routed_weight: f32,
    /// Use shared expert output as residual
    pub use_residual: bool,
}

impl SharedExpertConfig {
    pub fn new(d_model: usize, num_shared: usize, expert_hidden_dim: usize) -> Self {
        Self {
            d_model,
            num_shared,
            num_routed: 0,
            expert_hidden_dim,
            top_k: 2,
            shared_weight: 0.5,
            routed_weight: 0.5,
            use_residual: true,
        }
    }

    pub fn with_routed_experts(mut self, num: usize) -> Self {
        self.num_routed = num;
        self
    }

    pub fn with_top_k(mut self, k: usize) -> Self {
        self.top_k = k;
        self
    }

    pub fn with_weights(mut self, shared: f32, routed: f32) -> Self {
        self.shared_weight = shared;
        self.routed_weight = routed;
        self
    }

    pub fn with_residual(mut self, enabled: bool) -> Self {
        self.use_residual = enabled;
        self
    }

    /// Total number of experts.
    pub fn total_experts(&self) -> usize {
        self.num_shared + self.num_routed
    }

    /// Fraction of parameters active per token.
    pub fn active_fraction(&self) -> f32 {
        let active = self.num_shared + self.top_k;
        let total = self.total_experts();
        active as f32 / total as f32
    }
}

/// Statistics for shared expert layer.
#[derive(Debug, Clone)]
pub struct SharedExpertStats {
    pub num_tokens: usize,
    pub shared_expert_load: f32,
    pub routed_expert_load: Vec<f32>,
    pub avg_top_k_scores: f32,
    pub load_balance_score: f32,
}

/// Hybrid routing combining shared and routed experts.
pub struct HybridRouting {
    pub shared_gate: Vec<f32>,          // [num_tokens]
    pub routed_gates: Vec<f32>,         // [num_tokens, num_routed]
    pub top_k_indices: Vec<Vec<usize>>, // [num_tokens, top_k]
    pub top_k_weights: Vec<Vec<f32>>,   // [num_tokens, top_k]
}

/// Shared expert layer with always-on + routed experts.
pub struct SharedExpertLayer<B: Backend> {
    config: SharedExpertConfig,
    /// Shared experts (always active)
    shared_experts: Vec<ExpertMLP<B>>,
    /// Routed experts (selected via gating)
    routed_experts: Vec<ExpertMLP<B>>,
    /// Router for selecting routed experts
    router: Linear<B>,
    /// Output projection (shared across all experts)
    output_proj: Linear<B>,
}

/// Simple MLP expert.
pub struct ExpertMLP<B: Backend> {
    gate: Linear<B>,
    up: Linear<B>,
    down: Linear<B>,
}

impl<B: Backend> ExpertMLP<B> {
    pub fn new(backend: &B, d_model: usize, hidden_dim: usize) -> Result<Self> {
        Ok(Self {
            gate: Linear::new(backend, LinearConfig::new(d_model, hidden_dim))?,
            up: Linear::new(backend, LinearConfig::new(d_model, hidden_dim))?,
            down: Linear::new(backend, LinearConfig::new(hidden_dim, d_model))?,
        })
    }

    pub fn forward(&self, input: B::Tensor, ctx: &mut ForwardCtx<B>) -> Result<B::Tensor> {
        // SwiGLU-style activation: gate(x) * up(x)
        let gate_out = self.gate.forward(input.clone(), ctx)?;
        let up_out = self.up.forward(input, ctx)?;

        let ops = ctx.backend().ops();
        // Simplified: element-wise multiplication
        let activated = ops.mul(&gate_out, &up_out)?;

        self.down.forward(activated, ctx)
    }
}

impl<B: Backend> SharedExpertLayer<B>
where
    B::Tensor: Clone + AsRef<[f32]> + mnr_core::TensorShape,
{
    pub fn new(backend: &B, config: SharedExpertConfig, seed: u64) -> Result<Self> {
        let mut shared_experts = Vec::with_capacity(config.num_shared);
        for _ in 0..config.num_shared {
            shared_experts.push(ExpertMLP::new(backend, config.d_model, config.expert_hidden_dim)?);
        }

        let mut routed_experts = Vec::with_capacity(config.num_routed);
        for _ in 0..config.num_routed {
            routed_experts.push(ExpertMLP::new(backend, config.d_model, config.expert_hidden_dim)?);
        }

        let router =
            Linear::new(backend, LinearConfig::new(config.d_model, config.num_routed).with_bias(false))?;

        let output_proj = Linear::new(backend, LinearConfig::new(config.d_model, config.d_model))?;

        Ok(Self { config, shared_experts, routed_experts, router, output_proj })
    }

    /// Forward pass with hybrid routing.
    pub fn forward(
        &self,
        input: B::Tensor,
        ctx: &mut ForwardCtx<B>,
    ) -> Result<(B::Tensor, SharedExpertStats)> {
        let ops = ctx.backend().ops();
        let shape = ops.shape(&input);
        let num_tokens: usize = shape.iter().take(shape.len() - 1).product();

        // Flatten to [num_tokens, d_model]
        let flat_input = self.flatten(&input, ops)?;

        // 1. Compute shared expert output (always active)
        let mut shared_output = self.compute_shared(&flat_input, ctx)?;

        // 2. Route to selected experts
        let routing = self.compute_routing(&flat_input, ctx)?;

        // 3. Compute routed expert output
        let routed_output = if self.config.num_routed > 0 {
            self.compute_routed(&flat_input, &routing, ctx)?
        } else {
            shared_output.clone()
        };

        // 4. Combine shared + routed
        let combined = if self.config.use_residual {
            // Residual: input + weighted(shared + routed)
            let weighted_shared = ops.mul_scalar(&shared_output, self.config.shared_weight)?;
            let weighted_routed = ops.mul_scalar(&routed_output, self.config.routed_weight)?;
            let experts_sum = ops.add(&weighted_shared, &weighted_routed)?;
            ops.add(&flat_input, &experts_sum)?
        } else {
            let weighted_shared = ops.mul_scalar(&shared_output, self.config.shared_weight)?;
            let weighted_routed = ops.mul_scalar(&routed_output, self.config.routed_weight)?;
            ops.add(&weighted_shared, &weighted_routed)?
        };

        // 5. Output projection
        let output = self.output_proj.forward(combined, ctx)?;

        // Reshape back
        let final_output = ops.reshape(&output, &shape)?;

        // Compute statistics
        let stats = self.compute_stats(&routing, num_tokens);

        Ok((final_output, stats))
    }

    fn flatten(&self, input: &B::Tensor, ops: &dyn TensorOps<B>) -> Result<B::Tensor> {
        let shape = ops.shape(input);
        let num_tokens: usize = shape.iter().take(shape.len() - 1).product();
        let d_model = shape[shape.len() - 1];
        ops.reshape(input, &[num_tokens, d_model])
    }

    fn compute_shared(&self, input: &B::Tensor, ctx: &mut ForwardCtx<B>) -> Result<B::Tensor> {
        let ops = ctx.backend().ops();
        let mut output = ops.tensor_from_vec(vec![0.0f32; 1], &[1])?;
        let mut first = true;

        for expert in &self.shared_experts {
            let expert_out = expert.forward(input.clone(), ctx)?;
            if first {
                output = expert_out;
                first = false;
            } else {
                output = ops.add(&output, &expert_out)?;
            }
        }

        if !self.shared_experts.is_empty() {
            let count = self.shared_experts.len() as f32;
            output = ops.mul_scalar(&output, 1.0 / count)?;
        }

        Ok(output)
    }

    fn compute_routing(&self, input: &B::Tensor, ctx: &mut ForwardCtx<B>) -> Result<HybridRouting> {
        let ops = ctx.backend().ops();
        let shape = ops.shape(input);
        let num_tokens = shape[0];

        // Compute router logits
        let logits = self.router.forward(input.clone(), ctx)?;
        let logits_data: Vec<f32> = logits.as_ref().to_vec();

        // Softmax over routed experts
        let mut routed_gates = Vec::with_capacity(num_tokens * self.config.num_routed);
        let mut top_k_indices: Vec<Vec<usize>> = Vec::with_capacity(num_tokens);
        let mut top_k_weights: Vec<Vec<f32>> = Vec::with_capacity(num_tokens);

        for t in 0..num_tokens {
            let start = t * self.config.num_routed;
            let end = start + self.config.num_routed;

            // Softmax
            let max_logit = logits_data[start..end].iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut exps: Vec<f32> = logits_data[start..end].iter().map(|&x| (x - max_logit).exp()).collect();
            let sum_exp: f32 = exps.iter().sum();
            let probs: Vec<f32> = exps.iter().map(|&x| x / sum_exp).collect();

            // Select top-k
            let mut indexed: Vec<(usize, f32)> = probs.iter().enumerate().map(|(i, &p)| (i, p)).collect();
            indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

            let k = self.config.top_k.min(self.config.num_routed);
            let selected: Vec<usize> = indexed.iter().take(k).map(|(i, _)| *i).collect();
            let weights: Vec<f32> = indexed.iter().take(k).map(|(_, w)| *w).collect();

            top_k_indices.push(selected);
            top_k_weights.push(weights);
            routed_gates.extend_from_slice(&probs);
        }

        // Shared gate: learnable or fixed
        let shared_gate = vec![self.config.shared_weight; num_tokens];

        Ok(HybridRouting { shared_gate, routed_gates, top_k_indices, top_k_weights })
    }

    fn compute_routed(
        &self,
        input: &B::Tensor,
        routing: &HybridRouting,
        ctx: &mut ForwardCtx<B>,
    ) -> Result<B::Tensor> {
        let ops = ctx.backend().ops();
        let shape = ops.shape(input);
        let num_tokens = shape[0];

        let mut output_data: Vec<f32> = vec![0.0f32; shape.iter().product::<usize>()];
        let input_data: Vec<f32> = input.as_ref().to_vec();
        let d_model = self.config.d_model;

        // For each token, compute selected experts
        for (token_idx, (indices, weights)) in
            routing.top_k_indices.iter().zip(routing.top_k_weights.iter()).enumerate()
        {
            let mut token_output = vec![0.0f32; d_model];

            for (expert_idx_local, (&expert_global_idx, &weight)) in
                indices.iter().zip(weights.iter()).enumerate()
            {
                if expert_global_idx >= self.routed_experts.len() {
                    continue;
                }

                // Extract token input
                let token_start = token_idx * d_model;
                let token_input = &input_data[token_start..token_start + d_model];

                // Create single-token tensor
                let token_tensor = ops.tensor_from_vec(token_input.to_vec(), &[1, d_model])?;

                // Compute expert
                let expert_out = self.routed_experts[expert_global_idx].forward(token_tensor, ctx)?;
                let out_data: Vec<f32> = expert_out.as_ref().to_vec();

                // Accumulate weighted
                for d in 0..d_model {
                    token_output[d] += out_data[d] * weight;
                }
            }

            // Store
            let out_start = token_idx * d_model;
            output_data[out_start..out_start + d_model].copy_from_slice(&token_output);
        }

        ops.tensor_from_vec(output_data, &shape)
    }

    fn compute_stats(&self, routing: &HybridRouting, num_tokens: usize) -> SharedExpertStats {
        let mut expert_counts = vec![0usize; self.config.num_routed];

        for indices in &routing.top_k_indices {
            for &idx in indices {
                if idx < expert_counts.len() {
                    expert_counts[idx] += 1;
                }
            }
        }

        let avg_load = expert_counts.iter().sum::<usize>() as f32 / self.config.num_routed.max(1) as f32;
        let max_load = expert_counts.iter().cloned().max().unwrap_or(0) as f32;
        let balance_score = if avg_load > 0.0 { avg_load / max_load } else { 1.0 };

        let avg_scores: f32 = routing.top_k_weights.iter().flat_map(|w| w.iter()).sum::<f32>()
            / (num_tokens * self.config.top_k).max(1) as f32;

        SharedExpertStats {
            num_tokens,
            shared_expert_load: 1.0, // Always active
            routed_expert_load: expert_counts.iter().map(|&c| c as f32 / num_tokens as f32).collect(),
            avg_top_k_scores: avg_scores,
            load_balance_score: balance_score,
        }
    }
}

/// Configuration for combined shared + routed architecture.
pub struct SharedAndRoutedConfig {
    pub shared: SharedExpertConfig,
    pub use_expert_choice: bool,
}

impl SharedAndRoutedConfig {
    pub fn from_moe_config(d_model: usize, num_shared: usize, num_routed: usize, expert_dim: usize) -> Self {
        Self {
            shared: SharedExpertConfig::new(d_model, num_shared, expert_dim).with_routed_experts(num_routed),
            use_expert_choice: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mnr_core::Mode;
    use mnr_ndarray_backend::CpuBackend;

    #[test]
    fn test_shared_expert_config() {
        let config =
            SharedExpertConfig::new(512, 1, 2048).with_routed_experts(7).with_top_k(2).with_weights(0.3, 0.7);

        assert_eq!(config.total_experts(), 8);
        assert_eq!(config.top_k, 2);
        assert!((config.active_fraction() - 0.375).abs() < 0.01); // (1+2)/8
    }

    #[test]
    fn test_shared_expert_layer() {
        let backend = CpuBackend::default();
        let config = SharedExpertConfig::new(64, 1, 256).with_routed_experts(4).with_top_k(2);

        let layer = SharedExpertLayer::new(&backend, config, 42).unwrap();

        let input = backend.tensor_from_vec(vec![0.5f32; 4 * 64], &[2, 2, 64]).unwrap();
        let mut ctx = ForwardCtx::new(&backend, Mode::Inference);

        let (output, stats) = layer.forward(input, &mut ctx).unwrap();

        let out_shape = backend.ops().shape(&output);
        assert_eq!(out_shape, vec![2, 2, 64]);

        assert_eq!(stats.num_tokens, 4);
        assert_eq!(stats.shared_expert_load, 1.0);
        assert_eq!(stats.routed_expert_load.len(), 4);
    }

    #[test]
    fn test_expert_mlp() {
        let backend = CpuBackend::default();
        let mlp = ExpertMLP::new(&backend, 32, 128).unwrap();

        let input = backend.tensor_from_vec(vec![0.5f32; 32], &[1, 32]).unwrap();
        let mut ctx = ForwardCtx::new(&backend, Mode::Inference);

        let output = mlp.forward(input, &mut ctx).unwrap();
        let out_shape = backend.ops().shape(&output);

        assert_eq!(out_shape, vec![1, 32]);
    }

    #[test]
    fn test_shared_expert_config_with_weights() {
        let config = SharedExpertConfig::new(64, 2, 256).with_weights(0.4, 0.6);
        assert!((config.shared_weight - 0.4).abs() < 1e-5);
        assert!((config.routed_weight - 0.6).abs() < 1e-5);
    }

    #[test]
    fn test_shared_expert_config_active_fraction() {
        let config = SharedExpertConfig::new(64, 2, 256).with_routed_experts(6).with_top_k(3);
        // (2 + 3) / 8 = 0.625
        assert!((config.active_fraction() - 0.625).abs() < 0.01);
    }

    #[test]
    fn test_shared_expert_layer_from_config() {
        let backend = CpuBackend::default();
        let config = SharedExpertConfig::new(64, 1, 256).with_routed_experts(4).with_top_k(2);
        let _layer = SharedExpertLayer::new(&backend, config, 42).unwrap();
        // Layer created successfully
    }

    #[test]
    fn test_expert_mlp_forward() {
        let backend = CpuBackend::default();
        let mlp = ExpertMLP::new(&backend, 32, 128).unwrap();
        let input = backend.tensor_from_vec(vec![0.5f32; 32], &[1, 32]).unwrap();
        let mut ctx = ForwardCtx::new(&backend, Mode::Inference);
        let output = mlp.forward(input, &mut ctx).unwrap();
        assert_eq!(backend.ops().shape(&output), vec![1, 32]);
    }

    #[test]
    fn test_shared_and_routed_config() {
        let config = SharedAndRoutedConfig::from_moe_config(512, 1, 7, 2048);
        assert_eq!(config.shared.total_experts(), 8);
        assert!(!config.use_expert_choice);
    }

    #[test]
    fn test_routing_stats() {
        let backend = CpuBackend::default();
        let config = SharedExpertConfig::new(64, 1, 256).with_routed_experts(8).with_top_k(2);

        let layer = SharedExpertLayer::new(&backend, config, 42).unwrap();

        let input = backend.tensor_from_vec(vec![0.5f32; 10 * 64], &[10, 64]).unwrap();
        let mut ctx = ForwardCtx::new(&backend, Mode::Inference);

        let (_, stats) = layer.forward(input, &mut ctx).unwrap();

        assert_eq!(stats.num_tokens, 10);
        assert!(stats.load_balance_score > 0.0);
        assert!(stats.load_balance_score <= 1.0);

        // Routing happened (at least one expert has load)
        let max_load = stats.routed_expert_load.iter().cloned().fold(0.0f32, f32::max);
        assert!(max_load > 0.0, "No routing happened");
    }

    #[test]
    fn test_shared_expert_config_with_residual() {
        let config = SharedExpertConfig::new(64, 1, 256).with_residual(false);
        assert!(!config.use_residual);
    }

    #[test]
    fn test_shared_expert_layer_no_routed() {
        let backend = CpuBackend::default();
        let config = SharedExpertConfig::new(64, 1, 256).with_routed_experts(0).with_top_k(2);

        let layer = SharedExpertLayer::new(&backend, config, 42).unwrap();
        let input = backend.tensor_from_vec(vec![0.5f32; 4 * 64], &[2, 2, 64]).unwrap();
        let mut ctx = ForwardCtx::new(&backend, Mode::Inference);

        let (output, stats) = layer.forward(input, &mut ctx).unwrap();
        let out_shape = backend.ops().shape(&output);
        assert_eq!(out_shape, vec![2, 2, 64]);
        assert_eq!(stats.routed_expert_load.len(), 0);
    }

    #[test]
    fn test_shared_expert_layer_no_residual() {
        let backend = CpuBackend::default();
        let config =
            SharedExpertConfig::new(64, 1, 256).with_routed_experts(4).with_top_k(2).with_residual(false);

        let layer = SharedExpertLayer::new(&backend, config, 42).unwrap();
        let input = backend.tensor_from_vec(vec![0.5f32; 4 * 64], &[2, 2, 64]).unwrap();
        let mut ctx = ForwardCtx::new(&backend, Mode::Inference);

        let (output, _stats) = layer.forward(input, &mut ctx).unwrap();
        let out_shape = backend.ops().shape(&output);
        assert_eq!(out_shape, vec![2, 2, 64]);
    }

    #[test]
    fn test_shared_expert_layer_multiple_shared() {
        let backend = CpuBackend::default();
        let config = SharedExpertConfig::new(64, 2, 256).with_routed_experts(4).with_top_k(2);

        let layer = SharedExpertLayer::new(&backend, config, 42).unwrap();
        let input = backend.tensor_from_vec(vec![0.5f32; 4 * 64], &[2, 2, 64]).unwrap();
        let mut ctx = ForwardCtx::new(&backend, Mode::Inference);

        let (output, _stats) = layer.forward(input, &mut ctx).unwrap();
        let out_shape = backend.ops().shape(&output);
        assert_eq!(out_shape, vec![2, 2, 64]);
    }
}
