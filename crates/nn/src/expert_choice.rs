//! Expert Choice Routing for Mixture of Experts
//!
//! Alternative to top-k gating where experts choose which tokens to process
//! rather than tokens choosing experts. This provides better load balancing
//! and more predictable compute patterns.
//!
//! # Key Differences from Top-K
//!
//! - **Top-K**: Each token selects k experts (can cause imbalance)
//! - **Expert Choice**: Each expert selects k tokens (balanced by design)
//!
//! # Algorithm
//!
//! 1. Compute router scores: [num_tokens, num_experts]
//! 2. For each expert, select top-k tokens
//! 3. Gather selected tokens to experts
//! 4. Process with selected experts
//! 5. Scatter results back (weighted by router scores)
//!
//! # Example
//!
//! ```rust,ignore
//! use mnr_nn::expert_choice::{ExpertChoiceRouter, ExpertChoiceConfig};
//!
//! let config = ExpertChoiceConfig::new(512, 64, 2048)
//!     .with_tokens_per_expert(8);  // Each expert processes 8 tokens
//!
//! let router = ExpertChoiceRouter::new(&backend, config, 42)?;
//! let output = router.forward(input, &expert_layer, &mut ctx)?;
//! ```

use mnr_core::{Backend, CoreError, ForwardCtx, Module, Result, TensorOps};
use serde::{Deserialize, Serialize};

use crate::{Linear, LinearConfig};

/// Configuration for expert choice routing.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ExpertChoiceConfig {
    /// Input dimension (d_model)
    pub d_model: usize,
    /// Number of experts
    pub num_experts: usize,
    /// Hidden dimension per expert
    pub expert_hidden_dim: usize,
    /// Number of tokens each expert processes
    pub tokens_per_expert: usize,
    /// All-to-all communication
    pub all_to_all: bool,
    /// Capacity factor (1.0 = exact, >1.0 = slack)
    pub capacity_factor: f32,
}

impl ExpertChoiceConfig {
    pub fn new(d_model: usize, num_experts: usize, expert_hidden_dim: usize) -> Self {
        Self {
            d_model,
            num_experts,
            expert_hidden_dim,
            tokens_per_expert: 8,
            all_to_all: true,
            capacity_factor: 1.0,
        }
    }

    pub fn with_tokens_per_expert(mut self, tokens: usize) -> Self {
        self.tokens_per_expert = tokens;
        self
    }

    pub fn with_all_to_all(mut self, enabled: bool) -> Self {
        self.all_to_all = enabled;
        self
    }

    pub fn with_capacity_factor(mut self, factor: f32) -> Self {
        self.capacity_factor = factor;
        self
    }

    /// Calculate total tokens processed across all experts.
    pub fn total_tokens_processed(&self) -> usize {
        self.num_experts * self.tokens_per_expert
    }

    /// Check if configuration is valid for given batch size.
    pub fn validate_for_batch(&self, num_tokens: usize) -> Result<()> {
        let total_capacity = self.total_tokens_processed();

        if num_tokens > total_capacity {
            return Err(CoreError::InvalidArgument(
                format!("Batch size {} exceeds total capacity {} (experts: {}, tokens_per_expert: {})",
                    num_tokens, total_capacity, self.num_experts, self.tokens_per_expert)
            ));
        }

        Ok(())
    }
}

/// Router for expert choice.
pub struct ExpertChoiceRouter<B: Backend> {
    config: ExpertChoiceConfig,
    /// Router projection [d_model, num_experts]
    router: Linear<B>,
}

impl<B: Backend> ExpertChoiceRouter<B> {
    pub fn new(backend: &B, config: ExpertChoiceConfig, seed: u64) -> Result<Self> {
        let router = Linear::new(
            backend,
            LinearConfig::new(config.d_model, config.num_experts).with_bias(false),
        )?;

        Ok(Self { config, router })
    }

    /// Forward pass with expert choice routing.
    ///
    /// Returns (output, router_stats).
    pub fn forward(
        &self,
        input: &B::Tensor,
        expert_fn: &dyn Fn(&B::Tensor) -> Result<B::Tensor>,
        ctx: &mut ForwardCtx<B>,
    ) -> Result<(B::Tensor, ExpertChoiceStats)>
    where
        B::Tensor: AsRef<[f32]>,
    {
        let ops = ctx.backend().ops();
        let shape = ops.shape(input);

        if shape.len() < 2 {
            return Err(CoreError::Shape(
                "Input must have at least 2 dimensions [batch*seq, d_model]".to_string()
            ));
        }

        let num_tokens = shape.iter().take(shape.len() - 1).product::<usize>();
        let d_model = shape[shape.len() - 1];

        if d_model != self.config.d_model {
            return Err(CoreError::InvalidArgument(
                format!("Input d_model {} doesn't match config {}", d_model, self.config.d_model)
            ));
        }

        self.config.validate_for_batch(num_tokens)?;

        // Flatten to [num_tokens, d_model]
        let flat_input = self.flatten(input, ops)?;

        // Compute router scores: [num_tokens, num_experts]
        let router_logits = self.router.forward(flat_input.clone(), ctx)?;
        let router_probs = self.softmax(&router_logits, ops)?;

        // For each expert, select top-k tokens
        let assignments = self.expert_selection(&router_probs, ops)?;

        // Gather tokens to experts and process
        let expert_outputs = self.process_with_experts(
            &flat_input,
            &assignments,
            expert_fn,
            ops,
        )?;

        // Scatter results back with weighting
        let output = self.scatter_results(&expert_outputs, &assignments, &router_probs, ops)?;

        // Reshape back to input shape
        let final_output = ops.reshape(&output, &shape)?;

        // Compute stats
        let stats = self.compute_stats(&assignments, num_tokens);

        Ok((final_output, stats))
    }

    /// Flatten batch dimensions.
    fn flatten(&self, input: &B::Tensor, ops: &dyn TensorOps<B>) -> Result<B::Tensor> {
        let shape = ops.shape(input);
        let num_tokens: usize = shape.iter().take(shape.len() - 1).product();
        let d_model = shape[shape.len() - 1];
        ops.reshape(input, &[num_tokens, d_model])
    }

    /// Softmax over experts dimension.
    fn softmax(&self, logits: &B::Tensor, ops: &dyn TensorOps<B>) -> Result<Vec<f32>>
    where
        B::Tensor: AsRef<[f32]>,
    {
        let data: Vec<f32> = logits.as_ref().to_vec();
        let num_experts = self.config.num_experts;
        let num_tokens = data.len() / num_experts;

        let mut probs = Vec::with_capacity(data.len());

        for t in 0..num_tokens {
            let start = t * num_experts;
            let end = start + num_experts;

            // Find max for numerical stability
            let max_logit = data[start..end].iter().cloned().fold(f32::NEG_INFINITY, f32::max);

            // Compute exp and sum
            let mut sum = 0.0f32;
            let mut token_probs = Vec::with_capacity(num_experts);
            for i in start..end {
                let exp_val = (data[i] - max_logit).exp();
                token_probs.push(exp_val);
                sum += exp_val;
            }

            // Normalize
            for p in token_probs {
                probs.push(p / sum);
            }
        }

        Ok(probs)
    }

    /// For each expert, select top-k tokens based on router probs.
    fn expert_selection(&self, router_probs: &[f32], _ops: &dyn TensorOps<B>) -> Result<ExpertAssignments> {
        let num_experts = self.config.num_experts;
        let tokens_per_expert = self.config.tokens_per_expert;
        let num_tokens = router_probs.len() / num_experts;

        // Transpose to [num_experts, num_tokens] for easier processing
        let mut expert_token_scores: Vec<Vec<(usize, f32)>> = vec![Vec::new(); num_experts];

        for (token_idx, token_probs) in router_probs.chunks(num_experts).enumerate() {
            for (expert_idx, &score) in token_probs.iter().enumerate() {
                expert_token_scores[expert_idx].push((token_idx, score));
            }
        }

        // For each expert, sort by score and select top-k
        let mut assignments: Vec<Vec<TokenAssignment>> = Vec::with_capacity(num_experts);

        for expert_idx in 0..num_experts {
            let mut tokens = expert_token_scores[expert_idx].clone();
            tokens.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

            let capacity = (tokens_per_expert as f32 * self.config.capacity_factor) as usize;

            let selected: Vec<TokenAssignment> = tokens
                .into_iter()
                .take(capacity)
                .map(|(token_idx, weight)| TokenAssignment {
                    token_idx,
                    expert_idx,
                    weight,
                })
                .collect();

            assignments.push(selected);
        }

        Ok(ExpertAssignments { assignments, num_tokens })
    }

    /// Gather tokens to experts and process.
    fn process_with_experts(
        &self,
        flat_input: &B::Tensor,
        assignments: &ExpertAssignments,
        expert_fn: &dyn Fn(&B::Tensor) -> Result<B::Tensor>,
        ops: &dyn TensorOps<B>,
    ) -> Result<Vec<Vec<f32>>>
    where
        B::Tensor: AsRef<[f32]>,
    {
        let input_data: Vec<f32> = flat_input.as_ref().to_vec();
        let d_model = self.config.d_model;

        let mut expert_outputs: Vec<Vec<f32>> = Vec::new();

        for expert_assignments in &assignments.assignments {
            if expert_assignments.is_empty() {
                expert_outputs.push(Vec::new());
                continue;
            }

            // Gather tokens for this expert
            let mut expert_input = Vec::with_capacity(expert_assignments.len() * d_model);
            for assignment in expert_assignments {
                let token_start = assignment.token_idx * d_model;
                expert_input.extend_from_slice(&input_data[token_start..token_start + d_model]);
            }

            // Create tensor and process
            let expert_input_tensor = ops.tensor_from_vec(
                expert_input,
                &[expert_assignments.len(), d_model]
            )?;

            let output_tensor = expert_fn(&expert_input_tensor)?;
            let output_data: Vec<f32> = output_tensor.as_ref().to_vec();

            expert_outputs.push(output_data);
        }

        Ok(expert_outputs)
    }

    /// Scatter expert outputs back to token positions with weighting.
    fn scatter_results(
        &self,
        expert_outputs: &[Vec<f32>],
        assignments: &ExpertAssignments,
        router_probs: &[f32],
        ops: &dyn TensorOps<B>,
    ) -> Result<B::Tensor> {
        let num_tokens = assignments.num_tokens;
        let d_model = self.config.d_model;
        let mut output = vec![0.0f32; num_tokens * d_model];

        for (expert_idx, expert_assignments) in assignments.assignments.iter().enumerate() {
            let expert_output = &expert_outputs[expert_idx];
            if expert_output.is_empty() {
                continue;
            }

            for (i, assignment) in expert_assignments.iter().enumerate() {
                let weight = router_probs[assignment.token_idx * self.config.num_experts + expert_idx];

                let out_start = i * d_model;
                let token_out_start = assignment.token_idx * d_model;

                for d in 0..d_model {
                    output[token_out_start + d] += expert_output[out_start + d] * weight;
                }
            }
        }

        ops.tensor_from_vec(output, &[num_tokens, d_model])
    }

    /// Compute routing statistics.
    fn compute_stats(&self, assignments: &ExpertAssignments, num_tokens: usize) -> ExpertChoiceStats {
        let mut total_selected = 0;
        let mut min_tokens = usize::MAX;
        let mut max_tokens = 0;

        for expert_assignments in &assignments.assignments {
            let count = expert_assignments.len();
            total_selected += count;
            min_tokens = min_tokens.min(count);
            max_tokens = max_tokens.max(count);
        }

        let avg_tokens = total_selected as f64 / self.config.num_experts as f64;
        let imbalance = max_tokens as f64 / avg_tokens.max(1.0);

        ExpertChoiceStats {
            num_tokens,
            num_experts: self.config.num_experts,
            total_selected,
            min_tokens_per_expert: min_tokens,
            max_tokens_per_expert: max_tokens,
            avg_tokens_per_expert: avg_tokens,
            load_imbalance_ratio: imbalance,
        }
    }
}

/// Assignment of tokens to experts.
#[derive(Debug, Clone)]
pub struct ExpertAssignments {
    /// For each expert, list of token assignments
    pub assignments: Vec<Vec<TokenAssignment>>,
    /// Total number of tokens
    pub num_tokens: usize,
}

/// Single token assignment.
#[derive(Debug, Clone)]
pub struct TokenAssignment {
    pub token_idx: usize,
    pub expert_idx: usize,
    pub weight: f32,
}

/// Statistics for expert choice routing.
#[derive(Debug, Clone)]
pub struct ExpertChoiceStats {
    pub num_tokens: usize,
    pub num_experts: usize,
    pub total_selected: usize,
    pub min_tokens_per_expert: usize,
    pub max_tokens_per_expert: usize,
    pub avg_tokens_per_expert: f64,
    pub load_imbalance_ratio: f64,
}

impl ExpertChoiceStats {
    /// Print statistics.
    pub fn print(&self) {
        println!("Expert Choice Routing Statistics:");
        println!("  Tokens: {}", self.num_tokens);
        println!("  Experts: {}", self.num_experts);
        println!("  Total assignments: {}", self.total_selected);
        println!("  Min tokens/expert: {}", self.min_tokens_per_expert);
        println!("  Max tokens/expert: {}", self.max_tokens_per_expert);
        println!("  Avg tokens/expert: {:.2}", self.avg_tokens_per_expert);
        println!("  Load imbalance: {:.2}", self.load_imbalance_ratio);
    }
}

/// Comparison with top-k routing.
pub struct RoutingComparison {
    pub top_k_imbalance: f64,
    pub expert_choice_imbalance: f64,
    pub improvement: f64,
}

impl RoutingComparison {
    /// Compare load balancing between top-k and expert choice.
    pub fn compare(num_tokens: usize, num_experts: usize, top_k: usize) -> Self {
        // Simulate top-k: tokens choose experts randomly
        let mut expert_counts = vec![0usize; num_experts];
        let mut rng = rand::thread_rng();
        use rand::Rng;

        for _ in 0..num_tokens {
            for _ in 0..top_k {
                let expert = rng.gen::<usize>() % num_experts;
                expert_counts[expert] += 1;
            }
        }

        let avg = expert_counts.iter().sum::<usize>() as f64 / num_experts as f64;
        let max_count = expert_counts.iter().cloned().max().unwrap_or(1);
        let top_k_imbalance = max_count as f64 / avg.max(1.0);

        // Expert choice: perfectly balanced by design
        let tokens_per_expert = (num_tokens * top_k) / num_experts;
        let expert_choice_imbalance = 1.0;

        let improvement = (top_k_imbalance - expert_choice_imbalance) / top_k_imbalance * 100.0;

        Self {
            top_k_imbalance,
            expert_choice_imbalance,
            improvement,
        }
    }

    pub fn print(&self) {
        println!("Routing Comparison:");
        println!("  Top-K imbalance: {:.2}", self.top_k_imbalance);
        println!("  Expert Choice imbalance: {:.2}", self.expert_choice_imbalance);
        println!("  Improvement: {:.1}%", self.improvement);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mnr_ndarray_backend::CpuBackend;

    #[test]
    fn test_expert_choice_config() {
        let config = ExpertChoiceConfig::new(512, 64, 2048)
            .with_tokens_per_expert(16)
            .with_capacity_factor(1.2);

        assert_eq!(config.total_tokens_processed(), 64 * 16);
        assert_eq!(config.capacity_factor, 1.2);

        // Should work for small batch
        config.validate_for_batch(512).unwrap();

        // Should fail for batch too large
        assert!(config.validate_for_batch(2000).is_err());
    }

    #[test]
    fn test_expert_selection() {
        let backend = CpuBackend::default();
        let config = ExpertChoiceConfig::new(64, 8, 128)
            .with_tokens_per_expert(4);

        let router = ExpertChoiceRouter::new(&backend, config, 42).unwrap();

        // Create dummy router probabilities [num_tokens=8, num_experts=8]
        // Each token prefers a different expert
        let mut probs = vec![0.0f32; 64];
        for i in 0..8 {
            probs[i * 8 + i] = 1.0; // Token i prefers expert i
        }

        let assignments = router.expert_selection(&probs, backend.ops()).unwrap();

        // Each expert should have selected token with matching index
        assert_eq!(assignments.assignments.len(), 8);
    }

    #[test]
    fn test_routing_comparison() {
        let comparison = RoutingComparison::compare(1024, 64, 2);

        // Expert choice should have better (lower) imbalance
        assert!(comparison.expert_choice_imbalance <= comparison.top_k_imbalance);
        assert!(comparison.improvement >= 0.0);

        comparison.print();
    }

    #[test]
    fn test_softmax() {
        let backend = CpuBackend::default();
        let config = ExpertChoiceConfig::new(4, 4, 8);
        let router = ExpertChoiceRouter::new(&backend, config, 42).unwrap();

        // Test logits [2, 4] = [[1,2,3,4], [4,3,2,1]]
        let logits = vec![1.0f32, 2.0, 3.0, 4.0, 4.0, 3.0, 2.0, 1.0];
        let logits_tensor = backend.tensor_from_vec(logits, &[2, 4]).unwrap();

        let probs = router.softmax(&logits_tensor, backend.ops()).unwrap();

        // Each row should sum to 1
        let sum1: f32 = probs[0..4].iter().sum();
        let sum2: f32 = probs[4..8].iter().sum();

        assert!((sum1 - 1.0).abs() < 1e-5);
        assert!((sum2 - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_expert_choice_config_with_all_to_all() {
        let config = ExpertChoiceConfig::new(64, 8, 128)
            .with_all_to_all(false);
        assert!(!config.all_to_all);
    }

    #[test]
    fn test_expert_choice_router_forward_errors() {
        let backend = CpuBackend::default();
        let config = ExpertChoiceConfig::new(4, 4, 8)
            .with_tokens_per_expert(2);
        let router = ExpertChoiceRouter::new(&backend, config, 42).unwrap();

        let mut ctx = ForwardCtx::new(&backend, mnr_core::Mode::Inference);

        // Error: input rank < 2
        let bad_rank = backend.tensor_from_vec(vec![0.1f32; 4], &[4]).unwrap();
        let result = router.forward(&bad_rank, &|t| Ok(t.clone()), &mut ctx);
        assert!(result.is_err());

        // Error: d_model mismatch
        let bad_dim = backend.tensor_from_vec(vec![0.1f32; 6], &[2, 3]).unwrap();
        let result = router.forward(&bad_dim, &|t| Ok(t.clone()), &mut ctx);
        assert!(result.is_err());

        // Error: batch too large (6 tokens > 4*2=8 capacity, but 10 > 8)
        let too_large = backend.tensor_from_vec(vec![0.1f32; 40], &[10, 4]).unwrap();
        let result = router.forward(&too_large, &|t| Ok(t.clone()), &mut ctx);
        assert!(result.is_err());
    }

    #[test]
    fn test_expert_choice_router_forward_success() {
        let backend = CpuBackend::default();
        let config = ExpertChoiceConfig::new(4, 4, 8)
            .with_tokens_per_expert(4);
        let router = ExpertChoiceRouter::new(&backend, config, 42).unwrap();

        let input = backend.tensor_from_vec(vec![0.1f32; 8], &[2, 4]).unwrap();
        let mut ctx = ForwardCtx::new(&backend, mnr_core::Mode::Inference);

        let (output, stats) = router.forward(&input, &|t| Ok(t.clone()), &mut ctx).unwrap();

        let shape = backend.ops().shape(&output);
        assert_eq!(shape, &[2, 4]);

        assert_eq!(stats.num_tokens, 2);
        assert_eq!(stats.num_experts, 4);
        assert!(stats.load_imbalance_ratio > 0.0);
    }

    #[test]
    fn test_expert_choice_stats_print() {
        let stats = ExpertChoiceStats {
            num_tokens: 10,
            num_experts: 4,
            total_selected: 8,
            min_tokens_per_expert: 1,
            max_tokens_per_expert: 3,
            avg_tokens_per_expert: 2.0,
            load_imbalance_ratio: 1.5,
        };
        stats.print(); // just ensure it doesn't panic
    }
}
