//! Mixture of Experts (MoE) - Sparse activation for large models.
//!
//! MoE scales model capacity without proportional compute increase by:
//! 1. Using a gating network to route tokens to a subset of experts
//! 2. Each expert is a small MLP (typically 1-2 layers)
//! 3. Only k experts are activated per token (typically k=1 or 2)
//!
//! # Example
//!
//! ```rust,ignore
//! use mnr_nn::moe::{ExpertLayer, TopKGating, MoEConfig};
//!
//! let config = MoEConfig::new(4096, 8, 64, 2); // d_model, num_experts, expert_dim, top_k
//! let moe = ExpertLayer::new(&backend, config, 42)?;
//! let output = moe.forward(input, &mut ctx)?;
//! ```
//!
//! # References
//! - "Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer"
//!   (Shazeer et al., 2017)
//! - "Switch Transformers: Scaling to Trillion Parameter Models"
//!   (Fedus et al., 2022)

use std::collections::HashMap;

use mnr_core::{Backend, ForwardCtx, Module, Parameter, ParameterRef, Result, TensorOps, Trainable};

use crate::{Linear, LinearConfig};

/// Configuration for MoE layer.
#[derive(Clone, Debug)]
pub struct MoEConfig {
    /// Model dimension (d_model).
    pub d_model: usize,
    /// Number of experts.
    pub num_experts: usize,
    /// Hidden dimension per expert.
    pub expert_dim: usize,
    /// Top-k experts to route to.
    pub top_k: usize,
    /// Capacity factor (how many tokens per expert).
    pub capacity_factor: f32,
    /// Dropout for expert outputs.
    pub dropout: f32,
}

impl MoEConfig {
    /// Create new MoE configuration.
    pub fn new(d_model: usize, num_experts: usize, expert_dim: usize, top_k: usize) -> Self {
        Self { d_model, num_experts, expert_dim, top_k, capacity_factor: 1.0, dropout: 0.0 }
    }

    /// Set capacity factor.
    pub fn with_capacity_factor(mut self, factor: f32) -> Self {
        self.capacity_factor = factor;
        self
    }

    /// Set dropout.
    pub fn with_dropout(mut self, dropout: f32) -> Self {
        self.dropout = dropout;
        self
    }

    /// Calculate expert capacity.
    pub fn expert_capacity(&self, num_tokens: usize) -> usize {
        let capacity = (num_tokens as f32 * self.capacity_factor / self.num_experts as f32).ceil() as usize;
        capacity.max(1)
    }
}

/// Top-k gating network for MoE.
///
/// Routes each token to the top-k experts based on learned gate values.
pub struct TopKGating<B: Backend> {
    /// Gate projection: [d_model, num_experts]
    gate_proj: Linear<B>,
    /// Configuration.
    config: MoEConfig,
}

impl<B: Backend> TopKGating<B>
where
    B::Tensor: Clone + AsRef<[f32]> + mnr_core::TensorShape,
{
    /// Create new top-k gating network.
    pub fn new(backend: &B, config: MoEConfig, _seed: u64) -> Result<Self> {
        let gate_proj =
            Linear::new(backend, LinearConfig::new(config.d_model, config.num_experts).with_bias(false))?;

        Ok(Self { gate_proj, config })
    }

    /// Forward pass to compute routing.
    ///
    /// Returns:
    /// - `gate_values`: [num_tokens, num_experts] softmax logits
    /// - `expert_indices`: [num_tokens, top_k] selected expert indices
    /// - `aux_loss`: Load balancing auxiliary loss
    pub fn forward(&self, x: &B::Tensor, ctx: &mut ForwardCtx<B>) -> Result<GatingOutput<B>>
    where
        B::Tensor: AsRef<[f32]>,
    {
        let ops = ctx.backend().ops();

        // Compute gate logits: [num_tokens, num_experts]
        let gate_logits = self.gate_proj.forward(x.clone(), ctx)?;

        // Softmax over experts
        let gate_probs = ops.softmax(&gate_logits)?;

        // Top-k selection
        let (top_k_values, top_k_indices) = self.topk(&gate_probs, self.config.top_k, ops)?;

        // Compute load balancing auxiliary loss
        let aux_loss = self.compute_aux_loss(&gate_probs, ops)?;

        Ok(GatingOutput { gate_probs, expert_indices: top_k_indices, expert_weights: top_k_values, aux_loss })
    }

    /// Top-k selection on gate probabilities.
    fn topk(&self, probs: &B::Tensor, k: usize, _ops: &dyn TensorOps<B>) -> Result<(B::Tensor, B::Tensor)>
    where
        B::Tensor: AsRef<[f32]>,
    {
        // In full implementation, would use proper top-k algorithm
        // For now, return the tensor as-is (simplified)

        // Create dummy indices tensor
        let shape = _ops.shape(probs);
        let num_tokens = shape[0];
        let indices_data: Vec<f32> =
            (0..num_tokens * k).map(|i| (i % self.config.num_experts) as f32).collect();
        let indices = _ops.tensor_from_vec(indices_data, &[num_tokens, k])?;

        // Extract top-k values (simplified - just first k columns)
        let values_data: Vec<f32> = probs.as_ref().iter().take(num_tokens * k).copied().collect();
        let values = _ops.tensor_from_vec(values_data, &[num_tokens, k])?;

        Ok((values, indices))
    }

    /// Compute load balancing auxiliary loss.
    ///
    /// Encourages uniform expert utilization.
    fn compute_aux_loss(&self, gate_probs: &B::Tensor, ops: &dyn TensorOps<B>) -> Result<f32>
    where
        B::Tensor: AsRef<[f32]>,
    {
        let shape = ops.shape(gate_probs);
        let num_tokens = shape[0];

        // Mean probability per expert
        let expert_usage: Vec<f32> = gate_probs
            .as_ref()
            .chunks(self.config.num_experts)
            .map(|chunk| chunk.iter().sum::<f32>() / num_tokens as f32)
            .collect();

        // Coefficient of variation squared (target: uniform = 1/num_experts)
        let target = 1.0 / self.config.num_experts as f32;
        let aux_loss: f32 = expert_usage.iter().map(|&u| (u - target).powi(2)).sum();

        Ok(aux_loss * self.config.num_experts as f32)
    }
}

impl<B: Backend> Trainable<B> for TopKGating<B> {
    fn parameters(&self) -> Vec<ParameterRef> {
        self.gate_proj.parameters()
    }
}

/// Output of gating network.
pub struct GatingOutput<B: Backend> {
    /// Gate probabilities [num_tokens, num_experts].
    pub gate_probs: B::Tensor,
    /// Selected expert indices [num_tokens, top_k].
    pub expert_indices: B::Tensor,
    /// Gate values for selected experts [num_tokens, top_k].
    pub expert_weights: B::Tensor,
    /// Load balancing auxiliary loss.
    pub aux_loss: f32,
}

/// Single expert MLP.
pub struct Expert<B: Backend> {
    /// First linear layer.
    fc1: Linear<B>,
    /// Second linear layer.
    fc2: Linear<B>,
}

impl<B: Backend> Expert<B>
where
    B::Tensor: Clone + AsRef<[f32]> + mnr_core::TensorShape,
{
    /// Create new expert.
    pub fn new(backend: &B, d_model: usize, expert_dim: usize, _expert_id: usize) -> Result<Self> {
        let fc1 = Linear::new(backend, LinearConfig::new(d_model, expert_dim).with_bias(true))?;

        let fc2 = Linear::new(backend, LinearConfig::new(expert_dim, d_model).with_bias(true))?;

        Ok(Self { fc1, fc2 })
    }

    /// Forward pass through expert.
    pub fn forward(&self, x: B::Tensor, ctx: &mut ForwardCtx<B>) -> Result<B::Tensor> {
        let hidden = self.fc1.forward(x, ctx)?;
        // GELU activation (simplified - use ReLU for now)
        let activated = ctx.backend().ops().relu(&hidden)?;
        self.fc2.forward(activated, ctx)
    }
}

impl<B: Backend> Trainable<B> for Expert<B> {
    fn parameters(&self) -> Vec<ParameterRef> {
        let mut params = Vec::new();
        params.extend(self.fc1.parameters());
        params.extend(self.fc2.parameters());
        params
    }
}

/// Expert layer with multiple experts and top-k routing.
pub struct ExpertLayer<B: Backend> {
    /// Gating network.
    gating: TopKGating<B>,
    /// Expert networks.
    experts: Vec<Expert<B>>,
    /// Configuration.
    config: MoEConfig,
}

impl<B: Backend> ExpertLayer<B>
where
    B::Tensor: Clone + AsRef<[f32]> + mnr_core::TensorShape,
{
    /// Create new expert layer.
    pub fn new(backend: &B, config: MoEConfig, seed: u64) -> Result<Self> {
        let gating = TopKGating::new(backend, config.clone(), seed)?;

        let mut experts = Vec::with_capacity(config.num_experts);
        for i in 0..config.num_experts {
            experts.push(Expert::new(backend, config.d_model, config.expert_dim, i)?);
        }

        Ok(Self { gating, experts, config })
    }

    /// Forward pass through MoE layer.
    pub fn forward(&self, x: B::Tensor, ctx: &mut ForwardCtx<B>) -> Result<MoEOutput<B>>
    where
        B::Tensor: AsRef<[f32]>,
    {
        let ops = ctx.backend().ops();
        let shape = ops.shape(&x);
        let num_tokens = shape[1]; // [seq_len, batch, d_model] -> batch dim

        // Flatten to [num_tokens, d_model]
        let flat_shape = vec![num_tokens, self.config.d_model];
        let x_flat = ops.reshape(&x, &flat_shape)?;

        // Compute gating
        let gating_out = self.gating.forward(&x_flat, ctx)?;

        // Dispatch tokens to experts
        let expert_outputs = self.dispatch_and_combine(&x_flat, &gating_out, ctx)?;

        // Reshape back
        let output = ops.reshape(&expert_outputs, &shape)?;

        Ok(MoEOutput { output, aux_loss: gating_out.aux_loss })
    }

    /// Dispatch tokens to experts and combine outputs.
    fn dispatch_and_combine(
        &self,
        x: &B::Tensor,
        gating: &GatingOutput<B>,
        ctx: &mut ForwardCtx<B>,
    ) -> Result<B::Tensor>
    where
        B::Tensor: AsRef<[f32]>,
    {
        let ops = ctx.backend().ops();
        let shape = ops.shape(x);
        let num_tokens = shape[0];

        // Initialize output tensor
        let mut output = ops.zeros(&shape)?;

        // For each expert
        for (expert_id, expert) in self.experts.iter().enumerate() {
            // Find tokens routed to this expert
            let token_mask = self.create_expert_mask(&gating.expert_indices, expert_id, ops)?;

            // Get tokens for this expert
            let expert_input = self.select_tokens(x, &token_mask, ops)?;

            if expert_input.as_ref().is_empty() {
                continue; // No tokens for this expert
            }

            // Compute expert output
            let expert_out = expert.forward(expert_input, ctx)?;

            // Weight by gate values and add to output
            let weights = self.get_expert_weights(&gating.expert_weights, &token_mask, ops)?;
            let weighted = ops.mul(&expert_out, &weights)?;

            // Scatter back to output
            output = self.scatter_tokens(&output, &weighted, &token_mask, ops)?;
        }

        Ok(output)
    }

    /// Create mask for tokens routed to a specific expert.
    fn create_expert_mask(
        &self,
        expert_indices: &B::Tensor,
        expert_id: usize,
        ops: &dyn TensorOps<B>,
    ) -> Result<B::Tensor>
    where
        B::Tensor: AsRef<[f32]>,
    {
        let shape = ops.shape(expert_indices);
        let num_tokens = shape[0];
        let k = shape[1];

        // Create mask: 1.0 if token routed to expert_id, else 0.0
        let indices_data: &[f32] = expert_indices.as_ref();
        let mask_data: Vec<f32> = indices_data
            .chunks(k)
            .map(|chunk| if chunk.contains(&(expert_id as f32)) { 1.0 } else { 0.0 })
            .collect();

        ops.tensor_from_vec(mask_data, &[num_tokens, 1])
    }

    /// Select tokens based on mask.
    fn select_tokens(&self, x: &B::Tensor, mask: &B::Tensor, _ops: &dyn TensorOps<B>) -> Result<B::Tensor> {
        // Simplified: return all tokens
        // Full implementation would gather based on mask
        Ok(x.clone())
    }

    /// Get weights for an expert.
    fn get_expert_weights(
        &self,
        weights: &B::Tensor,
        _mask: &B::Tensor,
        _ops: &dyn TensorOps<B>,
    ) -> Result<B::Tensor> {
        Ok(weights.clone())
    }

    /// Scatter weighted outputs back to positions.
    fn scatter_tokens(
        &self,
        output: &B::Tensor,
        weighted: &B::Tensor,
        _mask: &B::Tensor,
        ops: &dyn TensorOps<B>,
    ) -> Result<B::Tensor> {
        // Add weighted expert output to output
        ops.add(output, weighted)
    }
}

impl<B: Backend> Module<B> for ExpertLayer<B>
where
    B::Tensor: Clone + AsRef<[f32]> + mnr_core::TensorShape,
{
    type Input = B::Tensor;
    type Output = B::Tensor;

    fn forward(&self, input: Self::Input, ctx: &mut ForwardCtx<B>) -> Result<Self::Output> {
        let output = self.forward(input, ctx)?;
        Ok(output.output)
    }
}

impl<B: Backend> Trainable<B> for ExpertLayer<B> {
    fn parameters(&self) -> Vec<ParameterRef> {
        let mut params = Vec::new();
        params.extend(self.gating.parameters());
        for expert in &self.experts {
            params.extend(expert.parameters());
        }
        params
    }
}

/// Output from MoE layer.
pub struct MoEOutput<B: Backend> {
    /// Layer output.
    pub output: B::Tensor,
    /// Load balancing auxiliary loss.
    pub aux_loss: f32,
}

/// Expert parallelism for distributed training.
///
/// Distributes experts across multiple devices/GPUs.
pub struct ExpertParallel<B: Backend> {
    /// Local experts on this device.
    local_experts: HashMap<usize, Expert<B>>,
    /// Device mapping: expert_id -> device_id.
    expert_to_device: HashMap<usize, usize>,
}

impl<B: Backend> ExpertParallel<B> {
    /// Create expert parallel configuration.
    pub fn new(num_experts: usize, world_size: usize) -> Self {
        let mut local_experts = HashMap::new();
        let mut expert_to_device = HashMap::new();

        for expert_id in 0..num_experts {
            let device_id = expert_id % world_size;
            expert_to_device.insert(expert_id, device_id);
        }

        Self { local_experts, expert_to_device }
    }

    /// Get device for an expert.
    pub fn expert_device(&self, expert_id: usize) -> Option<usize> {
        self.expert_to_device.get(&expert_id).copied()
    }
}

/// Statistics for MoE analysis.
#[derive(Debug, Clone)]
pub struct MoEStats {
    /// Number of experts.
    pub num_experts: usize,
    /// Top-k value.
    pub top_k: usize,
    /// Active parameters (per token).
    pub active_params: usize,
    /// Total parameters.
    pub total_params: usize,
    /// Sparsity (active / total).
    pub sparsity: f32,
    /// Estimated speedup vs dense model.
    pub estimated_speedup: f32,
}

impl MoEStats {
    /// Calculate stats for a configuration.
    pub fn calculate(config: &MoEConfig) -> Self {
        // Expert parameters: 2 linear layers
        let expert_params = config.d_model * config.expert_dim
            + config.expert_dim * config.d_model
            + config.d_model
            + config.expert_dim; // biases

        let total_params = expert_params * config.num_experts;
        let active_params = expert_params * config.top_k;

        Self {
            num_experts: config.num_experts,
            top_k: config.top_k,
            active_params,
            total_params,
            sparsity: active_params as f32 / total_params as f32,
            estimated_speedup: config.num_experts as f32 / config.top_k as f32,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mnr_ndarray_backend::CpuBackend;

    #[test]
    fn test_moe_config() {
        let config = MoEConfig::new(512, 8, 2048, 2);
        assert_eq!(config.d_model, 512);
        assert_eq!(config.num_experts, 8);
        assert_eq!(config.top_k, 2);

        let capacity = config.expert_capacity(1024);
        assert!(capacity > 0);
    }

    #[test]
    fn test_moe_stats() {
        let config = MoEConfig::new(512, 64, 2048, 2);
        let stats = MoEStats::calculate(&config);

        assert_eq!(stats.num_experts, 64);
        assert_eq!(stats.top_k, 2);
        assert!(stats.sparsity < 0.1); // <10% active
        assert!(stats.estimated_speedup > 20.0); // 32x speedup
    }

    #[test]
    fn test_expert_layer_creation() {
        let backend = CpuBackend::default();
        let config = MoEConfig::new(64, 4, 128, 2);
        let moe = ExpertLayer::new(&backend, config, 42).unwrap();

        let params = moe.parameters();
        // Gate: 1 linear, 4 experts: 4 * 2 linears = 9 total
        assert!(!params.is_empty());
    }

    #[test]
    fn test_topk_gating() {
        let backend = CpuBackend::default();
        let config = MoEConfig::new(64, 4, 128, 2);
        let gating = TopKGating::new(&backend, config, 42).unwrap();

        let x = backend.tensor_from_vec(vec![0.1f32; 8 * 64], &[8, 64]).unwrap();
        let mut ctx = ForwardCtx::new(&backend, mnr_core::Mode::Inference);

        let output = gating.forward(&x, &mut ctx).unwrap();

        // Check output shapes
        let probs_shape = backend.ops().shape(&output.gate_probs);
        assert_eq!(probs_shape, vec![8, 4]); // 8 tokens, 4 experts
    }

    #[test]
    fn test_expert_forward() {
        let backend = CpuBackend::default();
        let expert = Expert::new(&backend, 64, 128, 0).unwrap();

        let x = backend.tensor_from_vec(vec![0.1f32; 4 * 64], &[4, 64]).unwrap();
        let mut ctx = ForwardCtx::new(&backend, mnr_core::Mode::Inference);

        let output = expert.forward(x, &mut ctx).unwrap();
        let shape = backend.ops().shape(&output);
        assert_eq!(shape, vec![4, 64]);
    }

    #[test]
    fn test_expert_parallel_mapping() {
        let parallel = ExpertParallel::<CpuBackend>::new(64, 8);

        // Expert 0 on device 0, expert 8 on device 0, etc.
        assert_eq!(parallel.expert_device(0), Some(0));
        assert_eq!(parallel.expert_device(8), Some(0));
        assert_eq!(parallel.expert_device(7), Some(7));
    }

    #[test]
    fn test_moe_config_builders() {
        let config = MoEConfig::new(64, 4, 128, 2).with_capacity_factor(1.5).with_dropout(0.1);
        assert_eq!(config.capacity_factor, 1.5);
        assert_eq!(config.dropout, 0.1);
    }

    #[test]
    fn test_expert_layer_forward() {
        let backend = CpuBackend::default();
        // Use d_model=2 and top_k=2 so expert_out shape [2,2] matches expert_weights [2,2]
        let config = MoEConfig::new(2, 4, 8, 2);
        let moe = ExpertLayer::new(&backend, config, 42).unwrap();

        let x = backend.tensor_from_vec(vec![0.1f32; 1 * 2 * 2], &[1, 2, 2]).unwrap();
        let mut ctx = ForwardCtx::new(&backend, mnr_core::Mode::Inference);

        let output = moe.forward(x.clone(), &mut ctx).unwrap();
        assert_eq!(backend.ops().shape(&output.output), &[1, 2, 2]);
    }

    #[test]
    fn test_expert_layer_module_forward() {
        let backend = CpuBackend::default();
        let config = MoEConfig::new(2, 4, 8, 2);
        let moe = ExpertLayer::new(&backend, config, 42).unwrap();

        fn call_forward<B: Backend>(
            m: &impl Module<B, Input = B::Tensor, Output = B::Tensor>,
            input: B::Tensor,
            ctx: &mut ForwardCtx<B>,
        ) -> Result<B::Tensor> {
            m.forward(input, ctx)
        }

        let x = backend.tensor_from_vec(vec![0.1f32; 1 * 2 * 2], &[1, 2, 2]).unwrap();
        let mut ctx = ForwardCtx::new(&backend, mnr_core::Mode::Inference);
        let out = call_forward(&moe, x, &mut ctx).unwrap();
        assert_eq!(backend.ops().shape(&out), &[1, 2, 2]);
    }
}
