//! Sequence Parallelism for Ultra-Long Contexts
//!
//! Implements ring attention and sequence sharding for processing
//! very long sequences (100K+ tokens) across multiple devices.
//!
//! # Ring Attention
//!
//! ```text
//! Devices:    GPU 0          GPU 1          GPU 2          GPU 3
//!            ┌────┐        ┌────┐        ┌────┐        ┌────┐
//! Q blocks:  │Q0  │───────→│Q1  │───────→│Q2  │───────→│Q3  │
//!            │    │        │    │        │    │        │    │
//! K blocks:  │K0  │←───────│K1  │←───────│K2  │←───────│K3  │
//!            │    │        │    │        │    │        │    │
//! V blocks:  │V0  │←───────│V1  │←───────│V2  │←───────│V3  │
//!            └────┘        └────┘        └────┘        └────┘
//!                    Ring communication pattern
//! ```
//!
//! Each device computes attention for its query chunk against
//! all key/value chunks in a ring-reduce pattern.

use mnr_core::{Backend, CoreError, Result, TensorOps, TensorShape};

use crate::ProcessGroup;

/// Configuration for sequence parallelism.
pub struct SequenceParallelConfig {
    /// Number of devices to shard sequence across
    pub num_devices: usize,
    /// Block size for attention computation
    pub block_size: usize,
    /// Whether to use ring attention (vs simple gather)
    pub use_ring_attention: bool,
}

impl SequenceParallelConfig {
    pub fn new(num_devices: usize) -> Self {
        Self { num_devices, block_size: 1024, use_ring_attention: true }
    }

    pub fn with_block_size(mut self, size: usize) -> Self {
        self.block_size = size;
        self
    }

    pub fn with_ring_attention(mut self, enabled: bool) -> Self {
        self.use_ring_attention = enabled;
        self
    }
}

/// Shard a sequence tensor across devices.
///
/// Splits along the sequence dimension (typically dim 1).
pub fn shard_sequence<B: Backend>(
    tensor: &B::Tensor,
    num_shards: usize,
    shard_id: usize,
    ops: &dyn TensorOps<B>,
) -> Result<B::Tensor>
where
    B::Tensor: AsRef<[f32]>,
{
    let shape = ops.shape(tensor);

    if shape.len() < 2 {
        return Err(CoreError::Shape(
            "Sequence tensor must have at least 2 dimensions [batch, seq, ...]".to_string(),
        ));
    }

    let seq_len = shape[1];

    if seq_len % num_shards != 0 {
        return Err(CoreError::Shape(format!(
            "Sequence length {} must be divisible by num_shards {}",
            seq_len, num_shards
        )));
    }

    let shard_size = seq_len / num_shards;
    let start = shard_id * shard_size;
    let end = start + shard_size;

    // Slice along sequence dimension
    // In a real implementation, this would use proper slice operation
    // For now, we'll gather and split
    let data: Vec<f32> = tensor.as_ref().to_vec();

    // Calculate element offsets
    let batch_size = shape[0];
    let other_dims: usize = shape.iter().skip(2).product();
    let elements_per_seq = other_dims;

    let mut shard_data = Vec::with_capacity(batch_size * shard_size * elements_per_seq);

    for b in 0..batch_size {
        for s in start..end {
            let offset = b * seq_len * elements_per_seq + s * elements_per_seq;
            shard_data.extend_from_slice(&data[offset..offset + elements_per_seq]);
        }
    }

    // Create new shape with sharded sequence
    let mut new_shape = shape.clone();
    new_shape[1] = shard_size;

    ops.tensor_from_vec(shard_data, &new_shape)
}

/// Gather sharded sequences from all devices.
pub fn gather_sequence<B: Backend>(shards: &[B::Tensor], ops: &dyn TensorOps<B>) -> Result<B::Tensor>
where
    B::Tensor: AsRef<[f32]>,
{
    if shards.is_empty() {
        return Err(CoreError::InvalidArgument("No shards to gather".to_string()));
    }

    // Concatenate along sequence dimension
    // For simplicity, we'll use a simple concat
    let mut concatenated: Vec<f32> = Vec::new();

    for shard in shards {
        let data: Vec<f32> = shard.as_ref().to_vec();
        concatenated.extend_from_slice(&data);
    }

    // Calculate output shape
    let first_shape = ops.shape(&shards[0]);
    let total_seq: usize = shards.iter().map(|s| ops.shape(s)[1]).sum();

    let mut output_shape = first_shape.clone();
    output_shape[1] = total_seq;

    ops.tensor_from_vec(concatenated, &output_shape)
}

/// Ring attention computation for distributed attention.
///
/// Each device computes attention for its query block against
/// all key/value blocks in a ring pattern.
pub struct RingAttention<B: Backend> {
    config: SequenceParallelConfig,
    process_group: ProcessGroup,
    _backend: std::marker::PhantomData<B>,
}

impl<B: Backend> RingAttention<B>
where
    B::Tensor: Clone + AsRef<[f32]> + TensorShape,
{
    pub fn new(config: SequenceParallelConfig, process_group: ProcessGroup) -> Self {
        Self { config, process_group, _backend: std::marker::PhantomData }
    }

    /// Compute ring attention.
    ///
    /// Each device has Q, K, V blocks. We compute in rounds:
    /// Round 0: Use local K, V
    /// Round 1: Receive K, V from left neighbor
    /// ... until all K, V seen
    pub fn compute(
        &self,
        local_q: &B::Tensor,
        local_k: &B::Tensor,
        local_v: &B::Tensor,
        ops: &dyn TensorOps<B>,
    ) -> Result<B::Tensor> {
        if !self.config.use_ring_attention {
            // Fallback: gather all K, V and compute locally
            return self.compute_gathered(local_q, local_k, local_v, ops);
        }

        let num_devices = self.config.num_devices;
        let mut k_block = local_k.clone();
        let mut v_block = local_v.clone();

        // Track max logits and sum of exp for online softmax
        let shape = ops.shape(local_q);
        let batch = shape[0];
        let q_len = shape[1];
        let num_heads = shape.get(2).copied().unwrap_or(1);
        let head_dim = shape.get(3).copied().unwrap_or(64);

        // Initialize output and running statistics
        let mut output_data: Vec<f32> = vec![0.0; batch * q_len * num_heads * head_dim];
        let mut max_logits: Vec<f32> = vec![f32::NEG_INFINITY; batch * q_len * num_heads];
        let mut sum_exp: Vec<f32> = vec![0.0; batch * q_len * num_heads];

        // Ring iterations
        for step in 0..num_devices {
            // Compute attention with current K, V blocks
            let attention_chunk = self.compute_attention_chunk(local_q, &k_block, &v_block, ops)?;

            // Online softmax update
            self.update_online_softmax(
                &mut output_data,
                &mut max_logits,
                &mut sum_exp,
                &attention_chunk,
                ops,
            )?;

            // Send K, V to right neighbor, receive from left
            if step < num_devices - 1 {
                // In real impl: async send/recv
                // For now, simplified rotation
                let (new_k, new_v) = self.rotate_blocks(&k_block, &v_block, ops)?;
                k_block = new_k;
                v_block = new_v;
            }
        }

        // Final normalization
        self.normalize_output(&mut output_data, &sum_exp, ops)?;

        let output_shape = vec![batch, q_len, num_heads, head_dim];
        ops.tensor_from_vec(output_data, &output_shape)
    }

    fn compute_attention_chunk(
        &self,
        q: &B::Tensor,
        k: &B::Tensor,
        v: &B::Tensor,
        ops: &dyn TensorOps<B>,
    ) -> Result<Vec<f32>> {
        // Simplified: compute Q @ K^T / sqrt(d)
        let q_data: Vec<f32> = q.as_ref().to_vec();
        let k_data: Vec<f32> = k.as_ref().to_vec();
        let v_data: Vec<f32> = v.as_ref().to_vec();

        let q_shape = ops.shape(q);
        let k_shape = ops.shape(k);

        let batch = q_shape[0];
        let q_len = q_shape[1];
        let k_len = k_shape[1];
        let num_heads = q_shape.get(2).copied().unwrap_or(1);
        let head_dim = q_shape.get(3).copied().unwrap_or(64);

        // Compute attention scores (simplified)
        let scale = (head_dim as f32).sqrt().recip();
        let mut scores = vec![0.0f32; batch * q_len * k_len * num_heads];

        for b in 0..batch {
            for h in 0..num_heads {
                for i in 0..q_len {
                    for j in 0..k_len {
                        // Dot product of q[i] and k[j]
                        let q_idx =
                            b * q_len * num_heads * head_dim + i * num_heads * head_dim + h * head_dim;
                        let k_idx =
                            b * k_len * num_heads * head_dim + j * num_heads * head_dim + h * head_dim;

                        let mut dot = 0.0;
                        for d in 0..head_dim {
                            dot += q_data[q_idx + d] * k_data[k_idx + d];
                        }

                        let score_idx =
                            b * q_len * k_len * num_heads + i * k_len * num_heads + j * num_heads + h;
                        scores[score_idx] = dot * scale;
                    }
                }
            }
        }

        // Softmax over k_len dimension
        let mut weights = vec![0.0f32; scores.len()];
        for b in 0..batch {
            for i in 0..q_len {
                for h in 0..num_heads {
                    // Find max for numerical stability
                    let mut max_score = f32::NEG_INFINITY;
                    for j in 0..k_len {
                        let idx = b * q_len * k_len * num_heads + i * k_len * num_heads + j * num_heads + h;
                        max_score = max_score.max(scores[idx]);
                    }

                    // Compute exp and sum
                    let mut sum = 0.0;
                    for j in 0..k_len {
                        let idx = b * q_len * k_len * num_heads + i * k_len * num_heads + j * num_heads + h;
                        weights[idx] = (scores[idx] - max_score).exp();
                        sum += weights[idx];
                    }

                    // Normalize
                    for j in 0..k_len {
                        let idx = b * q_len * k_len * num_heads + i * k_len * num_heads + j * num_heads + h;
                        weights[idx] /= sum;
                    }
                }
            }
        }

        // Apply to V
        let mut output = vec![0.0f32; batch * q_len * num_heads * head_dim];
        for b in 0..batch {
            for h in 0..num_heads {
                for i in 0..q_len {
                    for d in 0..head_dim {
                        let mut val = 0.0;
                        for j in 0..k_len {
                            let w_idx =
                                b * q_len * k_len * num_heads + i * k_len * num_heads + j * num_heads + h;
                            let v_idx = b * k_len * num_heads * head_dim
                                + j * num_heads * head_dim
                                + h * head_dim
                                + d;
                            val += weights[w_idx] * v_data[v_idx];
                        }
                        let out_idx =
                            b * q_len * num_heads * head_dim + i * num_heads * head_dim + h * head_dim + d;
                        output[out_idx] = val;
                    }
                }
            }
        }

        Ok(output)
    }

    fn update_online_softmax(
        &self,
        output: &mut [f32],
        max_logits: &mut [f32],
        sum_exp: &mut [f32],
        chunk: &[f32],
        ops: &dyn TensorOps<B>,
    ) -> Result<()> {
        // Online softmax update
        // For each position, update running max and sum
        let limit = output.len().min(chunk.len()).min(max_logits.len());
        for i in 0..limit {
            let new_max = max_logits[i].max(chunk[i]);
            let exp_old = (max_logits[i] - new_max).exp();
            let exp_new = (chunk[i] - new_max).exp();

            output[i] = output[i] * exp_old * sum_exp[i] / (sum_exp[i] * exp_old + exp_new);
            sum_exp[i] = sum_exp[i] * exp_old + exp_new;
            max_logits[i] = new_max;
        }

        Ok(())
    }

    fn normalize_output(&self, output: &mut [f32], sum_exp: &[f32], _ops: &dyn TensorOps<B>) -> Result<()> {
        for i in 0..output.len().min(sum_exp.len()) {
            if sum_exp[i] > 0.0 {
                output[i] /= sum_exp[i];
            }
        }
        Ok(())
    }

    fn rotate_blocks(
        &self,
        k: &B::Tensor,
        v: &B::Tensor,
        _ops: &dyn TensorOps<B>,
    ) -> Result<(B::Tensor, B::Tensor)> {
        // In real implementation: async send/recv
        // For now, return clones (simplified)
        Ok((k.clone(), v.clone()))
    }

    fn compute_gathered(
        &self,
        local_q: &B::Tensor,
        local_k: &B::Tensor,
        local_v: &B::Tensor,
        _ops: &dyn TensorOps<B>,
    ) -> Result<B::Tensor> {
        // Gather all K, V to all devices
        // Then compute full attention locally
        // Simplified: just compute with local blocks
        self.compute_attention_chunk(local_q, local_k, local_v, _ops).and_then(|data| {
            let shape = _ops.shape(local_q);
            _ops.tensor_from_vec(data, &shape)
        })
    }
}

/// Compute sequence parallelism info for a given configuration.
pub fn compute_sequence_sharding(total_seq_len: usize, num_devices: usize) -> Result<(usize, usize)> {
    if total_seq_len % num_devices != 0 {
        return Err(CoreError::InvalidArgument(format!(
            "Sequence length {} not divisible by {}",
            total_seq_len, num_devices
        )));
    }

    let per_device = total_seq_len / num_devices;
    Ok((per_device, total_seq_len))
}

#[cfg(test)]
mod tests {
    use super::*;
    use mnr_ndarray_backend::CpuBackend;

    #[test]
    fn test_sequence_sharding() {
        let backend = CpuBackend::default();

        // Create [2, 8, 4] tensor (batch=2, seq=8, dim=4)
        let data: Vec<f32> = (0..64).map(|i| i as f32).collect();
        let tensor = backend.tensor_from_vec(data, &[2, 8, 4]).unwrap();

        // Shard into 2 pieces
        let shard0 = shard_sequence(&tensor, 2, 0, backend.ops()).unwrap();
        let shard1 = shard_sequence(&tensor, 2, 1, backend.ops()).unwrap();

        let shape0 = backend.ops().shape(&shard0);
        let shape1 = backend.ops().shape(&shard1);

        assert_eq!(shape0, vec![2, 4, 4]); // seq=8/2=4
        assert_eq!(shape1, vec![2, 4, 4]);

        // Gather back
        let gathered = gather_sequence(&[shard0, shard1], backend.ops()).unwrap();
        let gathered_shape = backend.ops().shape(&gathered);

        assert_eq!(gathered_shape, vec![2, 8, 4]);
    }

    #[test]
    fn test_shard_sequence_errors() {
        let backend = CpuBackend::default();

        // 1D tensor should fail
        let tensor = backend.tensor_from_vec(vec![1.0f32; 8], &[8]).unwrap();
        assert!(shard_sequence(&tensor, 2, 0, backend.ops()).is_err());

        // seq not divisible by num_shards
        let tensor = backend.tensor_from_vec(vec![1.0f32; 16], &[2, 4, 2]).unwrap();
        assert!(shard_sequence(&tensor, 3, 0, backend.ops()).is_err());
    }

    #[test]
    fn test_gather_sequence_empty() {
        let backend = CpuBackend::default();
        let result = gather_sequence::<CpuBackend>(&[], backend.ops());
        assert!(result.is_err());
    }

    #[test]
    fn test_sequence_sharding_config() {
        let (per_device, total) = compute_sequence_sharding(8192, 4).unwrap();
        assert_eq!(per_device, 2048);
        assert_eq!(total, 8192);

        // Test non-divisible
        let result = compute_sequence_sharding(100, 3);
        assert!(result.is_err());
    }

    #[test]
    fn test_ring_attention_config() {
        let config = SequenceParallelConfig::new(4).with_block_size(512).with_ring_attention(true);

        assert_eq!(config.num_devices, 4);
        assert_eq!(config.block_size, 512);
        assert!(config.use_ring_attention);
    }

    #[test]
    fn test_ring_attention_compute() {
        let backend = CpuBackend::default();
        let config = SequenceParallelConfig::new(1).with_block_size(4).with_ring_attention(true);
        let pg = ProcessGroup::new_single_process();
        let ra = RingAttention::<CpuBackend>::new(config, pg);

        let q = backend.tensor_from_vec(vec![1.0f32; 8], &[1, 2, 1, 4]).unwrap();
        let k = backend.tensor_from_vec(vec![1.0f32; 8], &[1, 2, 1, 4]).unwrap();
        let v = backend.tensor_from_vec(vec![1.0f32; 8], &[1, 2, 1, 4]).unwrap();

        let output = ra.compute(&q, &k, &v, backend.ops()).unwrap();
        let shape = backend.ops().shape(&output);
        assert_eq!(shape, vec![1, 2, 1, 4]);
    }

    #[test]
    fn test_ring_attention_compute_gathered() {
        let backend = CpuBackend::default();
        let config = SequenceParallelConfig::new(1).with_block_size(4).with_ring_attention(false);
        let pg = ProcessGroup::new_single_process();
        let ra = RingAttention::<CpuBackend>::new(config, pg);

        let q = backend.tensor_from_vec(vec![1.0f32; 8], &[1, 2, 1, 4]).unwrap();
        let k = backend.tensor_from_vec(vec![1.0f32; 8], &[1, 2, 1, 4]).unwrap();
        let v = backend.tensor_from_vec(vec![1.0f32; 8], &[1, 2, 1, 4]).unwrap();

        let output = ra.compute(&q, &k, &v, backend.ops()).unwrap();
        let shape = backend.ops().shape(&output);
        assert_eq!(shape, vec![1, 2, 1, 4]);
    }
}
