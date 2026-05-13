//! candle-core backend for Rustral.
//!
//! Provides both CPU and CUDA acceleration via the candle-core crate.
//! On NVIDIA hardware, this backend automatically uses CUDA when available
//! if compiled with the `cuda` feature.
//!
//! # Example
//!
//! ```rust,ignore
//! use rustral_candle_backend::CandleBackend;
//!
//! let backend = CandleBackend::cuda(0).unwrap_or_else(|_| CandleBackend::cpu());
//! let tensor = backend.tensor_from_vec(vec![1.0f32, 2.0, 3.0], &[3]).unwrap();
//! ```

use candle_core::{Device, IndexOp, Tensor};
use rustral_core::{
    AttentionOps, Backend, BackendCapabilities, ConvLayout, CoreError, FusionOps, Parameter, Result,
    TensorOps, TrainingDtype,
};
use thiserror::Error;

/// Errors specific to the candle backend.
#[derive(Debug, Error)]
pub enum CandleError {
    #[error("candle error: {0}")]
    Candle(#[from] candle_core::Error),

    #[error("CUDA not available: {0}")]
    CudaUnavailable(String),
}

/// Operation table for the candle backend.
#[derive(Clone, Debug)]
struct CandleOps {
    device: Device,
}

/// candle-core backend for Rustral.
#[derive(Clone, Debug)]
pub struct CandleBackend {
    device: Device,
    ops: CandleOps,
}

impl CandleBackend {
    /// Create a CPU backend.
    pub fn cpu() -> Self {
        let device = Device::Cpu;
        Self { device: device.clone(), ops: CandleOps { device } }
    }

    /// Create a CUDA backend on the given device index.
    pub fn cuda(_device_id: usize) -> Result<Self> {
        #[cfg(feature = "cuda")]
        {
            let device = Device::new_cuda(_device_id)
                .map_err(|e| CoreError::Backend(format!("CUDA init failed: {}", e)))?;
            Ok(Self { device: device.clone(), ops: CandleOps { device } })
        }
        #[cfg(not(feature = "cuda"))]
        {
            Err(CoreError::Backend("CUDA feature not enabled. Rebuild with --features cuda".to_string()))
        }
    }

    /// Create a Metal backend on the given device index (Apple Silicon / macOS).
    pub fn metal(_device_id: usize) -> Result<Self> {
        #[cfg(feature = "metal")]
        {
            let device = Device::new_metal(_device_id)
                .map_err(|e| CoreError::Backend(format!("Metal init failed: {}", e)))?;
            Ok(Self { device: device.clone(), ops: CandleOps { device } })
        }
        #[cfg(not(feature = "metal"))]
        {
            Err(CoreError::Backend("Metal feature not enabled. Rebuild with --features metal".to_string()))
        }
    }

    /// Create a backend, preferring CUDA if available.
    pub fn new() -> Self {
        #[cfg(feature = "cuda")]
        {
            if let Ok(dev) = Device::new_cuda(0) {
                return Self { device: dev.clone(), ops: CandleOps { device: dev } };
            }
        }
        Self::cpu()
    }

    /// Return the underlying candle device.
    pub fn candle_device(&self) -> &Device {
        &self.device
    }

    /// Create a tensor from host data.
    pub fn tensor_from_vec(&self, values: Vec<f32>, shape: &[usize]) -> Result<Tensor> {
        let tensor =
            Tensor::from_vec(values, shape, &self.device).map_err(|e| CoreError::Backend(e.to_string()))?;
        Ok(tensor)
    }

    /// Read tensor data back to host as a flat `Vec<f32>`.
    pub fn to_vec(&self, tensor: &Tensor) -> Vec<f32> {
        tensor.flatten_all().and_then(|t| t.to_vec1::<f32>()).unwrap_or_default()
    }
}

impl Default for CandleBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl Backend for CandleBackend {
    type Tensor = Tensor;
    type Device = Device;

    fn device(&self) -> Self::Device {
        self.device.clone()
    }

    fn ops(&self) -> &dyn TensorOps<Self> {
        &self.ops
    }

    fn fusion_ops(&self) -> Option<&dyn FusionOps<Self>> {
        Some(&self.ops)
    }

    fn attention_ops(&self) -> Option<&dyn AttentionOps<Self>> {
        Some(&self.ops)
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn capabilities(&self) -> BackendCapabilities {
        match &self.device {
            Device::Cuda(_) => BackendCapabilities {
                supports_fp16: true,
                supports_bf16: false, // Most RTX cards don't support BF16
                tensor_cores: true,
                optimal_batch_size: 32,
                optimal_chunk_size: 4096,
                max_allocation_size: usize::MAX,
                prefers_contiguous: true,
                supports_in_place: true,
                supports_mixed_precision: true,
                recommended_training_dtype: TrainingDtype::F16,
                supports_fast_fp16_tensor_cores: true,
                preferred_conv_layout: ConvLayout::NCHW, // CUDA prefers NCHW
                supports_strided_layouts: true,
                supports_packed_layouts: true,
            },
            Device::Metal(_) => BackendCapabilities {
                supports_fp16: true,
                supports_bf16: true,
                tensor_cores: true,
                optimal_batch_size: 16,
                optimal_chunk_size: 2048,
                max_allocation_size: usize::MAX,
                prefers_contiguous: true,
                supports_in_place: true,
                supports_mixed_precision: true,
                recommended_training_dtype: TrainingDtype::Bf16,
                supports_fast_fp16_tensor_cores: true,
                preferred_conv_layout: ConvLayout::NHWC, // Metal prefers NHWC
                supports_strided_layouts: true,
                supports_packed_layouts: true,
            },
            Device::Cpu => BackendCapabilities {
                supports_fp16: false,
                supports_bf16: false,
                tensor_cores: false,
                optimal_batch_size: 8,
                optimal_chunk_size: 1024,
                max_allocation_size: usize::MAX,
                prefers_contiguous: true,
                supports_in_place: false,
                supports_mixed_precision: false,
                recommended_training_dtype: TrainingDtype::F32,
                supports_fast_fp16_tensor_cores: false,
                preferred_conv_layout: ConvLayout::NCHW, // CPU default
                supports_strided_layouts: true,
                supports_packed_layouts: false,
            },
        }
    }

    fn normal_parameter(&self, name: &str, shape: &[usize], seed: u64, scale: f32) -> Result<Parameter<Self>>
    where
        Self: Sized,
    {
        use rand::rngs::StdRng;
        use rand::{Rng, SeedableRng};

        let mut rng = StdRng::seed_from_u64(seed);
        let values: Vec<f32> = if scale > 0.0 {
            (0..shape.iter().product::<usize>()).map(|_| rng.gen_range(-scale..scale)).collect()
        } else {
            vec![0.0; shape.iter().product()]
        };
        let tensor = self.tensor_from_vec(values, shape)?;
        Ok(Parameter::new(name, tensor))
    }

    fn parameter_from_vec(&self, name: &str, values: Vec<f32>, shape: &[usize]) -> Result<Parameter<Self>>
    where
        Self: Sized,
    {
        let tensor = self.tensor_from_vec(values, shape)?;
        Ok(Parameter::new(name, tensor))
    }
}

impl FusionOps<CandleBackend> for CandleOps {
    fn fused_linear_bias(
        &self,
        input: &Tensor,
        weight: &Parameter<CandleBackend>,
        bias: &Parameter<CandleBackend>,
    ) -> Result<Tensor> {
        let w = weight.tensor();
        let b = bias.tensor();

        // matmul + add_row_vector (fused into single operation sequence)
        let input = input.contiguous().map_err(|e| CoreError::Backend(e.to_string()))?;
        let w_t = w
            .t()
            .map_err(|e| CoreError::Backend(e.to_string()))?
            .contiguous()
            .map_err(|e| CoreError::Backend(e.to_string()))?;
        let output = input.matmul(&w_t).map_err(|e| CoreError::Backend(e.to_string()))?;
        let output = output.broadcast_add(b).map_err(|e| CoreError::Backend(e.to_string()))?;
        Ok(output)
    }

    fn fused_linear_bias_relu(
        &self,
        input: &Tensor,
        weight: &Parameter<CandleBackend>,
        bias: &Parameter<CandleBackend>,
    ) -> Result<Tensor> {
        let w = weight.tensor();
        let b = bias.tensor();

        // matmul + add_row_vector + relu (fused sequence)
        let input = input.contiguous().map_err(|e| CoreError::Backend(e.to_string()))?;
        let w_t = w
            .t()
            .map_err(|e| CoreError::Backend(e.to_string()))?
            .contiguous()
            .map_err(|e| CoreError::Backend(e.to_string()))?;
        let output = input.matmul(&w_t).map_err(|e| CoreError::Backend(e.to_string()))?;
        let output = output.broadcast_add(b).map_err(|e| CoreError::Backend(e.to_string()))?;
        let output = output.relu().map_err(|e| CoreError::Backend(e.to_string()))?;
        Ok(output)
    }

    fn fused_linear_bias_gelu(
        &self,
        input: &Tensor,
        weight: &Parameter<CandleBackend>,
        bias: &Parameter<CandleBackend>,
    ) -> Result<Tensor> {
        let w = weight.tensor();
        let b = bias.tensor();

        // matmul + add_row_vector + gelu (fused sequence)
        let input = input.contiguous().map_err(|e| CoreError::Backend(e.to_string()))?;
        let w_t = w
            .t()
            .map_err(|e| CoreError::Backend(e.to_string()))?
            .contiguous()
            .map_err(|e| CoreError::Backend(e.to_string()))?;
        let output = input.matmul(&w_t).map_err(|e| CoreError::Backend(e.to_string()))?;
        let output = output.broadcast_add(b).map_err(|e| CoreError::Backend(e.to_string()))?;

        // GELU: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        let x = &output;
        let x_cubed = (x * x * x).map_err(|e| CoreError::Backend(e.to_string()))?;
        let scalar_0_044715 = Tensor::from_vec(vec![0.044715f32], &[], &self.device)
            .map_err(|e| CoreError::Backend(e.to_string()))?;
        let scalar_0_044715_broadcasted =
            scalar_0_044715.broadcast_as(x.dims()).map_err(|e| CoreError::Backend(e.to_string()))?;
        let x_plus =
            (x + (&scalar_0_044715_broadcasted * x_cubed)).map_err(|e| CoreError::Backend(e.to_string()))?;
        let sqrt_2_pi = (2.0f32 / std::f32::consts::PI).sqrt();
        let scalar_sqrt_2_pi = Tensor::from_vec(vec![sqrt_2_pi], &[], &self.device)
            .map_err(|e| CoreError::Backend(e.to_string()))?;
        let scalar_sqrt_2_pi_broadcasted =
            scalar_sqrt_2_pi.broadcast_as(x.dims()).map_err(|e| CoreError::Backend(e.to_string()))?;
        let tanh_arg =
            (x_plus * scalar_sqrt_2_pi_broadcasted).map_err(|e| CoreError::Backend(e.to_string()))?;
        let tanh_val = tanh_arg.tanh().map_err(|e| CoreError::Backend(e.to_string()))?;
        let scalar_1 = Tensor::from_vec(vec![1.0f32], &[], &self.device)
            .map_err(|e| CoreError::Backend(e.to_string()))?;
        let scalar_1_broadcasted =
            scalar_1.broadcast_as(x.dims()).map_err(|e| CoreError::Backend(e.to_string()))?;
        let one_plus = (&scalar_1_broadcasted + tanh_val).map_err(|e| CoreError::Backend(e.to_string()))?;
        let scalar_0_5 = Tensor::from_vec(vec![0.5f32], &[], &self.device)
            .map_err(|e| CoreError::Backend(e.to_string()))?;
        let scalar_0_5_broadcasted =
            scalar_0_5.broadcast_as(x.dims()).map_err(|e| CoreError::Backend(e.to_string()))?;
        let gelu_output =
            ((&scalar_0_5_broadcasted * x) * one_plus).map_err(|e| CoreError::Backend(e.to_string()))?;

        Ok(gelu_output)
    }
}

impl AttentionOps<CandleBackend> for CandleOps {
    fn flash_attention_2(
        &self,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        causal_mask: bool,
    ) -> Result<Tensor> {
        // Note: This is a simplified implementation for demonstration.
        // A full Flash Attention 2 implementation would use memory-efficient tiling
        // and online softmax to achieve O(seq_len) memory complexity.
        // For production use, this should use candle's native flash attention kernels
        // when available, or custom CUDA kernels.

        // For now, fall back to standard attention with the pattern established
        // This allows the trait to be used while we develop the full implementation
        self.standard_attention(query, key, value, causal_mask)
    }

    fn flash_attention_2_with_mask(
        &self,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        mask: &Tensor,
    ) -> Result<Tensor> {
        // Simplified implementation - use standard attention with mask
        self.standard_attention_with_mask(query, key, value, mask)
    }
}

impl CandleOps {
    /// Standard scaled dot-product attention (fallback for Flash Attention).
    ///
    /// This is used as a fallback when Flash Attention kernels are not available.
    /// It has O(seq_len^2) memory complexity but provides correct results.
    fn standard_attention(
        &self,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        causal_mask: bool,
    ) -> Result<Tensor> {
        // query, key, value: [batch, seq_len, num_heads, head_dim]
        let q_dims = query.dims();
        let k_dims = key.dims();
        let v_dims = value.dims();

        if q_dims.len() != 4 || k_dims.len() != 4 || v_dims.len() != 4 {
            return Err(CoreError::InvalidArgument(format!(
                "Flash attention expects 4D tensors [batch, seq_len, num_heads, head_dim], got q:{:?}, k:{:?}, v:{:?}",
                q_dims, k_dims, v_dims
            )));
        }

        let _seq_len = q_dims[1];
        let head_dim = q_dims[3];

        // Compute attention scores: Q @ K^T / sqrt(head_dim)
        let k_t = key.transpose(2, 3).map_err(|e| CoreError::Backend(e.to_string()))?;
        let scores = query.matmul(&k_t).map_err(|e| CoreError::Backend(e.to_string()))?;
        let scale = Tensor::new((head_dim as f32).sqrt(), &self.device)
            .map_err(|e| CoreError::Backend(e.to_string()))?;
        let scores = (scores / scale).map_err(|e| CoreError::Backend(e.to_string()))?;

        // Apply causal mask if needed
        let scores = if causal_mask { self.apply_causal_mask(&scores)? } else { scores };

        // Apply softmax manually using candle-core operations
        // softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))
        let scores_max = scores.max(3).map_err(|e| CoreError::Backend(e.to_string()))?;
        let scores_max = scores_max.unsqueeze(3).map_err(|e| CoreError::Backend(e.to_string()))?;
        let scores_shifted = (scores - scores_max).map_err(|e| CoreError::Backend(e.to_string()))?;
        let exp_scores = scores_shifted.exp().map_err(|e| CoreError::Backend(e.to_string()))?;
        let sum_exp = exp_scores.sum(3).map_err(|e| CoreError::Backend(e.to_string()))?;
        let sum_exp = sum_exp.unsqueeze(3).map_err(|e| CoreError::Backend(e.to_string()))?;
        let attn_weights = (exp_scores / sum_exp).map_err(|e| CoreError::Backend(e.to_string()))?;

        // Apply attention to values: attn_weights @ V
        let output = attn_weights.matmul(value).map_err(|e| CoreError::Backend(e.to_string()))?;

        Ok(output)
    }

    /// Standard attention with custom mask.
    fn standard_attention_with_mask(
        &self,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        mask: &Tensor,
    ) -> Result<Tensor> {
        let q_dims = query.dims();
        let head_dim = q_dims[3];

        // Compute attention scores
        let k_t = key.transpose(2, 3).map_err(|e| CoreError::Backend(e.to_string()))?;
        let scores = query.matmul(&k_t).map_err(|e| CoreError::Backend(e.to_string()))?;
        let scale = Tensor::new((head_dim as f32).sqrt(), &self.device)
            .map_err(|e| CoreError::Backend(e.to_string()))?;
        let scores = (scores / scale).map_err(|e| CoreError::Backend(e.to_string()))?;

        // Apply custom mask (mask: 1.0 = attend, 0.0 = mask)
        // Convert to additive mask: -inf where mask is 0
        let neg_inf =
            Tensor::new(f32::NEG_INFINITY, &self.device).map_err(|e| CoreError::Backend(e.to_string()))?;
        let mask_broadcasted =
            mask.broadcast_as(scores.dims()).map_err(|e| CoreError::Backend(e.to_string()))?;
        let zero = Tensor::new(0.0, &self.device).map_err(|e| CoreError::Backend(e.to_string()))?;
        let additive_mask =
            mask_broadcasted.where_cond(&zero, &neg_inf).map_err(|e| CoreError::Backend(e.to_string()))?;
        let scores = (scores + additive_mask).map_err(|e| CoreError::Backend(e.to_string()))?;

        // Apply softmax manually
        let scores_max = scores.max(3).map_err(|e| CoreError::Backend(e.to_string()))?;
        let scores_max = scores_max.unsqueeze(3).map_err(|e| CoreError::Backend(e.to_string()))?;
        let scores_shifted = (scores - scores_max).map_err(|e| CoreError::Backend(e.to_string()))?;
        let exp_scores = scores_shifted.exp().map_err(|e| CoreError::Backend(e.to_string()))?;
        let sum_exp = exp_scores.sum(3).map_err(|e| CoreError::Backend(e.to_string()))?;
        let sum_exp = sum_exp.unsqueeze(3).map_err(|e| CoreError::Backend(e.to_string()))?;
        let attn_weights = (exp_scores / sum_exp).map_err(|e| CoreError::Backend(e.to_string()))?;

        // Apply attention to values
        let output = attn_weights.matmul(value).map_err(|e| CoreError::Backend(e.to_string()))?;

        Ok(output)
    }

    /// Apply causal mask to attention scores.
    fn apply_causal_mask(&self, scores: &Tensor) -> Result<Tensor> {
        let dims = scores.dims();
        let seq_len = dims[1];

        // Create causal mask: lower triangular matrix
        let mut mask_data = vec![0.0f32; seq_len * seq_len];
        for i in 0..seq_len {
            for j in 0..=i {
                mask_data[i * seq_len + j] = 1.0;
            }
        }

        let mask = Tensor::from_vec(mask_data, &[seq_len, seq_len], &self.device)
            .map_err(|e| CoreError::Backend(e.to_string()))?;

        // Broadcast mask to match scores shape
        let mask_broadcasted = mask.broadcast_as(dims).map_err(|e| CoreError::Backend(e.to_string()))?;

        // Apply mask: set masked positions to -inf
        let neg_inf =
            Tensor::new(f32::NEG_INFINITY, &self.device).map_err(|e| CoreError::Backend(e.to_string()))?;
        let zero = Tensor::new(0.0, &self.device).map_err(|e| CoreError::Backend(e.to_string()))?;
        let masked_scores =
            mask_broadcasted.where_cond(&zero, &neg_inf).map_err(|e| CoreError::Backend(e.to_string()))?;
        let masked_scores = (scores + masked_scores).map_err(|e| CoreError::Backend(e.to_string()))?;

        Ok(masked_scores)
    }
}

impl TensorOps<CandleBackend> for CandleOps {
    fn shape(&self, x: &Tensor) -> Vec<usize> {
        x.dims().to_vec()
    }

    fn tensor_from_vec(&self, values: Vec<f32>, shape: &[usize]) -> Result<Tensor> {
        Tensor::from_vec(values, shape, &self.device).map_err(|e| CoreError::Backend(e.to_string()))
    }

    fn zeros(&self, shape: &[usize]) -> Result<Tensor> {
        Tensor::zeros(shape, candle_core::DType::F32, &self.device)
            .map_err(|e| CoreError::Backend(e.to_string()))
    }

    fn matmul(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        let a = a.contiguous().map_err(|e| CoreError::Backend(e.to_string()))?;
        let b = b.contiguous().map_err(|e| CoreError::Backend(e.to_string()))?;
        a.matmul(&b).map_err(|e| CoreError::Backend(e.to_string()))
    }

    fn transpose(&self, x: &Tensor) -> Result<Tensor> {
        x.transpose(0, 1).map_err(|e| CoreError::Backend(e.to_string()))
    }

    fn add(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        (a + b).map_err(|e| CoreError::Backend(e.to_string()))
    }

    fn add_row_vector(&self, a: &Tensor, row: &Tensor) -> Result<Tensor> {
        let broadcasted = row.broadcast_as(a.dims()).map_err(|e| CoreError::Backend(e.to_string()))?;
        (a + broadcasted).map_err(|e| CoreError::Backend(e.to_string()))
    }

    fn relu(&self, x: &Tensor) -> Result<Tensor> {
        x.relu().map_err(|e| CoreError::Backend(e.to_string()))
    }

    fn softmax(&self, x: &Tensor) -> Result<Tensor> {
        let last_dim = x.dims().len().saturating_sub(1);
        candle_nn::ops::softmax(x, last_dim)
            .map_err(|e: candle_core::Error| CoreError::Backend(e.to_string()))
    }

    fn log_softmax(&self, x: &Tensor) -> Result<Tensor> {
        let last_dim = x.dims().len().saturating_sub(1);
        candle_nn::ops::log_softmax(x, last_dim)
            .map_err(|e: candle_core::Error| CoreError::Backend(e.to_string()))
    }

    fn argmax(&self, x: &Tensor) -> Result<usize> {
        let flat = x.flatten_all().map_err(|e| CoreError::Backend(e.to_string()))?;
        let idx = flat.argmax(0).map_err(|e| CoreError::Backend(e.to_string()))?;
        let val = idx.to_scalar::<i64>().map_err(|e| CoreError::Backend(e.to_string()))?;
        Ok(val as usize)
    }

    fn gather_rows(&self, table: &Parameter<CandleBackend>, ids: &[usize]) -> Result<Tensor> {
        let table_tensor = table.tensor();
        let ids_i64: Vec<i64> = ids.iter().map(|&id| id as i64).collect();
        let ids_tensor = Tensor::from_vec(ids_i64, &[ids.len()], &self.device)
            .map_err(|e| CoreError::Backend(e.to_string()))?;
        table_tensor.index_select(&ids_tensor, 0).map_err(|e| CoreError::Backend(e.to_string()))
    }

    fn linear(
        &self,
        input: &Tensor,
        weight: &Parameter<CandleBackend>,
        bias: Option<&Parameter<CandleBackend>>,
    ) -> Result<Tensor> {
        let w = weight.tensor();
        let input = input.contiguous().map_err(|e| CoreError::Backend(e.to_string()))?;
        let w_t = w
            .t()
            .map_err(|e| CoreError::Backend(e.to_string()))?
            .contiguous()
            .map_err(|e| CoreError::Backend(e.to_string()))?;
        let output = input.matmul(&w_t).map_err(|e| CoreError::Backend(e.to_string()))?;
        match bias {
            Some(b) => {
                let b_t =
                    b.tensor().broadcast_as(output.dims()).map_err(|e| CoreError::Backend(e.to_string()))?;
                (output + b_t).map_err(|e| CoreError::Backend(e.to_string()))
            }
            None => Ok(output),
        }
    }

    fn layer_norm(&self, x: &Tensor, gamma: &Tensor, beta: &Tensor, eps: f32) -> Result<Tensor> {
        candle_nn::ops::layer_norm(x, gamma, beta, eps)
            .map_err(|e: candle_core::Error| CoreError::Backend(e.to_string()))
    }

    fn sigmoid(&self, x: &Tensor) -> Result<Tensor> {
        (x.neg()
            .map_err(|e| CoreError::Backend(e.to_string()))?
            .exp()
            .map_err(|e| CoreError::Backend(e.to_string()))?
            + 1.0)
            .map_err(|e| CoreError::Backend(e.to_string()))?
            .recip()
            .map_err(|e| CoreError::Backend(e.to_string()))
    }

    fn tanh(&self, x: &Tensor) -> Result<Tensor> {
        x.tanh().map_err(|e| CoreError::Backend(e.to_string()))
    }

    fn gelu(&self, x: &Tensor) -> Result<Tensor> {
        x.gelu().map_err(|e| CoreError::Backend(e.to_string()))
    }

    fn mul(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        (a * b).map_err(|e| CoreError::Backend(e.to_string()))
    }

    fn dropout(&self, x: &Tensor, p: f32, training: bool) -> Result<Tensor> {
        if !training || p == 0.0 {
            return Ok(x.clone());
        }
        if p == 1.0 {
            return self.zeros(x.dims());
        }
        let scale = 1.0 / (1.0 - p);
        let rand_tensor = Tensor::rand(0.0f64, 1.0, x.shape(), &self.device)
            .map_err(|e| CoreError::Backend(e.to_string()))?;
        let mask = rand_tensor
            .ge(p as f64)
            .map_err(|e| CoreError::Backend(e.to_string()))?
            .to_dtype(candle_core::DType::F32)
            .map_err(|e| CoreError::Backend(e.to_string()))?;
        let scaled = (&mask * x).map_err(|e| CoreError::Backend(e.to_string()))?;
        self.mul_scalar(&scaled, scale)
    }

    fn concat(&self, tensors: &[&Tensor], dim: usize) -> Result<Tensor> {
        let owned: Vec<Tensor> = tensors.iter().map(|&t| t.clone()).collect();
        Tensor::cat(&owned, dim).map_err(|e| CoreError::Backend(e.to_string()))
    }

    fn slice(&self, x: &Tensor, start: usize, end: usize) -> Result<Tensor> {
        x.narrow(0, start, end - start).map_err(|e| CoreError::Backend(e.to_string()))
    }

    fn reshape(&self, x: &Tensor, shape: &[usize]) -> Result<Tensor> {
        x.reshape(shape).map_err(|e| CoreError::Backend(e.to_string()))
    }

    fn add_scalar(&self, x: &Tensor, scalar: f32) -> Result<Tensor> {
        (x + (scalar as f64)).map_err(|e| CoreError::Backend(e.to_string()))
    }

    fn mul_scalar(&self, x: &Tensor, scalar: f32) -> Result<Tensor> {
        (x * (scalar as f64)).map_err(|e| CoreError::Backend(e.to_string()))
    }

    fn broadcast(&self, x: &Tensor, shape: &[usize]) -> Result<Tensor> {
        x.broadcast_as(shape).map_err(|e| CoreError::Backend(e.to_string()))
    }

    fn neg(&self, x: &Tensor) -> Result<Tensor> {
        x.neg().map_err(|e| CoreError::Backend(e.to_string()))
    }

    fn sub(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        (a - b).map_err(|e| CoreError::Backend(e.to_string()))
    }

    fn sqrt(&self, x: &Tensor) -> Result<Tensor> {
        x.sqrt().map_err(|e| CoreError::Backend(e.to_string()))
    }

    fn div(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        (a / b).map_err(|e| CoreError::Backend(e.to_string()))
    }

    fn exp(&self, x: &Tensor) -> Result<Tensor> {
        x.exp().map_err(|e| CoreError::Backend(e.to_string()))
    }

    fn log(&self, x: &Tensor) -> Result<Tensor> {
        x.log().map_err(|e| CoreError::Backend(e.to_string()))
    }

    fn maximum(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        let mask = a.gt(b).map_err(|e| CoreError::Backend(e.to_string()))?;
        mask.where_cond(a, b).map_err(|e| CoreError::Backend(e.to_string()))
    }

    fn gt_scalar(&self, x: &Tensor, scalar: f32) -> Result<Tensor> {
        let mask = x.gt(scalar as f64).map_err(|e| CoreError::Backend(e.to_string()))?;
        mask.to_dtype(candle_core::DType::F32).map_err(|e| CoreError::Backend(e.to_string()))
    }

    fn sum_all(&self, x: &Tensor) -> Result<Tensor> {
        x.sum_all().map_err(|e| CoreError::Backend(e.to_string()))
    }

    fn sum_dim0(&self, x: &Tensor) -> Result<Tensor> {
        let dims = x.dims();
        if dims.len() != 2 {
            return Err(CoreError::InvalidArgument(format!(
                "sum_dim0 expects rank-2 tensor [batch, features], got shape {:?}",
                dims
            )));
        }
        x.sum(0).map_err(|e| CoreError::Backend(e.to_string()))
    }

    fn softmax_dim(&self, x: &Tensor, dim: usize) -> Result<Tensor> {
        candle_nn::ops::softmax(x, dim).map_err(|e: candle_core::Error| CoreError::Backend(e.to_string()))
    }

    fn log_softmax_dim(&self, x: &Tensor, dim: usize) -> Result<Tensor> {
        candle_nn::ops::log_softmax(x, dim).map_err(|e: candle_core::Error| CoreError::Backend(e.to_string()))
    }

    fn sum_dim(&self, x: &Tensor, dim: usize, keepdim: bool) -> Result<Tensor> {
        let s = x.sum(dim).map_err(|e| CoreError::Backend(e.to_string()))?;
        if keepdim {
            let mut new_shape = x.dims().to_vec();
            if dim >= new_shape.len() {
                return Err(CoreError::InvalidArgument(format!(
                    "sum_dim: dim {} out of range for shape {:?}",
                    dim,
                    x.dims()
                )));
            }
            new_shape[dim] = 1;
            s.reshape(new_shape).map_err(|e| CoreError::Backend(e.to_string()))
        } else {
            Ok(s)
        }
    }

    fn mean_dim(&self, x: &Tensor, dim: usize, keepdim: bool) -> Result<Tensor> {
        let dims = x.dims();
        let n = dims.get(dim).copied().ok_or_else(|| {
            CoreError::InvalidArgument(format!("mean_dim: dim {} out of range for shape {:?}", dim, dims))
        })?;
        if n == 0 {
            return Err(CoreError::InvalidArgument("mean_dim: axis size is 0".into()));
        }
        let s = self.sum_dim(x, dim, keepdim)?;
        let scale = 1.0 / (n as f32);
        self.mul_scalar(&s, scale)
    }

    fn var_dim(&self, x: &Tensor, dim: usize, unbiased: bool, keepdim: bool) -> Result<Tensor> {
        let dims = x.dims();
        let n = dims.get(dim).copied().ok_or_else(|| {
            CoreError::InvalidArgument(format!("var_dim: dim {} out of range for shape {:?}", dim, dims))
        })?;
        if n == 0 {
            return Err(CoreError::InvalidArgument("var_dim: axis size is 0".into()));
        }
        if unbiased && n < 2 {
            return Err(CoreError::InvalidArgument("var_dim: unbiased variance requires axis >= 2".into()));
        }

        let mean_keep = self.mean_dim(x, dim, true)?;
        let mean_b = mean_keep.broadcast_as(dims).map_err(|e| CoreError::Backend(e.to_string()))?;
        let diff = (x - mean_b).map_err(|e| CoreError::Backend(e.to_string()))?;
        let sq = (&diff * &diff).map_err(|e| CoreError::Backend(e.to_string()))?;

        let s = self.sum_dim(&sq, dim, keepdim)?;
        let denom = if unbiased { (n - 1) as f32 } else { n as f32 };
        let scale = 1.0 / denom;
        self.mul_scalar(&s, scale)
    }

    fn cross_entropy_with_indices(&self, logits: &Tensor, targets: &[usize]) -> Result<Tensor> {
        let targets_i64: Vec<i64> = targets.iter().map(|&t| t as i64).collect();
        let targets_tensor = Tensor::from_vec(targets_i64, &[targets.len()], &self.device)
            .map_err(|e| CoreError::Backend(e.to_string()))?;

        candle_nn::loss::cross_entropy(logits, &targets_tensor).map_err(|e| CoreError::Backend(e.to_string()))
    }

    fn transpose_axes(&self, x: &Tensor, dim0: usize, dim1: usize) -> Result<Tensor> {
        x.transpose(dim0, dim1).map_err(|e| CoreError::Backend(e.to_string()))
    }

    fn matmul_batched(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        a.matmul(b).map_err(|e| CoreError::Backend(e.to_string()))
    }

    fn broadcast_to(&self, x: &Tensor, shape: &[usize]) -> Result<Tensor> {
        x.broadcast_as(shape).map_err(|e| CoreError::Backend(e.to_string()))
    }

    fn tensor_to_vec(&self, x: &Tensor) -> Result<Vec<f32>> {
        x.flatten_all()
            .map_err(|e| CoreError::Backend(e.to_string()))?
            .to_vec1()
            .map_err(|e| CoreError::Backend(e.to_string()))
    }

    fn tensor_element(&self, x: &Tensor, index: usize) -> Result<f32> {
        let val = x.i(index).map_err(|e| CoreError::Backend(e.to_string()))?;
        val.to_scalar::<f32>().map_err(|e| CoreError::Backend(e.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_candle_backend_cpu() {
        let backend = CandleBackend::cpu();
        let tensor = backend.tensor_from_vec(vec![1.0f32, 2.0, 3.0], &[3]).unwrap();
        assert_eq!(tensor.dims().to_vec(), vec![3]);
    }

    #[test]
    fn test_candle_matmul() {
        let backend = CandleBackend::cpu();
        let a = backend.tensor_from_vec(vec![1.0f32, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let b = backend.tensor_from_vec(vec![5.0f32, 6.0, 7.0, 8.0], &[2, 2]).unwrap();
        let c = backend.ops().matmul(&a, &b).unwrap();
        let data = backend.to_vec(&c);
        assert_eq!(data, vec![19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn test_candle_element_wise_ops() {
        let backend = CandleBackend::cpu();
        let a = backend.tensor_from_vec(vec![1.0f32, 2.0, 3.0, 4.0], &[4]).unwrap();
        let b = backend.tensor_from_vec(vec![5.0f32, 4.0, 3.0, 2.0], &[4]).unwrap();

        let c = backend.ops().add(&a, &b).unwrap();
        assert_eq!(backend.to_vec(&c), vec![6.0, 6.0, 6.0, 6.0]);

        let c = backend.ops().mul(&a, &b).unwrap();
        assert_eq!(backend.to_vec(&c), vec![5.0, 8.0, 9.0, 8.0]);
    }

    #[test]
    fn test_candle_relu() {
        let backend = CandleBackend::cpu();
        let a = backend.tensor_from_vec(vec![-1.0f32, 0.0, 1.0, 2.0], &[4]).unwrap();
        let c = backend.ops().relu(&a).unwrap();
        assert_eq!(backend.to_vec(&c), vec![0.0, 0.0, 1.0, 2.0]);
    }

    #[test]
    fn test_candle_softmax() {
        let backend = CandleBackend::cpu();
        let a = backend.tensor_from_vec(vec![1.0f32, 2.0, 3.0], &[3]).unwrap();
        let c = backend.ops().softmax(&a).unwrap();
        let data = backend.to_vec(&c);
        let sum: f32 = data.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_candle_transpose() {
        let backend = CandleBackend::cpu();
        let a = backend.tensor_from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
        let c = backend.ops().transpose(&a).unwrap();
        assert_eq!(c.dims().to_vec(), vec![3, 2]);
    }

    #[test]
    fn test_candle_zeros() {
        let backend = CandleBackend::cpu();
        let c = backend.ops().zeros(&[2, 3]).unwrap();
        assert_eq!(c.dims().to_vec(), vec![2, 3]);
        let data = backend.to_vec(&c);
        assert!(data.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_candle_parameter() {
        let backend = CandleBackend::cpu();
        let param = backend.normal_parameter("test", &[3], 42, 1.0).unwrap();
        assert_eq!(param.name(), "test");
        assert_eq!(param.tensor().dims().to_vec(), vec![3]);
    }

    #[test]
    fn test_candle_matmul_rectangular() {
        let backend = CandleBackend::cpu();
        let a = backend.tensor_from_vec(vec![1.0f32; 8 * 40], &[8, 40]).unwrap();
        let b = backend.tensor_from_vec(vec![1.0f32; 40 * 64], &[40, 64]).unwrap();
        let c = backend.ops().matmul(&a, &b).unwrap();
        assert_eq!(c.dims().to_vec(), vec![8, 64]);
    }

    #[test]
    fn test_candle_log_softmax_2d() {
        let backend = CandleBackend::cpu();
        let a = backend.tensor_from_vec(vec![1.0f32, 2.0, 3.0], &[1, 3]).unwrap();
        let c = backend.ops().log_softmax(&a).unwrap();
        let data = backend.to_vec(&c);
        assert_eq!(data.len(), 3);
    }

    #[test]
    fn test_candle_full_pipeline() {
        let backend = CandleBackend::cpu();
        let ops = backend.ops();
        let vocab = 40usize;
        let d = 64usize;
        let h = 256usize;

        let w_emb = backend.normal_parameter("w_emb", &[vocab, d], 42, 0.1).unwrap();
        let w1 = backend.normal_parameter("w1", &[d, h], 43, 0.1).unwrap();
        let b1 = backend.normal_parameter("b1", &[h], 44, 0.0).unwrap();
        let w2 = backend.normal_parameter("w2", &[h, vocab], 45, 0.1).unwrap();
        let b2 = backend.normal_parameter("b2", &[vocab], 46, 0.0).unwrap();

        let x = backend.tensor_from_vec(vec![1.0f32; 8 * vocab], &[8, vocab]).unwrap();
        let emb = ops.matmul(&x, w_emb.tensor()).unwrap();
        let emb_data: Vec<f32> = emb.flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(emb_data.len(), 8 * d);

        let pooled = emb_data.chunks_exact(d).fold(vec![0.0f32; d], |mut acc, chunk| {
            for (i, &v) in chunk.iter().enumerate() {
                acc[i] += v / 8.0;
            }
            acc
        });
        let pooled_t = backend.tensor_from_vec(pooled, &[1, d]).unwrap();

        let h1 = ops.matmul(&pooled_t, w1.tensor()).unwrap();
        let h1_b = ops.add_row_vector(&h1, b1.tensor()).unwrap();
        let h1_relu = ops.relu(&h1_b).unwrap();
        let logits = ops.matmul(&h1_relu, w2.tensor()).unwrap();
        let logits_b = ops.add_row_vector(&logits, b2.tensor()).unwrap();
        let log_probs = ops.log_softmax(&logits_b).unwrap();
        let data = backend.to_vec(&log_probs);
        assert_eq!(data.len(), vocab);
    }

    #[test]
    fn test_candle_softmax_1_40() {
        let backend = CandleBackend::cpu();
        let ops = backend.ops();
        let a = backend.tensor_from_vec(vec![1.0f32; 40], &[1, 40]).unwrap();
        let c = ops.softmax(&a).unwrap();
        assert_eq!(c.dims().to_vec(), vec![1, 40]);
    }

    #[test]
    fn test_candle_softmax_dim_row_sums_to_one() {
        let backend = CandleBackend::cpu();
        let x = backend.tensor_from_vec(vec![1.0, 2.0, 3.0, 0.0, -1.0, 2.0], &[2, 3]).unwrap();
        let y = backend.ops.softmax_dim(&x, 1).unwrap();
        let v = backend.to_vec(&y);
        for row in 0..2 {
            let s: f32 = v[row * 3..row * 3 + 3].iter().sum();
            assert!((s - 1.0).abs() <= 1e-6, "row {row} sum={s}");
        }
    }

    #[test]
    fn test_candle_log_softmax_dim_matches_ln_softmax_dim() {
        let backend = CandleBackend::cpu();
        let x = backend.tensor_from_vec(vec![1.0, 2.0, 3.0, 0.0, -1.0, 2.0], &[2, 3]).unwrap();
        let ls = backend.ops.log_softmax_dim(&x, 1).unwrap();
        let s = backend.ops.softmax_dim(&x, 1).unwrap();
        let ln_s = s.log().unwrap();
        let a = backend.to_vec(&ls);
        let b = backend.to_vec(&ln_s);
        assert_eq!(a.len(), b.len());
        for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            let d = (x - y).abs();
            assert!(d <= 1e-5, "idx={i} x={x} y={y} diff={d}");
        }
    }
}
