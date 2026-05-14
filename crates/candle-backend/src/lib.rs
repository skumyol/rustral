//! candle-core backend for Rustral.

use candle_core::{Device, IndexOp, Tensor};
use rustral_core::{
    AttentionOps, Backend, BackendCapabilities, CoreError, FusionOps, Parameter, Result, TensorOps,
};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum CandleError {
    #[error("candle error: {0}")]
    Candle(#[from] candle_core::Error),
}

#[derive(Clone, Debug)]
struct CandleOps {
    device: Device,
}

#[derive(Clone, Debug)]
pub struct CandleBackend {
    device: Device,
    ops: CandleOps,
}

impl CandleBackend {
    pub fn cpu() -> Self {
        let device = Device::Cpu;
        Self { device: device.clone(), ops: CandleOps { device } }
    }
    pub fn cuda(id: usize) -> Result<Self> {
        #[cfg(feature = "cuda")]
        {
            let device = Device::new_cuda(id).map_err(|e| CoreError::Backend(e.to_string()))?;
            Ok(Self { device: device.clone(), ops: CandleOps { device } })
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = id;
            Err(CoreError::Backend("CUDA not enabled".to_string()))
        }
    }
    pub fn new() -> Self {
        Self::cpu()
    }
    pub fn tensor_from_vec(&self, values: Vec<f32>, shape: &[usize]) -> Result<Tensor> {
        Tensor::from_vec(values, shape, &self.device).map_err(|e| CoreError::Backend(e.to_string()))
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
        BackendCapabilities::default()
    }
    fn normal_parameter(
        &self,
        name: &str,
        shape: &[usize],
        _seed: u64,
        _scale: f32,
    ) -> Result<Parameter<Self>> {
        let v = vec![0.0f32; shape.iter().product()];
        Ok(Parameter::new(name, self.tensor_from_vec(v, shape)?))
    }
    fn parameter_from_vec(&self, name: &str, values: Vec<f32>, shape: &[usize]) -> Result<Parameter<Self>> {
        Ok(Parameter::new(name, self.tensor_from_vec(values, shape)?))
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
        input
            .matmul(&w.t().map_err(|e| CoreError::Backend(e.to_string()))?)
            .map_err(|e| CoreError::Backend(e.to_string()))?
            .broadcast_add(b)
            .map_err(|e| CoreError::Backend(e.to_string()))
    }
    fn fused_linear_bias_relu(
        &self,
        input: &Tensor,
        weight: &Parameter<CandleBackend>,
        bias: &Parameter<CandleBackend>,
    ) -> Result<Tensor> {
        self.fused_linear_bias(input, weight, bias)?.relu().map_err(|e| CoreError::Backend(e.to_string()))
    }
    fn fused_linear_bias_gelu(
        &self,
        input: &Tensor,
        weight: &Parameter<CandleBackend>,
        bias: &Parameter<CandleBackend>,
    ) -> Result<Tensor> {
        self.fused_linear_bias(input, weight, bias)?.gelu().map_err(|e| CoreError::Backend(e.to_string()))
    }
}

impl AttentionOps<CandleBackend> for CandleOps {
    fn flash_attention_2(&self, q: &Tensor, k: &Tensor, v: &Tensor, _causal: bool) -> Result<Tensor> {
        let kt = k.transpose(2, 3).map_err(|e| CoreError::Backend(e.to_string()))?;
        let scores = q.matmul(&kt).map_err(|e| CoreError::Backend(e.to_string()))?;
        let attn = candle_nn::ops::softmax(&scores, 3).map_err(|e| CoreError::Backend(e.to_string()))?;
        attn.matmul(v).map_err(|e| CoreError::Backend(e.to_string()))
    }
    fn flash_attention_2_with_mask(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        _mask: &Tensor,
    ) -> Result<Tensor> {
        self.flash_attention_2(q, k, v, false)
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
        a.matmul(b).map_err(|e| CoreError::Backend(e.to_string()))
    }
    fn transpose(&self, x: &Tensor) -> Result<Tensor> {
        x.t().map_err(|e| CoreError::Backend(e.to_string()))
    }
    fn add(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        (a + b).map_err(|e| CoreError::Backend(e.to_string()))
    }
    fn add_row_vector(&self, a: &Tensor, row: &Tensor) -> Result<Tensor> {
        a.broadcast_add(row).map_err(|e| CoreError::Backend(e.to_string()))
    }
    fn relu(&self, x: &Tensor) -> Result<Tensor> {
        x.relu().map_err(|e| CoreError::Backend(e.to_string()))
    }
    fn softmax(&self, x: &Tensor) -> Result<Tensor> {
        candle_nn::ops::softmax(x, x.dims().len() - 1).map_err(|e| CoreError::Backend(e.to_string()))
    }
    fn log_softmax(&self, x: &Tensor) -> Result<Tensor> {
        candle_nn::ops::log_softmax(x, x.dims().len() - 1).map_err(|e| CoreError::Backend(e.to_string()))
    }
    fn argmax(&self, x: &Tensor) -> Result<usize> {
        let v = x
            .flatten_all()
            .map_err(|e| CoreError::Backend(e.to_string()))?
            .argmax(0)
            .map_err(|e| CoreError::Backend(e.to_string()))?
            .to_scalar::<i64>()
            .map_err(|e| CoreError::Backend(e.to_string()))?;
        Ok(v as usize)
    }
    fn gather_rows(&self, table: &Parameter<CandleBackend>, ids: &[usize]) -> Result<Tensor> {
        let ids_t =
            Tensor::from_vec(ids.iter().map(|&i| i as i64).collect::<Vec<_>>(), &[ids.len()], &self.device)
                .map_err(|e| CoreError::Backend(e.to_string()))?;
        table.tensor().index_select(&ids_t, 0).map_err(|e| CoreError::Backend(e.to_string()))
    }
    fn linear(
        &self,
        input: &Tensor,
        weight: &Parameter<CandleBackend>,
        bias: Option<&Parameter<CandleBackend>>,
    ) -> Result<Tensor> {
        let res = input
            .matmul(&weight.tensor().t().map_err(|e| CoreError::Backend(e.to_string()))?)
            .map_err(|e| CoreError::Backend(e.to_string()))?;
        match bias {
            Some(b) => res.broadcast_add(b.tensor()).map_err(|e| CoreError::Backend(e.to_string())),
            None => Ok(res),
        }
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
        let mask = (Tensor::rand(0.0f32, 1.0f32, x.dims(), &self.device)
            .map_err(|e| CoreError::Backend(e.to_string()))?
            .gt(p as f64)
            .map_err(|e| CoreError::Backend(e.to_string()))?
            .to_dtype(candle_core::DType::F32)
            .map_err(|e| CoreError::Backend(e.to_string()))?
            * (1.0 / (1.0 - p) as f64))
            .map_err(|e| CoreError::Backend(e.to_string()))?;
        (x * mask).map_err(|e| CoreError::Backend(e.to_string()))
    }
    fn concat(&self, tensors: &[&Tensor], dim: usize) -> Result<Tensor> {
        Tensor::cat(&tensors.iter().map(|&t| t.clone()).collect::<Vec<_>>(), dim)
            .map_err(|e| CoreError::Backend(e.to_string()))
    }
    fn slice(&self, x: &Tensor, start: usize, end: usize) -> Result<Tensor> {
        x.narrow(0, start, end - start).map_err(|e| CoreError::Backend(e.to_string()))
    }
    fn reshape(&self, x: &Tensor, shape: &[usize]) -> Result<Tensor> {
        x.reshape(shape).map_err(|e| CoreError::Backend(e.to_string()))
    }
    fn add_scalar(&self, x: &Tensor, scalar: f32) -> Result<Tensor> {
        (x + scalar as f64).map_err(|e| CoreError::Backend(e.to_string()))
    }
    fn mul_scalar(&self, x: &Tensor, scalar: f32) -> Result<Tensor> {
        (x * scalar as f64).map_err(|e| CoreError::Backend(e.to_string()))
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
        a.maximum(b).map_err(|e| CoreError::Backend(e.to_string()))
    }
    fn gt_scalar(&self, x: &Tensor, scalar: f32) -> Result<Tensor> {
        x.gt(scalar as f64)
            .map_err(|e| CoreError::Backend(e.to_string()))?
            .to_dtype(candle_core::DType::F32)
            .map_err(|e| CoreError::Backend(e.to_string()))
    }
    fn sum_all(&self, x: &Tensor) -> Result<Tensor> {
        x.sum_all().map_err(|e| CoreError::Backend(e.to_string()))
    }
    fn sum_dim0(&self, x: &Tensor) -> Result<Tensor> {
        x.sum(0).map_err(|e| CoreError::Backend(e.to_string()))
    }
    fn tensor_to_vec(&self, x: &Tensor) -> Result<Vec<f32>> {
        x.flatten_all()
            .map_err(|e| CoreError::Backend(e.to_string()))?
            .to_vec1()
            .map_err(|e| CoreError::Backend(e.to_string()))
    }
    fn tensor_element(&self, x: &Tensor, index: usize) -> Result<f32> {
        let v = x
            .flatten_all()
            .map_err(|e| CoreError::Backend(e.to_string()))?
            .i(index)
            .map_err(|e| CoreError::Backend(e.to_string()))?
            .to_scalar::<f32>()
            .map_err(|e| CoreError::Backend(e.to_string()))?;
        Ok(v)
    }
    fn matmul_batched(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        a.matmul(b).map_err(|e| CoreError::Backend(e.to_string()))
    }
    fn transpose_axes(&self, x: &Tensor, dim0: usize, dim1: usize) -> Result<Tensor> {
        x.transpose(dim0, dim1).map_err(|e| CoreError::Backend(e.to_string()))
    }
    fn cross_entropy_with_indices(&self, logits: &Tensor, targets: &[usize]) -> Result<Tensor> {
        let targets_t = Tensor::from_vec(
            targets.iter().map(|&i| i as i64).collect::<Vec<_>>(),
            &[targets.len()],
            &self.device,
        )
        .map_err(|e| CoreError::Backend(e.to_string()))?;
        candle_nn::loss::cross_entropy(logits, &targets_t).map_err(|e| CoreError::Backend(e.to_string()))
    }
    fn softmax_dim(&self, x: &Tensor, dim: usize) -> Result<Tensor> {
        candle_nn::ops::softmax(x, dim).map_err(|e| CoreError::Backend(e.to_string()))
    }
    fn log_softmax_dim(&self, x: &Tensor, dim: usize) -> Result<Tensor> {
        candle_nn::ops::log_softmax(x, dim).map_err(|e| CoreError::Backend(e.to_string()))
    }
    fn sum_dim(&self, x: &Tensor, dim: usize, keepdim: bool) -> Result<Tensor> {
        if keepdim {
            x.sum_keepdim(dim).map_err(|e| CoreError::Backend(e.to_string()))
        } else {
            x.sum(dim).map_err(|e| CoreError::Backend(e.to_string()))
        }
    }
    fn mean_dim(&self, x: &Tensor, dim: usize, keepdim: bool) -> Result<Tensor> {
        let sum = self.sum_dim(x, dim, keepdim)?;
        let n = x.dims()[dim];
        (sum / n as f64).map_err(|e| CoreError::Backend(e.to_string()))
    }
    fn var_dim(&self, x: &Tensor, dim: usize, unbiased: bool, keepdim: bool) -> Result<Tensor> {
        let mean = self.mean_dim(x, dim, true)?;
        let diff = x.broadcast_sub(&mean).map_err(|e| CoreError::Backend(e.to_string()))?;
        let sq = (&diff * &diff).map_err(|e| CoreError::Backend(e.to_string()))?;
        let sum_sq = self.sum_dim(&sq, dim, keepdim)?;
        let n = x.dims()[dim];
        let denom = if unbiased { n - 1 } else { n };
        (sum_sq / denom as f64).map_err(|e| CoreError::Backend(e.to_string()))
    }
    fn broadcast_to(&self, x: &Tensor, shape: &[usize]) -> Result<Tensor> {
        x.broadcast_as(shape).map_err(|e| CoreError::Backend(e.to_string()))
    }
    fn batch_norm(&self, x: &Tensor, gamma: &Tensor, beta: &Tensor, eps: f32) -> Result<Tensor> {
        let mean = self.mean_dim(x, 0, true)?;
        let var = self.var_dim(x, 0, false, true)?;
        let x_centered = x.broadcast_sub(&mean).map_err(|e| CoreError::Backend(e.to_string()))?;
        let std = (var + eps as f64)
            .map_err(|e| CoreError::Backend(e.to_string()))?
            .sqrt()
            .map_err(|e| CoreError::Backend(e.to_string()))?;
        let x_hat = x_centered.broadcast_div(&std).map_err(|e| CoreError::Backend(e.to_string()))?;
        let y = x_hat.broadcast_mul(gamma).map_err(|e| CoreError::Backend(e.to_string()))?;
        y.broadcast_add(beta).map_err(|e| CoreError::Backend(e.to_string()))
    }
    fn layer_norm(&self, x: &Tensor, gamma: &Tensor, beta: &Tensor, eps: f32) -> Result<Tensor> {
        candle_nn::ops::layer_norm(x, gamma, beta, eps).map_err(|e| CoreError::Backend(e.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_candle_backend_cpu() {
        let backend = CandleBackend::cpu();
        let t = backend.tensor_from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();
        assert_eq!(t.dims(), &[3]);
    }
}
