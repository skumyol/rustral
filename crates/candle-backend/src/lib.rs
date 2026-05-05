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
use rustral_core::{Backend, CoreError, Parameter, Result, TensorOps};
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
        Self {
            device: device.clone(),
            ops: CandleOps { device },
        }
    }

    /// Create a CUDA backend on the given device index.
    pub fn cuda(_device_id: usize) -> Result<Self> {
        #[cfg(feature = "cuda")]
        {
            let device = Device::new_cuda(_device_id)
                .map_err(|e| CoreError::Backend(format!("CUDA init failed: {}", e)))?;
            Ok(Self {
                device: device.clone(),
                ops: CandleOps { device },
            })
        }
        #[cfg(not(feature = "cuda"))]
        {
            Err(CoreError::Backend(
                "CUDA feature not enabled. Rebuild with --features cuda".to_string(),
            ))
        }
    }

    /// Create a backend, preferring CUDA if available.
    pub fn new() -> Self {
        #[cfg(feature = "cuda")]
        {
            if let Ok(dev) = Device::new_cuda(0) {
                return Self {
                    device: dev.clone(),
                    ops: CandleOps { device: dev },
                };
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
        let tensor = Tensor::from_vec(values, shape, &self.device)
            .map_err(|e| CoreError::Backend(e.to_string()))?;
        Ok(tensor)
    }

    /// Read tensor data back to host as a flat Vec<f32>.
    pub fn to_vec(&self, tensor: &Tensor) -> Vec<f32> {
        tensor.flatten_all()
            .and_then(|t| t.to_vec1::<f32>())
            .unwrap_or_default()
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

    fn normal_parameter(
        &self,
        name: &str,
        shape: &[usize],
        seed: u64,
        scale: f32,
    ) -> Result<Parameter<Self>>
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

    fn parameter_from_vec(
        &self,
        name: &str,
        values: Vec<f32>,
        shape: &[usize],
    ) -> Result<Parameter<Self>>
    where
        Self: Sized,
    {
        let tensor = self.tensor_from_vec(values, shape)?;
        Ok(Parameter::new(name, tensor))
    }
}

impl TensorOps<CandleBackend> for CandleOps {
    fn shape(&self, x: &Tensor) -> Vec<usize> {
        x.dims().to_vec()
    }

    fn tensor_from_vec(&self, values: Vec<f32>, shape: &[usize]) -> Result<Tensor> {
        Tensor::from_vec(values, shape, &self.device)
            .map_err(|e| CoreError::Backend(e.to_string()))
    }

    fn zeros(&self, shape: &[usize]) -> Result<Tensor> {
        Tensor::zeros(shape, candle_core::DType::F32, &self.device)
            .map_err(|e| CoreError::Backend(e.to_string()))
    }

    fn matmul(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        a.matmul(b).map_err(|e| CoreError::Backend(e.to_string()))
    }

    fn transpose(&self, x: &Tensor) -> Result<Tensor> {
        x.transpose(0, 1).map_err(|e| CoreError::Backend(e.to_string()))
    }

    fn add(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        (a + b).map_err(|e| CoreError::Backend(e.to_string()))
    }

    fn add_row_vector(&self, a: &Tensor, row: &Tensor) -> Result<Tensor> {
        // Broadcast row to match a's shape, then add
        let broadcasted = row.broadcast_as(a.dims())
            .map_err(|e| CoreError::Backend(e.to_string()))?;
        (a + broadcasted).map_err(|e| CoreError::Backend(e.to_string()))
    }

    fn relu(&self, x: &Tensor) -> Result<Tensor> {
        x.relu().map_err(|e| CoreError::Backend(e.to_string()))
    }

    fn softmax(&self, x: &Tensor) -> Result<Tensor> {
        // Manual softmax over all elements
        let flat = x.flatten_all().map_err(|e| CoreError::Backend(e.to_string()))?;
        let max_val = flat.max(0).map_err(|e| CoreError::Backend(e.to_string()))?;
        let max_broadcast = max_val.broadcast_as(flat.dims())
            .map_err(|e| CoreError::Backend(e.to_string()))?;
        let shifted = (&flat - max_broadcast).map_err(|e| CoreError::Backend(e.to_string()))?;
        let exp_shifted = shifted.exp().map_err(|e| CoreError::Backend(e.to_string()))?;
        let sum_exp = exp_shifted.sum(0).map_err(|e| CoreError::Backend(e.to_string()))?;
        let sum_broadcast = sum_exp.broadcast_as(flat.dims())
            .map_err(|e| CoreError::Backend(e.to_string()))?;
        let result = (exp_shifted / sum_broadcast).map_err(|e| CoreError::Backend(e.to_string()))?;
        result.reshape(x.dims())
            .map_err(|e| CoreError::Backend(e.to_string()))
    }

    fn log_softmax(&self, x: &Tensor) -> Result<Tensor> {
        // Manual log-softmax over all elements
        let flat = x.flatten_all().map_err(|e| CoreError::Backend(e.to_string()))?;
        let max_val = flat.max(0).map_err(|e| CoreError::Backend(e.to_string()))?;
        let max_broadcast = max_val.broadcast_as(flat.dims())
            .map_err(|e| CoreError::Backend(e.to_string()))?;
        let shifted = (&flat - max_broadcast).map_err(|e| CoreError::Backend(e.to_string()))?;
        let exp_shifted = shifted.exp().map_err(|e| CoreError::Backend(e.to_string()))?;
        let sum_exp = exp_shifted.sum(0).map_err(|e| CoreError::Backend(e.to_string()))?;
        let log_sum = sum_exp.log().map_err(|e| CoreError::Backend(e.to_string()))?;
        let log_broadcast = log_sum.broadcast_as(flat.dims())
            .map_err(|e| CoreError::Backend(e.to_string()))?;
        let result = (flat - log_broadcast).map_err(|e| CoreError::Backend(e.to_string()))?;
        result.reshape(x.dims())
            .map_err(|e| CoreError::Backend(e.to_string()))
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
        table_tensor.index_select(&ids_tensor, 0)
            .map_err(|e| CoreError::Backend(e.to_string()))
    }

    fn linear(
        &self,
        input: &Tensor,
        weight: &Parameter<CandleBackend>,
        bias: Option<&Parameter<CandleBackend>>,
    ) -> Result<Tensor> {
        let w = weight.tensor();
        let output = input.matmul(&w.t().map_err(|e| CoreError::Backend(e.to_string()))?)
            .map_err(|e| CoreError::Backend(e.to_string()))?;
        match bias {
            Some(b) => {
                let b_t = b.tensor().broadcast_as(output.dims())
                    .map_err(|e| CoreError::Backend(e.to_string()))?;
                (output + b_t).map_err(|e| CoreError::Backend(e.to_string()))
            }
            None => Ok(output),
        }
    }

    fn sigmoid(&self, x: &Tensor) -> Result<Tensor> {
        // 1 / (1 + exp(-x))
        let neg_x = x.neg().map_err(|e| CoreError::Backend(e.to_string()))?;
        let exp_neg = neg_x.exp().map_err(|e| CoreError::Backend(e.to_string()))?;
        let one = Tensor::from_vec(vec![1.0f32], &[1], &self.device)
            .map_err(|e| CoreError::Backend(e.to_string()))?;
        let one_plus = (&one + exp_neg).map_err(|e| CoreError::Backend(e.to_string()))?;
        (one / one_plus).map_err(|e| CoreError::Backend(e.to_string()))
    }

    fn tanh(&self, x: &Tensor) -> Result<Tensor> {
        x.tanh().map_err(|e| CoreError::Backend(e.to_string()))
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
        // Inverted dropout: scale by 1/(1-p), randomly zero elements
        let scale = 1.0 / (1.0 - p);
        let rand_tensor = Tensor::rand(0.0f64, 1.0, x.shape(), &self.device)
            .map_err(|e| CoreError::Backend(e.to_string()))?;
        let mask = rand_tensor
            .ge(p as f64)
            .map_err(|e| CoreError::Backend(e.to_string()))?
            .to_dtype(candle_core::DType::F32)
            .map_err(|e| CoreError::Backend(e.to_string()))?;
        let scaled = (&mask * x).map_err(|e| CoreError::Backend(e.to_string()))?;
        let scale_t = Tensor::from_vec(vec![scale], &[1], &self.device)
            .map_err(|e| CoreError::Backend(e.to_string()))?;
        (scaled * scale_t).map_err(|e| CoreError::Backend(e.to_string()))
    }

    fn concat(&self, tensors: &[&Tensor], dim: usize) -> Result<Tensor> {
        let owned: Vec<Tensor> = tensors.iter().map(|&t| t.clone()).collect();
        Tensor::cat(&owned, dim)
            .map_err(|e| CoreError::Backend(e.to_string()))
    }

    fn slice(&self, x: &Tensor, start: usize, end: usize) -> Result<Tensor> {
        x.narrow(0, start, end - start)
            .map_err(|e| CoreError::Backend(e.to_string()))
    }

    fn reshape(&self, x: &Tensor, shape: &[usize]) -> Result<Tensor> {
        x.reshape(shape)
            .map_err(|e| CoreError::Backend(e.to_string()))
    }

    fn add_scalar(&self, x: &Tensor, scalar: f32) -> Result<Tensor> {
        let s = Tensor::from_vec(vec![scalar], &[], &self.device)
            .map_err(|e| CoreError::Backend(e.to_string()))?;
        let broadcasted = s.broadcast_as(x.dims())
            .map_err(|e| CoreError::Backend(e.to_string()))?;
        (x + broadcasted).map_err(|e| CoreError::Backend(e.to_string()))
    }

    fn mul_scalar(&self, x: &Tensor, scalar: f32) -> Result<Tensor> {
        let s = Tensor::from_vec(vec![scalar], &[], &self.device)
            .map_err(|e| CoreError::Backend(e.to_string()))?;
        let broadcasted = s.broadcast_as(x.dims())
            .map_err(|e| CoreError::Backend(e.to_string()))?;
        (x * broadcasted).map_err(|e| CoreError::Backend(e.to_string()))
    }

    fn broadcast(&self, x: &Tensor, shape: &[usize]) -> Result<Tensor> {
        x.broadcast_as(shape)
            .map_err(|e| CoreError::Backend(e.to_string()))
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
        // candle doesn't have element-wise maximum directly, use where_cond
        let mask = a.gt(b).map_err(|e| CoreError::Backend(e.to_string()))?;
        mask.where_cond(a, b).map_err(|e| CoreError::Backend(e.to_string()))
    }

    fn gt_scalar(&self, x: &Tensor, scalar: f32) -> Result<Tensor> {
        let mask = x.gt(scalar).map_err(|e| CoreError::Backend(e.to_string()))?;
        mask.to_dtype(candle_core::DType::F32)
            .map_err(|e| CoreError::Backend(e.to_string()))
    }

    fn sum_all(&self, x: &Tensor) -> Result<Tensor> {
        x.sum_all().map_err(|e| CoreError::Backend(e.to_string()))
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
}
