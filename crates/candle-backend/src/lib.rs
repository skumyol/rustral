//! candle-core backend for MNR.
//!
//! Provides both CPU and CUDA acceleration via the candle-core crate.
//! On NVIDIA hardware, this backend automatically uses CUDA when available.
//!
//! # Example
//!
//! ```rust,ignore
//! use mnr_candle_backend::CandleBackend;
//!
//! let backend = CandleBackend::cuda(0).unwrap_or_else(|_| CandleBackend::cpu());
//! let tensor = backend.tensor_from_vec(vec![1.0f32, 2.0, 3.0], &[3]).unwrap();
//! ```

use candle_core::{Device, Tensor};
use mnr_core::{Backend, CoreError, Parameter, Result, TensorOps};
use thiserror::Error;

/// Errors specific to the candle backend.
#[derive(Debug, Error)]
pub enum CandleError {
    #[error("candle error: {0}")]
    Candle(#[from] candle_core::Error),

    #[error("CUDA not available: {0}")]
    CudaUnavailable(String),
}

/// candle-core backend for MNR.
#[derive(Clone, Debug)]
pub struct CandleBackend {
    device: Device,
}

impl CandleBackend {
    /// Create a CPU backend.
    pub fn cpu() -> Self {
        Self {
            device: Device::Cpu,
        }
    }

    /// Create a CUDA backend on the given device index.
    pub fn cuda(device_id: usize) -> Result<Self> {
        #[cfg(feature = "cuda")]
        {
            let device = Device::new_cuda(device_id)
                .map_err(|e| CoreError::Backend(format!("CUDA init failed: {}", e)))?;
            Ok(Self { device })
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
                return Self { device: dev };
            }
        }
        Self::cpu()
    }

    /// Return the underlying candle device.
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Create a tensor from host data.
    pub fn tensor_from_vec(&self, values: Vec<f32>, shape: &[usize]) -> Result<Tensor> {
        let tensor = Tensor::from_vec(values, shape, &self.device)
            .map_err(|e| CoreError::Backend(e.to_string()))?;
        Ok(tensor)
    }

    /// Read tensor data back to host.
    pub fn to_vec(&self, tensor: &Tensor) -> Vec<f32> {
        tensor.to_vec1::<f32>().unwrap_or_default()
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
        // candle ops are methods on Tensor, so we use a unit struct as the ops table
        // and implement TensorOps for it with &CandleBackend context
        &CandleOps { device: self.device.clone() }
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

/// Operation table for the candle backend.
struct CandleOps {
    device: Device,
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
        // candle transpose takes specific dimensions
        x.transpose(0, 1)
            .map_err(|e| CoreError::Backend(e.to_string()))
    }

    fn add(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        (a + b).map_err(|e| CoreError::Backend(e.to_string()))
    }

    fn add_row_vector(&self, a: &Tensor, row: &Tensor) -> Result<Tensor> {
        // Broadcast row to match a's shape
        a.broadcast_add(row)
            .map_err(|e| CoreError::Backend(e.to_string()))
    }

    fn relu(&self, x: &Tensor) -> Result<Tensor> {
        x.relu().map_err(|e| CoreError::Backend(e.to_string()))
    }

    fn softmax(&self, x: &Tensor) -> Result<Tensor> {
        // Softmax over last dimension
        let last_dim = x.dims().len().saturating_sub(1);
        candle_core::ops::softmax(x, last_dim)
            .map_err(|e| CoreError::Backend(e.to_string()))
    }

    fn log_softmax(&self, x: &Tensor) -> Result<Tensor> {
        let last_dim = x.dims().len().saturating_sub(1);
        candle_core::ops::log_softmax(x, last_dim)
            .map_err(|e| CoreError::Backend(e.to_string()))
    }

    fn argmax(&self, x: &Tensor) -> Result<usize> {
        let flat = x.flatten_all().map_err(|e| CoreError::Backend(e.to_string()))?;
        let argmax = flat.argmax(0).map_err(|e| CoreError::Backend(e.to_string()))?;
        let idx = argmax.to_scalar::<i64>().map_err(|e| CoreError::Backend(e.to_string()))?;
        Ok(idx as usize)
    }

    fn gather_rows(&self, table: &Parameter<CandleBackend>, ids: &[usize]) -> Result<Tensor> {
        let table_tensor = table.tensor();
        let ids_tensor = Tensor::from_vec(ids.to_vec(), &[ids.len()], &self.device)
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
            Some(b) => output.broadcast_add(b.tensor())
                .map_err(|e| CoreError::Backend(e.to_string())),
            None => Ok(output),
        }
    }

    fn sigmoid(&self, x: &Tensor) -> Result<Tensor> {
        candle_core::ops::sigmoid(x)
            .map_err(|e| CoreError::Backend(e.to_string()))
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
        // candle has built-in dropout; scale is handled internally
        x.dropout(p)
            .map_err(|e| CoreError::Backend(e.to_string()))
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
        (x + scalar).map_err(|e| CoreError::Backend(e.to_string()))
    }

    fn mul_scalar(&self, x: &Tensor, scalar: f32) -> Result<Tensor> {
        (x * scalar).map_err(|e| CoreError::Backend(e.to_string()))
    }

    fn broadcast(&self, x: &Tensor, shape: &[usize]) -> Result<Tensor> {
        x.broadcast_as(shape)
            .map_err(|e| CoreError::Backend(e.to_string()))
    }

    fn neg(&self, x: &Tensor) -> Result<Tensor> {
        (-x).map_err(|e| CoreError::Backend(e.to_string()))
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
        x.gt(scalar).map_err(|e| CoreError::Backend(e.to_string()))?
            .to_dtype(candle_core::DType::F32)
            .map_err(|e| CoreError::Backend(e.to_string()))
    }

    fn sum_all(&self, x: &Tensor) -> Result<Tensor> {
        x.sum_all().map_err(|e| CoreError::Backend(e.to_string()))
    }

    fn tensor_element(&self, x: &Tensor, index: usize) -> Result<f32> {
        let flat = x.flatten_all().map_err(|e| CoreError::Backend(e.to_string()))?;
        let val = flat.i(index).map_err(|e| CoreError::Backend(e.to_string()))?;
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
}
