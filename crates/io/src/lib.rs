//! Serialization support for Rustral parameters and models.
//!
//! Provides saving/loading to the [Safetensors] format, which is a
//! simple, safe, fast tensor serialization format that requires no
//! `pickle` or arbitrary code execution.
//!
//! [Safetensors]: https://huggingface.co/docs/safetensors/index

use std::collections::HashMap;

use rustral_core::{Backend, Parameter, TensorShape};
use safetensors::{serialize, SafeTensors, View};
use thiserror::Error;

/// I/O errors specific to tensor serialization.
#[derive(Debug, Error)]
pub enum IoError {
    /// Underlying safetensors error.
    #[error("safetensors error: {0}")]
    SafeTensor(#[from] safetensors::SafeTensorError),

    /// Shape mismatch between stored tensor and expected shape.
    #[error("shape mismatch for '{name}': expected {expected:?}, got {actual:?}")]
    ShapeMismatch { name: String, expected: Vec<usize>, actual: Vec<usize> },

    /// A requested tensor was not found in the file.
    #[error("tensor '{0}' not found in checkpoint")]
    MissingTensor(String),

    /// A tensor had an unexpected dtype.
    #[error("dtype mismatch for '{name}': expected {expected:?}, got {actual:?}")]
    DTypeMismatch { name: String, expected: safetensors::Dtype, actual: safetensors::Dtype },
}

/// A loaded tensor entry from a safetensors state dict.
#[derive(Clone, Debug, PartialEq)]
pub struct StateTensor {
    pub shape: Vec<usize>,
    pub dtype: safetensors::Dtype,
    pub data: Vec<f32>,
}

/// Owned tensor data for serialization.
///
/// This stores the shape and data together to avoid lifetime issues
/// with the safetensors View trait.
struct TensorData {
    shape: Vec<usize>,
    data: Vec<u8>,
}

impl View for TensorData {
    fn dtype(&self) -> safetensors::Dtype {
        safetensors::Dtype::F32
    }

    fn shape(&self) -> &[usize] {
        &self.shape
    }

    fn data(&self) -> std::borrow::Cow<'_, [u8]> {
        std::borrow::Cow::Borrowed(&self.data)
    }

    fn data_len(&self) -> usize {
        self.data.len()
    }
}

/// Save a collection of named parameters to a Safetensors byte buffer.
///
/// # Example
/// ```ignore
/// let params = vec![
///     ("encoder.weight".to_string(), &encoder.weight),
///     ("encoder.bias".to_string(), encoder.bias.as_ref().unwrap()),
/// ];
/// let bytes = save_parameters::<CpuBackend>(&params).unwrap();
/// std::fs::write("model.safetensors", &bytes).unwrap();
/// ```
pub fn save_parameters<B: Backend>(params: &[(String, &Parameter<B>)]) -> Result<Vec<u8>, IoError>
where
    B::Tensor: AsRef<[f32]> + TensorShape,
{
    let mut views: HashMap<String, TensorData> = HashMap::with_capacity(params.len());
    for (name, param) in params {
        let tensor = param.tensor();
        let shape = tensor.shape().to_vec();
        let floats: &[f32] = tensor.as_ref();
        let bytes: Vec<u8> = bytemuck::cast_slice(floats).to_vec();
        views.insert(name.clone(), TensorData { shape, data: bytes });
    }

    Ok(serialize(views, &None)?)
}

/// Load a collection of parameters from a Safetensors byte buffer.
///
/// # Example
/// ```ignore
/// let bytes = std::fs::read("model.safetensors").unwrap();
/// let tensors = load_parameters(&bytes).unwrap();
/// let weight = tensors.get("encoder.weight").unwrap();
/// ```
pub fn load_parameters(data: &[u8]) -> Result<HashMap<String, Vec<f32>>, IoError> {
    let safe = SafeTensors::deserialize(data)?;
    let mut result = HashMap::with_capacity(safe.len());

    for (name, view) in safe.tensors() {
        if view.dtype() != safetensors::Dtype::F32 {
            // Skip non-f32 tensors or return an error depending on policy.
            // For now we just skip.
            continue;
        }
        let shape = view.shape().to_vec();
        let bytes = view.data();
        let floats: Vec<f32> = bytemuck::cast_slice(&bytes).to_vec();

        // Verify element count matches shape
        let expected_count: usize = shape.iter().product();
        if floats.len() != expected_count {
            return Err(IoError::ShapeMismatch {
                name: name.to_string(),
                expected: shape,
                actual: vec![floats.len()],
            });
        }

        result.insert(name.to_string(), floats);
    }

    Ok(result)
}

/// Save a typed state dictionary with shapes (and optional metadata).
pub fn save_state_dict_typed(
    state_dict: &HashMap<String, (Vec<f32>, Vec<usize>)>,
    metadata: Option<HashMap<String, String>>,
) -> Result<Vec<u8>, IoError> {
    let mut views: HashMap<String, TensorData> = HashMap::with_capacity(state_dict.len());
    for (name, (data, shape)) in state_dict {
        let bytes: Vec<u8> = bytemuck::cast_slice(data).to_vec();
        views.insert(name.clone(), TensorData { shape: shape.clone(), data: bytes });
    }
    Ok(serialize(views, &metadata)?)
}

/// Load a typed state dictionary preserving shapes and dtype.
///
/// This is the strict loader used for whole-model save/load.
pub fn load_state_dict_typed(data: &[u8]) -> Result<HashMap<String, StateTensor>, IoError> {
    let safe = SafeTensors::deserialize(data)?;
    let mut result: HashMap<String, StateTensor> = HashMap::with_capacity(safe.len());
    for (name, view) in safe.tensors() {
        let dtype = view.dtype();
        if dtype != safetensors::Dtype::F32 {
            return Err(IoError::DTypeMismatch {
                name: name.to_string(),
                expected: safetensors::Dtype::F32,
                actual: dtype,
            });
        }
        let shape = view.shape().to_vec();
        let bytes = view.data();
        let floats: Vec<f32> = bytemuck::cast_slice(&bytes).to_vec();
        let expected_count: usize = shape.iter().product();
        if floats.len() != expected_count {
            return Err(IoError::ShapeMismatch {
                name: name.to_string(),
                expected: shape,
                actual: vec![floats.len()],
            });
        }
        result.insert(name.to_string(), StateTensor { shape, dtype, data: floats });
    }

    // Note: safetensors metadata access is not exposed in our current dependency version.
    // We still allow writers to include metadata for inspection by external tools.
    Ok(result)
}

/// Save a state dictionary (name -> flat f32 values) to a Safetensors byte buffer.
///
/// Since the state dictionary does not carry shape information, each tensor
/// is stored as a 1-D array. The consumer (typically `Saveable::load_state_dict`)
/// is expected to know the expected shapes and reshape accordingly.
///
/// # Example
/// ```ignore
/// let mut dict = HashMap::new();
/// dict.insert("weight".to_string(), vec![1.0, 2.0, 3.0, 4.0]);
/// let bytes = save_state_dict(&dict).unwrap();
/// std::fs::write("model.safetensors", &bytes).unwrap();
/// ```
pub fn save_state_dict(state_dict: &HashMap<String, Vec<f32>>) -> Result<Vec<u8>, IoError> {
    let mut views: HashMap<String, TensorData> = HashMap::with_capacity(state_dict.len());
    for (name, data) in state_dict {
        let shape = vec![data.len()];
        let bytes: Vec<u8> = bytemuck::cast_slice(data).to_vec();
        views.insert(name.clone(), TensorData { shape, data: bytes });
    }
    Ok(serialize(views, &None)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustral_core::Backend;
    use rustral_ndarray_backend::CpuBackend;

    #[test]
    fn test_roundtrip_single_tensor() {
        let backend = CpuBackend::default();
        let param = backend.normal_parameter("test", &[2, 2], 42, 0.0).unwrap();

        let params = vec![("test".to_string(), &param)];
        let bytes = save_parameters::<CpuBackend>(&params).unwrap();

        let loaded = load_parameters(&bytes).unwrap();
        let loaded_test = loaded.get("test").unwrap();
        assert_eq!(loaded_test.len(), 4);
    }

    #[test]
    fn test_load_missing_tensor() {
        let backend = CpuBackend::default();
        let param = backend.normal_parameter("a", &[2], 1, 0.0).unwrap();
        let params = vec![("a".to_string(), &param)];
        let bytes = save_parameters::<CpuBackend>(&params).unwrap();

        let loaded = load_parameters(&bytes).unwrap();
        assert!(!loaded.contains_key("b"));
    }

    #[test]
    fn test_save_multiple_tensors() {
        let backend = CpuBackend::default();
        let p1 = backend.normal_parameter("w", &[2, 2], 1, 0.0).unwrap();
        let p2 = backend.normal_parameter("b", &[2], 2, 0.0).unwrap();
        let params = vec![("w".to_string(), &p1), ("b".to_string(), &p2)];
        let bytes = save_parameters::<CpuBackend>(&params).unwrap();

        let loaded = load_parameters(&bytes).unwrap();
        assert_eq!(loaded.len(), 2);
        assert_eq!(loaded.get("w").unwrap().len(), 4);
        assert_eq!(loaded.get("b").unwrap().len(), 2);
    }

    #[test]
    fn test_save_state_dict_roundtrip() {
        let mut dict = HashMap::new();
        dict.insert("weight".to_string(), vec![1.0f32, 2.0, 3.0, 4.0]);
        dict.insert("bias".to_string(), vec![0.5f32, 0.5f32]);

        let bytes = save_state_dict(&dict).unwrap();
        let loaded = load_parameters(&bytes).unwrap();

        assert_eq!(loaded.len(), 2);
        assert_eq!(loaded.get("weight").unwrap(), &vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(loaded.get("bias").unwrap(), &vec![0.5, 0.5]);
    }

    #[test]
    fn test_typed_state_dict_roundtrip_preserves_shape() {
        let mut dict: HashMap<String, (Vec<f32>, Vec<usize>)> = HashMap::new();
        dict.insert("w".to_string(), (vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]));
        let bytes = save_state_dict_typed(&dict, None).unwrap();
        let loaded = load_state_dict_typed(&bytes).unwrap();
        assert_eq!(
            loaded.get("w").unwrap(),
            &StateTensor { shape: vec![2, 2], dtype: safetensors::Dtype::F32, data: vec![1.0, 2.0, 3.0, 4.0] }
        );
    }
}
