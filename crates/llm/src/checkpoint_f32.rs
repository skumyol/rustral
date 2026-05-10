//! Shared F32 safetensors slice helpers for HF-style checkpoints (`rustral-io` [`TensorEntry`]).

use std::collections::HashMap;

use rustral_io::{MetaStateDict, TensorEntry};
use safetensors::Dtype;

use crate::LlmError;

pub(crate) fn tensor_entry_f32(entry: &TensorEntry) -> Result<Vec<f32>, LlmError> {
    if entry.dtype != Dtype::F32 {
        return Err(LlmError::UnsupportedCheckpointDtype {
            name: entry.name.clone(),
            dtype: format!("{:?}", entry.dtype),
        });
    }
    let n: usize = entry.shape.iter().product();
    if entry.data.len() != n * 4 {
        return Err(LlmError::InvalidArg(format!(
            "tensor '{}': expected {} f32 bytes for shape {:?}, got {}",
            entry.name,
            n * 4,
            entry.shape,
            entry.data.len()
        )));
    }
    Ok(entry
        .data
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

/// Row-major `[rows, cols]` → row-major `[cols, rows]`.
pub(crate) fn transpose_2d_row_major(data: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0f32; rows * cols];
    for i in 0..rows {
        for j in 0..cols {
            out[j * rows + i] = data[i * cols + j];
        }
    }
    out
}

/// HF `Linear` exports sometimes use the transposed shape vs [`rustral_nn::Linear`] storage `[out_dim, in_dim]`.
pub(crate) fn tensor_entry_f32_rustral_matrix(
    entry: &TensorEntry,
    rustral_shape: [usize; 2],
) -> Result<(Vec<f32>, Vec<usize>), LlmError> {
    let data = tensor_entry_f32(entry)?;
    let shape = entry.shape.clone();
    if shape == rustral_shape.to_vec() {
        return Ok((data, shape));
    }
    if shape.len() == 2 && shape[0] == rustral_shape[1] && shape[1] == rustral_shape[0] {
        let t = transpose_2d_row_major(&data, shape[0], shape[1]);
        return Ok((t, rustral_shape.to_vec()));
    }
    Err(LlmError::CheckpointShapeMismatch {
        name: entry.name.clone(),
        expected: rustral_shape.to_vec(),
        got: shape,
    })
}

pub(crate) fn vec_f32_to_tensor_entry(name: impl Into<String>, shape: Vec<usize>, data: Vec<f32>) -> TensorEntry {
    let mut bytes = Vec::with_capacity(data.len() * 4);
    for x in data {
        bytes.extend_from_slice(&x.to_le_bytes());
    }
    TensorEntry {
        name: name.into(),
        shape,
        dtype: Dtype::F32,
        data: bytes,
    }
}

pub(crate) fn insert_hf_tensor_if_present(
    out: &mut HashMap<String, (Vec<f32>, Vec<usize>)>,
    meta: &MetaStateDict,
    rustral: &str,
    hf: &str,
) -> Result<(), LlmError> {
    let Some(entry) = meta.tensors.get(hf) else {
        return Ok(());
    };
    let data = tensor_entry_f32(entry)?;
    out.insert(rustral.to_string(), (data, entry.shape.clone()));
    Ok(())
}
