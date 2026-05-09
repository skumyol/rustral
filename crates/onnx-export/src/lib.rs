//! Experimental ONNX model export (subset: single Linear as MatMul + Add).
//!
//! This is a **spike** for Track H Phase 5: supported op count is intentionally tiny.
//! Validate exports with ONNX Runtime or your deployment stack.

#![allow(clippy::large_enum_variant)]

include!(concat!(env!("OUT_DIR"), "/onnx.rs"));

use prost::Message;

/// ONNX tensor element types we use (`TensorProto.DataType.FLOAT`).
pub const TENSOR_ELEMENT_TYPE_FLOAT: i32 = 1;

#[derive(Debug, thiserror::Error)]
pub enum OnnxExportError {
    #[error("weight length mismatch: expected {expected}, got {got}")]
    WeightLen { expected: usize, got: usize },
    #[error("bias length mismatch: expected {expected}, got {got}")]
    BiasLen { expected: usize, got: usize },
    #[error("protobuf encode failed: {0}")]
    Encode(#[from] prost::EncodeError),
}

/// Export one linear layer: `Y = X @ W + B` with `X` shaped `[batch, in_features]`,
/// `W` row-major `[in_features, out_features]`, `B` length `out_features`.
///
/// The batch dimension is **symbolic** (`dim_param` `"batch"`).
/// Opset **17** for `MatMul` / `Add`.
pub fn export_linear_f32(
    in_features: i64,
    out_features: i64,
    weight_row_major: &[f32],
    bias: &[f32],
) -> Result<Vec<u8>, OnnxExportError> {
    let w_el = (in_features * out_features) as usize;
    if weight_row_major.len() != w_el {
        return Err(OnnxExportError::WeightLen {
            expected: w_el,
            got: weight_row_major.len(),
        });
    }
    if bias.len() != out_features as usize {
        return Err(OnnxExportError::BiasLen {
            expected: out_features as usize,
            got: bias.len(),
        });
    }

    let mut w_bytes = Vec::with_capacity(weight_row_major.len() * 4);
    for v in weight_row_major {
        w_bytes.extend_from_slice(&v.to_le_bytes());
    }
    let mut b_bytes = Vec::with_capacity(bias.len() * 4);
    for v in bias {
        b_bytes.extend_from_slice(&v.to_le_bytes());
    }

    let graph = GraphProto {
        node: vec![
            NodeProto {
                input: vec!["X".into(), "W".into()],
                output: vec!["mm_out".into()],
                name: Some("matmul".into()),
                op_type: Some("MatMul".into()),
                ..Default::default()
            },
            NodeProto {
                input: vec!["mm_out".into(), "B".into()],
                output: vec!["Y".into()],
                name: Some("add_bias".into()),
                op_type: Some("Add".into()),
                ..Default::default()
            },
        ],
        name: Some("linear_graph".into()),
        initializer: vec![
            tensor_f32("W", vec![in_features, out_features], w_bytes),
            tensor_f32("B", vec![out_features], b_bytes),
        ],
        input: vec![value_info_matrix("X", "batch", in_features)],
        output: vec![value_info_matrix("Y", "batch", out_features)],
        ..Default::default()
    };

    let model = ModelProto {
        ir_version: Some(9),
        opset_import: vec![OperatorSetIdProto {
            domain: Some(String::new()),
            version: Some(17),
        }],
        producer_name: Some("rustral-onnx-export".into()),
        producer_version: Some(env!("CARGO_PKG_VERSION").into()),
        graph: Some(graph),
        ..Default::default()
    };

    let mut buf = Vec::new();
    model.encode(&mut buf)?;
    Ok(buf)
}

fn tensor_f32(name: &str, dims: Vec<i64>, raw_data: Vec<u8>) -> TensorProto {
    TensorProto {
        dims,
        data_type: Some(TENSOR_ELEMENT_TYPE_FLOAT),
        name: Some(name.into()),
        raw_data: Some(raw_data),
        ..Default::default()
    }
}

fn value_info_matrix(name: &str, batch_param: &str, static_dim: i64) -> ValueInfoProto {
    let shape = TensorShapeProto {
        dim: vec![
            tensor_shape_proto::Dimension {
                denotation: None,
                value: Some(tensor_shape_proto::dimension::Value::DimParam(batch_param.into())),
            },
            tensor_shape_proto::Dimension {
                denotation: None,
                value: Some(tensor_shape_proto::dimension::Value::DimValue(static_dim)),
            },
        ],
    };
    ValueInfoProto {
        name: Some(name.into()),
        r#type: Some(TypeProto {
            denotation: None,
            value: Some(type_proto::Value::TensorType(type_proto::Tensor {
                elem_type: Some(TENSOR_ELEMENT_TYPE_FLOAT),
                shape: Some(shape),
            })),
        }),
        ..Default::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linear_roundtrip_decode() {
        let w = vec![1.0f32, 0.0, 0.0, 1.0];
        let b = vec![0.0f32, 0.0];
        let bytes = export_linear_f32(2, 2, &w, &b).unwrap();
        assert!(!bytes.is_empty());
        let m = ModelProto::decode(bytes.as_slice()).unwrap();
        let g = m.graph.expect("graph");
        assert_eq!(g.node.len(), 2);
    }
}
