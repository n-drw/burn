//! # MatMulNBits
//!
//! N-bit quantized matrix multiplication (QOperator format).
//!
//! **ONNX Spec**: <https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#com.microsoft.MatMulNBits>
//!
//! This is an ONNX Runtime contrib operator for efficient INT4 matrix multiplication.
//!
//! ## Formula
//! `Y = A @ dequantize(B, scales, zero_points)`
//!
//! Where B is packed N-bit weights and dequantization happens on-the-fly during matmul.
//!
//! ## Attributes
//! - **K**: Number of rows in B (before transposition)
//! - **N**: Number of columns in B (before transposition)  
//! - **bits**: Number of bits for quantization (typically 4)
//! - **block_size**: Block size for quantization (e.g., 32, 64, 128)

use derive_new::new;
use onnx_ir_derive::NodeBuilder;

use crate::ir::{ArgType, Argument, AttributeValue, Node, RawNode, TensorType};
use crate::processor::{
    InputSpec, NodeProcessor, NodeSpec, OutputPreferences, OutputSpec, ProcessError,
};

/// Configuration for MatMulNBits operations
#[derive(Debug, Clone, new)]
pub struct MatMulNBitsConfig {
    /// Number of rows in weight matrix B
    pub k: i64,
    /// Number of columns in weight matrix B
    pub n: i64,
    /// Number of bits for quantization (typically 4)
    pub bits: i64,
    /// Block size for block quantization
    pub block_size: i64,
}

impl Default for MatMulNBitsConfig {
    fn default() -> Self {
        Self {
            k: 0,
            n: 0,
            bits: 4,
            block_size: 128,
        }
    }
}

/// Node representation for MatMulNBits operation
#[derive(Debug, Clone, NodeBuilder)]
pub struct MatMulNBitsNode {
    pub name: String,
    pub inputs: Vec<Argument>,
    pub outputs: Vec<Argument>,
    pub config: MatMulNBitsConfig,
}

pub(crate) struct MatMulNBitsProcessor;

impl NodeProcessor for MatMulNBitsProcessor {
    type Config = MatMulNBitsConfig;

    fn spec(&self) -> NodeSpec {
        NodeSpec {
            min_opset: 1, // contrib op, no standard opset
            max_opset: None,
            inputs: InputSpec::Range(3, 6), // A, B, scales, [zero_points], [g_idx], [bias]
            outputs: OutputSpec::Exact(1),
        }
    }

    fn infer_types(
        &self,
        node: &mut RawNode,
        _opset: usize,
        _output_preferences: &OutputPreferences,
    ) -> Result<(), ProcessError> {
        // Get input A's type to determine output type
        let a_tensor = match &node.inputs[0].ty {
            ArgType::Tensor(t) => t.clone(),
            _ => return Err(ProcessError::TypeMismatch {
                expected: "Tensor".into(),
                actual: format!("{:?}", node.inputs[0].ty),
            }),
        };

        // Output shape: [..., M, N] where M comes from A and N from config
        let out_rank = a_tensor.rank;
        
        // Output dtype matches input A (typically F32 or F16)
        node.outputs[0].ty = ArgType::Tensor(TensorType {
            dtype: a_tensor.dtype,
            rank: out_rank,
            static_shape: None,
        });

        Ok(())
    }

    fn extract_config(&self, node: &RawNode, _opset: usize) -> Result<Self::Config, ProcessError> {
        let k = match node.attrs.get("K") {
            Some(AttributeValue::Int64(v)) => *v,
            None => 0,
            _ => {
                return Err(ProcessError::InvalidAttribute {
                    name: "K".to_string(),
                    reason: "must be Int64".to_string(),
                });
            }
        };

        let n = match node.attrs.get("N") {
            Some(AttributeValue::Int64(v)) => *v,
            None => 0,
            _ => {
                return Err(ProcessError::InvalidAttribute {
                    name: "N".to_string(),
                    reason: "must be Int64".to_string(),
                });
            }
        };

        let bits = match node.attrs.get("bits") {
            Some(AttributeValue::Int64(v)) => *v,
            None => 4,
            _ => {
                return Err(ProcessError::InvalidAttribute {
                    name: "bits".to_string(),
                    reason: "must be Int64".to_string(),
                });
            }
        };

        let block_size = match node.attrs.get("block_size") {
            Some(AttributeValue::Int64(v)) => *v,
            None => 128,
            _ => {
                return Err(ProcessError::InvalidAttribute {
                    name: "block_size".to_string(),
                    reason: "must be Int64".to_string(),
                });
            }
        };

        log::debug!(
            "MatMulNBits '{}': K={}, N={}, bits={}, block_size={}",
            node.name, k, n, bits, block_size
        );

        Ok(MatMulNBitsConfig::new(k, n, bits, block_size))
    }

    fn build_node(&self, builder: RawNode, opset: usize) -> Node {
        let config = self
            .extract_config(&builder, opset)
            .expect("Config extraction failed");

        Node::MatMulNBits(MatMulNBitsNode {
            name: builder.name,
            inputs: builder.inputs,
            outputs: builder.outputs,
            config,
        })
    }
}
