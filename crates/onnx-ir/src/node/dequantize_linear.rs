//! # DequantizeLinear
//!
//! Linear dequantization operator - converts quantized values to floating point.
//!
//! **ONNX Spec**: <https://onnx.ai/onnx/operators/onnx__DequantizeLinear.html>
//!
//! ## Formula
//! `y = (x - x_zero_point) * x_scale`
//!
//! ## Opset Versions
//! - **Opset 10**: Initial version
//! - **Opset 13**: Added per-axis quantization
//! - **Opset 19**: Extended type support
//! - **Opset 21**: Added INT4/UINT4 support and block quantization

use derive_new::new;
use onnx_ir_derive::NodeBuilder;

use crate::ir::{ArgType, Argument, DType, Node, RawNode, TensorType};
use crate::processor::{
    InputSpec, NodeProcessor, NodeSpec, OutputPreferences, OutputSpec, ProcessError,
};

/// Configuration for DequantizeLinear operations
#[derive(Debug, Clone, new)]
pub struct DequantizeLinearConfig {
    /// The axis of the dequantizing dimension (for per-axis quantization)
    /// Negative value means counting from the back.
    pub axis: i64,
    /// Block size for blocked quantization (0 means no blocking)
    pub block_size: i64,
}

impl Default for DequantizeLinearConfig {
    fn default() -> Self {
        Self {
            axis: 1,
            block_size: 0,
        }
    }
}

/// Node representation for DequantizeLinear operation
#[derive(Debug, Clone, NodeBuilder)]
pub struct DequantizeLinearNode {
    pub name: String,
    pub inputs: Vec<Argument>,
    pub outputs: Vec<Argument>,
    pub config: DequantizeLinearConfig,
}

pub(crate) struct DequantizeLinearProcessor;

impl NodeProcessor for DequantizeLinearProcessor {
    type Config = DequantizeLinearConfig;

    fn spec(&self) -> NodeSpec {
        NodeSpec {
            min_opset: 10,
            max_opset: None,
            inputs: InputSpec::Range(2, 3), // x, x_scale, x_zero_point (optional)
            outputs: OutputSpec::Exact(1),
        }
    }

    fn infer_types(
        &self,
        node: &mut RawNode,
        _opset: usize,
        _output_preferences: &OutputPreferences,
    ) -> Result<(), ProcessError> {
        // Get input tensor info
        let input_tensor = match &node.inputs[0].ty {
            ArgType::Tensor(tensor) => tensor.clone(),
            _ => {
                return Err(ProcessError::TypeMismatch {
                    expected: "Tensor".to_string(),
                    actual: format!("{:?}", node.inputs[0].ty),
                });
            }
        };

        // Get scale tensor to determine output type
        let scale_dtype = match &node.inputs[1].ty {
            ArgType::Tensor(tensor) => tensor.dtype,
            ArgType::Scalar(dtype) => *dtype,
            _ => {
                return Err(ProcessError::TypeMismatch {
                    expected: "Tensor or Scalar".to_string(),
                    actual: format!("{:?}", node.inputs[1].ty),
                });
            }
        };

        // Output has same shape as input, but dtype comes from scale
        // (FP16 scale -> FP16 output, FP32 scale -> FP32 output)
        node.outputs[0].ty = ArgType::Tensor(TensorType {
            dtype: scale_dtype,
            rank: input_tensor.rank,
            static_shape: input_tensor.static_shape,
        });

        Ok(())
    }

    fn extract_config(&self, node: &RawNode, _opset: usize) -> Result<Self::Config, ProcessError> {
        use crate::ir::AttributeValue;

        let axis = match node.attrs.get("axis") {
            Some(AttributeValue::Int64(v)) => *v,
            None => 1, // default
            _ => {
                return Err(ProcessError::InvalidAttribute {
                    name: "axis".to_string(),
                    reason: "must be Int64".to_string(),
                });
            }
        };

        let block_size = match node.attrs.get("block_size") {
            Some(AttributeValue::Int64(v)) => *v,
            None => 0, // default - no blocking
            _ => {
                return Err(ProcessError::InvalidAttribute {
                    name: "block_size".to_string(),
                    reason: "must be Int64".to_string(),
                });
            }
        };

        Ok(DequantizeLinearConfig::new(axis, block_size))
    }

    fn build_node(&self, builder: RawNode, opset: usize) -> Node {
        let config = self
            .extract_config(&builder, opset)
            .expect("Config extraction failed");

        Node::DequantizeLinear(DequantizeLinearNode {
            name: builder.name,
            inputs: builder.inputs,
            outputs: builder.outputs,
            config,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::NodeType;
    use crate::node::test_utils::TestNodeBuilder;

    fn create_test_node(input_rank: usize) -> RawNode {
        TestNodeBuilder::new(NodeType::DequantizeLinear, "test_dequantize")
            .input_tensor_i8("x", input_rank, None)
            .input_tensor_f32("x_scale", 0, None) // scalar scale
            .output_tensor_f32("y", input_rank, None)
            .build()
    }

    #[test]
    fn test_dequantize_linear_basic() {
        let mut node = create_test_node(2);
        
        let processor = DequantizeLinearProcessor;
        let prefs = OutputPreferences::new();
        processor.infer_types(&mut node, 21, &prefs).unwrap();

        match &node.outputs[0].ty {
            ArgType::Tensor(tensor) => {
                assert_eq!(tensor.dtype, DType::F32);
                assert_eq!(tensor.rank, 2);
            }
            _ => panic!("Expected tensor output"),
        }
    }

    #[test]
    fn test_dequantize_linear_config() {
        let node = TestNodeBuilder::new(NodeType::DequantizeLinear, "test")
            .input_tensor_i8("x", 2, None)
            .input_tensor_f32("x_scale", 1, None)
            .output_tensor_f32("y", 2, None)
            .attr_int("axis", 0)
            .attr_int("block_size", 128)
            .build();

        let processor = DequantizeLinearProcessor;
        let config = processor.extract_config(&node, 21).unwrap();

        assert_eq!(config.axis, 0);
        assert_eq!(config.block_size, 128);
    }
}
