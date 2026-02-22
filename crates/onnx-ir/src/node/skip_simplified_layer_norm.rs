//! # SkipSimplifiedLayerNormalization
//!
//! Microsoft-specific fused operator that combines a skip/residual connection with
//! SimplifiedLayerNormalization (RMSNorm).
//!
//! **Spec**: <https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#com.microsoft.SkipSimplifiedLayerNormalization>
//!
//! Computes: output = RMSNorm(input + skip, gamma, epsilon)
//!
//! ## Inputs
//! 0. input: (T) - Input tensor
//! 1. skip: (T) - Skip/residual tensor to add
//! 2. gamma: (T) - Scale tensor for normalization
//! 3. bias: (T) - Optional bias (added before normalization)
//!
//! ## Outputs
//! 0. output: (T) - Normalized result
//! 1. mean: (T) - Optional (not used in simplified version)
//! 2. inv_std_var: (T) - Optional
//! 3. input_skip_bias_sum: (T) - Optional, the sum input + skip (+ bias)

use derive_new::new;
use onnx_ir_derive::NodeBuilder;

use crate::ir::{ArgType, Argument, Node, RawNode, TensorType};
use crate::processor::{NodeProcessor, OutputPreferences, ProcessError};

#[derive(Debug, Clone, new)]
pub struct SkipSimplifiedLayerNormConfig {
    pub epsilon: f64,
}

/// Node representation for SkipSimplifiedLayerNormalization operation
#[derive(Debug, Clone, NodeBuilder)]
pub struct SkipSimplifiedLayerNormNode {
    pub name: String,
    pub inputs: Vec<Argument>,
    pub outputs: Vec<Argument>,
    pub config: SkipSimplifiedLayerNormConfig,
}

pub(crate) struct SkipSimplifiedLayerNormProcessor;

impl NodeProcessor for SkipSimplifiedLayerNormProcessor {
    type Config = SkipSimplifiedLayerNormConfig;

    fn spec(&self) -> crate::processor::NodeSpec {
        crate::processor::NodeSpec {
            min_opset: 1,
            max_opset: None,
            inputs: crate::processor::InputSpec::AtLeast(3),
            outputs: crate::processor::OutputSpec::Range(1, 4),
        }
    }

    fn infer_types(
        &self,
        node: &mut RawNode,
        _opset: usize,
        _output_preferences: &OutputPreferences,
    ) -> Result<(), ProcessError> {
        if node.inputs.len() < 3 {
            return Err(ProcessError::InvalidInputCount {
                expected: 3,
                actual: node.inputs.len(),
            });
        }

        let input_type = match &node.inputs[0].ty {
            ArgType::Tensor(t) => t.clone(),
            _ => {
                return Err(ProcessError::Custom(
                    "SkipSimplifiedLayerNormalization: input must be a tensor".to_string(),
                ))
            }
        };

        // Output 0: normalized result, same type as input
        node.outputs[0].ty = ArgType::Tensor(input_type.clone());

        // Output 3 (input_skip_bias_sum): same type as input, if present
        if let Some(skip_sum_out) = node.outputs.get_mut(3) {
            skip_sum_out.ty = ArgType::Tensor(input_type.clone());
        }

        // Outputs 1 and 2 (mean, inv_std_var) - set type if present
        for i in 1..=2 {
            if let Some(out) = node.outputs.get_mut(i) {
                out.ty = ArgType::Tensor(TensorType {
                    dtype: input_type.dtype,
                    rank: input_type.rank,
                    static_shape: None,
                });
            }
        }

        Ok(())
    }

    fn extract_config(&self, node: &RawNode, _opset: usize) -> Result<Self::Config, ProcessError> {
        let mut epsilon = 1e-5f64;

        for (key, value) in node.attrs.iter() {
            match key.as_str() {
                "epsilon" => epsilon = value.clone().into_f32() as f64,
                _ => {}
            }
        }

        Ok(SkipSimplifiedLayerNormConfig::new(epsilon))
    }

    fn build_node(&self, builder: RawNode, opset: usize) -> Node {
        let config = self
            .extract_config(&builder, opset)
            .expect("Config extraction failed");

        Node::SkipSimplifiedLayerNormalization(SkipSimplifiedLayerNormNode {
            name: builder.name,
            inputs: builder.inputs,
            outputs: builder.outputs,
            config,
        })
    }
}
