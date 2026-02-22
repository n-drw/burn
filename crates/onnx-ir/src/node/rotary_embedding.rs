//! # RotaryEmbedding
//!
//! Microsoft-specific RotaryEmbedding operator.
//!
//! **Spec**: <https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#com.microsoft.RotaryEmbedding>
//!
//! ## Inputs
//! 0. input: (batch, seq_len, hidden) or (batch, num_heads, seq_len, head_dim)
//! 1. position_ids: (batch, seq_len) or (1, seq_len)
//! 2. cos_cache: (max_seq_len, head_dim/2)
//! 3. sin_cache: (max_seq_len, head_dim/2)
//!
//! ## Outputs
//! 0. output: same shape as input

use derive_new::new;
use onnx_ir_derive::NodeBuilder;

use crate::ir::{Argument, Node, RawNode};
use crate::processor::{NodeProcessor, OutputPreferences, ProcessError};

#[derive(Debug, Clone, new)]
pub struct RotaryEmbeddingConfig {
    pub interleaved: bool,
    pub num_heads: Option<usize>,
    pub rotary_embedding_dim: Option<usize>,
    pub scale: Option<f64>,
}

/// Node representation for RotaryEmbedding operation
#[derive(Debug, Clone, NodeBuilder)]
pub struct RotaryEmbeddingNode {
    pub name: String,
    pub inputs: Vec<Argument>,
    pub outputs: Vec<Argument>,
    pub config: RotaryEmbeddingConfig,
}

pub(crate) struct RotaryEmbeddingProcessor;

impl NodeProcessor for RotaryEmbeddingProcessor {
    type Config = RotaryEmbeddingConfig;

    fn spec(&self) -> crate::processor::NodeSpec {
        crate::processor::NodeSpec {
            min_opset: 1,
            max_opset: None,
            inputs: crate::processor::InputSpec::Exact(4),
            outputs: crate::processor::OutputSpec::Exact(1),
        }
    }

    fn infer_types(
        &self,
        node: &mut RawNode,
        _opset: usize,
        _output_preferences: &OutputPreferences,
    ) -> Result<(), ProcessError> {
        if node.inputs.len() < 4 {
            return Err(ProcessError::InvalidInputCount {
                expected: 4,
                actual: node.inputs.len(),
            });
        }

        // Output has same type as input
        node.outputs[0].ty = node.inputs[0].ty.clone();

        Ok(())
    }

    fn extract_config(&self, node: &RawNode, _opset: usize) -> Result<Self::Config, ProcessError> {
        let mut interleaved = false;
        let mut num_heads = None;
        let mut rotary_embedding_dim = None;
        let mut scale = None;

        for (key, value) in node.attrs.iter() {
            match key.as_str() {
                "interleaved" => interleaved = value.clone().into_i64() != 0,
                "num_heads" => num_heads = Some(value.clone().into_i64() as usize),
                "rotary_embedding_dim" => {
                    rotary_embedding_dim = Some(value.clone().into_i64() as usize)
                }
                "scale" => scale = Some(value.clone().into_f32() as f64),
                _ => {}
            }
        }

        Ok(RotaryEmbeddingConfig::new(
            interleaved,
            num_heads,
            rotary_embedding_dim,
            scale,
        ))
    }

    fn build_node(&self, builder: RawNode, opset: usize) -> Node {
        let config = self
            .extract_config(&builder, opset)
            .expect("Config extraction failed");

        Node::RotaryEmbedding(RotaryEmbeddingNode {
            name: builder.name,
            inputs: builder.inputs,
            outputs: builder.outputs,
            config,
        })
    }
}
