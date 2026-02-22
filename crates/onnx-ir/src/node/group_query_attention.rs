//! # GroupQueryAttention
//!
//! Microsoft-specific fused Group Query Attention operator.
//!
//! **Spec**: <https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#com.microsoft.GroupQueryAttention>
//!
//! This operator fuses multi-head attention with optional rotary position embeddings (RoPE)
//! and KV cache management. It supports Grouped Query Attention (GQA) where the number of
//! KV heads can be less than Q heads.
//!
//! ## Inputs
//! 0. query: (batch, seq_len, num_heads * head_size)
//! 1. key: (batch, kv_seq_len, kv_num_heads * head_size)
//! 2. value: (batch, kv_seq_len, kv_num_heads * head_size)
//! 3. past_key: (batch, kv_num_heads, past_seq_len, head_size) - optional
//! 4. past_value: (batch, kv_num_heads, past_seq_len, head_size) - optional
//! 5. seqlens_k: (batch) - optional
//! 6. total_seq_len: scalar - optional
//! 7. cos_cache: (max_seq_len, head_size/2) - optional, for rotary embeddings
//! 8. sin_cache: (max_seq_len, head_size/2) - optional, for rotary embeddings
//!
//! ## Outputs
//! 0. output: (batch, seq_len, num_heads * head_size)
//! 1. present_key: (batch, kv_num_heads, total_seq_len, head_size)
//! 2. present_value: (batch, kv_num_heads, total_seq_len, head_size)

use derive_new::new;
use onnx_ir_derive::NodeBuilder;

use crate::ir::{ArgType, Argument, Node, RawNode, TensorType};
use crate::processor::{NodeProcessor, OutputPreferences, ProcessError};

#[derive(Debug, Clone, new)]
pub struct GroupQueryAttentionConfig {
    pub num_heads: usize,
    pub kv_num_heads: usize,
    pub scale: Option<f64>,
    pub do_rotary: bool,
    pub rotary_interleaved: bool,
    pub local_window_size: i64,
}

/// Node representation for GroupQueryAttention operation
#[derive(Debug, Clone, NodeBuilder)]
pub struct GroupQueryAttentionNode {
    pub name: String,
    pub inputs: Vec<Argument>,
    pub outputs: Vec<Argument>,
    pub config: GroupQueryAttentionConfig,
}

pub(crate) struct GroupQueryAttentionProcessor;

impl NodeProcessor for GroupQueryAttentionProcessor {
    type Config = GroupQueryAttentionConfig;

    fn spec(&self) -> crate::processor::NodeSpec {
        crate::processor::NodeSpec {
            min_opset: 1,
            max_opset: None,
            inputs: crate::processor::InputSpec::AtLeast(3),
            outputs: crate::processor::OutputSpec::Exact(3),
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

        let q_dtype = node.inputs[0].ty.elem_type();
        let q_rank = match &node.inputs[0].ty {
            ArgType::Tensor(t) => t.rank,
            _ => {
                return Err(ProcessError::Custom(
                    "GroupQueryAttention: query must be a tensor".to_string(),
                ))
            }
        };

        // Output: same rank and dtype as query
        node.outputs[0].ty = ArgType::Tensor(TensorType {
            dtype: q_dtype,
            rank: q_rank,
            static_shape: None,
        });

        // present_key: rank 4
        if let Some(present_key) = node.outputs.get_mut(1) {
            present_key.ty = ArgType::Tensor(TensorType {
                dtype: q_dtype,
                rank: 4,
                static_shape: None,
            });
        }

        // present_value: rank 4
        if let Some(present_value) = node.outputs.get_mut(2) {
            present_value.ty = ArgType::Tensor(TensorType {
                dtype: q_dtype,
                rank: 4,
                static_shape: None,
            });
        }

        Ok(())
    }

    fn extract_config(&self, node: &RawNode, _opset: usize) -> Result<Self::Config, ProcessError> {
        let mut num_heads = 0usize;
        let mut kv_num_heads = 0usize;
        let mut scale = None;
        let mut do_rotary = false;
        let mut rotary_interleaved = false;
        let mut local_window_size = -1i64;

        for (key, value) in node.attrs.iter() {
            match key.as_str() {
                "num_heads" => num_heads = value.clone().into_i64() as usize,
                "kv_num_heads" => kv_num_heads = value.clone().into_i64() as usize,
                "scale" => scale = Some(value.clone().into_f32() as f64),
                "do_rotary" => do_rotary = value.clone().into_i64() != 0,
                "rotary_interleaved" => rotary_interleaved = value.clone().into_i64() != 0,
                "local_window_size" => local_window_size = value.clone().into_i64(),
                // Ignore unknown attributes for contrib ops
                _ => {}
            }
        }

        if num_heads == 0 {
            return Err(ProcessError::Custom(
                "GroupQueryAttention: num_heads attribute is required".to_string(),
            ));
        }
        if kv_num_heads == 0 {
            kv_num_heads = num_heads;
        }

        Ok(GroupQueryAttentionConfig::new(
            num_heads,
            kv_num_heads,
            scale,
            do_rotary,
            rotary_interleaved,
            local_window_size,
        ))
    }

    fn build_node(&self, builder: RawNode, opset: usize) -> Node {
        let config = self
            .extract_config(&builder, opset)
            .expect("Config extraction failed");

        Node::GroupQueryAttention(GroupQueryAttentionNode {
            name: builder.name,
            inputs: builder.inputs,
            outputs: builder.outputs,
            config,
        })
    }
}
