use super::prelude::*;

impl NodeCodegen for onnx_ir::node::group_query_attention::GroupQueryAttentionNode {
    fn inputs(&self) -> &[Argument] {
        &self.inputs
    }

    fn outputs(&self) -> &[Argument] {
        &self.outputs
    }

    fn forward(&self, scope: &mut ScopeAtPosition<'_>) -> TokenStream {
        let num_heads = self.config.num_heads;
        let kv_num_heads = self.config.kv_num_heads;
        let do_rotary = self.config.do_rotary;
        let rotary_interleaved = self.config.rotary_interleaved;

        // Get Q, K, V inputs (required)
        let q = scope.arg(self.inputs.first().unwrap());
        let k = scope.arg(self.inputs.get(1).unwrap());
        let v = scope.arg(self.inputs.get(2).unwrap());

        // Get output names
        let output_y = arg_to_ident(self.outputs.first().unwrap());
        let present_k_out = arg_to_ident(self.outputs.get(1).unwrap());
        let present_v_out = arg_to_ident(self.outputs.get(2).unwrap());

        let mut body = TokenStream::new();

        // Reshape Q/K/V from [batch, seq, hidden] -> [batch, heads, seq, head_dim]
        body.extend(quote! {
            let q_input = #q;
            let k_input = #k;
            let v_input = #v;
            let [batch_size, q_seq_len, q_hidden] = q_input.dims();
            let [_batch, kv_seq_len, _kv_hidden] = k_input.dims();
            let head_dim = q_hidden / #num_heads;

            let q = q_input
                .reshape([batch_size, q_seq_len, #num_heads, head_dim])
                .permute([0, 2, 1, 3]);
            let k = k_input
                .reshape([batch_size, kv_seq_len, #kv_num_heads, head_dim])
                .permute([0, 2, 1, 3]);
            let v = v_input
                .reshape([batch_size, kv_seq_len, #kv_num_heads, head_dim])
                .permute([0, 2, 1, 3]);
        });

        // Handle RoPE if do_rotary is set
        if do_rotary {
            // cos_cache at input index 7, sin_cache at input index 8
            let has_cos_sin = self.inputs.len() > 8
                && self.inputs.get(7).is_some()
                && self.inputs.get(8).is_some();

            if has_cos_sin {
                let cos_cache = scope.arg(self.inputs.get(7).unwrap());
                let sin_cache = scope.arg(self.inputs.get(8).unwrap());

                if rotary_interleaved {
                    // Interleaved RoPE: pairs of (x0,x1), (x2,x3), ...
                    body.extend(quote! {
                        let cos = #cos_cache.narrow(0, 0, q_seq_len).reshape([1, 1, q_seq_len, head_dim / 2]);
                        let sin = #sin_cache.narrow(0, 0, q_seq_len).reshape([1, 1, q_seq_len, head_dim / 2]);

                        let q = {
                            let q_even = q.clone().slice([0..batch_size, 0..#num_heads, 0..q_seq_len, 0..head_dim / 2]);
                            let q_odd = q.slice([0..batch_size, 0..#num_heads, 0..q_seq_len, head_dim / 2..head_dim]);
                            let q_rot_even = q_even.clone() * cos.clone() - q_odd.clone() * sin.clone();
                            let q_rot_odd = q_even * sin.clone() + q_odd * cos.clone();
                            Tensor::cat([q_rot_even, q_rot_odd].to_vec(), 3)
                        };
                        let k = {
                            let cos_k = #cos_cache.narrow(0, 0, kv_seq_len).reshape([1, 1, kv_seq_len, head_dim / 2]);
                            let sin_k = #sin_cache.narrow(0, 0, kv_seq_len).reshape([1, 1, kv_seq_len, head_dim / 2]);
                            let k_even = k.clone().slice([0..batch_size, 0..#kv_num_heads, 0..kv_seq_len, 0..head_dim / 2]);
                            let k_odd = k.slice([0..batch_size, 0..#kv_num_heads, 0..kv_seq_len, head_dim / 2..head_dim]);
                            let k_rot_even = k_even.clone() * cos_k.clone() - k_odd.clone() * sin_k.clone();
                            let k_rot_odd = k_even * sin_k + k_odd * cos_k;
                            Tensor::cat([k_rot_even, k_rot_odd].to_vec(), 3)
                        };
                    });
                } else {
                    // Non-interleaved (half-rotary) RoPE: first half and second half
                    body.extend(quote! {
                        let cos = #cos_cache.narrow(0, 0, q_seq_len).reshape([1, 1, q_seq_len, head_dim / 2]);
                        let sin = #sin_cache.narrow(0, 0, q_seq_len).reshape([1, 1, q_seq_len, head_dim / 2]);

                        let q = {
                            let half = head_dim / 2;
                            let q1 = q.clone().narrow(3, 0, half);
                            let q2 = q.narrow(3, half, half);
                            let q_rot1 = q1.clone() * cos.clone() - q2.clone() * sin.clone();
                            let q_rot2 = q1 * sin.clone() + q2 * cos.clone();
                            Tensor::cat([q_rot1, q_rot2].to_vec(), 3)
                        };
                        let k = {
                            let half = head_dim / 2;
                            let cos_k = #cos_cache.narrow(0, 0, kv_seq_len).reshape([1, 1, kv_seq_len, half]);
                            let sin_k = #sin_cache.narrow(0, 0, kv_seq_len).reshape([1, 1, kv_seq_len, half]);
                            let k1 = k.clone().narrow(3, 0, half);
                            let k2 = k.narrow(3, half, half);
                            let k_rot1 = k1.clone() * cos_k.clone() - k2.clone() * sin_k.clone();
                            let k_rot2 = k1 * sin_k + k2 * cos_k;
                            Tensor::cat([k_rot1, k_rot2].to_vec(), 3)
                        };
                    });
                }
            }
        }

        // Handle past KV cache concatenation
        let has_past_kv = self.inputs.len() > 4
            && self.inputs.get(3).is_some()
            && self.inputs.get(4).is_some();

        if has_past_kv {
            let past_k = scope.arg(self.inputs.get(3).unwrap());
            let past_v = scope.arg(self.inputs.get(4).unwrap());

            body.extend(quote! {
                let k = Tensor::cat([#past_k, k].to_vec(), 2);
                let v = Tensor::cat([#past_v, v].to_vec(), 2);
            });
        }

        // Present KV = current K/V (after cache concat)
        // Also save total key length for causal mask before head expansion
        body.extend(quote! {
            let #present_k_out = k.clone();
            let #present_v_out = v.clone();
            let total_k_len = k.dims()[2];
            let past_len = total_k_len - q_seq_len;
        });

        // GQA head expansion: repeat KV heads to match Q heads
        if num_heads != kv_num_heads {
            let repeat_factor = num_heads / kv_num_heads;
            body.extend(quote! {
                // Expand KV heads to match Q heads for grouped query attention
                let k = {
                    let [b, kv_h, s, d] = k.dims();
                    k.unsqueeze_dim::<5>(2)
                        .expand([b, kv_h, #repeat_factor, s, d])
                        .reshape([b, #num_heads, s, d])
                };
                let v = {
                    let [b, kv_h, s, d] = v.dims();
                    v.unsqueeze_dim::<5>(2)
                        .expand([b, kv_h, #repeat_factor, s, d])
                        .reshape([b, #num_heads, s, d])
                };
            });
        }

        // Scaled dot-product attention with causal masking
        let scale_code = if let Some(scale) = self.config.scale {
            quote! { let scale = #scale; }
        } else {
            quote! { let scale = 1.0 / (head_dim as f64).sqrt(); }
        };

        body.extend(quote! {
            #scale_code

            // Perform entire attention in F32 to prevent F16 numerical overflow
            let attn_device = q.device();
            let attn_dtype = q.dtype();
            let q = q.cast(burn::tensor::DType::F32) * scale;
            let k = k.cast(burn::tensor::DType::F32);
            let v = v.cast(burn::tensor::DType::F32);

            let scores = q.matmul(k.transpose());

            // Causal masking: prevent attending to future tokens
            // Also mask position 0 when past_len > 0 (dummy zero KV cache token)
            let mut mask_data = vec![0.0f32; q_seq_len * total_k_len];
            for qi in 0..q_seq_len {
                let max_key = past_len + qi;
                for ki in 0..total_k_len {
                    if (ki == 0 && past_len > 0) || ki > max_key {
                        mask_data[qi * total_k_len + ki] = -1.0e9f32;
                    }
                }
            }
            let causal_bias = Tensor::<B, 2>::from_data_dtype(
                burn::tensor::TensorData::new(mask_data, [q_seq_len, total_k_len]),
                &attn_device,
                burn::tensor::DType::F32,
            ).unsqueeze::<4>();
            let scores = softmax(scores + causal_bias, 3);

            // Clamp to F16 range before casting back to prevent inf overflow
            let #output_y = scores.matmul(v).clamp(-65504.0, 65504.0).cast(attn_dtype);

            // Reshape back to [batch, seq, hidden]
            let #output_y = #output_y
                .permute([0, 2, 1, 3])
                .reshape([batch_size as i32, q_seq_len as i32, -1]);
        });

        quote! {
            let (#output_y, #present_k_out, #present_v_out) = {
                #body
                (#output_y, #present_k_out, #present_v_out)
            };
        }
    }

    fn register_imports(&self, imports: &mut BurnImports) {
        imports.register("burn::tensor::activation::softmax");
    }
}
