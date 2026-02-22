use super::prelude::*;

impl NodeCodegen for onnx_ir::node::rotary_embedding::RotaryEmbeddingNode {
    fn inputs(&self) -> &[Argument] {
        &self.inputs
    }

    fn outputs(&self) -> &[Argument] {
        &self.outputs
    }

    fn field(&self) -> Option<Field> {
        None
    }

    fn collect_snapshots(&self, _field_name: &str) -> Vec<burn_store::TensorSnapshot> {
        vec![]
    }

    fn forward(&self, scope: &mut ScopeAtPosition<'_>) -> TokenStream {
        let input = scope.arg(self.inputs.first().unwrap());
        let position_ids = scope.arg(self.inputs.get(1).unwrap());
        let cos_cache = scope.arg(self.inputs.get(2).unwrap());
        let sin_cache = scope.arg(self.inputs.get(3).unwrap());
        let output = arg_to_ident(self.outputs.first().unwrap());
        let interleaved = self.config.interleaved;

        if interleaved {
            // Interleaved RoPE: pairs (x0,x1,x2,x3,...) -> rotate adjacent pairs
            quote! {
                let #output = {
                    let x = #input;
                    let pos = #position_ids;
                    let cos_c = #cos_cache;
                    let sin_c = #sin_cache;

                    let dims = x.dims();
                    let seq_dim = dims.len() - 2;
                    let head_dim = *dims.last().unwrap();
                    let half = head_dim / 2;

                    // Gather cos/sin for the given positions
                    let pos_flat = pos.reshape([-1]);
                    let cos_vals = cos_c.select(0, pos_flat.clone()).reshape([-1, 1, half as i64]);
                    let sin_vals = sin_c.select(0, pos_flat).reshape([-1, 1, half as i64]);

                    // Split into even/odd
                    let x_even = x.clone().slice([0..dims[0], 0..dims[1], 0..dims[2], (0..head_dim).step_by(2)]);
                    let x_odd = x.slice([0..dims[0], 0..dims[1], 0..dims[2], (1..head_dim).step_by(2)]);

                    let rotated_even = x_even.clone() * cos_vals.clone() - x_odd.clone() * sin_vals.clone();
                    let rotated_odd = x_even * sin_vals + x_odd * cos_vals;

                    // Interleave back
                    Tensor::stack(vec![rotated_even, rotated_odd], dims.len())
                        .reshape(dims.to_vec())
                };
            }
        } else {
            // Non-interleaved (default): split first/second half of head_dim
            // Get input rank from ONNX IR to generate fixed-size reshape
            let input_rank = match &self.inputs.first().unwrap().ty {
                onnx_ir::ir::ArgType::Tensor(t) => t.rank,
                _ => 3, // fallback
            };

            if input_rank == 4 {
                // Rank 4: [batch, num_heads, seq, head_dim] — apply directly
                quote! {
                    let #output = {
                        let x = #input;
                        let pos = #position_ids;
                        let cos_c = #cos_cache;
                        let sin_c = #sin_cache;

                        let dims = x.dims();
                        let rot_half = cos_c.dims()[1]; // rotary_dim / 2

                        let pos_flat = pos.reshape([-1]);
                        let cos_gathered = cos_c.select(0, pos_flat.clone());
                        let sin_gathered = sin_c.select(0, pos_flat);

                        // Reshape to [batch, 1, seq, rot_half] for broadcasting across heads
                        let cos_vals = cos_gathered.reshape([dims[0], 1, dims[2], rot_half]);
                        let sin_vals = sin_gathered.reshape([dims[0], 1, dims[2], rot_half]);

                        let x1 = x.clone().narrow(3, 0, rot_half);
                        let x2 = x.narrow(3, rot_half, rot_half);

                        let rotated_x1 = x1.clone() * cos_vals.clone() - x2.clone() * sin_vals.clone();
                        let rotated_x2 = x1 * sin_vals + x2 * cos_vals;

                        Tensor::cat(vec![rotated_x1, rotated_x2], 3)
                    };
                }
            } else {
                // Rank 3: [batch, seq, hidden] — reshape to per-head, apply, reshape back
                quote! {
                    let #output = {
                        let x = #input;
                        let pos = #position_ids;
                        let cos_c = #cos_cache;
                        let sin_c = #sin_cache;

                        let dims = x.dims(); // [batch, seq, hidden]
                        let rot_half = cos_c.dims()[1]; // rotary_dim / 2
                        let rot_dim = rot_half * 2;
                        let num_heads = dims[2] / rot_dim;

                        // Reshape to [batch, seq, num_heads, rot_dim]
                        let x = x.reshape([dims[0], dims[1], num_heads, rot_dim]);

                        let pos_flat = pos.reshape([-1]);
                        let cos_gathered = cos_c.select(0, pos_flat.clone());
                        let sin_gathered = sin_c.select(0, pos_flat);

                        // Reshape to [batch, seq, 1, rot_half] for broadcasting across heads
                        let cos_vals = cos_gathered.reshape([dims[0], dims[1], 1, rot_half]);
                        let sin_vals = sin_gathered.reshape([dims[0], dims[1], 1, rot_half]);

                        // Split head_dim into two halves
                        let x1 = x.clone().narrow(3, 0, rot_half);
                        let x2 = x.narrow(3, rot_half, rot_half);

                        let rotated_x1 = x1.clone() * cos_vals.clone() - x2.clone() * sin_vals.clone();
                        let rotated_x2 = x1 * sin_vals + x2 * cos_vals;

                        // Concat and reshape back to [batch, seq, hidden]
                        Tensor::cat(vec![rotated_x1, rotated_x2], 3)
                            .reshape([dims[0], dims[1], dims[2]])
                    };
                }
            }
        }
    }

    fn register_imports(&self, _imports: &mut BurnImports) {}
}
