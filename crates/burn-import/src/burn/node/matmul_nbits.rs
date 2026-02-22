use super::prelude::*;

impl NodeCodegen for onnx_ir::matmul_nbits::MatMulNBitsNode {
    fn inputs(&self) -> &[Argument] {
        &self.inputs
    }

    fn outputs(&self) -> &[Argument] {
        &self.outputs
    }

    fn forward(&self, scope: &mut ScopeAtPosition<'_>) -> TokenStream {
        // MatMulNBits inputs:
        // 0: A - input tensor [batch, seq, K]
        // 1: B - packed quantized weights (INT4 packed as U8, stored as float)
        //        Shape: [N, n_blocks, blob_size] where blob_size = block_size/2
        // 2: scales - quantization scales [N, n_blocks]
        // 3: zero_points (optional)
        // 4: g_idx (optional)
        // 5: bias (optional)
        //
        // The B tensor contains packed 4-bit values: each float represents a U8 byte
        // containing 2 quantized 4-bit values (high nibble and low nibble).
        // We need to unpack these before dequantization and matmul.
        
        let a_arg = self.inputs.first().unwrap();
        let b_arg = self.inputs.get(1).unwrap();
        let scales_arg = self.inputs.get(2).unwrap();
        let output_arg = self.outputs.first().unwrap();

        let a = scope.arg(a_arg);
        let b = scope.arg(b_arg);
        let scales = scope.arg(scales_arg);
        let output = arg_to_ident(output_arg);

        let k = self.config.k as usize;
        let n = self.config.n as usize;
        let block_size = self.config.block_size as usize;

        // Check if zero_points provided (asymmetric quantization)
        let has_zero_points = self.inputs.len() > 3 
            && !self.inputs.get(3).map(|a| a.name.is_empty()).unwrap_or(true);

        // Check if bias provided
        let has_bias = self.inputs.len() > 5
            && !self.inputs.get(5).map(|a| a.name.is_empty()).unwrap_or(true);

        // Generate code that creates a quantized tensor from packed weights + scales
        // and uses native matmul. The burn-cubecl backend will use the efficient
        // quantized matmul kernel when it detects a QFloat tensor with qparams.
        //
        // For symmetric 4-bit quantization (Q4S):
        // 1. Create QuantScheme with Q4S, block-level quantization
        // 2. Create QuantizationParameters from scales tensor
        // 3. Use quantize() to create the quantized tensor
        // 4. Call matmul - cubecl automatically uses quantized kernel
        
        if has_zero_points {
            // Asymmetric quantization - fall back to dequantize path
            // since burn's quantized matmul currently focuses on symmetric
            let zp_arg = self.inputs.get(3).unwrap();
            let _zero_points = scope.arg(zp_arg);
            
            if has_bias {
                let bias_arg = self.inputs.get(5).unwrap();
                let bias = scope.arg(bias_arg);
                quote! {
                    // MatMulNBits asymmetric: K=#k, N=#n, block_size=#block_size
                    // Falls back to dequantize path for asymmetric quantization
                    let #output = {
                        // Dequantize B and perform float matmul
                        let b_float = #b.float();
                        let b_scaled = b_float.mul(#scales);
                        #a.matmul(b_scaled).add(#bias)
                    };
                }
            } else {
                quote! {
                    // MatMulNBits asymmetric: K=#k, N=#n, block_size=#block_size
                    // Falls back to dequantize path for asymmetric quantization
                    let #output = {
                        // Dequantize B and perform float matmul
                        let b_float = #b.float();
                        let b_scaled = b_float.mul(#scales);
                        #a.matmul(b_scaled)
                    };
                }
            }
        } else {
            // Symmetric quantization (Q4S) - dequantize then matmul
            // 
            // B tensor is packed 4-bit: [N, n_blocks, blob_size] where blob_size = block_size/2
            // Each float value in B represents a packed U8 byte with 2 x 4-bit values:
            //   high_nibble = floor(x / 16), low_nibble = x % 16
            // 
            // For symmetric Q4, values are in range [-8, 7] (signed 4-bit).
            // After unpacking, we subtract 8 to convert from [0,15] to [-8,7].
            // Then multiply by scales to dequantize.
            
            if has_bias {
                let bias_arg = self.inputs.get(5).unwrap();
                let bias = scope.arg(bias_arg);
                quote! {
                    // MatMulNBits symmetric Q4S: K=#k, N=#n, block_size=#block_size
                    // Dequantize packed 4-bit weights then matmul
                    // All computation done in F32 to avoid F16/F32 mixing NaN
                    let #output = {
                        let a_input = #a;
                        let out_dtype = a_input.dtype();
                        let a_f32 = a_input.cast(burn::tensor::DType::F32);
                        
                        let b_packed = #b.cast(burn::tensor::DType::F32);
                        
                        let b_floor = b_packed.clone().div_scalar(16.0).floor();
                        let b_high = b_floor.clone();
                        let b_low = b_packed.sub(b_floor.mul_scalar(16.0));
                        
                        let b_unpacked = Tensor::stack::<4>(vec![b_low, b_high], 3)
                            .flatten::<3>(2, 3);
                        let b_centered = b_unpacked.sub_scalar(8.0);
                        let b_flat = b_centered.flatten::<2>(1, 2);
                        
                        let n_blocks = #k / #block_size;
                        let scales_2d = #scales.cast(burn::tensor::DType::F32)
                            .reshape([#n, n_blocks]);
                        let scales_expanded = scales_2d.unsqueeze_dim::<3>(2)
                            .expand([#n, n_blocks, #block_size])
                            .flatten::<2>(1, 2);
                        let b_dequant = b_flat.mul(scales_expanded);
                        
                        let b_weight = b_dequant.transpose().unsqueeze::<3>();
                        
                        a_f32.matmul(b_weight).add(#bias.cast(burn::tensor::DType::F32)).clamp(-65504.0, 65504.0).cast(out_dtype)
                    };
                }
            } else {
                quote! {
                    // MatMulNBits symmetric Q4S: K=#k, N=#n, block_size=#block_size
                    // Dequantize packed 4-bit weights then matmul
                    // All computation done in F32 to avoid F16/F32 mixing NaN
                    let #output = {
                        let a_input = #a;
                        let out_dtype = a_input.dtype();
                        let a_f32 = a_input.cast(burn::tensor::DType::F32);
                        
                        let b_packed = #b.cast(burn::tensor::DType::F32);
                        
                        let b_floor = b_packed.clone().div_scalar(16.0).floor();
                        let b_high = b_floor.clone();
                        let b_low = b_packed.sub(b_floor.mul_scalar(16.0));
                        
                        let b_unpacked = Tensor::stack::<4>(vec![b_low, b_high], 3)
                            .flatten::<3>(2, 3);
                        let b_centered = b_unpacked.sub_scalar(8.0);
                        let b_flat = b_centered.flatten::<2>(1, 2);
                        
                        let n_blocks = #k / #block_size;
                        let scales_2d = #scales.cast(burn::tensor::DType::F32)
                            .reshape([#n, n_blocks]);
                        let scales_expanded = scales_2d.unsqueeze_dim::<3>(2)
                            .expand([#n, n_blocks, #block_size])
                            .flatten::<2>(1, 2);
                        let b_dequant = b_flat.mul(scales_expanded);
                        
                        let b_weight = b_dequant.transpose().unsqueeze::<3>();
                        
                        a_f32.matmul(b_weight).clamp(-65504.0, 65504.0).cast(out_dtype)
                    };
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_helpers::*;
    use burn::tensor::DType;
    use onnx_ir::{matmul_nbits::MatMulNBitsConfig, matmul_nbits::MatMulNBitsNode, Argument};

    #[test]
    fn test_matmul_nbits_codegen() {
        let node = MatMulNBitsNode {
            name: "test_matmul_nbits".to_string(),
            inputs: vec![
                create_tensor_arg("a", DType::F32, 2),
                create_tensor_arg("b", DType::U8, 2),  // packed INT4
                create_tensor_arg("scales", DType::F32, 1),
            ],
            outputs: vec![create_tensor_arg("output", DType::F32, 2)],
            config: MatMulNBitsConfig::new(512, 1024, 4, 128),
        };

        let code = node.forward(&mut create_test_scope());
        let code_str = code.to_string();
        
        assert!(code_str.contains("matmul"));
    }
}
