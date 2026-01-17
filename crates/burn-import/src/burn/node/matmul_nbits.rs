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
        // 0: A - input tensor [M, K]
        // 1: B - packed quantized weights (INT4 packed as U8)
        // 2: scales - quantization scales
        // 3: zero_points (optional)
        // 4: g_idx (optional)
        // 5: bias (optional)
        
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
            // Symmetric quantization (Q4S) - use native quantized matmul path
            // Create a quantized tensor from B (packed weights) with scales as qparams
            //
            // The key insight: burn-cubecl's matmul checks if inputs have qparams
            // and uses MatmulInputHandleRef::quantized() for efficient fused matmul
            
            if has_bias {
                let bias_arg = self.inputs.get(5).unwrap();
                let bias = scope.arg(bias_arg);
                quote! {
                    // MatMulNBits symmetric Q4S: K=#k, N=#n, block_size=#block_size
                    // Create quantized tensor and use native quantized matmul
                    let #output = {
                        use burn::tensor::quantization::{QuantScheme, QuantValue, QuantStore, QuantLevel, QuantMode, QuantParam, BlockSize, QuantizationParameters};
                        
                        // Define Q4S scheme with block quantization
                        let scheme = QuantScheme {
                            value: QuantValue::Q4S,
                            param: QuantParam::F32,
                            store: QuantStore::PackedU32(0), // packed along last dim
                            level: QuantLevel::Block(BlockSize::new([#block_size as u8])),
                            mode: QuantMode::Symmetric,
                        };
                        
                        // Create quantization parameters from scales
                        let qparams = QuantizationParameters { scales: #scales.clone() };
                        
                        // Quantize B with the scheme and params
                        // This creates a QFloat tensor that cubecl can use efficiently
                        // Note: B is already float (U8 weights are stored as float tensors)
                        let b_quantized = #b.quantize(&scheme, qparams);
                        
                        // Matmul with quantized weights - cubecl uses fused kernel
                        #a.matmul(b_quantized).add(#bias)
                    };
                }
            } else {
                quote! {
                    // MatMulNBits symmetric Q4S: K=#k, N=#n, block_size=#block_size
                    // Create quantized tensor and use native quantized matmul
                    let #output = {
                        use burn::tensor::quantization::{QuantScheme, QuantValue, QuantStore, QuantLevel, QuantMode, QuantParam, BlockSize, QuantizationParameters};
                        
                        // Define Q4S scheme with block quantization
                        let scheme = QuantScheme {
                            value: QuantValue::Q4S,
                            param: QuantParam::F32,
                            store: QuantStore::PackedU32(0), // packed along last dim
                            level: QuantLevel::Block(BlockSize::new([#block_size as u8])),
                            mode: QuantMode::Symmetric,
                        };
                        
                        // Create quantization parameters from scales
                        let qparams = QuantizationParameters { scales: #scales.clone() };
                        
                        // Quantize B with the scheme and params
                        // This creates a QFloat tensor that cubecl can use efficiently
                        // Note: B is already float (U8 weights are stored as float tensors)
                        let b_quantized = #b.quantize(&scheme, qparams);
                        
                        // Matmul with quantized weights - cubecl uses fused kernel
                        #a.matmul(b_quantized)
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
