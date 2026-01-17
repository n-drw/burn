use super::prelude::*;

impl NodeCodegen for onnx_ir::dequantize_linear::DequantizeLinearNode {
    fn inputs(&self) -> &[Argument] {
        &self.inputs
    }

    fn outputs(&self) -> &[Argument] {
        &self.outputs
    }

    fn forward(&self, scope: &mut ScopeAtPosition<'_>) -> TokenStream {
        let x = self.inputs.first().unwrap();
        let x_scale = self.inputs.get(1).unwrap();
        let output = arg_to_ident(self.outputs.first().unwrap());

        let x_val = scope.arg(x);
        let scale_val = scope.arg(x_scale);

        // Check if zero point is provided (3rd input is optional)
        let has_zero_point = self.inputs.len() > 2;

        // DequantizeLinear formula: y = (x - x_zero_point) * x_scale
        // For INT4/INT8 quantized weights, we:
        // 1. Cast quantized input to float
        // 2. Subtract zero point (if provided)
        // 3. Multiply by scale
        
        if has_zero_point {
            let x_zero_point = self.inputs.get(2).unwrap();
            let zero_point_val = scope.arg(x_zero_point);
            
            // Cast to float, subtract zero point, multiply by scale
            quote! {
                let #output = (#x_val.float() - #zero_point_val.float()) * #scale_val;
            }
        } else {
            // No zero point - just cast and multiply by scale
            quote! {
                let #output = #x_val.float() * #scale_val;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_helpers::*;
    use burn::tensor::DType;
    use insta::assert_snapshot;
    use onnx_ir::node::dequantize_linear::DequantizeLinearNodeBuilder;

    #[test]
    fn test_dequantize_linear_no_zero_point() {
        let node = DequantizeLinearNodeBuilder::new("dequant1")
            .input_tensor("x", 2, DType::I8)
            .input_tensor("x_scale", 0, DType::F32)
            .output_tensor("y", 2, DType::F32)
            .config(onnx_ir::node::dequantize_linear::DequantizeLinearConfig::new(1, 0))
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(
            &self,
            x: Tensor<B, 2, Int>,
            x_scale: Tensor<B, 0>,
        ) -> Tensor<B, 2> {
            let y = x.float() * x_scale;
            y
        }
        ");
    }

    #[test]
    fn test_dequantize_linear_with_zero_point() {
        let node = DequantizeLinearNodeBuilder::new("dequant2")
            .input_tensor("x", 2, DType::I8)
            .input_tensor("x_scale", 0, DType::F32)
            .input_tensor("x_zero_point", 0, DType::I8)
            .output_tensor("y", 2, DType::F32)
            .config(onnx_ir::node::dequantize_linear::DequantizeLinearConfig::new(1, 0))
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(
            &self,
            x: Tensor<B, 2, Int>,
            x_scale: Tensor<B, 0>,
            x_zero_point: Tensor<B, 0, Int>,
        ) -> Tensor<B, 2> {
            let y = (x.float() - x_zero_point.float()) * x_scale;
            y
        }
        ");
    }
}
