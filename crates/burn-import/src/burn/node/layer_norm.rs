use burn_store::TensorSnapshot;

use super::prelude::*;

impl NodeCodegen for onnx_ir::node::layer_norm::LayerNormalizationNode {
    fn inputs(&self) -> &[Argument] {
        &self.inputs
    }

    fn outputs(&self) -> &[Argument] {
        &self.outputs
    }

    fn field(&self) -> Option<Field> {
        if self.config.is_rms_norm {
            // RMSNorm: store scale as a parameter, no module
            let name = Ident::new(&self.name, Span::call_site());
            let num_features = self.config.d_model.to_tokens();
            Some(Field::new(
                self.name.clone(),
                quote! {
                    burn::module::Param<Tensor<B, 1>>
                },
                quote! {
                    let #name: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
                        burn::module::ParamId::new(),
                        move |device, _require_grad| Tensor::<B, 1>::ones([#num_features], device),
                        device.clone(),
                        false,
                        burn::tensor::Shape::new([#num_features]),
                    );
                },
            ))
        } else {
            let name = Ident::new(&self.name, Span::call_site());
            let num_features = self.config.d_model.to_tokens();
            let epsilon = self.config.epsilon;
            let has_bias = self.config.has_bias;

            Some(Field::new(
                self.name.clone(),
                quote! {
                    LayerNorm<B>
                },
                quote! {
                    let #name = LayerNormConfig::new(#num_features)
                        .with_epsilon(#epsilon)
                        .with_bias(#has_bias)
                        .init(device);
                },
            ))
        }
    }

    fn collect_snapshots(&self, field_name: &str) -> Vec<TensorSnapshot> {
        use crate::burn::node_traits::create_lazy_snapshot;

        let mut snapshots = vec![];

        if self.config.is_rms_norm {
            // RMSNorm: scale tensor stored directly as the field param
            if let Some(scale_input) = self.inputs.get(1) {
                if let Some(snapshot) = create_lazy_snapshot(scale_input, field_name, "RMSNorm") {
                    snapshots.push(snapshot);
                }
            }
        } else {
            // Gamma (scale) tensor at input index 1
            if let Some(gamma_input) = self.inputs.get(1) {
                let gamma_path = format!("{}.gamma", field_name);
                if let Some(snapshot) = create_lazy_snapshot(gamma_input, &gamma_path, "LayerNorm") {
                    snapshots.push(snapshot);
                }
            }

            // Beta (bias) tensor at input index 2 - only if ONNX model has bias
            if self.config.has_bias
                && let Some(beta_input) = self.inputs.get(2)
            {
                let beta_path = format!("{}.beta", field_name);
                if let Some(snapshot) = create_lazy_snapshot(beta_input, &beta_path, "LayerNorm") {
                    snapshots.push(snapshot);
                }
            }
        }

        snapshots
    }

    fn forward(&self, scope: &mut ScopeAtPosition<'_>) -> TokenStream {
        let input = scope.arg(self.inputs.first().unwrap());
        let output = arg_to_ident(self.outputs.first().unwrap());
        let field = Ident::new(&self.name, Span::call_site());

        if self.config.is_rms_norm {
            let epsilon = self.config.epsilon;
            if self.config.full_precision {
                quote! {
                    let #output = {
                        let dtype = #input.dtype();
                        let x = #input.cast(burn::tensor::DType::F32);
                        let variance = x.clone().powf_scalar(2.0).mean_dim(x.dims().len() - 1);
                        let rms = (variance + #epsilon).sqrt();
                        let normed = x / rms;
                        (normed * self.#field.val().unsqueeze().cast(burn::tensor::DType::F32)).cast(dtype)
                    };
                }
            } else {
                quote! {
                    let #output = {
                        let x = #input;
                        let variance = x.clone().powf_scalar(2.0).mean_dim(x.dims().len() - 1);
                        let rms = (variance + #epsilon).sqrt();
                        let normed = x / rms;
                        normed * self.#field.val().unsqueeze()
                    };
                }
            }
        } else if self.config.full_precision {
            quote! {
                let #output = {
                    let dtype = #input.dtype();
                    self.#field.forward(#input.cast(burn::tensor::DType::F32)).cast(dtype)
                };
            }
        } else {
            quote! {
                let #output = self.#field.forward(#input);
            }
        }
    }

    fn register_imports(&self, imports: &mut BurnImports) {
        if !self.config.is_rms_norm {
            imports.register("burn::nn::LayerNorm");
            imports.register("burn::nn::LayerNormConfig");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_helpers::*;
    use burn::tensor::DType;
    use insta::assert_snapshot;
    use onnx_ir::node::layer_norm::{
        LayerNormConfig, LayerNormalizationNode, LayerNormalizationNodeBuilder,
    };

    fn create_layer_norm_node(name: &str) -> LayerNormalizationNode {
        // has_bias = true (most common case with bias)
        let config = LayerNormConfig::new(512, 1e-5, true, true, false);

        LayerNormalizationNodeBuilder::new(name)
            .input_tensor("input", 3, DType::F32)
            .output_tensor("output", 3, DType::F32)
            .config(config)
            .build()
    }

    #[test]
    fn test_layer_norm_forward() {
        let node = create_layer_norm_node("layer_norm1");
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
            let output = {
                let dtype = input.dtype();
                self.layer_norm1.forward(input.cast(burn::tensor::DType::F32)).cast(dtype)
            };
            output
        }
        ");
    }

    #[test]
    fn test_layer_norm_forward_with_clone() {
        let node = create_layer_norm_node("layer_norm1");
        let code = codegen_forward_with_clone(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
            let output = {
                let dtype = input.clone().dtype();
                self.layer_norm1
                    .forward(input.clone().cast(burn::tensor::DType::F32))
                    .cast(dtype)
            };
            output
        }
        ");
    }
}
