use super::prelude::*;
use burn_store::TensorSnapshot;

impl NodeCodegen for onnx_ir::node::skip_simplified_layer_norm::SkipSimplifiedLayerNormNode {
    fn inputs(&self) -> &[Argument] {
        &self.inputs
    }

    fn outputs(&self) -> &[Argument] {
        &self.outputs
    }

    fn field(&self) -> Option<Field> {
        // RMSNorm scale (gamma) stored as a parameter
        let name = Ident::new(&self.name, Span::call_site());
        let gamma_input = self.inputs.get(2)?;
        let num_features = match &gamma_input.ty {
            onnx_ir::ir::ArgType::Tensor(t) => {
                t.static_shape.as_ref().and_then(|s| s.first().copied()).unwrap_or(0)
            }
            _ => 0,
        };

        if num_features > 0 {
            let num_features_tok = num_features.to_tokens();
            Some(Field::new(
                self.name.clone(),
                quote! {
                    burn::module::Param<Tensor<B, 1>>
                },
                quote! {
                    let #name: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
                        burn::module::ParamId::new(),
                        move |device, _require_grad| Tensor::<B, 1>::ones([#num_features_tok], device),
                        device.clone(),
                        false,
                        burn::tensor::Shape::new([#num_features_tok]),
                    );
                },
            ))
        } else {
            None
        }
    }

    fn collect_snapshots(&self, field_name: &str) -> Vec<TensorSnapshot> {
        use crate::burn::node_traits::create_lazy_snapshot;

        let mut snapshots = vec![];

        // gamma (scale) tensor at input index 2
        if let Some(gamma_input) = self.inputs.get(2) {
            if let Some(snapshot) =
                create_lazy_snapshot(gamma_input, field_name, "SkipSimplifiedLayerNorm")
            {
                snapshots.push(snapshot);
            }
        }

        snapshots
    }

    fn forward(&self, scope: &mut ScopeAtPosition<'_>) -> TokenStream {
        let input = scope.arg(self.inputs.first().unwrap());
        let skip = scope.arg(self.inputs.get(1).unwrap());
        let field = Ident::new(&self.name, Span::call_site());
        let epsilon = self.config.epsilon;

        let output = arg_to_ident(self.outputs.first().unwrap());

        // Check if output 3 (input_skip_bias_sum) is used
        let has_skip_sum_output = self.outputs.len() > 3 && self.outputs.get(3).is_some();

        if has_skip_sum_output {
            let skip_sum_out = arg_to_ident(self.outputs.get(3).unwrap());
            quote! {
                let (#output, #skip_sum_out) = {
                    let dtype = #input.dtype();
                    let skip_sum_f32 = #input.cast(burn::tensor::DType::F32) + #skip.cast(burn::tensor::DType::F32);
                    let skip_sum = skip_sum_f32.clone().clamp(-65504.0, 65504.0).cast(dtype);
                    let x = skip_sum_f32;
                    let variance = x.clone().powf_scalar(2.0).mean_dim(x.dims().len() - 1);
                    let rms = (variance + #epsilon).sqrt();
                    let normed = x / rms;
                    let result = (normed * self.#field.val().unsqueeze().cast(burn::tensor::DType::F32)).cast(dtype);
                    (result, skip_sum)
                };
            }
        } else {
            quote! {
                let #output = {
                    let dtype = #input.dtype();
                    let x = #input.cast(burn::tensor::DType::F32) + #skip.cast(burn::tensor::DType::F32);
                    let variance = x.clone().powf_scalar(2.0).mean_dim(x.dims().len() - 1);
                    let rms = (variance + #epsilon).sqrt();
                    let normed = x / rms;
                    (normed * self.#field.val().unsqueeze().cast(burn::tensor::DType::F32)).cast(dtype)
                };
            }
        }
    }

    fn register_imports(&self, _imports: &mut BurnImports) {}
}
