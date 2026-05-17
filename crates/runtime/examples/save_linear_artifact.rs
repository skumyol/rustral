//! Train the same tiny 1×1 linear as `tape_train_demo`, then write `save_model_to_path` for the inference server.
#[cfg(feature = "training")]
fn main() -> anyhow::Result<()> {
    use rustral_core::NamedParameters;
    use rustral_ndarray_backend::CpuBackend;
    use rustral_nn::tape::TapeModule;
    use rustral_nn::LinearBuilder;
    use rustral_optim::Sgd;
    use rustral_runtime::{save_model_to_path, TapeTrainer, TapeTrainerConfig};

    #[derive(Clone)]
    struct TinyModel<B: rustral_core::Backend> {
        lin: rustral_nn::Linear<B>,
    }

    impl<B: rustral_core::Backend> NamedParameters<B> for TinyModel<B> {
        fn visit_parameters(&self, f: &mut dyn FnMut(&str, &rustral_core::Parameter<B>)) {
            self.lin.visit_parameters(f);
        }

        fn visit_parameters_mut(&mut self, f: &mut dyn FnMut(&str, &mut rustral_core::Parameter<B>)) {
            self.lin.visit_parameters_mut(f);
        }
    }

    let out_path = std::env::args().nth(1).unwrap_or_else(|| "tiny_linear.safetensors".to_string());

    let backend = CpuBackend::default();
    let lin = LinearBuilder::new(1, 1).with_bias(true).seed(0).build(&backend)?;
    let mut model = TinyModel { lin };

    let data: Vec<(f32, f32)> = (-20..=20).map(|i| i as f32 / 10.0).map(|x| (x, 2.0 * x + 1.0)).collect();

    let config = TapeTrainerConfig { epochs: 50, batch_size: 16, ..Default::default() };
    let optimizer = Sgd::new(0.05);
    let mut trainer = TapeTrainer::<CpuBackend, _>::new(config, optimizer);

    trainer.train_model(&backend, &mut model, &data, |m, (x, y), tape, ctx| {
        let x_t = backend.tensor_from_vec(vec![*x], &[1, 1])?;
        let y_t = backend.tensor_from_vec(vec![*y], &[1, 1])?;
        let x_id = tape.watch(x_t);
        let y_id = tape.watch(y_t);

        let pred = m.lin.forward_tape(x_id, tape, ctx)?;
        tape.mse_loss(pred, y_id, ctx)
    })?;

    save_model_to_path(&out_path, &model, &backend)?;
    eprintln!("wrote artifact to {out_path}");
    Ok(())
}

#[cfg(not(feature = "training"))]
fn main() {
    eprintln!("This example requires `--features training`.");
}
