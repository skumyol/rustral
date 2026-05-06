#[cfg(feature = "training")]
fn main() -> anyhow::Result<()> {
    use rustral_core::{Backend, Module, NamedParameters};
    use rustral_ndarray_backend::CpuBackend;
    use rustral_nn::tape::TapeModule;
    use rustral_nn::LinearBuilder;
    use rustral_optim::Sgd;
    use rustral_runtime::{load_model, save_model, TapeTrainer, TapeTrainerConfig};

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

    let backend = CpuBackend::default();
    let lin = LinearBuilder::new(1, 1).with_bias(true).seed(0).build(&backend)?;
    let mut model = TinyModel { lin };

    // y = 2x + 1 training set
    let data: Vec<(f32, f32)> = (-20..=20).map(|i| i as f32 / 10.0).map(|x| (x, 2.0 * x + 1.0)).collect();

    let config = TapeTrainerConfig { epochs: 50, batch_size: 16, ..Default::default() };
    let optimizer = Sgd::new(0.05);
    let mut trainer = TapeTrainer::<CpuBackend, _>::new(config, optimizer);

    let stats = trainer.train_model(&backend, &mut model, &data, |m, (x, y), tape, ctx| {
        let x_t = backend.tensor_from_vec(vec![*x], &[1, 1])?;
        let y_t = backend.tensor_from_vec(vec![*y], &[1, 1])?;
        let x_id = tape.watch(x_t);
        let y_id = tape.watch(y_t);

        let pred = m.lin.forward_tape(x_id, tape, ctx)?;
        tape.mse_loss(pred, y_id, ctx)
    })?;

    let last = stats.last().cloned().unwrap();
    println!("done: epoch={} mean_loss={:.6}", last.epoch, last.mean_loss);

    // Save → load → infer roundtrip (byte-buffer model artifact).
    let bytes = save_model(&model, &backend)?;

    let lin2 = LinearBuilder::new(1, 1).with_bias(true).seed(123).build(&backend)?;
    let mut model2 = TinyModel { lin: lin2 };
    load_model(&mut model2, &backend, &bytes)?;

    // Run the same inference on both models and verify equality.
    let mut ctx = rustral_core::ForwardCtx::new(&backend, rustral_core::Mode::Inference);
    let x = backend.tensor_from_vec(vec![0.25], &[1, 1])?;
    let y1 = model.lin.forward(x.clone(), &mut ctx)?;
    let y1v = backend.ops().tensor_to_vec(&y1)?;

    let mut ctx2 = rustral_core::ForwardCtx::new(&backend, rustral_core::Mode::Inference);
    let y2 = model2.lin.forward(x, &mut ctx2)?;
    let y2v = backend.ops().tensor_to_vec(&y2)?;

    println!("roundtrip inference: y1={:?} y2={:?}", y1v, y2v);
    anyhow::ensure!(y1v == y2v, "save/load roundtrip changed outputs");
    Ok(())
}

#[cfg(not(feature = "training"))]
fn main() {
    eprintln!("This example requires `--features training`.");
}
