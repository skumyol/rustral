#[cfg(feature = "training")]
fn main() -> anyhow::Result<()> {
    use rustral_autodiff::Tape;
    use rustral_core::{ForwardCtx, Mode, Module, NamedParameters};
    use rustral_ndarray_backend::CpuBackend;
    use rustral_nn::tape::TapeModule;
    use rustral_nn::LinearBuilder;
    use rustral_optim::Adam;
    use rustral_runtime::{SupervisedTapeModel, TapeTrainer, TapeTrainerConfig};

    #[derive(Clone)]
    struct XorMlp<B: rustral_core::Backend> {
        l1: rustral_nn::Linear<B>,
        l2: rustral_nn::Linear<B>,
    }

    impl<B: rustral_core::Backend> NamedParameters<B> for XorMlp<B> {
        fn visit_parameters(&self, f: &mut dyn FnMut(&str, &rustral_core::Parameter<B>)) {
            self.l1.visit_parameters(&mut |name, p| f(&format!("l1.{name}"), p));
            self.l2.visit_parameters(&mut |name, p| f(&format!("l2.{name}"), p));
        }

        fn visit_parameters_mut(&mut self, f: &mut dyn FnMut(&str, &mut rustral_core::Parameter<B>)) {
            self.l1.visit_parameters_mut(&mut |name, p| f(&format!("l1.{name}"), p));
            self.l2.visit_parameters_mut(&mut |name, p| f(&format!("l2.{name}"), p));
        }
    }

    impl<B: rustral_core::Backend> SupervisedTapeModel<B, [f32; 2], usize> for XorMlp<B>
    where
        B::Tensor: Clone,
    {
        fn forward_tape(
            &mut self,
            input: [f32; 2],
            tape: &mut Tape<B>,
            ctx: &mut ForwardCtx<B>,
        ) -> rustral_core::Result<rustral_autodiff::TensorId> {
            let x = ctx.backend().ops().tensor_from_vec(vec![input[0], input[1]], &[1, 2])?;
            let x = tape.watch(x);
            let h = self.l1.forward_tape(x, tape, ctx)?;
            let h = tape.relu(h, ctx)?;
            self.l2.forward_tape(h, tape, ctx)
        }

        fn loss_tape(
            &mut self,
            logits: rustral_autodiff::TensorId,
            target: usize,
            tape: &mut Tape<B>,
            ctx: &mut ForwardCtx<B>,
        ) -> rustral_core::Result<rustral_autodiff::TensorId> {
            // Target index as f32 tensor [batch]
            let t = ctx.backend().ops().tensor_from_vec(vec![target as f32], &[1])?;
            let t = tape.watch(t);
            tape.cross_entropy_loss(logits, t, ctx)
        }
    }

    let backend = CpuBackend::default();
    let mut model = XorMlp {
        l1: LinearBuilder::new(2, 8).with_bias(true).seed(1).build(&backend)?,
        l2: LinearBuilder::new(8, 2).with_bias(true).seed(2).build(&backend)?,
    };

    let train: Vec<([f32; 2], usize)> =
        vec![([0.0, 0.0], 0), ([0.0, 1.0], 1), ([1.0, 0.0], 1), ([1.0, 1.0], 0)];

    let config =
        TapeTrainerConfig { epochs: 5000, batch_size: 4, shuffle: true, seed: 0, ..Default::default() };
    let optimizer = Adam::new(0.01);
    let mut trainer = TapeTrainer::<CpuBackend, _>::new(config, optimizer);

    let report = trainer.fit_classification(&backend, &mut model, &train)?;
    let last = report.epochs.last().cloned().unwrap();
    println!("done: epoch={} mean_loss={:.6} elapsed={:?}", last.epoch, last.mean_loss, last.elapsed);
    if let Some(acc) = report.accuracy.and_then(|v| v.last().copied()) {
        println!("train accuracy: {:.3}", acc);
    }

    // quick inference sanity check
    let mut ctx = ForwardCtx::new(&backend, Mode::Inference);
    let _ = model.l1.forward(backend.tensor_from_vec(vec![0.0, 1.0], &[1, 2])?, &mut ctx)?;
    Ok(())
}

#[cfg(not(feature = "training"))]
fn main() {
    eprintln!("This example requires `--features training`.");
}
