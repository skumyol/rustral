use std::sync::Arc;

use mnr_core::{Backend, ForwardCtx, Mode};
use mnr_ndarray_backend::CpuBackend;
use mnr_nn::{LinearBuilder, Readout};
use mnr_symbolic::Vocabulary;

fn main() -> anyhow::Result<()> {
    let backend = CpuBackend::default();
    let mut ctx = ForwardCtx::new(&backend, Mode::Inference);

    // Build a linear layer with the builder — no manual parameter creation.
    let linear = LinearBuilder::new(4, 2).with_bias(true).seed(42).build(&backend)?;

    let mut labels = Vocabulary::with_specials("unknown");
    labels.insert("accept")?;
    labels.freeze();

    let readout = Readout::new(Arc::new(labels), linear);
    let hidden = backend.tensor_from_vec(vec![1.0, 0.0, 0.5, -1.0], &[4])?;
    let logits = readout.logits(hidden, &mut ctx)?;
    println!("logits shape: {:?}", ctx.backend().ops().shape(&logits));

    Ok(())
}
