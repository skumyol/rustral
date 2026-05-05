//! Minimal multi-process **CPU** data-parallel smoke test using MPI.
//!
//! Requires OpenMPI or MPICH (`mpicc`, `mpirun`) and Rust built with `--features mpi`.
//!
//! ```text
//! cargo build -p rustral-distributed --example mpi_dp_cpu --features mpi
//! mpirun -n 4 target/debug/examples/mpi_dp_cpu
//! ```

#[cfg(feature = "mpi")]
fn main() {
    let _universe = mpi::initialize().expect("MPI_Init");

    use rustral_core::{ForwardCtx, Mode};
    use rustral_distributed::{DataParallelTrainer, ProcessGroup};
    use rustral_ndarray_backend::CpuBackend;
    use rustral_optim::{Gradient, Sgd};

    let pg = ProcessGroup::new_mpi().expect("process group");
    let backend = CpuBackend::default();
    let optimizer = Sgd::new(0.1);
    let mut trainer = DataParallelTrainer::new(pg.clone(), optimizer);

    let mut params =
        vec![rustral_core::Parameter::new("p0", backend.tensor_from_vec(vec![1.0f32], &[1]).unwrap())];
    let param_id = params[0].id();

    let batch = vec![backend.tensor_from_vec(vec![1.0f32], &[1]).unwrap()];
    let mut ctx = ForwardCtx::new(&backend, Mode::Train);

    let mut loss_fn =
        |_item: &<CpuBackend as rustral_core::Backend>::Tensor, _ctx: &mut ForwardCtx<CpuBackend>| {
            let grad_tensor = backend.tensor_from_vec(vec![0.25f32], &[1]).unwrap();
            Ok((0.25f32, vec![Gradient { param_id, tensor: grad_tensor }]))
        };

    let loss = trainer.step(&mut params, &batch, &mut loss_fn, &mut ctx).expect("step");
    if pg.is_primary() {
        println!("mpi_dp_cpu ok: rank {} / {}, reported loss {:.6}", pg.rank(), pg.world_size(), loss);
    }
}

#[cfg(not(feature = "mpi"))]
fn main() {
    eprintln!("Rebuild with: cargo build -p rustral-distributed --example mpi_dp_cpu --features mpi");
}
