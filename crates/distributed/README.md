# rustral-distributed

Data/tensor/pipeline-parallel and ZeRO-style APIs (single-process reference today).

Rustral is a Rust workspace for research and learning; see the [repository README](https://github.com/skumyol/rustral#readme) for install, examples, and status by backend.

## Multi-process CPU (MPI)

Enable with `--features mpi` (requires an MPI implementation: OpenMPI / MPICH / Intel MPI and dev packages so `mpicc` / pkg-config resolve).

Build the bundled smoke binary:

```bash
cargo build -p rustral-distributed --example mpi_dp_cpu --features mpi
mpirun -n 4 target/debug/examples/mpi_dp_cpu
```

`ProcessGroup::new_mpi()` wraps `MPI_COMM_WORLD`. Pair it with `DataParallelTrainer` and the CPU backend for correctness checks on a single host before scaling out.

CUDA/NVIDIA collectives are **not** wired here yet; MPI coordinates hosts, NCCL-style GPU collectives remain future work.
