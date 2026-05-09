//! Finite-difference gradient tests for autodiff correctness.
//!
//! This test validates that gradients computed by the Tape match numerical gradients
//! computed via finite differences. This is crucial for debugging overfitting issues
//! and ensuring the autodiff implementation is correct.

mod tests {
    use rustral_autodiff::Tape;
    use rustral_core::{Backend, ForwardCtx};
    use rustral_ndarray_backend::CpuBackend;

    /// Step size for symmetric finite differences; balances truncation vs rounding error on f32.
    const EPS: f32 = 5e-4;

    /// Compute numerical gradient using finite differences.
    fn numerical_gradient<B: Backend>(
        f: impl Fn(&B, &B::Tensor) -> std::result::Result<f32, Box<dyn std::error::Error>>,
        backend: &B,
        tensor: &B::Tensor,
        param_idx: usize,
    ) -> std::result::Result<f32, Box<dyn std::error::Error>> {
        let shape = backend.ops().shape(tensor);
        let flat = backend.ops().tensor_to_vec(tensor)?;
        let mut flat_plus = flat.clone();
        let mut flat_minus = flat.clone();

        flat_plus[param_idx] += EPS;
        flat_minus[param_idx] -= EPS;

        let tensor_plus = backend.ops().tensor_from_vec(flat_plus, &shape)?;
        let tensor_minus = backend.ops().tensor_from_vec(flat_minus, &shape)?;

        let f_plus = f(backend, &tensor_plus)?;
        let f_minus = f(backend, &tensor_minus)?;

        Ok((f_plus - f_minus) / (2.0 * EPS))
    }

    /// Test gradient of a simple linear operation: y = x^2
    #[test]
    fn test_gradient_square() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let backend = CpuBackend::default();
        let x = backend.tensor_from_vec(vec![2.0f32], &[1])?;

        let f = |b: &CpuBackend,
                 t: &<CpuBackend as Backend>::Tensor|
         -> std::result::Result<f32, Box<dyn std::error::Error>> {
            let mut ctx = ForwardCtx::new(b, rustral_core::Mode::Train);
            let mut tape = Tape::new();
            let x_id = tape.watch(t.clone());
            let y_id = tape.mul(x_id, x_id, &mut ctx)?;
            let y = tape.value(y_id).unwrap();
            let y_vec = b.ops().tensor_to_vec(y)?;
            Ok(y_vec.iter().sum())
        };

        let analytical = {
            let mut ctx = ForwardCtx::new(&backend, rustral_core::Mode::Train);
            let mut tape = Tape::new();
            let x_id = tape.watch(x.clone());
            let y_id = tape.mul(x_id, x_id, &mut ctx)?;
            let loss_id = tape.sum_all_tape(y_id, &mut ctx)?;
            let grads = tape.backward(loss_id, |v, s| backend.ops().tensor_from_vec(v, s), backend.ops())?;
            let grad = grads.get(&x_id).unwrap();
            let grad_vec = backend.ops().tensor_to_vec(grad)?;
            grad_vec[0]
        };

        let numerical = numerical_gradient(f, &backend, &x, 0)?;

        println!("Square gradient: analytical={}, numerical={}", analytical, numerical);
        assert!(
            (analytical - numerical).abs() < 1e-2,
            "Gradient mismatch: analytical={}, numerical={}",
            analytical,
            numerical
        );
        Ok(())
    }

    /// Test gradient of a simple linear operation: y = 2*x + 1
    #[test]
    fn test_gradient_linear() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let backend = CpuBackend::default();
        let x = backend.tensor_from_vec(vec![3.0f32], &[1])?;

        let f = |b: &CpuBackend,
                 t: &<CpuBackend as Backend>::Tensor|
         -> std::result::Result<f32, Box<dyn std::error::Error>> {
            let mut ctx = ForwardCtx::new(b, rustral_core::Mode::Train);
            let mut tape = Tape::new();
            let x_id = tape.watch(t.clone());
            let two = b.ops().tensor_from_vec(vec![2.0f32], &[1])?;
            let two_id = tape.watch(two);
            let one = b.ops().tensor_from_vec(vec![1.0f32], &[1])?;
            let one_id = tape.watch(one);
            let y_id = tape.mul(x_id, two_id, &mut ctx)?;
            let y_id = tape.add(y_id, one_id, &mut ctx)?;
            let y = tape.value(y_id).unwrap();
            let y_vec = b.ops().tensor_to_vec(y)?;
            Ok(y_vec.iter().sum())
        };

        let analytical = {
            let mut ctx = ForwardCtx::new(&backend, rustral_core::Mode::Train);
            let mut tape = Tape::new();
            let x_id = tape.watch(x.clone());
            let two = backend.ops().tensor_from_vec(vec![2.0f32], &[1])?;
            let two_id = tape.watch(two);
            let one = backend.ops().tensor_from_vec(vec![1.0f32], &[1])?;
            let one_id = tape.watch(one);
            let y_id = tape.mul(x_id, two_id, &mut ctx)?;
            let y_id = tape.add(y_id, one_id, &mut ctx)?;
            let loss_id = tape.sum_all_tape(y_id, &mut ctx)?;
            let grads = tape.backward(loss_id, |v, s| backend.ops().tensor_from_vec(v, s), backend.ops())?;
            let grad = grads.get(&x_id).unwrap();
            let grad_vec = backend.ops().tensor_to_vec(grad)?;
            grad_vec[0]
        };

        let numerical = numerical_gradient(f, &backend, &x, 0)?;

        println!("Linear gradient: analytical={}, numerical={}", analytical, numerical);
        assert!(
            (analytical - numerical).abs() < 1e-2,
            "Gradient mismatch: analytical={}, numerical={}",
            analytical,
            numerical
        );
        Ok(())
    }

    /// Test gradient of ReLU: y = relu(x)
    #[test]
    fn test_gradient_relu() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let backend = CpuBackend::default();

        // Test at x > 0 (gradient should be 1)
        let x_pos = backend.tensor_from_vec(vec![1.0f32], &[1])?;

        let f = |b: &CpuBackend,
                 t: &<CpuBackend as Backend>::Tensor|
         -> std::result::Result<f32, Box<dyn std::error::Error>> {
            let mut ctx = ForwardCtx::new(b, rustral_core::Mode::Train);
            let mut tape = Tape::new();
            let x_id = tape.watch(t.clone());
            let y_id = tape.relu(x_id, &mut ctx)?;
            let y = tape.value(y_id).unwrap();
            let y_vec = b.ops().tensor_to_vec(y)?;
            Ok(y_vec.iter().sum())
        };

        let analytical_pos = {
            let mut ctx = ForwardCtx::new(&backend, rustral_core::Mode::Train);
            let mut tape = Tape::new();
            let x_id = tape.watch(x_pos.clone());
            let y_id = tape.relu(x_id, &mut ctx)?;
            let loss_id = tape.sum_all_tape(y_id, &mut ctx)?;
            let grads = tape.backward(loss_id, |v, s| backend.ops().tensor_from_vec(v, s), backend.ops())?;
            let grad = grads.get(&x_id).unwrap();
            let grad_vec = backend.ops().tensor_to_vec(grad)?;
            grad_vec[0]
        };

        let numerical_pos = numerical_gradient(f, &backend, &x_pos, 0)?;

        println!("ReLU gradient (x>0): analytical={}, numerical={}", analytical_pos, numerical_pos);
        assert!(
            (analytical_pos - numerical_pos).abs() < 1e-2,
            "ReLU gradient mismatch (x>0): analytical={}, numerical={}",
            analytical_pos,
            numerical_pos
        );

        // Test at x < 0 (gradient should be 0)
        let x_neg = backend.tensor_from_vec(vec![-1.0f32], &[1])?;

        let analytical_neg = {
            let mut ctx = ForwardCtx::new(&backend, rustral_core::Mode::Train);
            let mut tape = Tape::new();
            let x_id = tape.watch(x_neg.clone());
            let y_id = tape.relu(x_id, &mut ctx)?;
            let loss_id = tape.sum_all_tape(y_id, &mut ctx)?;
            let grads = tape.backward(loss_id, |v, s| backend.ops().tensor_from_vec(v, s), backend.ops())?;
            let grad = grads.get(&x_id).unwrap();
            let grad_vec = backend.ops().tensor_to_vec(grad)?;
            grad_vec[0]
        };

        let numerical_neg = numerical_gradient(f, &backend, &x_neg, 0)?;

        println!("ReLU gradient (x<0): analytical={}, numerical={}", analytical_neg, numerical_neg);
        assert!(
            (analytical_neg - numerical_neg).abs() < 1e-2,
            "ReLU gradient mismatch (x<0): analytical={}, numerical={}",
            analytical_neg,
            numerical_neg
        );
        Ok(())
    }
}
