#[cfg(test)]
mod tabular_tests {
    use rustral_core::Backend;
    use crate::CpuBackend;

    #[test]
    fn test_bincount() {
        let backend = CpuBackend::default();
        let ops = backend.ops();
        let x = backend.tensor_from_vec(vec![1.0, 1.0, 2.0, 0.0, 1.0], &[5]).unwrap();
        let counts = ops.bincount(&x, 3).unwrap();
        let vals = ops.tensor_to_vec(&counts).unwrap();
        assert_eq!(vals, vec![1.0, 3.0, 1.0]);
    }

    #[test]
    fn test_topk_1d() {
        let backend = CpuBackend::default();
        let ops = backend.ops();
        let x = backend.tensor_from_vec(vec![1.0, 5.0, 2.0, 4.0, 3.0], &[5]).unwrap();
        let (vals, indices) = ops.topk(&x, 2, 0, true).unwrap();
        assert_eq!(ops.tensor_to_vec(&vals).unwrap(), vec![5.0, 4.0]);
        assert_eq!(ops.tensor_to_vec(&indices).unwrap(), vec![1.0, 3.0]);
    }

    #[test]
    fn test_unique_with_counts() {
        let backend = CpuBackend::default();
        let ops = backend.ops();
        let x = backend.tensor_from_vec(vec![1.0, 2.0, 1.0, 3.0, 2.0, 1.0], &[6]).unwrap();
        let (vals, counts) = ops.unique_with_counts(&x).unwrap();
        assert_eq!(ops.tensor_to_vec(&vals).unwrap(), vec![1.0, 2.0, 3.0]);
        assert_eq!(ops.tensor_to_vec(&counts).unwrap(), vec![3.0, 2.0, 1.0]);
    }
}
