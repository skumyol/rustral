use rustral_core::{DeviceType, OperationProfiler};

#[test]
fn snapshot_is_stable_and_serializable() {
    let mut p = OperationProfiler::new();
    p.record_operation_internal(
        "matmul",
        std::time::Duration::from_millis(10),
        None,
        None,
        DeviceType::Cpu,
        None,
    );
    p.record_operation_internal(
        "matmul",
        std::time::Duration::from_millis(20),
        None,
        None,
        DeviceType::Cpu,
        None,
    );

    let snap = p.snapshot(10);
    assert_eq!(snap.total_calls, 2);
    assert!(!snap.ops.is_empty());
    let json = serde_json::to_string(&snap).unwrap();
    assert!(json.contains("\"matmul\""));
}

