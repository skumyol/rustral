fn main() -> Result<(), Box<dyn std::error::Error>> {
    let proto = std::path::Path::new("proto/onnx.proto");
    println!("cargo:rerun-if-changed={}", proto.display());
    std::env::set_var("PROTOC", protoc_bin_vendored::protoc_bin_path().expect("vendored protoc"));
    prost_build::Config::new().compile_protos(&[proto], &["proto"])?;
    Ok(())
}
