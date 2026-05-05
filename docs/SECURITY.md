# Security Guidelines for Rustral

This document outlines security practices for Rustral.

## Security Philosophy

Rustral handles potentially sensitive data (model weights, training data) and runs with high privileges on GPU hardware. Security is critical at every layer.

## Current Security Posture

### Safe by Default

- **Memory Safety**: Rust's ownership model prevents most memory safety issues
- **No Unsafe**: Minimal use of `unsafe` (only for FFI to CUDA/NCCL)
- **Type Safety**: Strong typing prevents many logic errors

### Areas of Concern

1. **Serialization**: Model checkpoint format (Safetensors preferred over Pickle)
2. **Distributed Communication**: MPI/NCCL message handling
3. **FFI Boundaries**: CUDA, cuDNN, NCCL bindings
4. **Model Loading**: Untrusted model files from external sources

## Security Audit Checklist

Run before each release:

```bash
# Run automated security audit
./tools/security_audit.sh

# Manual checks
cargo audit                    # Dependency vulnerabilities
cargo clippy -- -W clippy::pedantic  # Additional lints
```

## Secure Coding Guidelines

### Unsafe Code

When `unsafe` is necessary:

```rust
// SAFETY: We hold the lock, and the pointer is valid
// because we just allocated it with the correct size.
unsafe { cuda_memcpy(dst, src, size, direction) }
```

Requirements:
1. Every `unsafe` block must have a `SAFETY:` comment
2. Code review required for all unsafe changes
3. Prefer safe abstractions (e.g., `std::slice::from_raw_parts`)

### Serialization

**Preferred: Safetensors**

```rust
// Safe - format is well-defined and limited
use rustral_io::SafetensorsLoader;
let tensors = SafetensorsLoader::new().load(path)?;
```

**Avoid: Pickle/Native formats for untrusted data**

```rust
// Dangerous - can execute arbitrary code
// let model: Model = serde_pickle::from_slice(data)?; // DON'T DO THIS
```

### Input Validation

Always validate user-controlled inputs:

```rust
pub fn load_dataset(path: &Path, max_size: usize) -> Result<Dataset> {
    // Validate path is within allowed directory
    let canonical = path.canonicalize()?;
    if !canonical.starts_with(&self.allowed_root) {
        return Err(Error::PathOutsideRoot);
    }

    // Validate size limits
    let size = std::fs::metadata(&canonical)?.len() as usize;
    if size > max_size {
        return Err(Error::DatasetTooLarge { size, max: max_size });
    }

    // ... proceed with loading
}
```

### Secrets Management

Never hardcode credentials:

```rust
// ❌ BAD
const API_KEY: &str = "sk-1234567890abcdef";

// ✅ GOOD
let api_key = std::env::var("RUSTRAL_API_KEY")
    .map_err(|_| Error::MissingApiKey)?;
```

## Vulnerability Disclosure

If you discover a security vulnerability in Rustral:

1. **DO NOT** open a public issue with exploit details
2. Prefer **[GitHub private vulnerability reporting](https://github.com/skumyol/rustral/security/advisories/new)** for the repository (enable “Private vulnerability reporting” in repo settings if needed).
3. Include:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
   - Suggested fix (if any)

We will acknowledge receipt within 48 hours and provide a timeline for a fix.

## Dependencies

### Security Scanning

We use `cargo audit` to track known vulnerabilities:

```bash
# Install
cargo install cargo-audit

# Run
cargo audit
```

### Update Policy

- Critical vulnerabilities: Update within 24 hours
- High severity: Update within 1 week
- Medium/Low: Update in next scheduled release

## Network Security

### Distributed Training

For multi-node training:

1. Use authenticated MPI (not TCP sockets)
2. Enable TLS for control plane communication
3. Isolate training network from public internet
4. Use network policies in Kubernetes environments

### Model Serving

If serving models:

1. Rate limiting on prediction endpoints
2. Input size validation
3. Timeout on model inference
4. Logging of suspicious inputs (careful with PII)

## Compliance

### Data Protection

When training on sensitive data:

- **Encryption at Rest**: Encrypt model checkpoints
- **Encryption in Transit**: TLS for distributed training
- **Access Control**: Limit checkpoint read access
- **Audit Logging**: Log model access and training runs

### Model Provenance

Track model lineage:

```rust
use rustral_metrics::ProvenanceTracker;

let tracker = ProvenanceTracker::new()
    .with_training_data("dataset-v1.2")
    .with_hyperparameters(&config)
    .with_git_commit(env!("VERGEN_GIT_SHA"));

// Saves metadata with checkpoint
tracker.save_with_model(&model, path)?;
```

## Incident Response

### Detection

Monitor for:

- Unexpected network connections
- Large data transfers
- Failed authentication attempts
- Model checkpoint tampering

### Response

If compromise is suspected:

1. Isolate affected nodes
2. Preserve logs
3. Rotate all credentials
4. Rebuild from known-good source
5. Audit all checkpoints loaded since compromise

## Security Testing

### Fuzz Testing

Run fuzz tests on deserialization:

```bash
cargo fuzz run fuzz_safetensors
cargo fuzz run fuzz_onnx
```

### Penetration Testing

Recommended before production:

- Third-party security audit
- Fuzz testing with custom mutators
- Chaos engineering (network partitions, crashes)

## References

- [Rust Secure Coding Guidelines](https://secure-coding.readthedocs.io/)
- [OWASP Top 10](https://owasp.org/www-project-top-ten/)
- [Safetensors Security](https://huggingface.co/docs/safetensors/security)
- [NVIDIA Security](https://www.nvidia.com/en-us/security/)

## Contact

Security: use [GitHub Security Advisories](https://github.com/skumyol/rustral/security) for responsible disclosure.  
General bugs and questions: [Issues](https://github.com/skumyol/rustral/issues).
