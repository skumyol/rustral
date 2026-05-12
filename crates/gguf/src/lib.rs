//! Minimal **GGUF** parsing: magic, version, tensor count, metadata KV count.
//!
//! Full tensor payload decoding (quantized weights) is future work; this crate establishes
//! the file boundary and lets CI validate tiny fixtures.

use thiserror::Error;

/// Little-endian `GGUF` magic (`0x4655_4747`).
pub const GGUF_MAGIC_LE: u32 = u32::from_le_bytes(*b"GGUF");

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GgufHeader {
    pub version: u32,
    pub tensor_count: u64,
    pub metadata_kv_count: u64,
}

#[derive(Debug, Error)]
pub enum GgufError {
    #[error("GGUF buffer too short (need at least {need} bytes, got {got})")]
    TooShort { need: usize, got: usize },
    #[error("invalid GGUF magic: expected 0x{expected:08x}, got 0x{got:08x}")]
    BadMagic { expected: u32, got: u32 },
}

/// Read the fixed-size GGUF preamble (magic, version, tensor count, metadata KV count).
pub fn read_gguf_header(data: &[u8]) -> Result<GgufHeader, GgufError> {
    const NEED: usize = 4 + 4 + 8 + 8;
    if data.len() < NEED {
        return Err(GgufError::TooShort { need: NEED, got: data.len() });
    }
    let magic = u32::from_le_bytes(data[0..4].try_into().unwrap());
    if magic != GGUF_MAGIC_LE {
        return Err(GgufError::BadMagic { expected: GGUF_MAGIC_LE, got: magic });
    }
    let version = u32::from_le_bytes(data[4..8].try_into().unwrap());
    let tensor_count = u64::from_le_bytes(data[8..16].try_into().unwrap());
    let metadata_kv_count = u64::from_le_bytes(data[16..24].try_into().unwrap());
    Ok(GgufHeader { version, tensor_count, metadata_kv_count })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_minimal_header() {
        let mut b = Vec::new();
        b.extend_from_slice(&GGUF_MAGIC_LE.to_le_bytes());
        b.extend_from_slice(&3u32.to_le_bytes());
        b.extend_from_slice(&7u64.to_le_bytes());
        b.extend_from_slice(&2u64.to_le_bytes());
        let h = read_gguf_header(&b).unwrap();
        assert_eq!(h.version, 3);
        assert_eq!(h.tensor_count, 7);
        assert_eq!(h.metadata_kv_count, 2);
    }

    #[test]
    fn rejects_bad_magic() {
        let mut b = vec![0u8; 24];
        b[0..4].copy_from_slice(&0u32.to_le_bytes());
        assert!(matches!(read_gguf_header(&b), Err(GgufError::BadMagic { .. })));
    }
}
