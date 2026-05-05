# WGPU upgrade path (experimental backend)

`rustral-wgpu-backend` is **experimental**. It remains behind explicit documentation and should not block CI until teardown and cross-platform tests are stable.

## Current pin

- Crate: `crates/wgpu-backend/Cargo.toml`
- Version: **`wgpu = "0.19"`** (see file for the exact pin at upgrade time)

## Recommended upgrade steps

1. **Bump `wgpu` and `pollster`** together (wgpu release notes often require matching `raw-window-handle` / `naga` transitive versions — let Cargo resolve after the bump).
2. **Run only wgpu-backend tests first**:
   ```bash
   cargo test -p rustral-wgpu-backend
   ```
3. **Watch for teardown hangs / abort-on-drop**: exercise drop paths (drop device/queue after work submission). Older wgpu versions occasionally needed explicit `pollster::block_on(device.poll(...))` patterns before dropping surfaces — re-verify against current wgpu docs.
4. **Vulkan validation**: on Linux CI, enable `VK_LAYER_KHRONOS_validation` locally when debugging descriptor/layout regressions (optional in CI to avoid flakes).
5. **Gate CI**: keep `rustral-wgpu-backend` out of default workspace checks until `cargo test -p rustral-wgpu-backend` is reliable on the CI image (GPU/Vulkan optional).

## Optional execution upgrade

If compilation pulls in incompatible `naga`/`wgpu-hal` APIs:

- Prefer upgrading to the **latest patch** of the target minor line before jumping majors.
- Read `wgpu` changelog for breaking changes in `Device::poll`, `Queue::submit`, and buffer mapping.

## Status

Until the steps above pass on developer and CI machines, treat WebGPU as **inference / experimentation only**, not primary training.
