
#[cfg(any(target_arch = "aarch64", target_arch = "arm64ec"))]
mod aarch64;

#[cfg(any(target_arch = "aarch64", target_arch = "arm64ec"))]
pub use aarch64::*;
