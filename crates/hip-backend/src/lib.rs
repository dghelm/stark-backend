//! HIP/ROCm backend for STARK proving.
//!
//! This crate provides a HIP-based GPU backend for the STARK proving system,
//! mirroring the structure of `openvm-cuda-backend` for AMD GPU support.
//!
//! # Status
//!
//! All components are fully implemented and wired up:
//!
//! - [x] HIP kernels (NTT, FRI, Poseidon2, LDE, Merkle, Quotient, etc.)
//! - [x] Field arithmetic headers (fp.h, fpext.h with HIP intrinsics)
//! - [x] Kernel launchers (launcher.cuh)
//! - [x] Rust kernel bindings (hip/kernels.rs, hip/ntt.rs)
//! - [x] DeviceDataTransporter implementation (data_transporter.rs)
//! - [x] LDE (Low Degree Extension) - lde module
//! - [x] Merkle tree - merkle_tree module
//! - [x] TraceCommitter trait - wired to committer module
//! - [x] QuotientCommitter trait - wired to quotient module
//! - [x] RapPartialProver trait - wired to fri_log_up module
//! - [x] OpeningProver trait - wired to opener module
//! - [x] FriLogUpPhaseGpu - integrated into HipDevice
//!
//! The kernel sources in `cuda-backend/cuda/` have been ported to be HIP-compatible
//! using `__HIPCC__` preprocessor guards. They are compiled with hipcc during build.
//!
//! # Testing
//!
//! All core modules have been tested on AMD Radeon 8060S (gfx1151 / Strix Halo):
//! - Matrix transpose: 7 tests
//! - Data transport roundtrip: 8 tests
//! - LDE computation: 7 tests
//! - Merkle tree: 5 tests
//! - Trace committer: 4 tests
//! - Transpiler codec: 8 tests
//!
//! ## Running Tests
//!
//! Due to a bug in the HIP/ROCm runtime's cleanup handlers, running many tests
//! without the `HIP_FORCE_EXIT=1` environment variable may cause a SIGSEGV after
//! tests complete (during process teardown). To run tests cleanly:
//!
//! ```bash
//! HIP_FORCE_EXIT=1 HIP_ARCH=gfx1151 cargo test --package openvm-hip-backend
//! ```
//!
//! The `HIP_FORCE_EXIT=1` flag tells the HIP backend to call `_exit(0)` before
//! the buggy HIP runtime cleanup handlers can run. This is safe for tests but
//! should not be used in production (where you want normal process teardown).

// Ensure CUDA and ROCm backends are not enabled simultaneously.
// This would cause link-time conflicts between cudart and amdhip64.
#[cfg(feature = "cuda")]
compile_error!(
    "Features `cuda` and `rocm` are mutually exclusive. \
     The HIP backend (openvm-hip-backend) cannot be used alongside CUDA. \
     Please enable only one GPU backend."
);

pub mod base;
pub mod types;

// Device and backend types
pub mod engine;
pub mod hip_device;
pub mod prover_backend;

// HIP kernel bindings (now functional with ported kernels)
pub mod hip;

// Data transfer between host and device
pub mod data_transporter;

// LDE (Low Degree Extension) module
pub mod lde;

// Merkle tree for polynomial commitment
pub mod merkle_tree;

// Trace commitment
mod committer;

// Constraint transpiler (compiles SymbolicConstraintsDag to GPU rules)
mod transpiler;

// Quotient polynomial evaluation
mod quotient;

// FRI log-up for permutation trace generation
mod fri_log_up;

// Opening prover (FRI opening)
mod opener;

pub mod prelude {
    pub use crate::types::prelude::*;
}

// Re-export main types for convenience
pub use engine::HipBabyBearPoseidon2Engine;
pub use hip_device::{HipConfig, HipDevice};
pub use prover_backend::{HipBackend, HipPcsData};
