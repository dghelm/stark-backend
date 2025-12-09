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
//! **Required environment variables:**
//!
//! ```bash
//! HIP_FORCE_EXIT=1 HIP_ARCH=gfx1151 cargo test -p openvm-hip-backend
//! ```
//!
//! - `HIP_ARCH`: Set to your GPU architecture (e.g., `gfx1151` for Strix Halo,
//!   `gfx1100` for RDNA3, `gfx90a` for MI200)
//! - `HIP_FORCE_EXIT=1`: **Required** to avoid SIGSEGV on exit (see below)
//!
//! ## Known Issue: SIGSEGV on Exit
//!
//! Without `HIP_FORCE_EXIT=1`, tests will pass but the process crashes with
//! SIGSEGV during cleanup:
//!
//! ```text
//! test result: ok. 40 passed; 0 failed
//! error: process didn't exit successfully (signal: 11, SIGSEGV)
//! ```
//!
//! This is caused by a bug in HIP/ROCm's internal atexit handlers that crash
//! when cleaning up VPMM (Virtual Memory Pool) allocations. The crash happens
//! *inside HIP's code*, not ours. Setting `HIP_FORCE_EXIT=1` calls `_exit(0)`
//! before HIP's buggy cleanup runs.
//!
//! # Production Usage
//!
//! For production binaries, do **not** use `HIP_FORCE_EXIT`. Instead, call
//! [`openvm_hip_common::hip_runtime_shutdown()`] explicitly before exit:
//!
//! ```ignore
//! fn main() {
//!     // ... your proving code ...
//!
//!     // Clean shutdown before exit
//!     openvm_hip_common::hip_runtime_shutdown();
//! }
//! ```
//!
//! This properly releases VPMM resources before HIP's cleanup runs, avoiding
//! the crash without the nuclear option of `_exit(0)`.

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
