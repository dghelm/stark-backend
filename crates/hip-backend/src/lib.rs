//! HIP/ROCm backend for STARK proving.
//!
//! This crate provides a HIP-based GPU backend for the STARK proving system,
//! mirroring the structure of `openvm-cuda-backend` for AMD GPU support.
//!
//! # Status
//!
//! This crate is a work in progress. The following components need to be ported:
//!
//! - [ ] HIP kernels (NTT, FRI, Poseidon2, etc.)
//! - [ ] Field arithmetic headers (fp.h, fpext.h with HIP intrinsics)
//! - [ ] Kernel launchers
//!
//! Currently, only the Rust-side infrastructure is implemented.

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

// TODO: Enable these modules once kernels are ported
// pub mod chip;
// mod committer;
// pub mod hip;
// pub mod fri_log_up;
// mod lde;
// mod merkle_tree;
// mod opener;
// mod quotient;
// mod transpiler;

pub mod prelude {
    pub use crate::types::prelude::*;
}

// TODO: Enable once kernel infrastructure is in place
// pub mod data_transporter;
// pub mod engine;
// pub mod hip_device;
// pub mod prover_backend;
