// Ensure CUDA and ROCm backends are not enabled simultaneously.
// This would cause link-time conflicts between cudart and amdhip64.
#[cfg(feature = "rocm")]
compile_error!(
    "Features `cuda` and `rocm` are mutually exclusive. \
     The CUDA backend (openvm-cuda-backend) cannot be used alongside HIP/ROCm. \
     Please enable only one GPU backend."
);

pub mod base;
pub mod chip;
mod committer;
pub mod cuda;
pub mod fri_log_up;
mod lde;
mod merkle_tree;
mod opener;
mod quotient;
mod transpiler;
pub mod types;

pub mod prelude {
    pub use crate::types::prelude::*;
}
pub mod data_transporter;
pub mod engine;
pub mod gpu_device;
pub mod prover_backend;
