//! HIP kernel module.
//!
//! This module provides safe Rust wrappers for the HIP kernels compiled from
//! the CUDA kernel sources (which have been ported to be HIP-compatible).

#![allow(clippy::missing_safety_doc)]
pub mod kernels;
pub mod ntt;
