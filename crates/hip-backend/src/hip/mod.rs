//! HIP kernel module.
//!
//! This module provides safe Rust wrappers for the HIP kernels compiled from
//! the CUDA kernel sources (which have been ported to be HIP-compatible).

#![allow(clippy::missing_safety_doc)]
pub(crate) mod kernels;
pub(crate) mod ntt;
