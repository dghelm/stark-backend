//! Common utilities for HIP/ROCm GPU programming.
//!
//! This crate provides shared infrastructure for the HIP backend:
//! - Device memory management with virtual memory pooling (VPMM)
//! - Async stream management
//! - Device buffer abstractions
//! - Error handling utilities
//!
//! # Shutdown and Cleanup
//!
//! HIP/ROCm has a known bug where its internal atexit cleanup handlers crash
//! (SIGSEGV) when cleaning up Virtual Memory Pool Manager (VPMM) resources.
//! This crate provides two mechanisms to handle this:
//!
//! ## For Production Binaries
//!
//! Call [`hip_runtime_shutdown()`] explicitly before your program exits:
//!
//! ```ignore
//! fn main() {
//!     // ... your HIP code ...
//!
//!     // Clean shutdown - properly releases VPMM resources
//!     openvm_hip_common::hip_runtime_shutdown();
//! }
//! ```
//!
//! This releases all GPU resources before HIP's buggy cleanup runs.
//!
//! ## For Tests
//!
//! Set `HIP_FORCE_EXIT=1` environment variable:
//!
//! ```bash
//! HIP_FORCE_EXIT=1 HIP_ARCH=gfx1151 cargo test -p openvm-hip-backend
//! ```
//!
//! This calls `_exit(0)` after tests complete, skipping HIP's cleanup entirely.
//! **Do not use `HIP_FORCE_EXIT` in production** - it skips all Rust destructors.
//!
//! # Architecture
//!
//! The memory manager uses a "leak-on-exit" pattern (`Box::leak`) to prevent
//! its `Drop` from running during static destruction. Combined with the
//! `#[ctor::dtor]` early shutdown mechanism, this provides robust cleanup
//! in most scenarios without requiring `HIP_FORCE_EXIT`.

pub mod common;
pub mod copy;
pub mod d_buffer;
pub mod error;
pub mod memory_manager;
pub mod stream;

// Re-export hip_runtime_shutdown for convenient access
pub use memory_manager::hip_runtime_shutdown;
