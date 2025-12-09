//! Common utilities for HIP backend benchmarks.
//!
//! This module provides shared functionality for all performance benchmarks:
//! - Random data generation for BabyBear field elements
//! - Matrix creation utilities
//! - Benchmark helper macros and functions

use openvm_hip_common::stream::device_synchronize;
use p3_baby_bear::BabyBear;
use p3_field::FieldAlgebra;
use rand::{rngs::StdRng, Rng, SeedableRng};

/// Create a seeded RNG for reproducible benchmarks.
pub fn create_seeded_rng() -> StdRng {
    StdRng::seed_from_u64(0x1234_5678_9ABC_DEF0)
}

/// Generate random BabyBear field elements.
pub fn random_field_elements(count: usize) -> Vec<BabyBear> {
    let mut rng = create_seeded_rng();
    (0..count)
        .map(|_| BabyBear::from_wrapped_u32(rng.gen::<u32>()))
        .collect()
}

/// Generate random u32 values (for simpler benchmarks).
pub fn random_u32_elements(count: usize) -> Vec<u32> {
    let mut rng = create_seeded_rng();
    (0..count).map(|_| rng.gen::<u32>()).collect()
}

/// Force GPU synchronization and return elapsed time.
pub fn sync_gpu() {
    device_synchronize().expect("GPU sync failed");
}

/// Benchmark sizes (in log2).
pub const LOG_SIZES: &[usize] = &[10, 12, 14, 16, 18, 20];

/// Get human-readable size string.
pub fn size_string(log_size: usize) -> String {
    let size = 1usize << log_size;
    let bytes = size * 4; // BabyBear is 4 bytes
    if bytes >= 1024 * 1024 {
        format!("{}MB", bytes / (1024 * 1024))
    } else if bytes >= 1024 {
        format!("{}KB", bytes / 1024)
    } else {
        format!("{}B", bytes)
    }
}
