//! Trace committer tests for HIP backend.
//!
//! Tests for GPU-accelerated trace commitment.

use std::sync::Arc;

use openvm_hip_backend::{
    data_transporter::transport_matrix_to_device,
    hip_device::{HipConfig, HipDevice},
    lde::GpuLdeImpl,
};
use openvm_stark_sdk::config::FriParameters;
use p3_baby_bear::BabyBear;
use p3_field::FieldAlgebra;
use p3_matrix::dense::RowMajorMatrix;

fn skip_if_no_hip() -> bool {
    if !openvm_hip_builder::hip_available() {
        eprintln!("Skipping: HIP not available");
        true
    } else {
        false
    }
}

/// Create a simple test matrix with known values
fn create_test_matrix(height: usize, width: usize) -> Arc<RowMajorMatrix<BabyBear>> {
    let values: Vec<BabyBear> = (0..height * width)
        .map(|i| BabyBear::from_canonical_u32((i % 1000) as u32))
        .collect();
    Arc::new(RowMajorMatrix::new(values, width))
}

fn create_hip_device() -> HipDevice {
    let fri = FriParameters::standard_fast();
    let config = HipConfig::new(fri, BabyBear::ONE);
    HipDevice::new(config)
}

#[test]
fn test_commit_trace() {
    if skip_if_no_hip() {
        return;
    }

    let hip_device = create_hip_device();

    let height = 16;
    let width = 4;

    let host_matrix = create_test_matrix(height, width);
    let device_matrix = transport_matrix_to_device(host_matrix);

    let (log_heights, merkle_tree) = hip_device.commit_trace::<GpuLdeImpl>(device_matrix);

    assert_eq!(log_heights.len(), 1);
    assert_eq!(log_heights[0], 4); // log2(16) = 4

    let _root = merkle_tree.root();
}

#[test]
fn test_commit_traces_with_lde() {
    if skip_if_no_hip() {
        return;
    }

    let hip_device = create_hip_device();

    let height = 16;
    let width = 4;
    let log_blowup = 1; // 2x blowup

    let host_matrix = create_test_matrix(height, width);
    let device_matrix = transport_matrix_to_device(host_matrix);

    let traces_with_shifts = vec![(device_matrix, BabyBear::ONE)];

    let (log_heights, merkle_tree) =
        hip_device.commit_traces_with_lde::<GpuLdeImpl>(traces_with_shifts, log_blowup);

    assert_eq!(log_heights.len(), 1);
    assert_eq!(log_heights[0], 4); // log2(16) = 4

    let _root = merkle_tree.root();

    // LDE height should be 2x original
    assert_eq!(merkle_tree.get_max_height(), height << log_blowup);
}

#[test]
fn test_commit_multiple_traces() {
    if skip_if_no_hip() {
        return;
    }

    let hip_device = create_hip_device();
    let log_blowup = 1;

    // Create two traces with same height
    let matrix1 = create_test_matrix(16, 4);
    let matrix2 = create_test_matrix(16, 8);

    let device_matrix1 = transport_matrix_to_device(matrix1);
    let device_matrix2 = transport_matrix_to_device(matrix2);

    let traces_with_shifts = vec![
        (device_matrix1, BabyBear::ONE),
        (device_matrix2, BabyBear::ONE),
    ];

    let (log_heights, merkle_tree) =
        hip_device.commit_traces_with_lde::<GpuLdeImpl>(traces_with_shifts, log_blowup);

    assert_eq!(log_heights.len(), 2);
    assert_eq!(log_heights[0], 4); // log2(16) = 4
    assert_eq!(log_heights[1], 4); // log2(16) = 4

    let _root = merkle_tree.root();
}

#[test]
fn test_commit_deterministic() {
    if skip_if_no_hip() {
        return;
    }

    let hip_device = create_hip_device();

    let height = 8;
    let width = 4;

    // Commit the same trace twice
    let matrix1 = create_test_matrix(height, width);
    let matrix2 = create_test_matrix(height, width);

    let device_matrix1 = transport_matrix_to_device(matrix1);
    let device_matrix2 = transport_matrix_to_device(matrix2);

    let (_, tree1) = hip_device.commit_trace::<GpuLdeImpl>(device_matrix1);
    let (_, tree2) = hip_device.commit_trace::<GpuLdeImpl>(device_matrix2);

    let root1 = tree1.root();
    let root2 = tree2.root();

    assert_eq!(root1, root2, "Same trace should produce same commitment");
}
