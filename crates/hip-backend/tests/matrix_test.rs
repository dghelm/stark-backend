//! Matrix transpose kernel tests for HIP backend.
//!
//! These tests verify the correctness of the `matrix_transpose` kernel
//! by comparing GPU results against expected values.

use std::sync::Arc;

use openvm_hip_backend::{base::DeviceMatrix, data_transporter::transport_matrix_to_device};
use openvm_hip_common::copy::MemCopyD2H;
use openvm_stark_backend::prover::hal::MatrixDimensions;
use p3_baby_bear::BabyBear;
use p3_field::FieldAlgebra;
use p3_matrix::{dense::RowMajorMatrix, Matrix};

/// Skip test if HIP is not available.
fn skip_if_no_hip() -> bool {
    if !openvm_hip_builder::hip_available() {
        eprintln!("Skipping: HIP not available");
        true
    } else {
        false
    }
}

/// Create a test matrix with deterministic values for verification.
/// Each element at (row, col) = row * width + col (as a field element).
fn create_test_matrix(height: usize, width: usize) -> RowMajorMatrix<BabyBear> {
    let values: Vec<BabyBear> = (0..height * width)
        .map(|i| BabyBear::from_canonical_u32(i as u32))
        .collect();
    RowMajorMatrix::new(values, width)
}

/// Verify that the device matrix (column-major) matches expected transpose of host matrix.
fn verify_transpose(host: &RowMajorMatrix<BabyBear>, device: &DeviceMatrix<BabyBear>) {
    assert_eq!(device.height(), host.height(), "Height mismatch");
    assert_eq!(device.width(), host.width(), "Width mismatch");

    // Device matrix is stored in column-major order after transpose
    let device_data = device.to_host().unwrap();

    for row in 0..host.height() {
        for col in 0..host.width() {
            // Column-major index: col * height + row
            let device_idx = col * host.height() + row;
            let expected = host.get(row, col);
            let actual = device_data[device_idx];
            assert_eq!(
                actual, expected,
                "Mismatch at row={}, col={}: expected {:?}, got {:?}",
                row, col, expected, actual
            );
        }
    }
}

#[test]
fn test_matrix_transpose_small() {
    if skip_if_no_hip() {
        return;
    }

    // 4x3 matrix (4 rows, 3 columns)
    let host_matrix = create_test_matrix(4, 3);
    let device_matrix = transport_matrix_to_device(Arc::new(host_matrix.clone()));

    verify_transpose(&host_matrix, &device_matrix);
}

#[test]
fn test_matrix_transpose_square() {
    if skip_if_no_hip() {
        return;
    }

    // 32x32 square matrix
    let host_matrix = create_test_matrix(32, 32);
    let device_matrix = transport_matrix_to_device(Arc::new(host_matrix.clone()));

    verify_transpose(&host_matrix, &device_matrix);
}

#[test]
fn test_matrix_transpose_wide() {
    if skip_if_no_hip() {
        return;
    }

    // 16x1024 wide matrix (16 rows, 1024 columns)
    let host_matrix = create_test_matrix(16, 1024);
    let device_matrix = transport_matrix_to_device(Arc::new(host_matrix.clone()));

    verify_transpose(&host_matrix, &device_matrix);
}

#[test]
fn test_matrix_transpose_tall() {
    if skip_if_no_hip() {
        return;
    }

    // 1024x16 tall matrix (1024 rows, 16 columns)
    let host_matrix = create_test_matrix(1024, 16);
    let device_matrix = transport_matrix_to_device(Arc::new(host_matrix.clone()));

    verify_transpose(&host_matrix, &device_matrix);
}

#[test]
fn test_matrix_transpose_large() {
    if skip_if_no_hip() {
        return;
    }

    // 256x256 larger matrix
    let host_matrix = create_test_matrix(256, 256);
    let device_matrix = transport_matrix_to_device(Arc::new(host_matrix.clone()));

    verify_transpose(&host_matrix, &device_matrix);
}

#[test]
fn test_matrix_transpose_power_of_two() {
    if skip_if_no_hip() {
        return;
    }

    // 64x128 - common power-of-two dimensions
    let host_matrix = create_test_matrix(64, 128);
    let device_matrix = transport_matrix_to_device(Arc::new(host_matrix.clone()));

    verify_transpose(&host_matrix, &device_matrix);
}

#[test]
fn test_matrix_transpose_non_power_of_two() {
    if skip_if_no_hip() {
        return;
    }

    // 17x23 - non-power-of-two dimensions to test edge cases
    let host_matrix = create_test_matrix(17, 23);
    let device_matrix = transport_matrix_to_device(Arc::new(host_matrix.clone()));

    verify_transpose(&host_matrix, &device_matrix);
}
