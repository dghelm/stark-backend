//! Data transport tests - H2D and D2H roundtrip verification.
//!
//! These tests verify that data can be transported to the GPU and back
//! without corruption, testing the `transport_matrix_to_device` and
//! `transport_device_matrix_to_host` functions.

use std::sync::Arc;

use openvm_hip_backend::data_transporter::{
    assert_eq_host_and_device_matrix, transport_device_matrix_to_host, transport_matrix_to_device,
};
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

/// Create a test matrix with deterministic values.
fn create_test_matrix(height: usize, width: usize) -> RowMajorMatrix<BabyBear> {
    let values: Vec<BabyBear> = (0..height * width)
        .map(|i| BabyBear::from_canonical_u32(i as u32))
        .collect();
    RowMajorMatrix::new(values, width)
}

/// Create a matrix with specific BabyBear field elements to test field operations.
fn create_field_test_matrix(height: usize, width: usize) -> RowMajorMatrix<BabyBear> {
    let values: Vec<BabyBear> = (0..height * width)
        .map(|i| {
            // Use various field values including edge cases
            match i % 7 {
                0 => BabyBear::ZERO,
                1 => BabyBear::ONE,
                2 => BabyBear::NEG_ONE,
                3 => BabyBear::TWO,
                4 => BabyBear::from_canonical_u32(0x78000000), // Near field modulus
                5 => BabyBear::from_canonical_u32(1234567),
                _ => BabyBear::from_canonical_u32((i * 7919) as u32 % 0x78000001),
            }
        })
        .collect();
    RowMajorMatrix::new(values, width)
}

#[test]
fn test_roundtrip_small_matrix() {
    if skip_if_no_hip() {
        return;
    }

    // 8x4 small matrix
    let host_matrix = create_test_matrix(8, 4);
    let host_arc = Arc::new(host_matrix.clone());

    // Transport to device
    let device_matrix = transport_matrix_to_device(host_arc.clone());

    // Transport back to host
    let result = transport_device_matrix_to_host(&device_matrix);

    // Verify dimensions
    assert_eq!(result.height(), host_matrix.height(), "Height mismatch");
    assert_eq!(result.width(), host_matrix.width(), "Width mismatch");

    // Verify all values match
    for row in 0..host_matrix.height() {
        for col in 0..host_matrix.width() {
            assert_eq!(
                result.get(row, col),
                host_matrix.get(row, col),
                "Mismatch at row={}, col={}",
                row,
                col
            );
        }
    }
}

#[test]
fn test_roundtrip_large_matrix() {
    if skip_if_no_hip() {
        return;
    }

    // 1024x256 large matrix
    let host_matrix = create_test_matrix(1024, 256);
    let host_arc = Arc::new(host_matrix.clone());

    let device_matrix = transport_matrix_to_device(host_arc.clone());
    let result = transport_device_matrix_to_host(&device_matrix);

    assert_eq!(result.height(), host_matrix.height());
    assert_eq!(result.width(), host_matrix.width());

    // Spot-check some values to avoid O(n^2) comparison for large matrices
    let check_points = [
        (0, 0),
        (0, 255),
        (1023, 0),
        (1023, 255),
        (512, 128),
        (100, 50),
    ];

    for (row, col) in check_points {
        assert_eq!(
            result.get(row, col),
            host_matrix.get(row, col),
            "Mismatch at row={}, col={}",
            row,
            col
        );
    }
}

#[test]
fn test_roundtrip_preserves_field_elements() {
    if skip_if_no_hip() {
        return;
    }

    // Test with specific field element values including edge cases
    let host_matrix = create_field_test_matrix(32, 16);
    let host_arc = Arc::new(host_matrix.clone());

    let device_matrix = transport_matrix_to_device(host_arc.clone());
    let result = transport_device_matrix_to_host(&device_matrix);

    // Verify all values including special field elements
    for row in 0..host_matrix.height() {
        for col in 0..host_matrix.width() {
            let expected = host_matrix.get(row, col);
            let actual = result.get(row, col);
            assert_eq!(
                actual, expected,
                "Field element mismatch at row={}, col={}: expected {:?}, got {:?}",
                row, col, expected, actual
            );
        }
    }
}

#[test]
fn test_roundtrip_power_of_two_dimensions() {
    if skip_if_no_hip() {
        return;
    }

    // 128x64 - common FFT-friendly dimensions
    let host_matrix = create_test_matrix(128, 64);
    let host_arc = Arc::new(host_matrix.clone());

    let device_matrix = transport_matrix_to_device(host_arc.clone());
    let result = transport_device_matrix_to_host(&device_matrix);

    assert_eq!(result.height(), 128);
    assert_eq!(result.width(), 64);

    // Full verification for medium-sized matrix
    for row in 0..host_matrix.height() {
        for col in 0..host_matrix.width() {
            assert_eq!(
                result.get(row, col),
                host_matrix.get(row, col),
                "Mismatch at row={}, col={}",
                row,
                col
            );
        }
    }
}

#[test]
fn test_roundtrip_non_power_of_two() {
    if skip_if_no_hip() {
        return;
    }

    // 37x19 - non-power-of-two to catch edge cases in kernels
    let host_matrix = create_test_matrix(37, 19);
    let host_arc = Arc::new(host_matrix.clone());

    let device_matrix = transport_matrix_to_device(host_arc.clone());
    let result = transport_device_matrix_to_host(&device_matrix);

    for row in 0..host_matrix.height() {
        for col in 0..host_matrix.width() {
            assert_eq!(
                result.get(row, col),
                host_matrix.get(row, col),
                "Mismatch at row={}, col={}",
                row,
                col
            );
        }
    }
}

#[test]
fn test_assert_eq_host_and_device_matrix() {
    if skip_if_no_hip() {
        return;
    }

    // Test the helper function that compares host and device matrices directly
    let host_matrix = create_test_matrix(16, 8);
    let host_arc = Arc::new(host_matrix);

    let device_matrix = transport_matrix_to_device(host_arc.clone());

    // This should not panic if the matrices match
    assert_eq_host_and_device_matrix(host_arc, &device_matrix);
}

#[test]
fn test_roundtrip_single_row() {
    if skip_if_no_hip() {
        return;
    }

    // 1x64 - single row matrix (degenerate case)
    let host_matrix = create_test_matrix(1, 64);
    let host_arc = Arc::new(host_matrix.clone());

    let device_matrix = transport_matrix_to_device(host_arc.clone());
    let result = transport_device_matrix_to_host(&device_matrix);

    assert_eq!(result.height(), 1);
    assert_eq!(result.width(), 64);

    for col in 0..64 {
        assert_eq!(result.get(0, col), host_matrix.get(0, col));
    }
}

#[test]
fn test_roundtrip_single_column() {
    if skip_if_no_hip() {
        return;
    }

    // 64x1 - single column matrix (degenerate case)
    let host_matrix = create_test_matrix(64, 1);
    let host_arc = Arc::new(host_matrix.clone());

    let device_matrix = transport_matrix_to_device(host_arc.clone());
    let result = transport_device_matrix_to_host(&device_matrix);

    assert_eq!(result.height(), 64);
    assert_eq!(result.width(), 1);

    for row in 0..64 {
        assert_eq!(result.get(row, 0), host_matrix.get(row, 0));
    }
}
