//! LDE module tests for HIP backend.
//!
//! Tests for Low Degree Extension (LDE) computation on GPU.

use std::sync::Arc;

use openvm_hip_backend::{
    base::DeviceMatrix,
    data_transporter::transport_matrix_to_device,
    lde::{GpuLde, GpuLdeImpl, LdeCommon},
};
use openvm_hip_common::copy::MemCopyD2H;
use openvm_stark_backend::prover::hal::MatrixDimensions;
use p3_baby_bear::BabyBear;
use p3_field::{FieldAlgebra, TwoAdicField};
use p3_matrix::{dense::RowMajorMatrix, Matrix};

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

#[test]
fn test_lde_basic_dimensions() {
    if skip_if_no_hip() {
        return;
    }

    let height = 8; // Power of 2
    let width = 4;
    let added_bits = 2; // 4x blowup

    let host_matrix = create_test_matrix(height, width);
    let device_matrix = transport_matrix_to_device(host_matrix.clone());

    let shift = BabyBear::ONE;
    let lde = GpuLdeImpl::new(device_matrix, added_bits, shift);

    assert_eq!(lde.width(), width, "Width should be preserved");
    assert_eq!(
        lde.height(),
        height << added_bits,
        "Height should be expanded by blowup factor"
    );
    assert_eq!(
        lde.trace_height(),
        height,
        "Trace height should be original height"
    );
    assert_eq!(lde.shift(), shift, "Shift should be preserved");
}

#[test]
fn test_lde_zero_blowup() {
    if skip_if_no_hip() {
        return;
    }

    let height = 16;
    let width = 8;
    let added_bits = 0; // No blowup

    let host_matrix = create_test_matrix(height, width);
    let device_matrix = transport_matrix_to_device(host_matrix.clone());

    let shift = BabyBear::ONE;
    let lde = GpuLdeImpl::new(device_matrix, added_bits, shift);

    assert_eq!(lde.width(), width);
    assert_eq!(lde.height(), height, "Height unchanged with zero blowup");
    assert_eq!(lde.trace_height(), height);
}

#[test]
fn test_lde_take_lde() {
    if skip_if_no_hip() {
        return;
    }

    let height = 8;
    let width = 4;
    let added_bits = 1; // 2x blowup

    let host_matrix = create_test_matrix(height, width);
    let device_matrix = transport_matrix_to_device(host_matrix.clone());

    let shift = BabyBear::ONE;
    let lde = GpuLdeImpl::new(device_matrix, added_bits, shift);

    let lde_height = height << added_bits;
    let taken_lde = lde.take_lde(lde_height);

    assert_eq!(taken_lde.width(), width);
    assert_eq!(taken_lde.height(), lde_height);
}

#[test]
fn test_lde_get_rows() {
    if skip_if_no_hip() {
        return;
    }

    let height = 16;
    let width = 4;
    let added_bits = 1; // 2x blowup

    let host_matrix = create_test_matrix(height, width);
    let device_matrix = transport_matrix_to_device(host_matrix.clone());

    let shift = BabyBear::ONE;
    let lde = GpuLdeImpl::new(device_matrix, added_bits, shift);

    // Get a few rows
    let row_indices = vec![0, 5, 10, 15];
    let rows = lde.get_lde_rows(&row_indices);

    assert_eq!(rows.height(), row_indices.len());
    assert_eq!(rows.width(), width);

    // Verify we can read the data back
    let host_rows = rows.to_host().unwrap();
    assert_eq!(host_rows.len(), row_indices.len() * width);
}

#[test]
fn test_lde_with_shift() {
    if skip_if_no_hip() {
        return;
    }

    let height = 8;
    let width = 4;
    let added_bits = 2;

    let host_matrix = create_test_matrix(height, width);
    let device_matrix = transport_matrix_to_device(host_matrix.clone());

    // Use a non-trivial shift (generator of the multiplicative group)
    let shift = BabyBear::two_adic_generator(3);
    let lde = GpuLdeImpl::new(device_matrix, added_bits, shift);

    assert_eq!(lde.shift(), shift);
    assert_eq!(lde.height(), height << added_bits);
    assert_eq!(lde.width(), width);

    // Verify we can read the LDE back
    let lde_matrix = lde.take_lde(lde.height());
    let host_lde = lde_matrix.to_host().unwrap();
    assert_eq!(host_lde.len(), lde.height() * lde.width());
}

#[test]
fn test_lde_larger_matrix() {
    if skip_if_no_hip() {
        return;
    }

    let height = 256;
    let width = 16;
    let added_bits = 2;

    let host_matrix = create_test_matrix(height, width);
    let device_matrix = transport_matrix_to_device(host_matrix.clone());

    let shift = BabyBear::ONE;
    let lde = GpuLdeImpl::new(device_matrix, added_bits, shift);

    assert_eq!(lde.height(), height << added_bits);
    assert_eq!(lde.width(), width);
    assert_eq!(lde.trace_height(), height);

    // Get some rows and verify dimensions
    let row_indices: Vec<usize> = (0..32).collect();
    let rows = lde.get_lde_rows(&row_indices);
    assert_eq!(rows.height(), 32);
    assert_eq!(rows.width(), width);
}

#[test]
fn test_lde_clone() {
    if skip_if_no_hip() {
        return;
    }

    let height = 16;
    let width = 4;
    let added_bits = 1;

    let host_matrix = create_test_matrix(height, width);
    let device_matrix = transport_matrix_to_device(host_matrix.clone());

    let shift = BabyBear::ONE;
    let lde = GpuLdeImpl::new(device_matrix, added_bits, shift);
    let lde_clone = lde.clone();

    // Both should have the same dimensions
    assert_eq!(lde.height(), lde_clone.height());
    assert_eq!(lde.width(), lde_clone.width());
    assert_eq!(lde.trace_height(), lde_clone.trace_height());
    assert_eq!(lde.shift(), lde_clone.shift());
}
