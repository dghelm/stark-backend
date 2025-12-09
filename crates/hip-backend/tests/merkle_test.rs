//! Merkle tree tests for HIP backend.
//!
//! Tests for GPU-accelerated Merkle tree construction using Poseidon2.

use std::sync::Arc;

use openvm_hip_backend::{
    data_transporter::transport_matrix_to_device,
    hip_device::{HipConfig, HipDevice},
    lde::{GpuLde, GpuLdeImpl},
    merkle_tree::GpuMerkleTree,
};
use openvm_hip_common::copy::MemCopyD2H;
use openvm_stark_backend::prover::hal::MatrixDimensions;
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
fn test_merkle_tree_single_matrix() {
    if skip_if_no_hip() {
        return;
    }

    let height = 16; // Power of 2
    let width = 4;
    let added_bits = 1; // 2x blowup for LDE

    let host_matrix = create_test_matrix(height, width);
    let device_matrix = transport_matrix_to_device(host_matrix);

    let shift = BabyBear::ONE;
    let lde = GpuLdeImpl::new(device_matrix, added_bits, shift);

    let hip_device = create_hip_device();
    let merkle_tree = GpuMerkleTree::new(vec![lde], &hip_device).unwrap();

    // Verify the root can be computed
    let _root = merkle_tree.root();
    // Root is [F; 8] which is automatically verified by the type system

    // Verify the tree has the expected number of layers
    let max_height = merkle_tree.get_max_height();
    assert_eq!(max_height, height << added_bits);
}

#[test]
fn test_merkle_tree_multiple_matrices() {
    if skip_if_no_hip() {
        return;
    }

    let hip_device = create_hip_device();

    // Create matrices with different heights
    let matrix1 = create_test_matrix(16, 4);
    let matrix2 = create_test_matrix(16, 8);

    let device_matrix1 = transport_matrix_to_device(matrix1);
    let device_matrix2 = transport_matrix_to_device(matrix2);

    let shift = BabyBear::ONE;
    let lde1 = GpuLdeImpl::new(device_matrix1, 1, shift); // 32 height
    let lde2 = GpuLdeImpl::new(device_matrix2, 1, shift); // 32 height

    let merkle_tree = GpuMerkleTree::new(vec![lde1, lde2], &hip_device).unwrap();

    let _root = merkle_tree.root();
    // Root is [F; 8] which is automatically verified by the type system
}

#[test]
fn test_merkle_tree_open_batch() {
    if skip_if_no_hip() {
        return;
    }

    let hip_device = create_hip_device();

    let height = 16;
    let width = 4;
    let added_bits = 1;

    let host_matrix = create_test_matrix(height, width);
    let device_matrix = transport_matrix_to_device(host_matrix);

    let shift = BabyBear::ONE;
    let lde = GpuLdeImpl::new(device_matrix, added_bits, shift);

    let merkle_tree = GpuMerkleTree::new(vec![lde], &hip_device).unwrap();

    // Query at a few indices
    let indices = vec![0, 5, 10, 15];
    let openings = merkle_tree
        .open_batch_at_multiple_indices(&indices)
        .unwrap();

    assert_eq!(
        openings.len(),
        indices.len(),
        "Should have one opening per index"
    );

    for (opening, proof) in &openings {
        assert_eq!(opening.len(), 1, "Should have one leaf per opening");
        assert_eq!(
            opening[0].len(),
            width,
            "Opening width should match matrix width"
        );

        // Proof should have log2(height << added_bits) layers
        let expected_proof_len = (height << added_bits).trailing_zeros() as usize;
        assert_eq!(
            proof.len(),
            expected_proof_len,
            "Proof should have correct number of layers"
        );
    }
}

#[test]
fn test_merkle_tree_deterministic_root() {
    if skip_if_no_hip() {
        return;
    }

    let hip_device = create_hip_device();

    let height = 8;
    let width = 4;
    let added_bits = 1;

    // Create the same matrix twice
    let host_matrix1 = create_test_matrix(height, width);
    let host_matrix2 = create_test_matrix(height, width);

    let device_matrix1 = transport_matrix_to_device(host_matrix1);
    let device_matrix2 = transport_matrix_to_device(host_matrix2);

    let shift = BabyBear::ONE;
    let lde1 = GpuLdeImpl::new(device_matrix1, added_bits, shift);
    let lde2 = GpuLdeImpl::new(device_matrix2, added_bits, shift);

    let tree1 = GpuMerkleTree::new(vec![lde1], &hip_device).unwrap();
    let tree2 = GpuMerkleTree::new(vec![lde2], &hip_device).unwrap();

    let root1 = tree1.root();
    let root2 = tree2.root();

    assert_eq!(root1, root2, "Same input should produce same root");
}

#[test]
fn test_merkle_tree_different_inputs_different_roots() {
    if skip_if_no_hip() {
        return;
    }

    let hip_device = create_hip_device();

    let height = 8;
    let width = 4;
    let added_bits = 1;

    // Create two different matrices
    let values1: Vec<BabyBear> = (0..height * width)
        .map(|i| BabyBear::from_canonical_u32(i as u32))
        .collect();
    let values2: Vec<BabyBear> = (0..height * width)
        .map(|i| BabyBear::from_canonical_u32((i + 1000) as u32))
        .collect();

    let host_matrix1 = Arc::new(RowMajorMatrix::new(values1, width));
    let host_matrix2 = Arc::new(RowMajorMatrix::new(values2, width));

    let device_matrix1 = transport_matrix_to_device(host_matrix1);
    let device_matrix2 = transport_matrix_to_device(host_matrix2);

    let shift = BabyBear::ONE;
    let lde1 = GpuLdeImpl::new(device_matrix1, added_bits, shift);
    let lde2 = GpuLdeImpl::new(device_matrix2, added_bits, shift);

    let tree1 = GpuMerkleTree::new(vec![lde1], &hip_device).unwrap();
    let tree2 = GpuMerkleTree::new(vec![lde2], &hip_device).unwrap();

    let root1 = tree1.root();
    let root2 = tree2.root();

    assert_ne!(
        root1, root2,
        "Different inputs should produce different roots"
    );
}
