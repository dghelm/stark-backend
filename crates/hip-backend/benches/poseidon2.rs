//! Benchmark: Poseidon2 Hashing Performance
//!
//! This benchmark measures Poseidon2 hash throughput, which is critical for
//! Merkle tree construction during trace commitment.
//!
//! Key findings expected:
//! - Hashes per second at various batch sizes
//! - CPU vs GPU crossover point
//! - Overhead of H2D metadata copies (the 3 separate copies in hash_matrices)
//! - Impact of matrix width on throughput

mod common;

use std::sync::Arc;

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use openvm_hip_backend::{
    base::DeviceMatrix,
    data_transporter::transport_matrix_to_device,
    hip::kernels::poseidon2::*,
};
use openvm_hip_common::{
    copy::{MemCopyD2H, MemCopyH2D},
    d_buffer::DeviceBuffer,
};
use p3_baby_bear::BabyBear;
use p3_matrix::dense::RowMajorMatrix;

use common::{random_field_elements, size_string, sync_gpu};

const DIGEST_WIDTH: usize = 8;
type H = [BabyBear; DIGEST_WIDTH];

/// Benchmark poseidon2_rows_p3_multi (row hashing for leaves).
fn bench_poseidon2_rows(c: &mut Criterion) {
    let mut group = c.benchmark_group("poseidon2/rows");

    // Test different matrix heights
    for log_height in [10, 12, 14, 16, 18] {
        let height = 1 << log_height;
        let width = 64; // Typical trace width

        let data = random_field_elements(height * width);
        let d_matrix = transport_matrix_to_device(Arc::new(RowMajorMatrix::new(data, width)));
        sync_gpu();

        let digests = DeviceBuffer::<H>::with_capacity(height);

        // Prepare metadata arrays
        let matrices_ptr = vec![d_matrix.buffer().as_ptr() as u64];
        let matrices_col = vec![width as u64];
        let matrices_row = vec![height as u64];
        let d_matrices_ptr = matrices_ptr.to_device().unwrap();
        let d_matrices_col = matrices_col.to_device().unwrap();
        let d_matrices_row = matrices_row.to_device().unwrap();
        sync_gpu();

        // Hashes per iteration
        group.throughput(Throughput::Elements(height as u64));

        group.bench_with_input(
            BenchmarkId::new("height", size_string(log_height)),
            &height,
            |b, _| {
                b.iter(|| unsafe {
                    poseidon2_rows_p3_multi(
                        &digests,
                        &d_matrices_ptr,
                        &d_matrices_col,
                        &d_matrices_row,
                        height as u64,
                        1,
                    )
                    .unwrap();
                    sync_gpu();
                });
            },
        );
    }

    group.finish();
}

/// Benchmark poseidon2_compress (internal Merkle tree nodes).
fn bench_poseidon2_compress(c: &mut Criterion) {
    let mut group = c.benchmark_group("poseidon2/compress");

    // Test different layer sizes
    for log_size in [10, 12, 14, 16, 18] {
        let size = 1 << log_size;

        // Create input layer (digests)
        let input_data = random_field_elements(size * 2 * DIGEST_WIDTH);
        let d_input: DeviceBuffer<H> = input_data
            .chunks(DIGEST_WIDTH)
            .map(|chunk| {
                let arr: [BabyBear; DIGEST_WIDTH] = chunk.try_into().unwrap();
                arr
            })
            .collect::<Vec<_>>()
            .to_device()
            .unwrap();
        sync_gpu();

        let d_output = DeviceBuffer::<H>::with_capacity(size);

        // Compressions per iteration
        group.throughput(Throughput::Elements(size as u64));

        group.bench_with_input(
            BenchmarkId::new("layer_size", size_string(log_size)),
            &size,
            |b, &size| {
                b.iter(|| unsafe {
                    poseidon2_compress(&d_output, &d_input, size as u32, false).unwrap();
                    sync_gpu();
                });
            },
        );
    }

    group.finish();
}

/// Benchmark the metadata H2D overhead separately.
///
/// This isolates the cost of the 3 separate H2D copies in hash_matrices.
fn bench_metadata_h2d_overhead(c: &mut Criterion) {
    let mut group = c.benchmark_group("poseidon2/metadata_h2d");

    for num_matrices in [1, 4, 16, 64] {
        // Simulate metadata for multiple matrices
        let matrices_ptr: Vec<u64> = (0..num_matrices).map(|i| i as u64 * 0x1000).collect();
        let matrices_col: Vec<u64> = (0..num_matrices).map(|_| 64).collect();
        let matrices_row: Vec<u64> = (0..num_matrices).map(|_| 1024).collect();

        group.bench_with_input(
            BenchmarkId::new("num_matrices", num_matrices),
            &num_matrices,
            |b, _| {
                b.iter(|| {
                    // Current approach: 3 separate H2D copies
                    let d_ptr = matrices_ptr.to_device().unwrap();
                    let d_col = matrices_col.to_device().unwrap();
                    let d_row = matrices_row.to_device().unwrap();
                    sync_gpu();
                    black_box((d_ptr, d_col, d_row))
                });
            },
        );
    }

    group.finish();
}

/// Benchmark varying matrix widths (affects row hashing).
fn bench_poseidon2_varying_width(c: &mut Criterion) {
    let mut group = c.benchmark_group("poseidon2/varying_width");

    let height = 1 << 14; // Fixed height

    for width in [16, 32, 64, 128, 256] {
        let data = random_field_elements(height * width);
        let d_matrix = transport_matrix_to_device(Arc::new(RowMajorMatrix::new(data, width)));
        sync_gpu();

        let digests = DeviceBuffer::<H>::with_capacity(height);

        let matrices_ptr = vec![d_matrix.buffer().as_ptr() as u64];
        let matrices_col = vec![width as u64];
        let matrices_row = vec![height as u64];
        let d_matrices_ptr = matrices_ptr.to_device().unwrap();
        let d_matrices_col = matrices_col.to_device().unwrap();
        let d_matrices_row = matrices_row.to_device().unwrap();
        sync_gpu();

        // Total elements processed
        group.throughput(Throughput::Elements((height * width) as u64));

        group.bench_with_input(BenchmarkId::new("width", width), &width, |b, _| {
            b.iter(|| unsafe {
                poseidon2_rows_p3_multi(
                    &digests,
                    &d_matrices_ptr,
                    &d_matrices_col,
                    &d_matrices_row,
                    height as u64,
                    1,
                )
                .unwrap();
                sync_gpu();
            });
        });
    }

    group.finish();
}

/// Benchmark full Merkle layer construction (rows + compress).
fn bench_merkle_layer(c: &mut Criterion) {
    let mut group = c.benchmark_group("poseidon2/merkle_layer");

    for log_height in [12, 14, 16] {
        let height = 1 << log_height;
        let width = 64;

        let data = random_field_elements(height * width);
        let d_matrix = transport_matrix_to_device(Arc::new(RowMajorMatrix::new(data, width)));
        sync_gpu();

        let d_digests = DeviceBuffer::<H>::with_capacity(height);
        let d_compressed = DeviceBuffer::<H>::with_capacity(height / 2);

        let matrices_ptr = vec![d_matrix.buffer().as_ptr() as u64];
        let matrices_col = vec![width as u64];
        let matrices_row = vec![height as u64];
        let d_matrices_ptr = matrices_ptr.to_device().unwrap();
        let d_matrices_col = matrices_col.to_device().unwrap();
        let d_matrices_row = matrices_row.to_device().unwrap();
        sync_gpu();

        group.bench_with_input(
            BenchmarkId::new("full_layer", size_string(log_height)),
            &height,
            |b, &height| {
                b.iter(|| unsafe {
                    // Hash rows
                    poseidon2_rows_p3_multi(
                        &d_digests,
                        &d_matrices_ptr,
                        &d_matrices_col,
                        &d_matrices_row,
                        height as u64,
                        1,
                    )
                    .unwrap();

                    // Compress
                    poseidon2_compress(&d_compressed, &d_digests, (height / 2) as u32, false)
                        .unwrap();

                    sync_gpu();
                });
            },
        );
    }

    group.finish();
}

/// Benchmark multiple matrices (as happens with multiple AIR traces).
fn bench_poseidon2_multi_matrix(c: &mut Criterion) {
    let mut group = c.benchmark_group("poseidon2/multi_matrix");

    let height = 1 << 14;
    let width = 64;

    for num_matrices in [1, 2, 4, 8] {
        // Create multiple matrices
        let matrices: Vec<DeviceMatrix<BabyBear>> = (0..num_matrices)
            .map(|_| {
                let data = random_field_elements(height * width);
                transport_matrix_to_device(Arc::new(RowMajorMatrix::new(data, width)))
            })
            .collect();
        sync_gpu();

        let d_digests = DeviceBuffer::<H>::with_capacity(height);

        let matrices_ptr: Vec<u64> = matrices
            .iter()
            .map(|m| m.buffer().as_ptr() as u64)
            .collect();
        let matrices_col: Vec<u64> = matrices.iter().map(|_| width as u64).collect();
        let matrices_row: Vec<u64> = matrices.iter().map(|_| height as u64).collect();
        let d_matrices_ptr = matrices_ptr.to_device().unwrap();
        let d_matrices_col = matrices_col.to_device().unwrap();
        let d_matrices_row = matrices_row.to_device().unwrap();
        sync_gpu();

        // Total hashes
        group.throughput(Throughput::Elements(height as u64));

        group.bench_with_input(
            BenchmarkId::new("num_matrices", num_matrices),
            &num_matrices,
            |b, &num_matrices| {
                b.iter(|| unsafe {
                    poseidon2_rows_p3_multi(
                        &d_digests,
                        &d_matrices_ptr,
                        &d_matrices_col,
                        &d_matrices_row,
                        height as u64,
                        num_matrices as u64,
                    )
                    .unwrap();
                    sync_gpu();
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_poseidon2_rows,
    bench_poseidon2_compress,
    bench_metadata_h2d_overhead,
    bench_poseidon2_varying_width,
    bench_merkle_layer,
    bench_poseidon2_multi_matrix,
);

criterion_main!(benches);
