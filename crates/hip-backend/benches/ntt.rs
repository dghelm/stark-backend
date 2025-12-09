//! Benchmark: NTT (Number Theoretic Transform) Performance
//!
//! This benchmark measures GPU NTT performance and compares it to CPU NTT.
//! NTT is the core operation in the LDE pipeline and is critical for overall prover speed.
//!
//! Key findings expected:
//! - GPU vs CPU crossover point (where GPU becomes faster)
//! - NTT scaling behavior (should be O(n log n))
//! - Overhead of twiddle factor initialization
//! - Impact of batch size on throughput

mod common;

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use openvm_hip_backend::lde::ntt::batch_ntt;
use openvm_hip_common::{copy::MemCopyH2D, d_buffer::DeviceBuffer};
use p3_baby_bear::BabyBear;
use p3_dft::{Radix2Dit, TwoAdicSubgroupDft};
use p3_matrix::dense::RowMajorMatrix;

use common::{random_field_elements, size_string, sync_gpu};

/// Benchmark forward NTT on GPU at various sizes.
fn bench_ntt_forward_gpu(c: &mut Criterion) {
    let mut group = c.benchmark_group("ntt/forward_gpu");

    for log_n in [10, 12, 14, 16, 18, 20] {
        let n = 1 << log_n;
        let data = random_field_elements(n);
        let d_data = data.to_device().unwrap();
        sync_gpu();

        let bytes = n * std::mem::size_of::<BabyBear>();
        group.throughput(Throughput::Bytes(bytes as u64));

        group.bench_with_input(BenchmarkId::new("size", size_string(log_n)), &log_n, |b, _| {
            b.iter(|| {
                // Note: batch_ntt works on a buffer with width=1 (single polynomial)
                batch_ntt(&d_data, log_n as u32, 0, 1, false, false);
                sync_gpu();
            });
        });
    }

    group.finish();
}

/// Benchmark inverse NTT on GPU at various sizes.
fn bench_ntt_inverse_gpu(c: &mut Criterion) {
    let mut group = c.benchmark_group("ntt/inverse_gpu");

    for log_n in [10, 12, 14, 16, 18, 20] {
        let n = 1 << log_n;
        let data = random_field_elements(n);
        let d_data = data.to_device().unwrap();
        sync_gpu();

        let bytes = n * std::mem::size_of::<BabyBear>();
        group.throughput(Throughput::Bytes(bytes as u64));

        group.bench_with_input(BenchmarkId::new("size", size_string(log_n)), &log_n, |b, _| {
            b.iter(|| {
                batch_ntt(&d_data, log_n as u32, 0, 1, false, true);
                sync_gpu();
            });
        });
    }

    group.finish();
}

/// Benchmark forward NTT on CPU for comparison.
fn bench_ntt_forward_cpu(c: &mut Criterion) {
    let mut group = c.benchmark_group("ntt/forward_cpu");
    let dft = Radix2Dit::default();

    for log_n in [10, 12, 14, 16, 18] {
        // Skip 20 for CPU - too slow
        let n = 1 << log_n;
        let data = random_field_elements(n);
        let matrix = RowMajorMatrix::new(data.clone(), 1);

        let bytes = n * std::mem::size_of::<BabyBear>();
        group.throughput(Throughput::Bytes(bytes as u64));

        group.bench_with_input(
            BenchmarkId::new("size", size_string(log_n)),
            &matrix,
            |b, matrix| {
                b.iter(|| {
                    let result = dft.dft_batch(matrix.clone());
                    black_box(result)
                });
            },
        );
    }

    group.finish();
}

/// Benchmark batch NTT (multiple polynomials at once).
fn bench_ntt_batch_gpu(c: &mut Criterion) {
    let mut group = c.benchmark_group("ntt/batch_gpu");

    let log_n = 16; // Fixed polynomial size
    let n = 1 << log_n;

    for width in [1, 4, 16, 64, 256] {
        let total = n * width;
        let data = random_field_elements(total);
        let d_data = data.to_device().unwrap();
        sync_gpu();

        let bytes = total * std::mem::size_of::<BabyBear>();
        group.throughput(Throughput::Bytes(bytes as u64));

        group.bench_with_input(BenchmarkId::new("width", width), &width, |b, &width| {
            b.iter(|| {
                batch_ntt(&d_data, log_n as u32, 0, width as u32, false, false);
                sync_gpu();
            });
        });
    }

    group.finish();
}

/// Compare GPU vs CPU at the same sizes.
fn bench_ntt_gpu_vs_cpu(c: &mut Criterion) {
    let mut group = c.benchmark_group("ntt/gpu_vs_cpu");
    let dft = Radix2Dit::default();

    for log_n in [12, 14, 16] {
        let n = 1 << log_n;
        let data = random_field_elements(n);

        // GPU benchmark
        let d_data = data.to_device().unwrap();
        sync_gpu();

        group.bench_with_input(
            BenchmarkId::new("gpu", size_string(log_n)),
            &log_n,
            |b, _| {
                b.iter(|| {
                    batch_ntt(&d_data, log_n as u32, 0, 1, false, false);
                    sync_gpu();
                });
            },
        );

        // CPU benchmark
        let matrix = RowMajorMatrix::new(data.clone(), 1);
        group.bench_with_input(
            BenchmarkId::new("cpu", size_string(log_n)),
            &matrix,
            |b, matrix| {
                b.iter(|| {
                    let result = dft.dft_batch(matrix.clone());
                    black_box(result)
                });
            },
        );
    }

    group.finish();
}

/// Measure NTT with bit-reverse step (used in actual LDE).
fn bench_ntt_with_bitrev(c: &mut Criterion) {
    let mut group = c.benchmark_group("ntt/with_bitrev");

    for log_n in [12, 14, 16, 18] {
        let n = 1 << log_n;
        let data = random_field_elements(n);
        let d_data = data.to_device().unwrap();
        sync_gpu();

        let bytes = n * std::mem::size_of::<BabyBear>();
        group.throughput(Throughput::Bytes(bytes as u64));

        // Without bit-reverse
        group.bench_with_input(
            BenchmarkId::new("no_bitrev", size_string(log_n)),
            &log_n,
            |b, _| {
                b.iter(|| {
                    batch_ntt(&d_data, log_n as u32, 0, 1, false, false);
                    sync_gpu();
                });
            },
        );

        // With bit-reverse
        group.bench_with_input(
            BenchmarkId::new("with_bitrev", size_string(log_n)),
            &log_n,
            |b, _| {
                b.iter(|| {
                    batch_ntt(&d_data, log_n as u32, 0, 1, true, false);
                    sync_gpu();
                });
            },
        );
    }

    group.finish();
}

/// Measure H2D + NTT + D2H full pipeline latency.
fn bench_ntt_full_pipeline(c: &mut Criterion) {
    let mut group = c.benchmark_group("ntt/full_pipeline");

    for log_n in [12, 14, 16] {
        let n = 1 << log_n;
        let data = random_field_elements(n);

        group.bench_with_input(
            BenchmarkId::new("h2d_ntt_d2h", size_string(log_n)),
            &data,
            |b, data| {
                b.iter(|| {
                    // H2D
                    let d_data = data.to_device().unwrap();
                    sync_gpu();

                    // NTT
                    batch_ntt(&d_data, log_n as u32, 0, 1, false, false);
                    sync_gpu();

                    // D2H
                    use openvm_hip_common::copy::MemCopyD2H;
                    let result = d_data.to_host().unwrap();
                    black_box(result)
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_ntt_forward_gpu,
    bench_ntt_inverse_gpu,
    bench_ntt_forward_cpu,
    bench_ntt_batch_gpu,
    bench_ntt_gpu_vs_cpu,
    bench_ntt_with_bitrev,
    bench_ntt_full_pipeline,
);

criterion_main!(benches);
