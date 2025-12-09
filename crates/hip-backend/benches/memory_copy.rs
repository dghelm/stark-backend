//! Benchmark: Memory Copy Throughput
//!
//! This benchmark measures H2D and D2H transfer throughput to understand:
//! - Raw PCIe bandwidth achievable
//! - Overhead of record_and_wait() in D2H copies
//! - Impact of batching many small copies vs fewer large copies
//!
//! Key findings expected:
//! - Baseline throughput for large transfers
//! - Overhead per transfer (latency)
//! - Batching benefit for small transfers

mod common;

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use openvm_hip_common::{
    copy::{MemCopyD2D, MemCopyD2H, MemCopyH2D},
    d_buffer::DeviceBuffer,
};
use p3_baby_bear::BabyBear;

use common::{random_field_elements, size_string, sync_gpu, LOG_SIZES};

/// Benchmark H2D (Host-to-Device) throughput at various sizes.
fn bench_h2d_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory/h2d");

    for &log_size in LOG_SIZES {
        let size = 1 << log_size;
        let data = random_field_elements(size);
        let bytes = size * std::mem::size_of::<BabyBear>();

        group.throughput(Throughput::Bytes(bytes as u64));
        group.bench_with_input(
            BenchmarkId::new("throughput", size_string(log_size)),
            &data,
            |b, data| {
                b.iter(|| {
                    let buf = data.to_device().unwrap();
                    sync_gpu();
                    black_box(buf)
                });
            },
        );
    }

    group.finish();
}

/// Benchmark D2H (Device-to-Host) throughput at various sizes.
///
/// This is critical because every D2H uses record_and_wait() which adds sync overhead.
fn bench_d2h_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory/d2h");

    for &log_size in LOG_SIZES {
        let size = 1 << log_size;
        let data = random_field_elements(size);
        let bytes = size * std::mem::size_of::<BabyBear>();

        // Pre-upload data
        let d_buf = data.to_device().unwrap();
        sync_gpu();

        group.throughput(Throughput::Bytes(bytes as u64));
        group.bench_with_input(
            BenchmarkId::new("throughput", size_string(log_size)),
            &d_buf,
            |b, d_buf| {
                b.iter(|| {
                    let host = d_buf.to_host().unwrap();
                    black_box(host)
                });
            },
        );
    }

    group.finish();
}

/// Benchmark D2D (Device-to-Device) copy throughput.
fn bench_d2d_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory/d2d");

    for &log_size in LOG_SIZES {
        let size = 1 << log_size;
        let data = random_field_elements(size);
        let bytes = size * std::mem::size_of::<BabyBear>();

        // Pre-upload data
        let d_buf = data.to_device().unwrap();
        sync_gpu();

        group.throughput(Throughput::Bytes(bytes as u64));
        group.bench_with_input(
            BenchmarkId::new("throughput", size_string(log_size)),
            &d_buf,
            |b, d_buf| {
                b.iter(|| {
                    let copy = d_buf.device_copy().unwrap();
                    sync_gpu();
                    black_box(copy)
                });
            },
        );
    }

    group.finish();
}

/// Compare many small H2D copies vs one large batched copy.
///
/// This demonstrates the benefit of batching transfers.
fn bench_batched_vs_individual_h2d(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory/batch_h2d");

    // Total data size: 1MB (fixed)
    let total_elements = 256 * 1024; // 1MB of BabyBear
    let total_bytes = total_elements * std::mem::size_of::<BabyBear>();

    for num_copies in [1, 4, 16, 64, 256] {
        let elements_per_copy = total_elements / num_copies;
        let chunks: Vec<Vec<BabyBear>> = (0..num_copies)
            .map(|_| random_field_elements(elements_per_copy))
            .collect();

        group.throughput(Throughput::Bytes(total_bytes as u64));
        group.bench_with_input(
            BenchmarkId::new("copies", num_copies),
            &chunks,
            |b, chunks| {
                b.iter(|| {
                    let buffers: Vec<_> = chunks.iter().map(|c| c.to_device().unwrap()).collect();
                    sync_gpu();
                    black_box(buffers)
                });
            },
        );
    }

    group.finish();
}

/// Compare many small D2H copies vs one large batched copy.
///
/// This is especially important because each D2H has record_and_wait() overhead.
fn bench_batched_vs_individual_d2h(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory/batch_d2h");

    // Total data size: 1MB (fixed)
    let total_elements = 256 * 1024; // 1MB of BabyBear
    let total_bytes = total_elements * std::mem::size_of::<BabyBear>();

    for num_copies in [1, 4, 16, 64, 256] {
        let elements_per_copy = total_elements / num_copies;

        // Pre-allocate device buffers
        let d_buffers: Vec<DeviceBuffer<BabyBear>> = (0..num_copies)
            .map(|_| {
                let data = random_field_elements(elements_per_copy);
                data.to_device().unwrap()
            })
            .collect();
        sync_gpu();

        group.throughput(Throughput::Bytes(total_bytes as u64));
        group.bench_with_input(
            BenchmarkId::new("copies", num_copies),
            &d_buffers,
            |b, d_buffers| {
                b.iter(|| {
                    let host_vecs: Vec<_> = d_buffers.iter().map(|d| d.to_host().unwrap()).collect();
                    black_box(host_vecs)
                });
            },
        );
    }

    group.finish();
}

/// Measure allocation overhead separately from copy.
fn bench_allocation_overhead(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory/allocation");

    for &log_size in LOG_SIZES {
        let size = 1 << log_size;

        group.bench_with_input(
            BenchmarkId::new("device_buffer", size_string(log_size)),
            &size,
            |b, &size| {
                b.iter(|| {
                    let buf = DeviceBuffer::<BabyBear>::with_capacity(size);
                    sync_gpu();
                    black_box(buf)
                });
            },
        );
    }

    group.finish();
}

/// Roundtrip benchmark: H2D + D2H.
fn bench_roundtrip(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory/roundtrip");

    for &log_size in &[12, 14, 16, 18] {
        let size = 1 << log_size;
        let data = random_field_elements(size);
        let bytes = size * std::mem::size_of::<BabyBear>();

        group.throughput(Throughput::Bytes((bytes * 2) as u64)); // H2D + D2H
        group.bench_with_input(
            BenchmarkId::new("h2d_d2h", size_string(log_size)),
            &data,
            |b, data| {
                b.iter(|| {
                    let d_buf = data.to_device().unwrap();
                    sync_gpu();
                    let result = d_buf.to_host().unwrap();
                    black_box(result)
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_h2d_throughput,
    bench_d2h_throughput,
    bench_d2d_throughput,
    bench_batched_vs_individual_h2d,
    bench_batched_vs_individual_d2h,
    bench_allocation_overhead,
    bench_roundtrip,
);

criterion_main!(benches);
