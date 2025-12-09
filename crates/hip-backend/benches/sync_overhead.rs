//! Benchmark: Stream Synchronization Overhead
//!
//! This benchmark measures the overhead of various HIP synchronization primitives
//! to understand how much time is spent waiting for GPU operations to complete.
//!
//! Key findings expected:
//! - Base latency of hipDeviceSynchronize()
//! - Base latency of hipStreamSynchronize()
//! - Overhead of record_and_wait() (used in every D2H copy)
//! - Accumulated cost of per-operation syncs vs batch sync

mod common;

use std::time::{Duration, Instant};

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use openvm_hip_common::{
    copy::MemCopyH2D,
    d_buffer::DeviceBuffer,
    stream::{device_synchronize, HipEvent, HipStream},
};
use p3_baby_bear::BabyBear;

use common::{random_field_elements, sync_gpu};

/// Measure raw hipDeviceSynchronize() latency with no pending work.
fn bench_device_sync_idle(c: &mut Criterion) {
    let mut group = c.benchmark_group("sync/device_sync_idle");

    // Warm up GPU
    sync_gpu();

    group.bench_function("hipDeviceSynchronize", |b| {
        b.iter(|| {
            device_synchronize().unwrap();
        });
    });

    group.finish();
}

/// Measure HipEvent record_and_wait() latency (the bottleneck in D2H copies).
fn bench_event_record_wait(c: &mut Criterion) {
    let mut group = c.benchmark_group("sync/event_record_wait");

    // Warm up GPU
    sync_gpu();

    let event = HipEvent::new().unwrap();

    group.bench_function("record_and_wait", |b| {
        b.iter(|| unsafe {
            event
                .record_and_wait(openvm_hip_common::stream::hipStreamPerThread)
                .unwrap();
        });
    });

    group.finish();
}

/// Measure HipStream synchronize() latency.
fn bench_stream_sync(c: &mut Criterion) {
    let mut group = c.benchmark_group("sync/stream_sync");

    // Warm up GPU
    sync_gpu();

    let stream = HipStream::new().unwrap();

    group.bench_function("hipStreamSynchronize", |b| {
        b.iter(|| {
            stream.synchronize().unwrap();
        });
    });

    group.finish();
}

/// Compare accumulated syncs vs single final sync.
///
/// This simulates the pattern in LDE where we might sync after each kernel
/// vs syncing only at the end.
fn bench_accumulated_vs_batch_sync(c: &mut Criterion) {
    let mut group = c.benchmark_group("sync/accumulated_vs_batch");

    // Small data to ensure minimal actual work
    let data = random_field_elements(1024);

    for n_ops in [1, 5, 10, 20, 50] {
        // Pattern 1: Sync after each small H2D copy
        group.bench_with_input(
            BenchmarkId::new("sync_each", n_ops),
            &n_ops,
            |b, &n_ops| {
                b.iter(|| {
                    for _ in 0..n_ops {
                        let _ = black_box(data.to_device().unwrap());
                        sync_gpu();
                    }
                });
            },
        );

        // Pattern 2: Batch all copies, sync once at end
        group.bench_with_input(
            BenchmarkId::new("sync_batch", n_ops),
            &n_ops,
            |b, &n_ops| {
                b.iter(|| {
                    let buffers: Vec<_> = (0..n_ops)
                        .map(|_| black_box(data.to_device().unwrap()))
                        .collect();
                    sync_gpu();
                    black_box(buffers);
                });
            },
        );
    }

    group.finish();
}

/// Measure sync overhead as percentage of a real operation.
///
/// This helps understand how significant sync overhead is relative to
/// actual GPU work.
fn bench_sync_vs_work_ratio(c: &mut Criterion) {
    let mut group = c.benchmark_group("sync/work_ratio");

    for log_size in [12, 14, 16, 18] {
        let size = 1 << log_size;
        let data = random_field_elements(size);

        // Just the H2D copy (no explicit sync beyond what's in the API)
        group.bench_with_input(
            BenchmarkId::new("h2d_copy_only", log_size),
            &data,
            |b, data| {
                b.iter(|| {
                    black_box(data.to_device().unwrap());
                });
            },
        );

        // H2D copy + explicit sync
        group.bench_with_input(
            BenchmarkId::new("h2d_copy_with_sync", log_size),
            &data,
            |b, data| {
                b.iter(|| {
                    let buf = black_box(data.to_device().unwrap());
                    sync_gpu();
                    black_box(buf);
                });
            },
        );
    }

    group.finish();
}

/// Measure event creation overhead.
fn bench_event_creation(c: &mut Criterion) {
    let mut group = c.benchmark_group("sync/event_creation");

    group.bench_function("create_event", |b| {
        b.iter(|| {
            let event = HipEvent::new().unwrap();
            black_box(event);
        });
    });

    group.bench_function("create_stream", |b| {
        b.iter(|| {
            let stream = HipStream::new().unwrap();
            black_box(stream);
        });
    });

    group.finish();
}

/// Manual timing to get raw numbers without criterion overhead.
fn bench_raw_sync_timing(c: &mut Criterion) {
    let mut group = c.benchmark_group("sync/raw_timing");

    // Pre-warm
    sync_gpu();

    group.bench_function("device_sync_1000x", |b| {
        b.iter(|| {
            let start = Instant::now();
            for _ in 0..1000 {
                device_synchronize().unwrap();
            }
            black_box(start.elapsed())
        });
    });

    let event = HipEvent::new().unwrap();
    group.bench_function("event_record_wait_1000x", |b| {
        b.iter(|| {
            let start = Instant::now();
            for _ in 0..1000 {
                unsafe {
                    event
                        .record_and_wait(openvm_hip_common::stream::hipStreamPerThread)
                        .unwrap();
                }
            }
            black_box(start.elapsed())
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_device_sync_idle,
    bench_event_record_wait,
    bench_stream_sync,
    bench_accumulated_vs_batch_sync,
    bench_sync_vs_work_ratio,
    bench_event_creation,
    bench_raw_sync_timing,
);

criterion_main!(benches);
