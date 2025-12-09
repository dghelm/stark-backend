//! Benchmark: LDE (Low-Degree Extension) Pipeline
//!
//! This benchmark measures the full LDE pipeline and individual steps to identify
//! which operations dominate and where optimizations should focus.
//!
//! The LDE pipeline consists of:
//! 1. batch_expand_pad - Expand trace to larger domain with zero padding
//! 2. batch_ntt (forward) - Forward NTT
//! 3. zk_shift - Apply coset shift for zero-knowledge
//! 4. batch_bit_reverse - Reorder for final NTT
//! 5. batch_ntt (inverse) - Inverse NTT for evaluation form
//!
//! Key findings expected:
//! - Which step dominates total LDE time
//! - Sync overhead between steps
//! - GPU vs CPU LDE comparison

mod common;

use std::sync::Arc;

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use openvm_hip_backend::{
    base::DeviceMatrix,
    data_transporter::transport_matrix_to_device,
    hip::kernels::lde::{batch_bit_reverse, batch_expand_pad, zk_shift},
    lde::ntt::batch_ntt,
};
use openvm_hip_common::{
    copy::{MemCopyD2H, MemCopyH2D},
    d_buffer::DeviceBuffer,
};
use p3_baby_bear::BabyBear;
use p3_dft::{Radix2Dit, TwoAdicSubgroupDft};
use p3_field::{Field, FieldAlgebra, PrimeField32};
use p3_matrix::dense::RowMajorMatrix;

use common::{random_field_elements, size_string, sync_gpu};

const LOG_BLOWUP: u32 = 2; // 4x blowup is typical

/// Benchmark the full LDE pipeline (all 5 steps).
fn bench_lde_full(c: &mut Criterion) {
    let mut group = c.benchmark_group("lde/full");

    for log_height in [10, 12, 14, 16, 18] {
        let height = 1 << log_height;
        let width = 64; // Typical trace width
        let lde_height = height << LOG_BLOWUP;
        let shift = BabyBear::GENERATOR;

        let data = random_field_elements(height * width);
        let d_trace = transport_matrix_to_device(Arc::new(RowMajorMatrix::new(data, width)));
        sync_gpu();

        // Output buffer
        let d_lde = DeviceMatrix::<BabyBear>::with_capacity(lde_height, width);

        let input_bytes = height * width * std::mem::size_of::<BabyBear>();
        group.throughput(Throughput::Bytes(input_bytes as u64));

        group.bench_with_input(
            BenchmarkId::new("pipeline", size_string(log_height)),
            &log_height,
            |b, _| {
                b.iter(|| {
                    // Step 1: Expand and pad
                    unsafe {
                        batch_expand_pad(
                            d_lde.buffer(),
                            d_trace.buffer(),
                            width as u32,
                            lde_height as u32,
                            height as u32,
                        )
                        .unwrap();
                    }

                    // Step 2: Forward NTT
                    batch_ntt(
                        d_lde.buffer(),
                        log_height as u32,
                        LOG_BLOWUP,
                        width as u32,
                        true,
                        true,
                    );

                    // Step 3: ZK shift
                    let lde_size = (lde_height * width) as u32;
                    let log_lde_height = (log_height as u32) + LOG_BLOWUP;
                    unsafe {
                        zk_shift(
                            d_lde.buffer(),
                            lde_size,
                            log_lde_height,
                            shift.as_canonical_u32(),
                        )
                        .unwrap();
                    }

                    // Step 4: Bit reverse
                    unsafe {
                        batch_bit_reverse(d_lde.buffer(), log_lde_height, lde_size).unwrap();
                    }

                    // Step 5: Inverse NTT
                    batch_ntt(d_lde.buffer(), log_lde_height, 0, width as u32, false, false);

                    sync_gpu();
                });
            },
        );
    }

    group.finish();
}

/// Benchmark individual LDE steps to identify bottlenecks.
fn bench_lde_steps(c: &mut Criterion) {
    let mut group = c.benchmark_group("lde/steps");

    let log_height = 16; // Fixed size for step comparison
    let height = 1 << log_height;
    let width = 64;
    let lde_height = height << LOG_BLOWUP;
    let log_lde_height = log_height + LOG_BLOWUP as usize;
    let lde_size = (lde_height * width) as u32;
    let shift = BabyBear::GENERATOR;

    let data = random_field_elements(height * width);
    let d_trace = transport_matrix_to_device(Arc::new(RowMajorMatrix::new(data, width)));
    sync_gpu();

    let d_lde = DeviceMatrix::<BabyBear>::with_capacity(lde_height, width);

    // Step 1: batch_expand_pad
    group.bench_function("expand_pad", |b| {
        b.iter(|| {
            unsafe {
                batch_expand_pad(
                    d_lde.buffer(),
                    d_trace.buffer(),
                    width as u32,
                    lde_height as u32,
                    height as u32,
                )
                .unwrap();
            }
            sync_gpu();
        });
    });

    // Prepare buffer for subsequent steps
    unsafe {
        batch_expand_pad(
            d_lde.buffer(),
            d_trace.buffer(),
            width as u32,
            lde_height as u32,
            height as u32,
        )
        .unwrap();
    }
    sync_gpu();

    // Step 2: Forward NTT (with bit-reverse)
    group.bench_function("ntt_forward", |b| {
        b.iter(|| {
            batch_ntt(
                d_lde.buffer(),
                log_height as u32,
                LOG_BLOWUP,
                width as u32,
                true,
                true,
            );
            sync_gpu();
        });
    });

    // Step 3: ZK shift
    group.bench_function("zk_shift", |b| {
        b.iter(|| {
            unsafe {
                zk_shift(
                    d_lde.buffer(),
                    lde_size,
                    log_lde_height as u32,
                    shift.as_canonical_u32(),
                )
                .unwrap();
            }
            sync_gpu();
        });
    });

    // Step 4: Bit reverse
    group.bench_function("bit_reverse", |b| {
        b.iter(|| {
            unsafe {
                batch_bit_reverse(d_lde.buffer(), log_lde_height as u32, lde_size).unwrap();
            }
            sync_gpu();
        });
    });

    // Step 5: Inverse NTT
    group.bench_function("ntt_inverse", |b| {
        b.iter(|| {
            batch_ntt(
                d_lde.buffer(),
                log_lde_height as u32,
                0,
                width as u32,
                false,
                false,
            );
            sync_gpu();
        });
    });

    group.finish();
}

/// Benchmark LDE with different blowup factors.
fn bench_lde_blowup(c: &mut Criterion) {
    let mut group = c.benchmark_group("lde/blowup");

    let log_height = 14;
    let height = 1 << log_height;
    let width = 64;
    let shift = BabyBear::GENERATOR;

    let data = random_field_elements(height * width);
    let d_trace = transport_matrix_to_device(Arc::new(RowMajorMatrix::new(data, width)));
    sync_gpu();

    for log_blowup in [1, 2, 3, 4] {
        let lde_height = height << log_blowup;
        let log_lde_height = log_height + log_blowup;
        let lde_size = (lde_height * width) as u32;

        let d_lde = DeviceMatrix::<BabyBear>::with_capacity(lde_height, width);

        group.bench_with_input(
            BenchmarkId::new("blowup", 1 << log_blowup),
            &log_blowup,
            |b, &log_blowup| {
                b.iter(|| {
                    unsafe {
                        batch_expand_pad(
                            d_lde.buffer(),
                            d_trace.buffer(),
                            width as u32,
                            lde_height as u32,
                            height as u32,
                        )
                        .unwrap();
                    }
                    batch_ntt(
                        d_lde.buffer(),
                        log_height as u32,
                        log_blowup as u32,
                        width as u32,
                        true,
                        true,
                    );
                    unsafe {
                        zk_shift(
                            d_lde.buffer(),
                            lde_size,
                            log_lde_height as u32,
                            shift.as_canonical_u32(),
                        )
                        .unwrap();
                        batch_bit_reverse(d_lde.buffer(), log_lde_height as u32, lde_size).unwrap();
                    }
                    batch_ntt(
                        d_lde.buffer(),
                        log_lde_height as u32,
                        0,
                        width as u32,
                        false,
                        false,
                    );
                    sync_gpu();
                });
            },
        );
    }

    group.finish();
}

/// Benchmark LDE with different trace widths.
fn bench_lde_width(c: &mut Criterion) {
    let mut group = c.benchmark_group("lde/width");

    let log_height = 14;
    let height = 1 << log_height;
    let lde_height = height << LOG_BLOWUP;
    let log_lde_height = log_height + LOG_BLOWUP as usize;
    let shift = BabyBear::GENERATOR;

    for width in [16, 32, 64, 128, 256] {
        let lde_size = (lde_height * width) as u32;

        let data = random_field_elements(height * width);
        let d_trace = transport_matrix_to_device(Arc::new(RowMajorMatrix::new(data, width)));
        sync_gpu();

        let d_lde = DeviceMatrix::<BabyBear>::with_capacity(lde_height, width);

        let bytes = height * width * std::mem::size_of::<BabyBear>();
        group.throughput(Throughput::Bytes(bytes as u64));

        group.bench_with_input(BenchmarkId::new("width", width), &width, |b, &width| {
            b.iter(|| {
                unsafe {
                    batch_expand_pad(
                        d_lde.buffer(),
                        d_trace.buffer(),
                        width as u32,
                        lde_height as u32,
                        height as u32,
                    )
                    .unwrap();
                }
                batch_ntt(
                    d_lde.buffer(),
                    log_height as u32,
                    LOG_BLOWUP,
                    width as u32,
                    true,
                    true,
                );
                unsafe {
                    zk_shift(
                        d_lde.buffer(),
                        lde_size,
                        log_lde_height as u32,
                        shift.as_canonical_u32(),
                    )
                    .unwrap();
                    batch_bit_reverse(d_lde.buffer(), log_lde_height as u32, lde_size).unwrap();
                }
                batch_ntt(
                    d_lde.buffer(),
                    log_lde_height as u32,
                    0,
                    width as u32,
                    false,
                    false,
                );
                sync_gpu();
            });
        });
    }

    group.finish();
}

/// Benchmark full H2D -> LDE -> D2H pipeline latency.
fn bench_lde_with_transfers(c: &mut Criterion) {
    let mut group = c.benchmark_group("lde/with_transfers");

    for log_height in [12, 14, 16] {
        let height = 1 << log_height;
        let width = 64;
        let lde_height = height << LOG_BLOWUP;
        let log_lde_height = log_height + LOG_BLOWUP as usize;
        let lde_size = (lde_height * width) as u32;
        let shift = BabyBear::GENERATOR;

        let data = random_field_elements(height * width);

        group.bench_with_input(
            BenchmarkId::new("h2d_lde_d2h", size_string(log_height)),
            &data,
            |b, data| {
                b.iter(|| {
                    // H2D
                    let d_trace =
                        transport_matrix_to_device(Arc::new(RowMajorMatrix::new(data.clone(), width)));
                    let d_lde = DeviceMatrix::<BabyBear>::with_capacity(lde_height, width);
                    sync_gpu();

                    // LDE pipeline
                    unsafe {
                        batch_expand_pad(
                            d_lde.buffer(),
                            d_trace.buffer(),
                            width as u32,
                            lde_height as u32,
                            height as u32,
                        )
                        .unwrap();
                    }
                    batch_ntt(
                        d_lde.buffer(),
                        log_height as u32,
                        LOG_BLOWUP,
                        width as u32,
                        true,
                        true,
                    );
                    unsafe {
                        zk_shift(
                            d_lde.buffer(),
                            lde_size,
                            log_lde_height as u32,
                            shift.as_canonical_u32(),
                        )
                        .unwrap();
                        batch_bit_reverse(d_lde.buffer(), log_lde_height as u32, lde_size).unwrap();
                    }
                    batch_ntt(
                        d_lde.buffer(),
                        log_lde_height as u32,
                        0,
                        width as u32,
                        false,
                        false,
                    );
                    sync_gpu();

                    // D2H (typically not needed in full pipeline, but included for completeness)
                    let result = d_lde.buffer().to_host().unwrap();
                    black_box(result)
                });
            },
        );
    }

    group.finish();
}

/// Compare GPU LDE vs CPU LDE (using p3_dft).
fn bench_lde_gpu_vs_cpu(c: &mut Criterion) {
    let mut group = c.benchmark_group("lde/gpu_vs_cpu");

    let dft = Radix2Dit::default();

    for log_height in [10, 12, 14] {
        // Smaller sizes for CPU comparison
        let height = 1 << log_height;
        let width = 64;
        let lde_height = height << LOG_BLOWUP;
        let log_lde_height = log_height + LOG_BLOWUP as usize;
        let lde_size = (lde_height * width) as u32;
        let shift = BabyBear::GENERATOR;

        let data = random_field_elements(height * width);

        // GPU benchmark
        let d_trace =
            transport_matrix_to_device(Arc::new(RowMajorMatrix::new(data.clone(), width)));
        let d_lde = DeviceMatrix::<BabyBear>::with_capacity(lde_height, width);
        sync_gpu();

        group.bench_with_input(
            BenchmarkId::new("gpu", size_string(log_height)),
            &log_height,
            |b, _| {
                b.iter(|| {
                    unsafe {
                        batch_expand_pad(
                            d_lde.buffer(),
                            d_trace.buffer(),
                            width as u32,
                            lde_height as u32,
                            height as u32,
                        )
                        .unwrap();
                    }
                    batch_ntt(
                        d_lde.buffer(),
                        log_height as u32,
                        LOG_BLOWUP,
                        width as u32,
                        true,
                        true,
                    );
                    unsafe {
                        zk_shift(
                            d_lde.buffer(),
                            lde_size,
                            log_lde_height as u32,
                            shift.as_canonical_u32(),
                        )
                        .unwrap();
                        batch_bit_reverse(d_lde.buffer(), log_lde_height as u32, lde_size).unwrap();
                    }
                    batch_ntt(
                        d_lde.buffer(),
                        log_lde_height as u32,
                        0,
                        width as u32,
                        false,
                        false,
                    );
                    sync_gpu();
                });
            },
        );

        // CPU benchmark (using coset_lde_batch from p3_dft)
        let matrix = RowMajorMatrix::new(data.clone(), width);
        group.bench_with_input(
            BenchmarkId::new("cpu", size_string(log_height)),
            &matrix,
            |b, matrix| {
                b.iter(|| {
                    let result = dft.coset_lde_batch(matrix.clone(), LOG_BLOWUP as usize, shift);
                    black_box(result)
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_lde_full,
    bench_lde_steps,
    bench_lde_blowup,
    bench_lde_width,
    bench_lde_with_transfers,
    bench_lde_gpu_vs_cpu,
);

criterion_main!(benches);
