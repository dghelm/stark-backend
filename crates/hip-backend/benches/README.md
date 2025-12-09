# HIP Backend Performance Benchmarks

This directory contains micro-benchmarks for investigating GPU performance bottlenecks in the HIP backend.

## Background

The HIP backend shows **severe performance degradation** at larger input sizes:

| log_height | Input Size | CPU Time | HIP Time | Speedup |
|------------|-----------|----------|----------|---------|
| 16 | 6 MB | 388ms | 350ms | **1.1x** ✓ |
| 18 | 24 MB | 1.56s | 3.31s | **0.47x** ✗ |
| 20 | 96 MB | 7.82s | 51.28s | **0.15x** ✗ |

At log_height=20, HIP is **6.5x slower than CPU**. These benchmarks help identify and quantify the bottlenecks.

## Running Benchmarks

```bash
# Run all benchmarks
HIP_FORCE_EXIT=1 HIP_ARCH=gfx1151 cargo bench -p openvm-hip-backend

# Run specific benchmark
HIP_FORCE_EXIT=1 HIP_ARCH=gfx1151 cargo bench -p openvm-hip-backend --bench sync_overhead

# Run with filter
HIP_FORCE_EXIT=1 HIP_ARCH=gfx1151 cargo bench -p openvm-hip-backend --bench lde -- "lde/steps"

# Test mode (quick validation)
HIP_FORCE_EXIT=1 HIP_ARCH=gfx1151 cargo bench -p openvm-hip-backend --bench sync_overhead -- --test
```

Replace `gfx1151` with your GPU architecture (e.g., `gfx1100` for RDNA3, `gfx90a` for MI200).

## Benchmark Descriptions

### 1. `sync_overhead.rs` - Stream Synchronization

Measures the overhead of HIP synchronization primitives.

**Key findings:**
- `hipDeviceSynchronize()` idle: ~74ns
- `record_and_wait()`: **~5.4µs (73x slower!)**
- Batched vs individual syncs: 2x overhead for individual syncs

| Operations | sync_each | sync_batch | Overhead |
|------------|-----------|------------|----------|
| 10 | 106.6µs | 57.2µs | 1.86x |
| 20 | 212.6µs | 106.7µs | 2.0x |
| 50 | 532.4µs | 246.5µs | 2.16x |

**Implication:** Every D2H copy uses `record_and_wait()`, adding ~5µs overhead. With hundreds of D2H calls in a proof, this accumulates to significant overhead.

### 2. `memory_copy.rs` - Memory Transfer Throughput

Measures H2D/D2H/D2D transfer performance.

**Test cases:**
- H2D throughput at various sizes (1KB to 100MB)
- D2H throughput (includes sync overhead)
- D2D copy throughput
- Batched vs individual transfers
- Allocation overhead

### 3. `ntt.rs` - Number Theoretic Transform

Measures NTT performance, the core operation in LDE.

**Test cases:**
- Forward NTT (GPU) at sizes 2^10 to 2^20
- Inverse NTT (GPU)
- Forward NTT (CPU) for comparison
- Batch NTT with varying widths
- NTT with/without bit-reverse
- Full H2D → NTT → D2H pipeline

### 4. `poseidon2.rs` - Poseidon2 Hashing

Measures Poseidon2 hash throughput for Merkle tree construction.

**Test cases:**
- Row hashing (`poseidon2_rows_p3_multi`)
- Compression (`poseidon2_compress`)
- Metadata H2D overhead (3 separate copies per `hash_matrices`)
- Varying matrix widths
- Full Merkle layer (rows + compress)
- Multiple matrices

### 5. `lde.rs` - Low-Degree Extension Pipeline

Measures the full LDE pipeline and individual steps.

**LDE pipeline steps:**
1. `batch_expand_pad` - Expand trace with zero padding
2. `batch_ntt` (forward) - Forward NTT
3. `zk_shift` - Apply coset shift
4. `batch_bit_reverse` - Reorder for final NTT
5. `batch_ntt` (inverse) - Inverse NTT

**Test cases:**
- Full pipeline at various sizes
- Individual step timing
- Different blowup factors (2x, 4x, 8x, 16x)
- Different trace widths
- H2D → LDE → D2H full latency
- GPU vs CPU comparison

## Identified Bottlenecks

### 1. Implicit Stream Synchronization (40% of overhead) - HIGH

**Location:** `hip-common/src/copy.rs:100-107`

Every `to_host()` call blocks via `record_and_wait()`:
```rust
get_copy_event().lock().unwrap().record_and_wait(hipStreamPerThread)?;
```

### 2. Serial Kernel Execution (30% of overhead) - HIGH

**Location:** `hip-backend/src/lde/ops.rs:32-70`

LDE launches 5 kernels sequentially with no overlap opportunity.

### 3. Excessive Metadata H2D Copies (20% of overhead) - MEDIUM

**Location:** `hip-backend/src/merkle_tree.rs:96-98`

Three separate H2D copies per `hash_matrices()` call.

### 4. Single Stream Architecture (10% of overhead) - MEDIUM

**Location:** `hip-common/src/stream.rs:68`

All operations use `hipStreamPerThread`, preventing overlap.

## Recommended Fixes

| Fix | Impact | Complexity | Priority |
|-----|--------|------------|----------|
| Remove unnecessary D2H syncs | High | Low | P0 |
| Batch metadata H2D copies | Medium | Low | P1 |
| Multi-stream for overlap | High | High | P2 |
| Kernel fusion | Medium | High | P3 |

## Output Location

Criterion reports are generated in `target/criterion/`. Open `target/criterion/report/index.html` for detailed analysis with graphs.
