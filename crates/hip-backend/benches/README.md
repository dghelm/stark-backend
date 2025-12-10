# HIP Backend Performance Benchmarks

This directory contains micro-benchmarks for investigating GPU performance bottlenecks in the HIP backend.

## Quick Start - Critical Performance Fix

**For consumer AMD GPUs (RX 7000/9000 series), always set `HIP_DISABLE_VPMM=1`:**

```bash
HIP_FORCE_EXIT=1 HIP_ARCH=gfx1151 HIP_DISABLE_VPMM=1 ./your_hip_binary
```

This bypasses the slow Virtual Memory Management (VMM) allocator and provides **217x speedup** on consumer GPUs.

| log_height=20 | VPMM Enabled | VPMM Disabled | Improvement |
|---------------|--------------|---------------|-------------|
| HIP Time | 50.46s | 231.84ms | **217x faster** |
| vs CPU | 0.12x (8x slower) | **27.1x faster** | |

## Background

The HIP backend showed **severe performance degradation** at larger input sizes with VPMM enabled:

| log_height | Input Size | CPU Time | HIP Time (VPMM) | HIP Time (no VPMM) | Speedup |
|------------|-----------|----------|-----------------|-------------------|---------|
| 16 | 6 MB | 390ms | 401ms | 188ms | **2.1x** ✓ |
| 20 | 96 MB | 6.3s | 50.5s | 232ms | **27x** ✓ |

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `HIP_DISABLE_VPMM=1` | **CRITICAL:** Disable VMM allocator. Use on consumer GPUs. | off |
| `HIP_FORCE_EXIT=1` | Force exit to avoid SIGSEGV on cleanup | off |
| `HIP_ARCH=gfxXXXX` | Target GPU architecture | auto-detect |
| `VPMM_PAGE_SIZE` | VMM page size (if VPMM enabled) | device default |
| `VPMM_PAGES` | Pre-allocate N pages at startup | 0 |

## Running Benchmarks

```bash
# Run all benchmarks (with VPMM disabled for accurate results)
HIP_FORCE_EXIT=1 HIP_ARCH=gfx1151 HIP_DISABLE_VPMM=1 cargo bench -p openvm-hip-backend

# Run specific benchmark
HIP_FORCE_EXIT=1 HIP_ARCH=gfx1151 HIP_DISABLE_VPMM=1 cargo bench -p openvm-hip-backend --bench sync_overhead

# Run with filter
HIP_FORCE_EXIT=1 HIP_ARCH=gfx1151 HIP_DISABLE_VPMM=1 cargo bench -p openvm-hip-backend --bench lde -- "lde/steps"

# Test mode (quick validation)
HIP_FORCE_EXIT=1 HIP_ARCH=gfx1151 HIP_DISABLE_VPMM=1 cargo bench -p openvm-hip-backend --bench sync_overhead -- --test
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

### 1. VMM Allocator Overhead (99% of overhead) - CRITICAL [FIXED]

**Location:** `hip-common/src/memory_manager/vm_pool.rs`

The Virtual Memory Management (VMM) allocator uses `hipMemMap`, `hipMemCreate`, `hipMemUnmap`, `hipMemRelease` which are **extremely slow on consumer AMD GPUs**:

| API Call | Total Time | Calls | % of Runtime |
|----------|------------|-------|--------------|
| hipMemMap | 313ms | 6,207 | 29% |
| hipMemUnmap | 290ms | 6,016 | 27% |
| hipMemRelease | 152ms | 6,015 | 14% |
| hipMemCreate | 146ms | 6,015 | 14% |

While kernel execution was only **8ms** (0.7%), memory management consumed **1,072ms** (99.3%)!

**Fix:** Set `HIP_DISABLE_VPMM=1` to use simple `hipMalloc`/`hipFree` instead.

### 2. Implicit Stream Synchronization - LOW (after VPMM fix)

**Location:** `hip-common/src/copy.rs`

Every `to_host()` call blocks via `record_and_wait()`. Use `to_host_fast()` for better performance (stream sync vs event sync).

### 3. Excessive Metadata H2D Copies - LOW (fixed)

**Location:** `hip-backend/src/merkle_tree.rs`

Three separate H2D copies per `hash_matrices()` call have been batched into a single transfer.

## Recommended Fixes

| Fix | Impact | Complexity | Status |
|-----|--------|------------|--------|
| Disable VPMM via `HIP_DISABLE_VPMM=1` | **217x** | Trivial | **DONE** |
| Use `to_host_fast()` instead of `to_host()` | Low | Low | DONE |
| Batch metadata H2D copies | Low | Low | DONE |
| Multi-stream for overlap | Medium | High | TODO |
| Kernel fusion | Medium | High | TODO |

## Output Location

Criterion reports are generated in `target/criterion/`. Open `target/criterion/report/index.html` for detailed analysis with graphs.
