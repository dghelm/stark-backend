use std::{
    collections::HashMap,
    ffi::c_void,
    ptr::NonNull,
    sync::{Mutex, OnceLock},
};

use bytesize::ByteSize;
use cubecl_hip_sys::{hipFree, hipFreeAsync, hipMalloc, hipMallocAsync};

use crate::{
    error::{check, MemoryError},
    stream::{current_stream_id, hipStreamPerThread},
};

/// Simple free-list pool for reusing allocations when VPMM is disabled.
/// This avoids the ~7ms overhead of hipMallocAsync on consumer AMD GPUs.
///
/// Buffers are bucketed by size class (power of 2), and we keep up to
/// MAX_CACHED_PER_SIZE buffers per size class.
mod simple_pool {
    use std::{collections::HashMap, ffi::c_void, ptr::NonNull};

    /// Maximum buffers to cache per size class
    const MAX_CACHED_PER_SIZE: usize = 4;

    /// Minimum allocation size (smaller allocations are rounded up)
    const MIN_ALLOC_SIZE: usize = 256;

    pub struct SimplePool {
        /// Map from size class (power of 2) to list of free buffers
        free_lists: HashMap<usize, Vec<NonNull<c_void>>>,
        /// Total bytes currently in the pool
        pool_bytes: usize,
    }

    impl SimplePool {
        pub fn new() -> Self {
            Self {
                free_lists: HashMap::new(),
                pool_bytes: 0,
            }
        }

        /// Round size up to the nearest power of 2 (with minimum)
        fn size_class(size: usize) -> usize {
            let size = size.max(MIN_ALLOC_SIZE);
            size.next_power_of_two()
        }

        /// Try to get a buffer from the pool that's >= requested size
        /// Returns (pointer, actual_size) if found
        pub fn try_get(&mut self, size: usize) -> Option<(NonNull<c_void>, usize)> {
            let size_class = Self::size_class(size);

            if let Some(list) = self.free_lists.get_mut(&size_class) {
                if let Some(ptr) = list.pop() {
                    self.pool_bytes -= size_class;
                    return Some((ptr, size_class));
                }
            }
            None
        }

        /// Return a buffer to the pool
        /// Returns false if the pool is full for this size class (caller should free)
        pub fn try_put(&mut self, ptr: NonNull<c_void>, size: usize) -> bool {
            let size_class = Self::size_class(size);

            let list = self.free_lists.entry(size_class).or_default();
            if list.len() >= MAX_CACHED_PER_SIZE {
                return false; // Pool full, caller should free
            }

            list.push(ptr);
            self.pool_bytes += size_class;
            true
        }

        /// Get all buffers for shutdown cleanup
        pub fn drain_all(&mut self) -> Vec<NonNull<c_void>> {
            let mut all = Vec::new();
            for (_, list) in self.free_lists.drain() {
                all.extend(list);
            }
            self.pool_bytes = 0;
            all
        }

        /// Current pool size in bytes
        #[allow(dead_code)]
        pub fn pool_bytes(&self) -> usize {
            self.pool_bytes
        }
    }

    impl Default for SimplePool {
        fn default() -> Self {
            Self::new()
        }
    }
}

mod hip;
mod vm_pool;
use simple_pool::SimplePool;
use vm_pool::VirtualMemoryPool;

#[cfg(test)]
mod tests;

use std::sync::atomic::{AtomicBool, Ordering};

// NOTE: We use `&'static` with Box::leak to prevent Drop from running during
// static destruction. This avoids SIGSEGV crashes caused by calling HIP APIs
// after the ROCm runtime has already started shutting down. The OS will reclaim
// all GPU resources when the process exits.
static MEMORY_MANAGER: OnceLock<&'static Mutex<MemoryManager>> = OnceLock::new();

// Flag to indicate that we're in shutdown mode and should skip HIP API calls.
// This is set by the #[ctor::dtor] function before static destruction begins.
static SHUTDOWN_IN_PROGRESS: AtomicBool = AtomicBool::new(false);

/// Returns true if the process is shutting down and HIP APIs should be avoided.
pub fn is_shutting_down() -> bool {
    SHUTDOWN_IN_PROGRESS.load(Ordering::Relaxed)
}

/// atexit handler that calls _exit(0) to skip remaining atexit handlers.
/// This prevents HIP runtime's buggy cleanup from running and causing SIGSEGV.
///
/// # For Testing Only
///
/// This escape hatch exists because HIP/ROCm has a bug where its internal atexit
/// cleanup crashes (SIGSEGV) when cleaning up VPMM allocations. The crash happens
/// *inside HIP's code*, not ours, so we can't fix it directly.
///
/// **Production binaries** should call [`hip_runtime_shutdown()`] explicitly
/// before exit instead of relying on this mechanism.
///
/// # Environment Variable
///
/// - `HIP_FORCE_EXIT=1`: Enable force exit (required for openvm-hip-backend tests)
/// - Default (no env var): Normal exit, `#[ctor::dtor]` cleanup runs
///
/// The force exit is **OFF by default** because calling `_exit()` skips Rust
/// destructors, can mask test failures, and bypasses normal cleanup. Only enable
/// it when running HIP tests that would otherwise crash during cleanup.
///
/// # Example
///
/// ```bash
/// # Run hip-backend tests without SIGSEGV on exit
/// HIP_FORCE_EXIT=1 HIP_ARCH=gfx1151 cargo test -p openvm-hip-backend
/// ```
extern "C" fn force_exit() {
    // Only force exit if explicitly requested via environment variable
    let force_exit_enabled = std::env::var("HIP_FORCE_EXIT")
        .map(|v| v == "1")
        .unwrap_or(false);

    if force_exit_enabled {
        // _exit() immediately terminates without running remaining atexit handlers
        // or flushing stdio buffers. This is necessary because HIP/ROCm runtime's
        // cleanup handlers have a bug that causes SIGSEGV when cleaning up resources
        // from virtual memory pool (VPMM) allocations.
        unsafe {
            libc::_exit(0);
        }
    }
    // Otherwise, let normal exit proceed (dtor cleanup will run)
}

/// atexit handler to set the shutdown flag.
/// This signals that HIP APIs should be avoided during remaining cleanup.
extern "C" fn shutdown_flag_setter() {
    SHUTDOWN_IN_PROGRESS.store(true, Ordering::Relaxed);
}

#[ctor::ctor]
fn init() {
    // Register shutdown_flag_setter FIRST (runs LAST due to LIFO order).
    unsafe {
        libc::atexit(shutdown_flag_setter);
    }

    // Box::leak gives us 'static lifetime - MemoryManager will never be dropped.
    // This is intentional: static destructor order vs HIP runtime teardown is
    // undefined and causes SIGSEGVs at process exit on some platforms.
    let manager = Box::leak(Box::new(Mutex::new(MemoryManager::new())));
    let _ = MEMORY_MANAGER.set(manager);
    tracing::info!("Memory manager initialized (leak-on-exit pattern)");

    // Register force_exit LAST (runs FIRST due to LIFO order).
    // Only takes effect if HIP_FORCE_EXIT=1 is set - see force_exit() docs.
    unsafe {
        libc::atexit(force_exit);
    }
}

/// Best-effort early shutdown via destructor.
///
/// This runs during process teardown and attempts to clean up VPMM resources
/// before HIP's internal atexit handlers run. The order of dtor execution
/// relative to HIP's cleanup is not guaranteed, but this provides a clean
/// shutdown path when it runs early enough.
///
/// Combined with the opt-in `HIP_FORCE_EXIT=1` fallback, this gives us:
/// - Normal exit: dtor tries to clean up properly
/// - Force exit: `_exit(0)` skips everything (guaranteed no crash)
#[ctor::dtor]
fn early_shutdown() {
    // Don't attempt shutdown if force_exit will handle it
    let force_exit_enabled = std::env::var("HIP_FORCE_EXIT")
        .map(|v| v == "1")
        .unwrap_or(false);

    if force_exit_enabled {
        // force_exit() will call _exit(0) and skip all cleanup anyway
        return;
    }

    // Attempt early cleanup
    if let Some(mm) = MEMORY_MANAGER.get() {
        if let Ok(mut guard) = mm.lock() {
            guard.shutdown();
        }
    }
}

pub struct MemoryManager {
    pool: VirtualMemoryPool,
    /// Simple free-list pool for reusing small allocations (when VPMM disabled)
    simple_pool: SimplePool,
    allocated_ptrs: HashMap<NonNull<c_void>, usize>,
    current_size: usize,
    max_used_size: usize,
    shutdown_done: bool,
}

/// # Safety
/// `MemoryManager` is not internally synchronized. These impls are safe because
/// the singleton instance is wrapped in `Mutex` via `MEMORY_MANAGER`.
unsafe impl Send for MemoryManager {}
unsafe impl Sync for MemoryManager {}

impl MemoryManager {
    pub fn new() -> Self {
        // Create virtual memory pool
        let pool = VirtualMemoryPool::default();

        Self {
            pool,
            simple_pool: SimplePool::new(),
            allocated_ptrs: HashMap::new(),
            current_size: 0,
            max_used_size: 0,
            shutdown_done: false,
        }
    }

    /// Explicitly release all GPU resources.
    ///
    /// This method properly cleans up GPU resources before process exit,
    /// avoiding the SIGSEGV crash that occurs when HIP's internal cleanup
    /// runs after we've allocated VPMM objects.
    ///
    /// # Idempotency
    /// Safe to call multiple times - subsequent calls are no-ops.
    pub fn shutdown(&mut self) {
        if self.shutdown_done {
            return;
        }
        self.shutdown_done = true;

        tracing::debug!(
            "MemoryManager::shutdown() - releasing {} small allocations",
            self.allocated_ptrs.len()
        );

        // Free small allocations (allocated via hipMalloc)
        for (ptr, _size) in self.allocated_ptrs.drain() {
            // Best effort - don't fail on errors during shutdown
            let result = unsafe { hipFreeAsync(ptr.as_ptr(), hipStreamPerThread) };
            if result != 0 {
                tracing::warn!(
                    "hipFreeAsync failed during shutdown: ptr={:p}, error={}",
                    ptr.as_ptr(),
                    result
                );
            }
        }

        // Drain and free the simple pool
        for ptr in self.simple_pool.drain_all() {
            let result = unsafe { hipFreeAsync(ptr.as_ptr(), hipStreamPerThread) };
            if result != 0 {
                tracing::warn!(
                    "hipFreeAsync (pool) failed during shutdown: ptr={:p}, error={}",
                    ptr.as_ptr(),
                    result
                );
            }
        }

        // Shutdown the VM pool
        self.pool.shutdown();

        tracing::debug!("MemoryManager::shutdown() completed");
    }

    fn d_malloc(&mut self, size: usize) -> Result<*mut c_void, MemoryError> {
        assert!(size != 0, "Requested size must be non-zero");

        let mut tracked_size = size;
        let ptr = if size < self.pool.page_size {
            // When VPMM is disabled (page_size = usize::MAX), all allocations come here.
            // First, try to get a buffer from the simple pool (avoids ~7ms hipMallocAsync)
            if let Some((pooled_ptr, pooled_size)) = self.simple_pool.try_get(size) {
                // pooled_size is the size class (power of 2), which is >= size
                tracked_size = pooled_size;
                self.allocated_ptrs.insert(pooled_ptr, pooled_size);
                pooled_ptr.as_ptr()
            } else {
                // No pooled buffer available, allocate fresh
                // Round up to size class for consistency with pool reuse
                let alloc_size = size.max(256).next_power_of_two();
                let mut ptr: *mut c_void = std::ptr::null_mut();
                // Use hipMalloc (sync) instead of hipMallocAsync - may have less overhead
                // on consumer AMD GPUs where async memory APIs are slow
                check(unsafe { hipMalloc(&mut ptr, alloc_size) }).map_err(|e| {
                    tracing::error!("hipMalloc failed: size={}: {:?}", alloc_size, e);
                    MemoryError::from(e)
                })?;
                tracked_size = alloc_size;
                self.allocated_ptrs.insert(
                    NonNull::new(ptr).expect("BUG: hipMalloc returned null"),
                    alloc_size,
                );
                ptr
            }
        } else {
            tracked_size = size.next_multiple_of(self.pool.page_size);
            let stream_id = current_stream_id()?;
            self.pool.malloc_internal(tracked_size, stream_id)?
        };

        self.current_size += tracked_size;
        if self.current_size > self.max_used_size {
            self.max_used_size = self.current_size;
        }
        Ok(ptr)
    }

    /// # Safety
    /// The pointer `ptr` must be a valid, previously allocated device pointer.
    /// The caller must ensure that `ptr` is not used after this function is called.
    unsafe fn d_free(&mut self, ptr: *mut c_void) -> Result<(), MemoryError> {
        let nn = NonNull::new(ptr).ok_or(MemoryError::NullPointer)?;

        if let Some(size) = self.allocated_ptrs.remove(&nn) {
            self.current_size -= size;
            // Try to return to the simple pool for reuse (avoids ~7ms hipMallocAsync next time)
            if !self.simple_pool.try_put(nn, size) {
                // Pool is full for this size class, actually free
                check(unsafe { hipFreeAsync(ptr, hipStreamPerThread) }).map_err(|e| {
                    tracing::error!("hipFreeAsync failed: ptr={:p}: {:?}", ptr, e);
                    MemoryError::from(e)
                })?;
            }
        } else {
            let stream_id = current_stream_id()?;
            let freed_size = self.pool.free_internal(ptr, stream_id)?;
            self.current_size -= freed_size;
        }

        Ok(())
    }
}

impl Drop for MemoryManager {
    fn drop(&mut self) {
        // NOTE: We intentionally avoid calling HIP APIs from Drop.
        // Static destructor order vs HIP runtime teardown is undefined and caused
        // SIGSEGVs at process exit. The OS will reclaim GPU resources when the
        // process exits, so we accept this small "leak" for robustness.
        //
        // In practice, this Drop should never run because we use Box::leak in
        // the ctor to give MemoryManager a 'static lifetime.
        tracing::debug!(
            "MemoryManager::drop() called - skipping HIP cleanup (leak-on-exit pattern)"
        );
    }
}

impl Default for MemoryManager {
    fn default() -> Self {
        Self::new()
    }
}

pub fn d_malloc(size: usize) -> Result<*mut c_void, MemoryError> {
    let manager = MEMORY_MANAGER.get().unwrap();
    let mut manager = manager.lock().map_err(|_| MemoryError::LockError)?;
    manager.d_malloc(size)
}

/// # Safety
/// The pointer `ptr` must be a valid, previously allocated device pointer.
/// The caller must ensure that `ptr` is not used after this function is called.
pub unsafe fn d_free(ptr: *mut c_void) -> Result<(), MemoryError> {
    let manager = MEMORY_MANAGER.get().unwrap();
    let mut manager = manager.lock().map_err(|_| MemoryError::LockError)?;
    manager.d_free(ptr)
}

/// Explicitly shutdown the HIP memory manager and release all GPU resources.
///
/// Call this from `main()` before process exit for clean teardown.
/// This properly releases VPMM resources before HIP's buggy internal cleanup
/// runs, avoiding SIGSEGV crashes on process exit.
///
/// # Idempotency
/// Safe to call multiple times - subsequent calls are no-ops.
///
/// # Usage
///
/// For production binaries that want clean shutdown:
/// ```ignore
/// fn main() {
///     // ... your code ...
///
///     openvm_hip_common::hip_runtime_shutdown();
/// }
/// ```
///
/// For tests, you can alternatively use `HIP_FORCE_EXIT=1` environment variable
/// which terminates the process immediately after tests complete, skipping
/// HIP's buggy cleanup entirely.
pub fn hip_runtime_shutdown() {
    if is_shutting_down() {
        // Already in shutdown - avoid re-entering
        return;
    }

    if let Some(mm) = MEMORY_MANAGER.get() {
        if let Ok(mut guard) = mm.lock() {
            guard.shutdown();
        } else {
            tracing::warn!("hip_runtime_shutdown: failed to acquire lock");
        }
    }
}

#[derive(Debug, Clone)]
pub struct MemTracker {
    current: usize,
    label: &'static str,
}

impl MemTracker {
    pub fn start(label: &'static str) -> Self {
        let current = MEMORY_MANAGER
            .get()
            .and_then(|m| m.lock().ok())
            .map(|m| m.current_size)
            .unwrap_or(0);

        Self { current, label }
    }

    #[inline]
    pub fn tracing_info(&self, msg: impl Into<Option<&'static str>>) {
        let Some(manager) = MEMORY_MANAGER.get().and_then(|m| m.lock().ok()) else {
            tracing::error!("Memory manager not available");
            return;
        };
        let current = manager.current_size;
        let peak = manager.max_used_size;
        let used = current as isize - self.current as isize;
        let sign = if used >= 0 { "+" } else { "-" };
        let pool_usage = manager.pool.memory_usage();
        tracing::info!(
            "GPU mem: used={}{}, current={}, peak={}, in pool={} ({})",
            sign,
            ByteSize::b(used.unsigned_abs() as u64),
            ByteSize::b(current as u64),
            ByteSize::b(peak as u64),
            ByteSize::b(pool_usage as u64),
            msg.into()
                .map_or(self.label.to_string(), |m| format!("{}:{}", self.label, m))
        );
    }

    pub fn reset_peak(&mut self) {
        if let Some(mut manager) = MEMORY_MANAGER.get().and_then(|m| m.lock().ok()) {
            manager.max_used_size = manager.current_size;
        }
    }
}

impl Drop for MemTracker {
    fn drop(&mut self) {
        self.tracing_info(None);
    }
}
