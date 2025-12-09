use std::{
    collections::HashMap,
    ffi::c_void,
    ptr::NonNull,
    sync::{Mutex, OnceLock},
};

use bytesize::ByteSize;
use cubecl_hip_sys::{hipFreeAsync, hipMallocAsync};

use crate::{
    error::{check, MemoryError},
    stream::{current_stream_id, hipStreamPerThread},
};

mod hip;
mod vm_pool;
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
/// Set HIP_NO_FORCE_EXIT=1 to disable this workaround for debugging.
extern "C" fn force_exit() {
    // Allow opt-out via environment variable for debugging
    if std::env::var("HIP_NO_FORCE_EXIT").is_ok() {
        eprintln!("[HIP] HIP_NO_FORCE_EXIT set - skipping force exit (may crash)");
        return;
    }
    // _exit() immediately terminates without running remaining atexit handlers
    // or flushing stdio buffers. This is necessary because HIP/ROCm runtime's
    // cleanup handlers have a bug that causes SIGSEGV when cleaning up resources
    // from virtual memory pool (VPMM) allocations.
    unsafe {
        libc::_exit(0);
    }
}

/// atexit handler to set the shutdown flag.
/// This is used when HIP_NO_FORCE_EXIT is set, to signal that HIP APIs should be avoided
/// during the remaining cleanup (though this path typically crashes due to HIP runtime bugs).
extern "C" fn shutdown_flag_setter() {
    SHUTDOWN_IN_PROGRESS.store(true, Ordering::Relaxed);
}

#[ctor::ctor]
fn init() {
    // Register shutdown_flag_setter FIRST (runs LAST due to LIFO order).
    // This is only reached if HIP_NO_FORCE_EXIT is set and force_exit doesn't terminate.
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
    // It immediately exits the process, preventing HIP's buggy cleanup from running.
    unsafe {
        libc::atexit(force_exit);
    }
}

pub struct MemoryManager {
    pool: VirtualMemoryPool,
    allocated_ptrs: HashMap<NonNull<c_void>, usize>,
    current_size: usize,
    max_used_size: usize,
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
            allocated_ptrs: HashMap::new(),
            current_size: 0,
            max_used_size: 0,
        }
    }

    fn d_malloc(&mut self, size: usize) -> Result<*mut c_void, MemoryError> {
        assert!(size != 0, "Requested size must be non-zero");

        let mut tracked_size = size;
        let ptr = if size < self.pool.page_size {
            let mut ptr: *mut c_void = std::ptr::null_mut();
            check(unsafe { hipMallocAsync(&mut ptr, size, hipStreamPerThread) }).map_err(|e| {
                tracing::error!("hipMallocAsync failed: size={}: {:?}", size, e);
                MemoryError::from(e)
            })?;
            self.allocated_ptrs.insert(
                NonNull::new(ptr).expect("BUG: hipMallocAsync returned null"),
                size,
            );
            ptr
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
            check(unsafe { hipFreeAsync(ptr, hipStreamPerThread) }).map_err(|e| {
                tracing::error!("hipFreeAsync failed: ptr={:p}: {:?}", ptr, e);
                MemoryError::from(e)
            })?;
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
