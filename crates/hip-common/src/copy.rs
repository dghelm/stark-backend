use std::{ffi::c_void, sync::Mutex, sync::OnceLock};

use cubecl_hip_sys::{
    hipMemcpyAsync, hipMemcpyKind, hipMemcpyKind_hipMemcpyDeviceToDevice,
    hipMemcpyKind_hipMemcpyDeviceToHost, hipMemcpyKind_hipMemcpyHostToDevice,
    hipMemcpyKind_hipMemcpyHostToHost,
};

use crate::{
    d_buffer::DeviceBuffer,
    error::{check, MemCopyError},
    stream::{current_stream_sync, hipStreamPerThread, HipEvent},
};

// NOTE: We use Box::leak + OnceLock instead of lazy_static to prevent Drop from
// running during static destruction. This avoids SIGSEGV crashes caused by calling
// HIP APIs after the ROCm runtime has already started shutting down.
static COPY_EVENT: OnceLock<&'static Mutex<HipEvent>> = OnceLock::new();

fn get_copy_event() -> &'static Mutex<HipEvent> {
    COPY_EVENT.get_or_init(|| {
        Box::leak(Box::new(Mutex::new(HipEvent::new().unwrap())))
    })
}

/// FFI binding for the `hipMemcpyAsync` function on the default hip stream.
///
/// # Safety
/// Must follow the rules of the `hipMemcpyAsync` function from the HIP runtime API.
pub unsafe fn hip_memcpy<const SRC_DEVICE: bool, const DST_DEVICE: bool>(
    dst: *mut c_void,
    src: *const c_void,
    size_bytes: usize,
) -> Result<(), MemCopyError> {
    let kind: hipMemcpyKind = match (SRC_DEVICE, DST_DEVICE) {
        (false, false) => hipMemcpyKind_hipMemcpyHostToHost,
        (false, true) => hipMemcpyKind_hipMemcpyHostToDevice,
        (true, false) => hipMemcpyKind_hipMemcpyDeviceToHost,
        (true, true) => hipMemcpyKind_hipMemcpyDeviceToDevice,
    };

    check(unsafe { hipMemcpyAsync(dst, src, size_bytes, kind, hipStreamPerThread) })
        .map_err(MemCopyError::from)
}

// Host -> Device
pub trait MemCopyH2D<T> {
    fn copy_to(&self, dst: &mut DeviceBuffer<T>) -> Result<(), MemCopyError>;
    fn to_device(&self) -> Result<DeviceBuffer<T>, MemCopyError>;
}

impl<T> MemCopyH2D<T> for [T] {
    fn copy_to(&self, dst: &mut DeviceBuffer<T>) -> Result<(), MemCopyError> {
        if self.len() > dst.len() {
            return Err(MemCopyError::SizeMismatch {
                operation: "copy_to_device",
                host_len: self.len(),
                device_len: dst.len(),
            });
        }
        let size_bytes = std::mem::size_of_val(self);
        check(unsafe {
            hipMemcpyAsync(
                dst.as_mut_raw_ptr(),
                self.as_ptr() as *const c_void,
                size_bytes,
                hipMemcpyKind_hipMemcpyHostToDevice,
                hipStreamPerThread,
            )
        })
        .map_err(MemCopyError::from)
    }

    fn to_device(&self) -> Result<DeviceBuffer<T>, MemCopyError> {
        let mut dst = DeviceBuffer::with_capacity(self.len());
        self.copy_to(&mut dst)?;
        Ok(dst)
    }
}

// Device -> Host
pub trait MemCopyD2H<T> {
    /// Copy device buffer to host, using event-based synchronization.
    ///
    /// This uses `record_and_wait()` which has ~5µs overhead but is safe
    /// when multiple streams are in use.
    fn to_host(&self) -> Result<Vec<T>, MemCopyError>;

    /// Copy device buffer to host, using stream synchronization.
    ///
    /// This uses `hipStreamSynchronize()` which is faster (~70ns) but
    /// synchronizes the entire stream. Use when you know all prior work
    /// on the stream should complete before reading.
    fn to_host_fast(&self) -> Result<Vec<T>, MemCopyError>;
}

impl<T> MemCopyD2H<T> for DeviceBuffer<T> {
    fn to_host(&self) -> Result<Vec<T>, MemCopyError> {
        let mut host_vec = Vec::with_capacity(self.len());
        let size_bytes = std::mem::size_of::<T>() * self.len();

        check(unsafe {
            hipMemcpyAsync(
                host_vec.as_mut_ptr() as *mut c_void,
                self.as_raw_ptr(),
                size_bytes,
                hipMemcpyKind_hipMemcpyDeviceToHost,
                hipStreamPerThread,
            )
        })?;
        unsafe {
            get_copy_event()
                .lock()
                .unwrap()
                .record_and_wait(hipStreamPerThread)?;

            host_vec.set_len(self.len());
        }

        Ok(host_vec)
    }

    fn to_host_fast(&self) -> Result<Vec<T>, MemCopyError> {
        let mut host_vec = Vec::with_capacity(self.len());
        let size_bytes = std::mem::size_of::<T>() * self.len();

        check(unsafe {
            hipMemcpyAsync(
                host_vec.as_mut_ptr() as *mut c_void,
                self.as_raw_ptr(),
                size_bytes,
                hipMemcpyKind_hipMemcpyDeviceToHost,
                hipStreamPerThread,
            )
        })?;

        // Use stream sync instead of event sync - much faster
        current_stream_sync()?;

        unsafe {
            host_vec.set_len(self.len());
        }

        Ok(host_vec)
    }
}

// Async Device -> Host (deferred synchronization)

/// Pending D2H transfer that syncs lazily when data is accessed.
///
/// This allows GPU work to continue while the memcpy is in flight.
/// Call `.wait()` to synchronize and get the data.
pub struct PendingD2H<T> {
    host_vec: Vec<T>,
    len: usize,
}

impl<T> PendingD2H<T> {
    /// Force synchronization and get the data.
    ///
    /// This will block until the D2H transfer is complete, then return
    /// the copied data.
    pub fn wait(mut self) -> Result<Vec<T>, MemCopyError> {
        unsafe {
            get_copy_event()
                .lock()
                .unwrap()
                .record_and_wait(hipStreamPerThread)?;
            self.host_vec.set_len(self.len);
        }
        Ok(self.host_vec)
    }
}

/// Async D2H copy that returns a pending result.
///
/// Use this when you want to defer synchronization, allowing GPU work
/// to continue while the memcpy is in flight.
pub trait MemCopyD2HAsync<T> {
    /// Start async D2H copy, returns pending result.
    ///
    /// The transfer starts immediately but does not block.
    /// Call `.wait()` on the result to synchronize and get the data.
    fn to_host_async(&self) -> Result<PendingD2H<T>, MemCopyError>;
}

impl<T> MemCopyD2HAsync<T> for DeviceBuffer<T> {
    fn to_host_async(&self) -> Result<PendingD2H<T>, MemCopyError> {
        let mut host_vec = Vec::with_capacity(self.len());
        let size_bytes = std::mem::size_of::<T>() * self.len();

        check(unsafe {
            hipMemcpyAsync(
                host_vec.as_mut_ptr() as *mut c_void,
                self.as_raw_ptr(),
                size_bytes,
                hipMemcpyKind_hipMemcpyDeviceToHost,
                hipStreamPerThread,
            )
        })?;

        Ok(PendingD2H {
            host_vec,
            len: self.len(),
        })
    }
}

pub trait MemCopyD2D<T> {
    fn device_copy(&self) -> Result<DeviceBuffer<T>, MemCopyError>;
    fn device_copy_to(&self, dst: &mut DeviceBuffer<T>) -> Result<(), MemCopyError>;
}

impl<T> MemCopyD2D<T> for DeviceBuffer<T> {
    fn device_copy(&self) -> Result<DeviceBuffer<T>, MemCopyError> {
        let mut dst = DeviceBuffer::<T>::with_capacity(self.len());
        self.device_copy_to(&mut dst)?;
        Ok(dst)
    }

    fn device_copy_to(&self, dst: &mut DeviceBuffer<T>) -> Result<(), MemCopyError> {
        let size_bytes = std::mem::size_of::<T>() * self.len();

        check(unsafe {
            hipMemcpyAsync(
                dst.as_mut_raw_ptr(),
                self.as_raw_ptr(),
                size_bytes,
                hipMemcpyKind_hipMemcpyDeviceToDevice,
                hipStreamPerThread,
            )
        })
        .map_err(MemCopyError::from)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::d_buffer::DeviceBuffer;

    #[test]
    fn test_mem_copy() {
        // Our source data on the host
        let h = vec![1, 2, 3, 4, 5];

        // 1) Copy to a newly allocated device buffer
        let d1 = h.to_device().unwrap();

        // 2) Create another device buffer of the same size
        let mut d2 = DeviceBuffer::<i32>::with_capacity(h.len());

        // 3) Copy into that existing buffer
        h.copy_to(&mut d2).unwrap();

        // 4) Copy both buffers back to host
        let h1 = d1.to_host().unwrap();
        let h2 = d2.to_host().unwrap();

        assert_eq!(h, h1, "First device buffer mismatch");
        assert_eq!(h, h2, "Second device buffer mismatch");
    }

    #[test]
    fn test_mem_copy_async() {
        // Test async D2H copy
        let h = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];

        // Copy to device
        let d = h.to_device().unwrap();

        // Start async copy back (does not block)
        let pending = d.to_host_async().unwrap();

        // Now wait and get the data
        let result = pending.wait().unwrap();

        assert_eq!(h, result, "Async D2H mismatch");
    }

    #[test]
    fn test_mem_copy_async_multiple() {
        // Test multiple async copies - demonstrates deferred sync benefit
        let h1 = vec![1u32, 2, 3, 4];
        let h2 = vec![5u32, 6, 7, 8];
        let h3 = vec![9u32, 10, 11, 12];

        let d1 = h1.to_device().unwrap();
        let d2 = h2.to_device().unwrap();
        let d3 = h3.to_device().unwrap();

        // Start all three async copies (only final sync blocks)
        let p1 = d1.to_host_async().unwrap();
        let p2 = d2.to_host_async().unwrap();
        let p3 = d3.to_host_async().unwrap();

        // Wait for all - only one sync needed at the end
        let r1 = p1.wait().unwrap();
        let r2 = p2.wait().unwrap();
        let r3 = p3.wait().unwrap();

        assert_eq!(h1, r1);
        assert_eq!(h2, r2);
        assert_eq!(h3, r3);
    }

    #[test]
    fn test_mem_copy_fast() {
        // Test fast D2H copy (stream sync instead of event sync)
        let h = vec![1, 2, 3, 4, 5, 6, 7, 8];

        let d = h.to_device().unwrap();
        let result = d.to_host_fast().unwrap();

        assert_eq!(h, result, "Fast D2H mismatch");
    }
}
