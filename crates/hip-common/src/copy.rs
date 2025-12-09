use std::{ffi::c_void, sync::Mutex, sync::OnceLock};

use cubecl_hip_sys::{
    hipMemcpyAsync, hipMemcpyKind, hipMemcpyKind_hipMemcpyDeviceToDevice,
    hipMemcpyKind_hipMemcpyDeviceToHost, hipMemcpyKind_hipMemcpyHostToDevice,
    hipMemcpyKind_hipMemcpyHostToHost,
};

use crate::{
    d_buffer::DeviceBuffer,
    error::{check, MemCopyError},
    stream::{hipStreamPerThread, HipEvent},
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
    fn to_host(&self) -> Result<Vec<T>, MemCopyError>;
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
}
