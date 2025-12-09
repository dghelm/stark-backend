use std::ffi::c_void;

use cubecl_hip_sys::{hipDeviceReset, hipFree, hipGetDevice, hipSetDevice};

use crate::error::{check, HipError};

pub fn get_device() -> Result<i32, HipError> {
    let mut device = 0;
    unsafe {
        check(hipGetDevice(&mut device))?;
    }
    assert!(device >= 0);
    Ok(device)
}

pub fn set_device() -> Result<i32, HipError> {
    let mut device = 0;
    unsafe {
        // 1. Create a context (hipFree(nullptr) initializes the HIP runtime)
        check(hipFree(std::ptr::null_mut() as *mut c_void))?;
        // 2. Get and set the device
        check(hipGetDevice(&mut device))?;
        check(hipSetDevice(device))?;
    }
    Ok(device)
}

pub fn reset_device() -> Result<(), HipError> {
    check(unsafe { hipDeviceReset() })
}
