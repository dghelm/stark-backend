use std::ffi::CStr;

use cubecl_hip_sys::{hipGetErrorName, hipGetErrorString};
use thiserror::Error;

/// Safely convert a C string pointer returned by HIP into a Rust `String`.
fn cstr_to_string(ptr: *const i8) -> String {
    if ptr.is_null() {
        return "Unknown HIP error (null pointer)".to_string();
    }
    unsafe { CStr::from_ptr(ptr).to_string_lossy().into_owned() }
}

/// Returns the symbolic error name (e.g. "hipErrorOutOfMemory")
pub fn get_hip_error_name(error_code: u32) -> String {
    let name_ptr = unsafe { hipGetErrorName(error_code) };
    cstr_to_string(name_ptr)
}

/// Returns a descriptive error string (e.g. "out of memory")
pub fn get_hip_error_string(error_code: u32) -> String {
    let str_ptr = unsafe { hipGetErrorString(error_code) };
    cstr_to_string(str_ptr)
}

/// A HIP error with code, name, and message
#[derive(Error, Debug)]
#[error("{message} ({name})")]
pub struct HipError {
    pub code: u32,
    pub name: String,
    pub message: String,
}

impl HipError {
    /// Construct from a raw HIP error code (non-zero).
    pub fn new(code: u32) -> Self {
        HipError {
            code,
            name: get_hip_error_name(code),
            message: get_hip_error_string(code),
        }
    }

    /// Returns `Ok(())` if `code == 0` (hipSuccess), or `Err(HipError)` if non-zero.
    pub fn from_result(code: u32) -> Result<(), Self> {
        if code == 0 {
            Ok(())
        } else {
            Err(Self::new(code))
        }
    }

    /// Returns `true` if the error is hipErrorOutOfMemory
    #[inline]
    pub fn is_out_of_memory(&self) -> bool {
        // hipErrorOutOfMemory = 2 in HIP (same as CUDA)
        self.code == 2
    }
}

#[inline]
pub fn check(code: u32) -> Result<(), HipError> {
    HipError::from_result(code)
}

#[derive(Error, Debug)]
pub enum MemoryError {
    #[error(transparent)]
    Hip(#[from] HipError),

    #[error("Attempted to free null pointer")]
    NullPointer,

    #[error("Attempted to free untracked pointer")]
    UntrackedPointer,

    #[error("Failed to acquire memory manager lock")]
    LockError,

    #[error("Invalid memory size: {size}")]
    InvalidMemorySize { size: usize },

    #[error(
        "Out of memory in pool (size requested: {requested} bytes, available: {available} bytes)"
    )]
    OutOfMemory { requested: usize, available: usize },

    #[error("Invalid pointer: pointer not found in allocation table")]
    InvalidPointer,

    #[error("Failed to reserve virtual address space (bytes: {size}, page size: {page_size})")]
    ReserveFailed { size: usize, page_size: usize },
}

#[derive(Error, Debug)]
pub enum MemCopyError {
    #[error(transparent)]
    Hip(#[from] HipError),
    #[error("Size mismatch in {operation}: host len={host_len}, device len={device_len}")]
    SizeMismatch {
        operation: &'static str,
        host_len: usize,
        device_len: usize,
    },
}

#[derive(Error, Debug)]
pub enum KernelError {
    #[error(transparent)]
    Hip(#[from] HipError),

    #[error("Unsupported type size {size}")]
    UnsupportedTypeSize { size: usize },
}
