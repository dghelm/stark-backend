use std::borrow::Cow;

use cubecl_hip_sys::{
    hipDeviceSynchronize, hipEventCreate, hipEventDestroy, hipEventElapsedTime, hipEventQuery,
    hipEventRecord, hipEventSynchronize, hipStreamCreate, hipStreamDestroy, hipStreamSynchronize,
    hipStreamWaitEvent, hipEvent_t, hipStream_t,
};

use crate::error::{check, HipError};

pub fn device_synchronize() -> Result<(), HipError> {
    check(unsafe { hipDeviceSynchronize() })
}

pub struct HipStream {
    stream: hipStream_t,
}

unsafe impl Send for HipStream {}
unsafe impl Sync for HipStream {}

impl HipStream {
    /// Creates a new non-blocking HIP stream.
    pub fn new() -> Result<Self, HipError> {
        let mut stream: hipStream_t = std::ptr::null_mut();
        check(unsafe { hipStreamCreate(&mut stream) })?;
        Ok(Self { stream })
    }

    /// Get the raw HIP stream handle.
    #[inline]
    pub fn as_raw(&self) -> hipStream_t {
        self.stream
    }

    /// Synchronize this stream.
    pub fn synchronize(&self) -> Result<(), HipError> {
        check(unsafe { hipStreamSynchronize(self.stream) })
    }

    /// Wait for the given event.
    pub fn wait(&self, event: &HipEvent) -> Result<(), HipError> {
        check(unsafe { hipStreamWaitEvent(self.stream, event.event, 0) })
    }
}

impl Drop for HipStream {
    fn drop(&mut self) {
        if !self.stream.is_null() {
            self.synchronize().unwrap();
            let _ = unsafe { hipStreamDestroy(self.stream) };
            self.stream = std::ptr::null_mut();
        }
    }
}

/// Per-thread default stream constant for HIP
/// HIP uses 0x2 as hipStreamPerThread (same as CUDA)
#[allow(non_upper_case_globals)]
pub const hipStreamPerThread: hipStream_t = 0x02 as hipStream_t;

pub type HipStreamId = u64;

/// Get the current stream ID.
/// Note: HIP doesn't have a direct equivalent to cudaStreamGetId.
/// We use the stream pointer as an identifier.
pub fn current_stream_id() -> Result<HipStreamId, HipError> {
    // For per-thread stream, use a constant ID based on thread
    Ok(hipStreamPerThread as HipStreamId)
}

pub fn current_stream_sync() -> Result<(), HipError> {
    check(unsafe { hipStreamSynchronize(hipStreamPerThread) })
}

#[derive(Debug)]
pub enum HipEventStatus {
    Completed,
    NotReady,
    Error(HipError),
}

impl PartialEq for HipEventStatus {
    fn eq(&self, other: &Self) -> bool {
        use HipEventStatus::*;
        matches!((self, other), (Completed, Completed) | (NotReady, NotReady))
    }
}

impl Eq for HipEventStatus {}

impl PartialOrd for HipEventStatus {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

// Completed < NotReady < Error
impl Ord for HipEventStatus {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        use std::cmp::Ordering;

        use HipEventStatus::*;

        match (self, other) {
            (Completed, Completed) => Ordering::Equal,
            (Completed, _) => Ordering::Less,
            (_, Completed) => Ordering::Greater,
            (NotReady, NotReady) => Ordering::Equal,
            (NotReady, Error(_)) => Ordering::Less,
            (Error(_), NotReady) => Ordering::Greater,
            (Error(_), Error(_)) => Ordering::Equal,
        }
    }
}

#[derive(Debug, Clone)]
pub struct HipEvent {
    event: hipEvent_t,
}

pub fn default_stream_wait(event: &HipEvent) -> Result<(), HipError> {
    check(unsafe { hipStreamWaitEvent(hipStreamPerThread, event.event, 0) })
}

unsafe impl Send for HipEvent {}
unsafe impl Sync for HipEvent {}

impl HipEvent {
    pub fn new() -> Result<Self, HipError> {
        let mut event: hipEvent_t = std::ptr::null_mut();
        check(unsafe { hipEventCreate(&mut event) })?;
        Ok(Self { event })
    }

    /// # Safety
    /// The caller must ensure that `stream` is a valid stream.
    pub unsafe fn record(&self, stream: hipStream_t) -> Result<(), HipError> {
        check(hipEventRecord(self.event, stream))
    }

    pub fn record_on_this(&self) -> Result<(), HipError> {
        check(unsafe { hipEventRecord(self.event, hipStreamPerThread) })
    }

    pub fn synchronize(&self) -> Result<(), HipError> {
        check(unsafe { hipEventSynchronize(self.event) })
    }

    /// # Safety
    /// The caller must ensure that `stream` is a valid stream.
    pub unsafe fn record_and_wait(&self, stream: hipStream_t) -> Result<(), HipError> {
        self.record(stream)?;
        check(hipEventSynchronize(self.event))
    }

    pub fn status(&self) -> HipEventStatus {
        let status = unsafe { hipEventQuery(self.event) };
        match status {
            0 => HipEventStatus::Completed,   // hipSuccess
            600 => HipEventStatus::NotReady,  // hipErrorNotReady
            _ => HipEventStatus::Error(HipError::new(status)),
        }
    }

    pub fn completed(&self) -> bool {
        self.status() == HipEventStatus::Completed
    }
}

impl Drop for HipEvent {
    fn drop(&mut self) {
        unsafe { hipEventDestroy(self.event) };
    }
}

/// A GPU-aware span that collects a gauge metric using HIP events.
pub fn gpu_metrics_span<R, F: FnOnce() -> R>(
    name: impl Into<Cow<'static, str>>,
    f: F,
) -> Result<R, HipError> {
    let start = HipEvent::new()?;
    let stop = HipEvent::new()?;
    unsafe {
        check(hipEventRecord(start.event, hipStreamPerThread))?;
    }
    let res = f();
    unsafe { stop.record_and_wait(hipStreamPerThread)? };

    let mut elapsed_ms = 0f32;
    unsafe {
        check(hipEventElapsedTime(
            &mut elapsed_ms,
            start.event,
            stop.event,
        ))?
    };

    metrics::gauge!(name.into()).set(elapsed_ms as f64);
    Ok(res)
}
