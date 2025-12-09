use crate::error::HipError;

/// HIP device pointer type (same as CUDA)
#[allow(non_camel_case_types)]
pub(super) type hipDeviceptr_t = u64;

/// HIP memory generic allocation handle
#[allow(non_camel_case_types)]
pub(super) type hipMemGenericAllocationHandle_t = u64;

// HIP VMM functions - these are available in ROCm 5.0+
// The function names mirror CUDA's driver API but with hip prefix
extern "C" {
    // Check if VMM is supported on the device
    fn hipMemGetAllocationGranularity(
        granularity: *mut usize,
        prop: *const HipMemAllocationProp,
        option: u32,
    ) -> u32;

    // Reserve virtual address space
    fn hipMemAddressReserve(
        ptr: *mut hipDeviceptr_t,
        size: usize,
        alignment: usize,
        addr: hipDeviceptr_t,
        flags: u64,
    ) -> u32;

    // Free virtual address space
    fn hipMemAddressFree(ptr: hipDeviceptr_t, size: usize) -> u32;

    // Create physical memory allocation
    fn hipMemCreate(
        handle: *mut hipMemGenericAllocationHandle_t,
        size: usize,
        prop: *const HipMemAllocationProp,
        flags: u64,
    ) -> u32;

    // Map physical memory to virtual address
    fn hipMemMap(
        ptr: hipDeviceptr_t,
        size: usize,
        offset: usize,
        handle: hipMemGenericAllocationHandle_t,
        flags: u64,
    ) -> u32;

    // Unmap memory from virtual address
    fn hipMemUnmap(ptr: hipDeviceptr_t, size: usize) -> u32;

    // Release physical memory handle
    fn hipMemRelease(handle: hipMemGenericAllocationHandle_t) -> u32;

    // Set memory access flags
    fn hipMemSetAccess(
        ptr: hipDeviceptr_t,
        size: usize,
        desc: *const HipMemAccessDesc,
        count: usize,
    ) -> u32;
}

/// Memory allocation properties for HIP VMM
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub(super) struct HipMemAllocationProp {
    pub allocation_type: u32, // hipMemAllocationType
    pub handle_type: u32,     // hipMemAllocationHandleType
    pub location_type: u32,   // hipMemLocationType
    pub location_id: i32,     // device ID
    pub win32_handle_meta_data: *const std::ffi::c_void,
    pub reserved: [u64; 8],
}

impl Default for HipMemAllocationProp {
    fn default() -> Self {
        Self {
            allocation_type: 1, // hipMemAllocationTypePinned
            handle_type: 0,     // hipMemHandleTypeNone
            location_type: 1,   // hipMemLocationTypeDevice
            location_id: 0,
            win32_handle_meta_data: std::ptr::null(),
            reserved: [0; 8],
        }
    }
}

/// Memory access description for HIP VMM
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub(super) struct HipMemAccessDesc {
    pub location_type: u32,
    pub location_id: i32,
    pub flags: u32,
}

impl HipMemAccessDesc {
    pub fn new_read_write(device_id: i32) -> Self {
        Self {
            location_type: 1, // hipMemLocationTypeDevice
            location_id: device_id,
            flags: 3, // hipMemAccessFlagsProtReadWrite
        }
    }
}

/// Check if HIP VMM is supported on the given device
pub(super) unsafe fn vpmm_check_support(device_ordinal: i32) -> Result<(), HipError> {
    let mut granularity: usize = 0;
    let mut prop = HipMemAllocationProp::default();
    prop.location_id = device_ordinal;

    // Try to get granularity - if this fails, VMM is not supported
    let result = hipMemGetAllocationGranularity(
        &mut granularity,
        &prop,
        1, // hipMemAllocationGranularityRecommended
    );
    HipError::from_result(result)
}

/// Get the minimum allocation granularity for the device
pub(super) unsafe fn vpmm_min_granularity(device_ordinal: i32) -> Result<usize, HipError> {
    let mut granularity: usize = 0;
    let mut prop = HipMemAllocationProp::default();
    prop.location_id = device_ordinal;

    HipError::from_result(hipMemGetAllocationGranularity(
        &mut granularity,
        &prop,
        0, // hipMemAllocationGranularityMinimum
    ))?;
    Ok(granularity)
}

/// Reserve virtual address space
pub(super) unsafe fn vpmm_reserve(size: usize, align: usize) -> Result<hipDeviceptr_t, HipError> {
    let mut va_base: hipDeviceptr_t = 0;
    HipError::from_result(hipMemAddressReserve(&mut va_base, size, align, 0, 0))?;
    Ok(va_base)
}

/// Release virtual address space
pub(super) unsafe fn vpmm_release_va(base: hipDeviceptr_t, size: usize) -> Result<(), HipError> {
    HipError::from_result(hipMemAddressFree(base, size))
}

/// Create physical memory allocation
pub(super) unsafe fn vpmm_create_physical(
    device_ordinal: i32,
    bytes: usize,
) -> Result<hipMemGenericAllocationHandle_t, HipError> {
    let mut handle: hipMemGenericAllocationHandle_t = 0;
    let mut prop = HipMemAllocationProp::default();
    prop.location_id = device_ordinal;

    HipError::from_result(hipMemCreate(&mut handle, bytes, &prop, 0))?;
    Ok(handle)
}

/// Map physical memory to virtual address
pub(super) unsafe fn vpmm_map(
    va: hipDeviceptr_t,
    bytes: usize,
    handle: hipMemGenericAllocationHandle_t,
) -> Result<(), HipError> {
    HipError::from_result(hipMemMap(va, bytes, 0, handle, 0))
}

/// Set access flags for mapped memory
pub(super) unsafe fn vpmm_set_access(
    va: hipDeviceptr_t,
    bytes: usize,
    device_ordinal: i32,
) -> Result<(), HipError> {
    let desc = HipMemAccessDesc::new_read_write(device_ordinal);
    HipError::from_result(hipMemSetAccess(va, bytes, &desc, 1))
}

/// Unmap memory from virtual address
pub(super) unsafe fn vpmm_unmap(va: hipDeviceptr_t, bytes: usize) -> Result<(), HipError> {
    HipError::from_result(hipMemUnmap(va, bytes))
}

/// Release physical memory handle
pub(super) unsafe fn vpmm_release(handle: hipMemGenericAllocationHandle_t) -> Result<(), HipError> {
    HipError::from_result(hipMemRelease(handle))
}
