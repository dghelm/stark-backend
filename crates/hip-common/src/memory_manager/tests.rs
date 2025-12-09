use super::*;

#[test]
fn test_d_malloc_and_free() {
    let size = 1024;
    let ptr = d_malloc(size).expect("Allocation failed");
    assert!(!ptr.is_null());

    unsafe {
        d_free(ptr).expect("Free failed");
    }
}

#[test]
fn test_multiple_allocations() {
    let sizes = [512, 1024, 2048, 4096];
    let mut ptrs = Vec::new();

    for &size in &sizes {
        let ptr = d_malloc(size).expect("Allocation failed");
        assert!(!ptr.is_null());
        ptrs.push(ptr);
    }

    for ptr in ptrs {
        unsafe {
            d_free(ptr).expect("Free failed");
        }
    }
}

#[test]
fn test_mem_tracker() {
    let tracker = MemTracker::start("test");

    let ptr = d_malloc(4096).expect("Allocation failed");
    assert!(!ptr.is_null());

    tracker.tracing_info(Some("after alloc"));

    unsafe {
        d_free(ptr).expect("Free failed");
    }

    // Tracker will log on drop
}
