//! Smoke test for hip-builder.
//!
//! This test verifies that HipBuilder can successfully compile a trivial HIP file
//! using hipcc. It requires ROCm to be installed on the system.

use openvm_hip_builder::{detect_hip_arch, hip_available, HipBuilder};
use std::env;
use std::path::PathBuf;

/// Test that hipcc is available on the system
#[test]
fn test_hipcc_available() {
    if !hip_available() {
        eprintln!("Skipping test: hipcc not available");
        return;
    }
    assert!(hip_available());
}

/// Test that we can detect the HIP architecture
#[test]
fn test_arch_detection() {
    if !hip_available() {
        eprintln!("Skipping test: hipcc not available");
        return;
    }

    let arch = detect_hip_arch();
    assert!(
        arch.starts_with("gfx"),
        "Expected architecture to start with 'gfx', got: {}",
        arch
    );
}

/// Smoke test: compile a dummy HIP file
#[test]
fn test_compile_dummy_hip() {
    if !hip_available() {
        eprintln!("Skipping test: hipcc not available");
        return;
    }

    // Get the path to the fixtures directory
    let manifest_dir = env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR not set");
    let fixture_path = PathBuf::from(&manifest_dir)
        .join("tests")
        .join("fixtures")
        .join("dummy.hip");

    assert!(
        fixture_path.exists(),
        "Fixture file not found: {}",
        fixture_path.display()
    );

    // Set OUT_DIR for the test (normally set by cargo during build)
    let out_dir = PathBuf::from(&manifest_dir).join("target").join("test-out");
    std::fs::create_dir_all(&out_dir).expect("Failed to create output directory");
    env::set_var("OUT_DIR", &out_dir);

    // Build the dummy HIP file
    HipBuilder::new()
        .library_name("dummy_hip_test")
        .file(&fixture_path)
        .build();

    // Check that the library was created
    let lib_path = out_dir.join("libdummy_hip_test.a");
    assert!(
        lib_path.exists(),
        "Expected library not found: {}",
        lib_path.display()
    );

    // Clean up
    let _ = std::fs::remove_file(&lib_path);
}
