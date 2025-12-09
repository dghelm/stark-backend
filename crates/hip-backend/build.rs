use std::process::exit;

use openvm_hip_builder::{hip_available, HipBuilder};

fn main() {
    if !hip_available() {
        eprintln!("cargo:warning=HIP/ROCm is not available");
        exit(1);
    }

    let common = HipBuilder::new().include_from_dep("DEP_HIP_COMMON_INCLUDE");

    common.emit_link_directives();

    // TODO: Build HIP kernels once they are ported from CUDA
    // For now, we just emit link directives for the HIP runtime.
    //
    // The kernel porting requires:
    // 1. Port fp.h and fpext.h (field arithmetic with HIP-compatible intrinsics)
    // 2. Port launcher.cuh (kernel launch helpers)
    // 3. Port all .cu files to .hip equivalents
    //
    // common
    //     .clone()
    //     .library_name("stark_backend_hip")
    //     .include("hip/include")
    //     .files_from_glob("hip/src/*.hip")
    //     .build();
    //
    // common
    //     .clone()
    //     .library_name("supra_ntt_hip")
    //     .include("hip/supra/include")
    //     .files_from_glob("hip/supra/*.hip")
    //     .build();
}
