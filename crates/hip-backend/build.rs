use std::process::exit;

use openvm_hip_builder::{hip_available, HipBuilder};

fn main() {
    if !hip_available() {
        eprintln!("cargo:warning=HIP/ROCm is not available");
        exit(1);
    }

    // Paths for shared CUDA/HIP source code
    // The headers in cuda-common have been ported to support both CUDA and HIP
    // via __HIPCC__ preprocessor guards
    let cuda_common_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../cuda-common");
    let cuda_backend_path =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../cuda-backend");

    // Base builder with cuda-common includes (fp.h, fpext.h, launcher.cuh, etc.)
    let common = HipBuilder::new().include(cuda_common_path.join("include").to_str().unwrap());

    common.emit_link_directives();

    // Build the main kernels from cuda-backend (ported to be HIP-compatible)
    // The .cu files use __HIPCC__ guards to provide HIP-compatible code paths
    //
    // Note: We previously used -fgpu-rdc (relocatable device code) for cross-TU
    // symbol resolution, but this requires device-linking with hipcc at link time.
    // Since Cargo uses the system linker (cc), we cannot do device linking.
    // Each .cu file must be self-contained (no cross-TU __device__ calls).
    common
        .clone()
        .library_name("stark_backend_hip")
        .include(cuda_backend_path.join("cuda/include").to_str().unwrap())
        .files_from_glob(cuda_backend_path.join("cuda/src/*.cu").to_str().unwrap())
        .build();

    // Build the NTT kernels from supra (ported to be HIP-compatible)
    //
    // The NTT code uses __constant__ device symbols defined in ntt_params.cu
    // that are referenced in ntt.cu. Without -fgpu-rdc, these cross-TU references
    // fail. We use a unified compilation unit (ntt_all.cu) that #includes all
    // NTT sources to avoid the need for RDC.
    let hip_backend_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    common
        .clone()
        .library_name("supra_ntt_hip")
        .include(
            cuda_backend_path
                .join("cuda/supra/include")
                .to_str()
                .unwrap(),
        )
        .include(cuda_backend_path.join("cuda/supra").to_str().unwrap()) // For #include "ntt_*.cu"
        .file(hip_backend_path.join("hip/ntt_all.cu").to_str().unwrap())
        .build();
}
