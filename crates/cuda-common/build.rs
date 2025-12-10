use std::{env, path::PathBuf};

use openvm_cuda_builder::{cuda_available, CudaBuilder};

fn main() {
    // Always export the include path - needed for shared headers (launcher.cuh, fp.h, etc.)
    // Used by both CUDA and HIP builds
    let include_path = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap()).join("include");
    println!("cargo:include={}", include_path.display()); // -> DEP_CUDA_COMMON_INCLUDE

    // Only compile CUDA-specific code (vpmm_shim) when CUDA is available
    if cuda_available() {
        println!("cargo:rerun-if-changed=cuda");
        println!("cargo:rerun-if-changed=include");

        let builder = CudaBuilder::new()
            .library_name("vmm_shim")
            .flag("-Xcompiler=-fPIC")
            .file("cuda/src/vpmm_shim.cu");

        builder.clone().build();
        builder.emit_link_directives();
    } else {
        // CUDA not available - only headers will be used (e.g., for HIP builds)
        println!("cargo:warning=CUDA is not available, skipping CUDA compilation (headers-only mode)");
    }
}
