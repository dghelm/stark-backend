use std::{env, path::PathBuf, process::exit};

use openvm_hip_builder::{hip_available, HipBuilder};

fn main() {
    if hip_available() {
        println!("cargo:rerun-if-changed=hip");
        println!("cargo:rerun-if-changed=include");

        // For now, we don't have a VMM shim to build for HIP.
        // The HIP VMM APIs are used directly via cubecl-hip-sys.
        // If we need a custom shim later, uncomment and adapt:
        //
        // let builder = HipBuilder::new()
        //     .library_name("vmm_shim")
        //     .flag("-fPIC")
        //     .file("hip/src/vpmm_shim.cpp");
        //
        // builder.clone().build();
        // builder.emit_link_directives();

        // Export include path for dependent crates
        let include_path = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap()).join("include");
        println!("cargo:include={}", include_path.display()); // -> DEP_HIP_COMMON_INCLUDE

        // Emit link directives for HIP runtime
        HipBuilder::new().emit_link_directives();
    } else {
        eprintln!("cargo:warning=HIP/ROCm is not available");
        exit(1);
    }
}
