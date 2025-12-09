use std::{env, path::Path, process::Command};

/// HIP builder configuration
#[derive(Debug, Clone)]
pub struct HipBuilder {
    include_paths: Vec<String>,
    source_files: Vec<String>,
    watch_paths: Vec<String>,
    watch_globs: Vec<String>,
    library_name: String,
    hip_arch: Vec<String>,
    hip_opt_level: Option<String>,
    custom_flags: Vec<String>,
    link_libraries: Vec<String>,
    link_search_paths: Vec<String>,
}

impl Default for HipBuilder {
    fn default() -> Self {
        // Determine library search path: HIP_PATH/lib -> ROCM_PATH/lib -> /opt/rocm/lib
        let mut link_search_paths = Vec::new();
        if let Ok(hip_path) = env::var("HIP_PATH") {
            link_search_paths.push(format!("{}/lib", hip_path));
        }
        if let Ok(rocm_path) = env::var("ROCM_PATH") {
            link_search_paths.push(format!("{}/lib", rocm_path));
        }
        if link_search_paths.is_empty() {
            link_search_paths.push("/opt/rocm/lib".to_string());
        }

        Self {
            include_paths: Vec::new(),
            source_files: Vec::new(),
            watch_paths: vec!["build.rs".to_string()],
            watch_globs: Vec::new(),
            library_name: String::new(),
            hip_arch: Vec::new(),
            hip_opt_level: None,
            // Only --std=c++17 by default. Do NOT include CUDA-specific flags like
            // --expt-relaxed-constexpr, -Xfatbin, --default-stream=per-thread
            custom_flags: vec!["--std=c++17".to_string()],
            link_libraries: vec!["amdhip64".to_string()],
            link_search_paths,
        }
    }
}

impl HipBuilder {
    /// Create a new HipBuilder
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the library name (useful when cloning from a template)
    pub fn library_name(mut self, name: &str) -> Self {
        self.library_name = name.to_string();
        self
    }

    /// Add include path
    pub fn include<P: AsRef<Path>>(mut self, path: P) -> Self {
        let path_str = path.as_ref().to_string_lossy().to_string();
        self.include_paths.push(path_str.clone());
        self.watch_paths.push(path_str);
        self
    }

    /// Add include path from another crate's exported include
    pub fn include_from_dep(mut self, dep_env_var: &str) -> Self {
        if let Ok(path) = env::var(dep_env_var) {
            self.include_paths.push(path);
        }
        self
    }

    /// Add source file
    pub fn file<P: AsRef<Path>>(mut self, path: P) -> Self {
        let path_str = path.as_ref().to_string_lossy().to_string();
        self.source_files.push(path_str.clone());
        self.watch_paths.push(path_str);
        self
    }

    /// Add multiple source files
    pub fn files<P: AsRef<Path>, I: IntoIterator<Item = P>>(mut self, paths: I) -> Self {
        for path in paths {
            let path_str = path.as_ref().to_string_lossy().to_string();
            self.source_files.push(path_str.clone());
            self.watch_paths.push(path_str);
        }
        self
    }

    /// Add multiple source files matching a glob pattern
    /// Matches .hip, .cpp, and .cu files
    pub fn files_from_glob(mut self, pattern: &str) -> Self {
        self.watch_globs.push(pattern.to_string());
        for path in glob::glob(pattern).expect("Invalid glob pattern").flatten() {
            if path.is_file() {
                let ext = path.extension().and_then(|e| e.to_str());
                if matches!(ext, Some("hip") | Some("cpp") | Some("cu")) {
                    self.source_files.push(path.to_string_lossy().to_string());
                }
            }
        }
        self
    }

    /// Watch a specific path for changes
    pub fn watch<P: AsRef<Path>>(mut self, path: P) -> Self {
        self.watch_paths
            .push(path.as_ref().to_string_lossy().to_string());
        self
    }

    /// Watch paths matching a glob pattern
    pub fn watch_glob(mut self, pattern: &str) -> Self {
        self.watch_globs.push(pattern.to_string());
        self
    }

    /// Set HIP architecture (e.g., "gfx906", "gfx1100")
    pub fn hip_arch(mut self, arch: &str) -> Self {
        self.hip_arch = vec![arch.to_string()];
        self
    }

    /// Set multiple HIP architectures
    pub fn hip_archs(mut self, archs: Vec<&str>) -> Self {
        self.hip_arch = archs.iter().map(|s| s.to_string()).collect();
        self
    }

    /// Set HIP optimization level (0-3)
    pub fn hip_opt_level(mut self, level: u8) -> Self {
        self.hip_opt_level = Some(level.to_string());
        self
    }

    /// Add custom compiler flag
    pub fn flag(mut self, flag: &str) -> Self {
        self.custom_flags.push(flag.to_string());
        self
    }

    /// Add library to link
    pub fn link_lib(mut self, lib: &str) -> Self {
        self.link_libraries.push(lib.to_string());
        self
    }

    /// Add library search path
    pub fn link_search<P: AsRef<Path>>(mut self, path: P) -> Self {
        self.link_search_paths
            .push(path.as_ref().to_string_lossy().to_string());
        self
    }

    /// Build the HIP library
    pub fn build(self) {
        // Validation
        self.validate();

        // Set up rerun conditions
        self.setup_rerun_conditions();

        // Get or detect HIP architecture
        let hip_archs = self.get_hip_arch();

        // Create cc::Build - use compiler("hipcc") instead of .cuda(true)
        // to avoid NVCC-specific flag injection
        let mut builder = cc::Build::new();
        let hipcc_path = find_hipcc().expect(
            "hipcc not found. Make sure ROCm is installed and either hipcc is in PATH, \
             or set HIP_PATH/ROCM_PATH environment variable."
        );
        builder.compiler(&hipcc_path);

        // Handle HIP_DEBUG=1
        self.handle_debug_shortcuts(&mut builder);

        // Get optimization level
        let hip_opt_level = self.get_hip_opt_level();

        // Add include paths
        for include in &self.include_paths {
            builder.include(include);
        }

        // Add HIP_PATH/ROCM_PATH include if available
        if let Ok(hip_path) = env::var("HIP_PATH") {
            builder.include(format!("{}/include", hip_path));
        } else if let Ok(rocm_path) = env::var("ROCM_PATH") {
            builder.include(format!("{}/include", rocm_path));
        }

        // Add custom flags
        for flag in &self.custom_flags {
            builder.flag(flag);
        }

        // Add offload-arch for each architecture
        for arch in &hip_archs {
            builder.flag(&format!("--offload-arch={}", arch));
        }

        // Add parallel jobs flag
        builder.flag(&hipcc_parallel_jobs());

        // Set optimization and debug flags
        if hip_opt_level == "0" {
            builder.debug(true).flag("-O0");
        } else {
            builder.debug(false).flag(&format!("-O{}", hip_opt_level));
        }

        // Add source files
        for file in &self.source_files {
            builder.file(file);
        }

        // Compile
        builder.compile(&self.library_name);
    }

    /// Validate the builder configuration
    fn validate(&self) {
        if self.library_name.is_empty() {
            panic!(
                "Library name must be set using .library_name(\"name\") before calling .build()"
            );
        }

        if self.source_files.is_empty() {
            panic!("At least one source file must be added using .file() or .files() before calling .build()");
        }

        // Validate that source files exist (optional, but helpful)
        for file in &self.source_files {
            if !Path::new(file).exists() {
                eprintln!("cargo:warning=HIP source file does not exist: {}", file);
            }
        }

        // Validate include paths exist (optional warning)
        for include in &self.include_paths {
            if !Path::new(include).exists() {
                eprintln!("cargo:warning=Include path does not exist: {}", include);
            }
        }
    }

    pub fn emit_link_directives(&self) {
        for path in &self.link_search_paths {
            println!("cargo:rustc-link-search=native={}", path);
        }
        for lib in &self.link_libraries {
            println!("cargo:rustc-link-lib={}", lib);
        }
    }

    fn setup_rerun_conditions(&self) {
        // Standard rerun conditions for HIP
        println!("cargo:rerun-if-env-changed=HIP_ARCH");
        println!("cargo:rerun-if-env-changed=HIP_OPT_LEVEL");
        println!("cargo:rerun-if-env-changed=HIP_DEBUG");
        println!("cargo:rerun-if-env-changed=HIP_THREADS");
        println!("cargo:rerun-if-env-changed=HIP_PATH");
        println!("cargo:rerun-if-env-changed=ROCM_PATH");

        // Watch specific paths
        for path in &self.watch_paths {
            println!("cargo:rerun-if-changed={}", path);
        }

        // Watch glob patterns
        for pattern in &self.watch_globs {
            watch_glob(pattern);
        }
    }

    fn get_hip_arch(&self) -> Vec<String> {
        if !self.hip_arch.is_empty() {
            return self.hip_arch.clone();
        }

        // Check environment variable (comma-separated for multiple archs)
        if let Ok(env_archs) = env::var("HIP_ARCH") {
            return env_archs
                .split(',')
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect();
        }

        // Auto-detect current GPU
        vec![detect_hip_arch()]
    }

    fn get_hip_opt_level(&self) -> String {
        if let Some(level) = &self.hip_opt_level {
            return level.clone();
        }

        env::var("HIP_OPT_LEVEL").unwrap_or_else(|_| "3".to_string())
    }

    fn handle_debug_shortcuts(&self, builder: &mut cc::Build) {
        if env::var("HIP_DEBUG").map(|v| v == "1").unwrap_or(false) {
            env::set_var("HIP_OPT_LEVEL", "0");

            println!("cargo:warning=HIP_DEBUG=1 → Enabling comprehensive debugging:");
            println!("cargo:warning=  → HIP_OPT_LEVEL=0 (no optimization)");
            println!("cargo:warning=  → Debug symbols enabled");
            println!("cargo:warning=  → HIP_DEBUG macro defined for preprocessor");

            builder.flag("-g"); // Debug symbols
            builder.flag("-O0"); // No optimization
            builder.define("HIP_DEBUG", "1"); // Define HIP_DEBUG macro
        }
    }
}

/// Check if HIP is available on the system.
/// Checks for hipcc in PATH, then standard ROCm locations.
pub fn hip_available() -> bool {
    // First check if hipcc is in PATH
    if Command::new("hipcc").arg("--version").output().is_ok() {
        return true;
    }

    // Check standard ROCm locations
    let standard_paths = [
        "/opt/rocm/bin/hipcc",
        "/usr/local/rocm/bin/hipcc",
    ];

    for path in &standard_paths {
        if std::path::Path::new(path).exists() {
            return true;
        }
    }

    // Check HIP_PATH and ROCM_PATH environment variables
    if let Ok(hip_path) = env::var("HIP_PATH") {
        let hipcc = format!("{}/bin/hipcc", hip_path);
        if std::path::Path::new(&hipcc).exists() {
            return true;
        }
    }

    if let Ok(rocm_path) = env::var("ROCM_PATH") {
        let hipcc = format!("{}/bin/hipcc", rocm_path);
        if std::path::Path::new(&hipcc).exists() {
            return true;
        }
    }

    false
}

/// Get the path to hipcc, checking PATH first then standard locations.
pub fn find_hipcc() -> Option<String> {
    // First check if hipcc is in PATH
    if Command::new("hipcc").arg("--version").output().is_ok() {
        return Some("hipcc".to_string());
    }

    // Check HIP_PATH and ROCM_PATH environment variables first
    if let Ok(hip_path) = env::var("HIP_PATH") {
        let hipcc = format!("{}/bin/hipcc", hip_path);
        if std::path::Path::new(&hipcc).exists() {
            return Some(hipcc);
        }
    }

    if let Ok(rocm_path) = env::var("ROCM_PATH") {
        let hipcc = format!("{}/bin/hipcc", rocm_path);
        if std::path::Path::new(&hipcc).exists() {
            return Some(hipcc);
        }
    }

    // Check standard ROCm locations
    let standard_paths = [
        "/opt/rocm/bin/hipcc",
        "/usr/local/rocm/bin/hipcc",
    ];

    for path in &standard_paths {
        if std::path::Path::new(path).exists() {
            return Some(path.to_string());
        }
    }

    None
}

/// Find hipconfig binary path
fn find_hipconfig() -> Option<String> {
    // Check if hipconfig is in PATH
    if Command::new("hipconfig").arg("--version").output().is_ok() {
        return Some("hipconfig".to_string());
    }

    // Check standard ROCm locations
    let standard_paths = [
        "/opt/rocm/bin/hipconfig",
        "/usr/local/rocm/bin/hipconfig",
    ];

    for path in &standard_paths {
        if std::path::Path::new(path).exists() {
            return Some(path.to_string());
        }
    }

    // Check HIP_PATH and ROCM_PATH
    if let Ok(hip_path) = env::var("HIP_PATH") {
        let hipconfig = format!("{}/bin/hipconfig", hip_path);
        if std::path::Path::new(&hipconfig).exists() {
            return Some(hipconfig);
        }
    }

    if let Ok(rocm_path) = env::var("ROCM_PATH") {
        let hipconfig = format!("{}/bin/hipconfig", rocm_path);
        if std::path::Path::new(&hipconfig).exists() {
            return Some(hipconfig);
        }
    }

    None
}

/// Detect HIP architecture using hipconfig (preferred) or rocminfo (fallback)
pub fn detect_hip_arch() -> String {
    // Try hipconfig --amdgpu-target first (simple, maintained)
    let hipconfig = find_hipconfig();
    if let Some(hipconfig_path) = hipconfig {
        if let Ok(output) = Command::new(&hipconfig_path)
            .arg("--amdgpu-target")
            .output()
        {
            if output.status.success() {
                let arch = String::from_utf8_lossy(&output.stdout)
                    .trim()
                    .to_string();
                if !arch.is_empty() && arch.starts_with("gfx") {
                    // Set both cargo env and process env
                    println!("cargo:rustc-env=HIP_ARCH={}", arch);
                    env::set_var("HIP_ARCH", &arch);
                    return arch;
                }
            }
        }
    }

    // Fallback to rocminfo - parse cautiously as output can vary
    if let Ok(output) = Command::new("rocminfo").output() {
        if output.status.success() {
            let stdout = String::from_utf8_lossy(&output.stdout);
            // Look for lines like "  Name:                    gfx1100"
            for line in stdout.lines() {
                let line = line.trim();
                if line.starts_with("Name:") {
                    let parts: Vec<&str> = line.splitn(2, ':').collect();
                    if parts.len() == 2 {
                        let name = parts[1].trim();
                        if name.starts_with("gfx") {
                            println!("cargo:rustc-env=HIP_ARCH={}", name);
                            env::set_var("HIP_ARCH", name);
                            return name.to_string();
                        }
                    }
                }
            }
        }
    }

    panic!(
        "Failed to detect HIP architecture. Make sure ROCm is installed and either \
         'hipconfig --amdgpu-target' or 'rocminfo' works. Alternatively, set HIP_ARCH \
         environment variable (e.g., HIP_ARCH=gfx1100)."
    );
}

/// Calculate optimal number of parallel hipcc jobs
pub fn hipcc_parallel_jobs() -> String {
    let threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);

    let threads = env::var("HIP_THREADS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(threads);

    format!("-j{}", threads)
}

/// Watch files matching a glob pattern
fn watch_glob(pattern: &str) {
    for path in glob::glob(pattern).expect("Invalid glob pattern").flatten() {
        if path.is_file() {
            println!("cargo:rerun-if-changed={}", path.display());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hip_available() {
        // This test just ensures the function doesn't panic
        let _ = hip_available();
    }

    #[test]
    fn test_hipcc_parallel_jobs() {
        let jobs = hipcc_parallel_jobs();
        assert!(jobs.starts_with("-j"));
    }

    #[test]
    fn test_default_builder() {
        let builder = HipBuilder::new();
        assert!(builder.library_name.is_empty());
        assert!(builder.source_files.is_empty());
        assert!(builder.custom_flags.contains(&"--std=c++17".to_string()));
        assert!(builder.link_libraries.contains(&"amdhip64".to_string()));
    }

    #[test]
    fn test_builder_chain() {
        let builder = HipBuilder::new()
            .library_name("test_lib")
            .hip_arch("gfx1100")
            .hip_opt_level(2)
            .flag("-Wall");

        assert_eq!(builder.library_name, "test_lib");
        assert_eq!(builder.hip_arch, vec!["gfx1100"]);
        assert_eq!(builder.hip_opt_level, Some("2".to_string()));
        assert!(builder.custom_flags.contains(&"-Wall".to_string()));
    }
}
