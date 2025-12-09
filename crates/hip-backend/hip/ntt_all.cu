// HIP NTT unified compilation unit
//
// The NTT kernels use __constant__ device memory symbols that are defined in
// ntt_params.cu but referenced in ntt.cu. This requires all NTT files to be
// compiled together as a single translation unit to avoid RDC linking issues.
//
// Without RDC: Each .cu file is compiled to a self-contained object with its
// own device code. Cross-TU __constant__ symbol references fail at link time.
//
// Solution: Include all NTT sources here so they compile as one unit.

// Order matters: ntt_params.cu defines the symbols, must come before ntt.cu
#include "ntt_params.cu"
#include "ntt.cu"
#include "ntt_bitrev.cu"
