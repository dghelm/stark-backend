//! HIP device configuration and management.
//!
//! This module mirrors `cuda-backend/src/gpu_device.rs` for AMD GPUs.

use derivative::Derivative;
use openvm_hip_common::common::get_device;
use openvm_stark_sdk::config::FriParameters;
use p3_baby_bear::BabyBear;
use p3_commit::TwoAdicMultiplicativeCoset;
use p3_field::FieldAlgebra;
use p3_util::log2_strict_usize;

use crate::fri_log_up::FriLogUpPhaseGpu;

/// Configuration for HIP device proving.
#[derive(Derivative, derive_new::new, Clone, Copy, Debug)]
pub struct HipConfig {
    pub fri: FriParameters,
    pub shift: BabyBear,
}

/// HIP device handle for GPU proving operations.
#[derive(Derivative, Clone, Debug)]
pub struct HipDevice {
    pub config: HipConfig,
    pub id: u32,
    rap_phase_seq: Option<FriLogUpPhaseGpu>,
}

impl HipDevice {
    /// Create a new HIP device with the given configuration.
    pub fn new(config: HipConfig, rap_phase_seq: Option<FriLogUpPhaseGpu>) -> Self {
        Self {
            config,
            id: get_device().unwrap() as u32,
            rap_phase_seq,
        }
    }

    pub fn rap_phase_seq(&self) -> &FriLogUpPhaseGpu {
        self.rap_phase_seq
            .as_ref()
            .expect("FriLogUpPhaseGpu is not initialized")
    }

    /// Get the natural domain for a given degree.
    pub fn natural_domain_for_degree(&self, degree: usize) -> TwoAdicMultiplicativeCoset<BabyBear> {
        let log_n = log2_strict_usize(degree);
        TwoAdicMultiplicativeCoset {
            log_n,
            shift: BabyBear::ONE,
        }
    }
}
