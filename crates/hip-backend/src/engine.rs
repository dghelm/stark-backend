//! HIP STARK engine implementation.
//!
//! This module mirrors `cuda-backend/src/engine.rs` for AMD GPUs.
//! Currently a stub - full implementation requires ported kernels.

use openvm_hip_common::memory_manager::MemTracker;
#[cfg(feature = "touchemall")]
use openvm_stark_backend::prover::types::AirProvingContext;
use openvm_stark_backend::{
    config::StarkGenericConfig,
    proof::Proof,
    prover::{
        coordinator::Coordinator,
        types::{DeviceMultiStarkProvingKey, ProvingContext},
        Prover,
    },
};
use openvm_stark_sdk::{
    config::{
        baby_bear_poseidon2::{config_from_perm, default_perm, BabyBearPoseidon2Config},
        fri_params::SecurityParameters,
        log_up_params::log_up_security_params_baby_bear_100_bits,
        FriParameters,
    },
    engine::{StarkEngine, StarkFriEngine},
};
use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
use p3_field::Field;

use crate::{
    fri_log_up::FriLogUpPhaseGpu,
    hip_device::{HipConfig, HipDevice},
    prelude::{SC, WIDTH},
    prover_backend::HipBackend,
};

/// Multi-trace STARK prover type alias for HIP backend.
pub type MultiTraceStarkProverHip = Coordinator<SC, HipBackend, HipDevice>;

/// HIP-based BabyBear Poseidon2 STARK engine.
///
/// This is the AMD GPU equivalent of `GpuBabyBearPoseidon2Engine`.
pub struct HipBabyBearPoseidon2Engine {
    device: HipDevice,
    config: BabyBearPoseidon2Config,
    perm: Poseidon2BabyBear<WIDTH>,
}

impl StarkFriEngine for HipBabyBearPoseidon2Engine {
    fn new(fri_params: FriParameters) -> Self {
        let perm = default_perm();
        let log_up_params = log_up_security_params_baby_bear_100_bits();
        Self {
            device: HipDevice::new(
                HipConfig::new(fri_params, BabyBear::GENERATOR),
                Some(FriLogUpPhaseGpu::new(log_up_params.clone())),
            ),
            config: config_from_perm(
                &perm,
                SecurityParameters {
                    fri_params,
                    log_up_params,
                },
            ),
            perm,
        }
    }

    fn fri_params(&self) -> FriParameters {
        self.device.config.fri
    }
}

impl StarkEngine for HipBabyBearPoseidon2Engine {
    type SC = BabyBearPoseidon2Config;
    type PB = HipBackend;
    type PD = HipDevice;

    fn config(&self) -> &SC {
        &self.config
    }

    fn max_constraint_degree(&self) -> Option<usize> {
        Some(self.device.config.fri.max_constraint_degree())
    }

    fn new_challenger(&self) -> <SC as StarkGenericConfig>::Challenger {
        <SC as StarkGenericConfig>::Challenger::new(self.perm.clone())
    }

    fn device(&self) -> &Self::PD {
        &self.device
    }

    fn prover(&self) -> MultiTraceStarkProverHip {
        MultiTraceStarkProverHip::new(
            HipBackend::default(),
            self.device.clone(),
            self.new_challenger(),
        )
    }

    fn prove(
        &self,
        pk: &DeviceMultiStarkProvingKey<Self::PB>,
        ctx: ProvingContext<Self::PB>,
    ) -> Proof<Self::SC> {
        let mut mem = MemTracker::start("prove");
        mem.reset_peak();

        let mpk_view = pk.view(ctx.air_ids());
        #[cfg(feature = "touchemall")]
        {
            for (air_id, air_ctx) in ctx.per_air.iter() {
                check_trace_validity(air_ctx, &pk.per_air[*air_id].air_name);
            }
        }
        let mut prover = self.prover();
        let proof = prover.prove(mpk_view, ctx);
        proof.into()
    }
}

#[cfg(feature = "touchemall")]
pub fn check_trace_validity(proving_ctx: &AirProvingContext<HipBackend>, name: &str) {
    use openvm_hip_common::copy::MemCopyD2H;
    use openvm_stark_backend::prover::hal::MatrixDimensions;

    use crate::types::F;

    let trace = proving_ctx.common_main.as_ref().unwrap();
    let height = trace.height();
    let width = trace.width();
    let trace = trace.to_host().unwrap();
    for r in 0..height {
        for c in 0..width {
            let value = trace[c * height + r];
            let value_u32 = unsafe { *(&value as *const F as *const u32) };
            assert!(
                value_u32 != 0xffffffff,
                "potentially untouched value at ({r}, {c}) of a trace of size {height}x{width} for air {name}"
            );
        }
    }
}
