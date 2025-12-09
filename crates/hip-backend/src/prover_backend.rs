//! HIP prover backend implementation.
//!
//! This module mirrors `cuda-backend/src/prover_backend.rs` for AMD GPUs.
//! Currently contains stub implementations - full functionality requires ported kernels.

use std::sync::Arc;

use openvm_stark_backend::{
    config::{Com, PcsProof, PcsProverData, RapPartialProvingKey, RapPhaseSeqPartialProof, Val},
    keygen::types::MultiStarkProvingKey,
    p3_challenger::DuplexChallenger,
    p3_matrix::dense::RowMajorMatrix,
    proof::OpeningProof,
    prover::{
        hal::{
            DeviceDataTransporter, OpeningProver, ProverBackend, ProverDevice,
            QuotientCommitter, RapPartialProver, TraceCommitter,
        },
        types::{
            AirView, CommittedTraceData, DeviceMultiStarkProvingKey,
            DeviceMultiStarkProvingKeyView, DeviceStarkProvingKey, ProverDataAfterRapPhases,
        },
    },
};
use p3_baby_bear::Poseidon2BabyBear;

use crate::{
    base::DeviceMatrix,
    hip_device::HipDevice,
    prelude::*,
};

/// HIP backend implementation for STARK proving system.
///
/// This is the AMD GPU equivalent of `GpuBackend` from cuda-backend.
#[derive(Clone, Copy, Default, Debug)]
pub struct HipBackend;

impl ProverBackend for HipBackend {
    const CHALLENGE_EXT_DEGREE: u8 = 4;

    // Host Types
    type Val = F;
    type Challenge = EF;
    type OpeningProof = OpeningProof<PcsProof<SC>, Self::Challenge>;
    type RapPartialProof = Option<RapPhaseSeqPartialProof<SC>>;
    type Commitment = Com<SC>;
    type Challenger = DuplexChallenger<F, Poseidon2BabyBear<WIDTH>, WIDTH, RATE>;

    // Device Types
    type Matrix = DeviceMatrix<F>;
    type PcsData = HipPcsData;
    type RapPartialProvingKey = RapPartialProvingKey<SC>;
}

/// PCS data for HIP backend.
///
/// TODO: This is a placeholder. Full implementation requires:
/// - HipMerkleTree (ported from GpuMerkleTree)
/// - HipLdeImpl (ported from GpuLdeImpl)
#[derive(Clone, Debug)]
pub struct HipPcsData {
    // TODO: Replace with actual merkle tree once LDE/kernels are ported
    // pub data: HipMerkleTree<HipLdeImpl>,
    pub log_trace_heights: Vec<u8>,
}

impl ProverDevice<HipBackend> for HipDevice {}

impl TraceCommitter<HipBackend> for HipDevice {
    fn commit(&self, _traces: &[DeviceMatrix<F>]) -> (Com<SC>, HipPcsData) {
        unimplemented!(
            "HIP TraceCommitter requires ported kernels (LDE, Merkle tree). \
             See docs/rocm-stark-backend-plan.md for remaining work."
        )
    }
}

impl RapPartialProver<HipBackend> for HipDevice {
    fn partially_prove(
        &self,
        _challenger: &mut <HipBackend as ProverBackend>::Challenger,
        _mpk: &DeviceMultiStarkProvingKeyView<'_, HipBackend>,
        _trace_views: Vec<AirView<DeviceMatrix<F>, F>>,
    ) -> (
        <HipBackend as ProverBackend>::RapPartialProof,
        ProverDataAfterRapPhases<HipBackend>,
    ) {
        unimplemented!(
            "HIP RapPartialProver requires ported kernels (permutation trace, FRI log-up). \
             See docs/rocm-stark-backend-plan.md for remaining work."
        )
    }
}

impl QuotientCommitter<HipBackend> for HipDevice {
    fn eval_and_commit_quotient(
        &self,
        _challenger: &mut <HipBackend as ProverBackend>::Challenger,
        _pk_views: &[&DeviceStarkProvingKey<HipBackend>],
        _public_values: &[Vec<F>],
        _cached_pcs_datas_per_air: &[Vec<HipPcsData>],
        _common_main_pcs_data: &HipPcsData,
        _prover_data_after: &ProverDataAfterRapPhases<HipBackend>,
    ) -> (Com<SC>, HipPcsData) {
        unimplemented!(
            "HIP QuotientCommitter requires ported kernels (quotient polynomial evaluation). \
             See docs/rocm-stark-backend-plan.md for remaining work."
        )
    }
}

impl OpeningProver<HipBackend> for HipDevice {
    fn open(
        &self,
        _challenger: &mut <HipBackend as ProverBackend>::Challenger,
        _preprocessed: Vec<&HipPcsData>,
        _main: Vec<HipPcsData>,
        _after_phase: Vec<HipPcsData>,
        _quotient_data: HipPcsData,
        _quotient_degrees: &[u8],
    ) -> <HipBackend as ProverBackend>::OpeningProof {
        unimplemented!(
            "HIP OpeningProver requires ported kernels (FRI opening). \
             See docs/rocm-stark-backend-plan.md for remaining work."
        )
    }
}

impl DeviceDataTransporter<SC, HipBackend> for HipDevice {
    fn transport_pk_to_device(
        &self,
        _mpk: &MultiStarkProvingKey<SC>,
    ) -> DeviceMultiStarkProvingKey<HipBackend> {
        unimplemented!(
            "HIP DeviceDataTransporter requires device memory infrastructure. \
             See docs/rocm-stark-backend-plan.md for remaining work."
        )
    }

    fn transport_matrix_to_device(&self, _matrix: &Arc<RowMajorMatrix<Val<SC>>>) -> DeviceMatrix<F> {
        unimplemented!(
            "HIP DeviceDataTransporter requires device memory infrastructure. \
             See docs/rocm-stark-backend-plan.md for remaining work."
        )
    }

    fn transport_committed_trace_to_device(
        &self,
        _commitment: Com<SC>,
        _trace: &Arc<RowMajorMatrix<Val<SC>>>,
        _prover_data: &Arc<PcsProverData<SC>>,
    ) -> CommittedTraceData<HipBackend> {
        unimplemented!(
            "HIP DeviceDataTransporter requires device memory infrastructure. \
             See docs/rocm-stark-backend-plan.md for remaining work."
        )
    }

    fn transport_matrix_from_device_to_host(
        &self,
        _matrix: &DeviceMatrix<F>,
    ) -> Arc<RowMajorMatrix<Val<SC>>> {
        unimplemented!(
            "HIP DeviceDataTransporter requires device memory infrastructure. \
             See docs/rocm-stark-backend-plan.md for remaining work."
        )
    }
}
