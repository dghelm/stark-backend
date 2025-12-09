//! HIP data transporter implementation.
//!
//! This module mirrors `cuda-backend/src/data_transporter.rs` for AMD GPUs.
//!
//! # Status
//!
//! - `transport_matrix_to_device` and `transport_matrix_from_device_to_host` are fully functional
//! - `transport_pk_to_device` and `transport_committed_trace_to_device` require TraceCommitter
//!   (LDE + Merkle tree) to be implemented first

use std::{fmt::Debug, sync::Arc};

use openvm_hip_common::{
    copy::{MemCopyD2H, MemCopyH2D},
    d_buffer::DeviceBuffer,
};
use openvm_stark_backend::{
    config::{Com, PcsProverData, Val},
    keygen::types::MultiStarkProvingKey,
    prover::{
        hal::{DeviceDataTransporter, MatrixDimensions},
        types::{CommittedTraceData, DeviceMultiStarkProvingKey},
    },
};
use p3_matrix::{dense::RowMajorMatrix, Matrix};

use crate::{
    base::DeviceMatrix,
    hip::kernels::matrix::matrix_transpose,
    hip_device::HipDevice,
    prelude::{F, SC},
    prover_backend::HipBackend,
};

impl DeviceDataTransporter<SC, HipBackend> for HipDevice {
    fn transport_pk_to_device(
        &self,
        _mpk: &MultiStarkProvingKey<SC>,
    ) -> DeviceMultiStarkProvingKey<HipBackend> {
        // TODO: Implement once TraceCommitter is available
        // This requires self.commit() which needs LDE + Merkle tree
        unimplemented!(
            "HIP transport_pk_to_device requires TraceCommitter (LDE + Merkle tree). \
             See hip-backend/src/lib.rs for porting status."
        )
    }

    fn transport_matrix_to_device(&self, matrix: &Arc<RowMajorMatrix<F>>) -> DeviceMatrix<F> {
        transport_matrix_to_device(matrix.clone())
    }

    fn transport_committed_trace_to_device(
        &self,
        _commitment: Com<SC>,
        _trace: &Arc<RowMajorMatrix<Val<SC>>>,
        _prover_data: &Arc<PcsProverData<SC>>,
    ) -> CommittedTraceData<HipBackend> {
        // TODO: Implement once TraceCommitter is available
        // This requires self.commit() which needs LDE + Merkle tree
        unimplemented!(
            "HIP transport_committed_trace_to_device requires TraceCommitter (LDE + Merkle tree). \
             See hip-backend/src/lib.rs for porting status."
        )
    }

    fn transport_matrix_from_device_to_host(
        &self,
        matrix: &DeviceMatrix<F>,
    ) -> Arc<RowMajorMatrix<F>> {
        let matrix_host = transport_device_matrix_to_host(matrix);
        Arc::new(matrix_host)
    }
}

pub fn transport_matrix_to_device(matrix: Arc<RowMajorMatrix<F>>) -> DeviceMatrix<F> {
    let data = matrix.values.as_slice();
    let input_buffer = data.to_device().unwrap();
    let output = DeviceMatrix::<F>::with_capacity(matrix.height(), matrix.width());
    unsafe {
        matrix_transpose::<F>(
            output.buffer(),
            &input_buffer,
            matrix.width(),
            matrix.height(),
        )
        .unwrap();
    }
    assert_eq!(output.strong_count(), 1);
    output
}

pub fn transport_device_matrix_to_host<T: Clone + Send + Sync>(
    matrix: &DeviceMatrix<T>,
) -> RowMajorMatrix<T> {
    let matrix_buffer = DeviceBuffer::<T>::with_capacity(matrix.height() * matrix.width());
    unsafe {
        matrix_transpose::<T>(
            &matrix_buffer,
            matrix.buffer(),
            matrix.height(),
            matrix.width(),
        )
        .unwrap();
    }
    RowMajorMatrix::<T>::new(matrix_buffer.to_host().unwrap(), matrix.width())
}

pub fn assert_eq_device_matrix<T: Clone + Send + Sync + PartialEq + Debug>(
    a: &DeviceMatrix<T>,
    b: &DeviceMatrix<T>,
) {
    assert_eq!(a.height(), b.height());
    assert_eq!(a.width(), b.width());
    assert_eq!(a.buffer().len(), b.buffer().len());
    let a_host = a.to_host().unwrap();
    let b_host = b.to_host().unwrap();
    for r in 0..a.height() {
        for c in 0..a.width() {
            assert_eq!(
                a_host[c * a.height() + r],
                b_host[c * b.height() + r],
                "Mismatch at row {} column {}",
                r,
                c
            );
        }
    }
}

pub fn assert_eq_host_and_device_matrix<T: Clone + Send + Sync + PartialEq + Debug>(
    cpu: Arc<RowMajorMatrix<T>>,
    gpu: &DeviceMatrix<T>,
) {
    assert_eq!(gpu.width(), cpu.width());
    assert_eq!(gpu.height(), cpu.height());
    let gpu = gpu.to_host().unwrap();
    for r in 0..cpu.height() {
        for c in 0..cpu.width() {
            assert_eq!(
                gpu[c * cpu.height() + r],
                cpu.get(r, c),
                "Mismatch at row {} column {}",
                r,
                c
            );
        }
    }
}
