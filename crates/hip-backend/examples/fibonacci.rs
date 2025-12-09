use std::sync::Arc;

use itertools::zip_eq;
use openvm_hip_backend::{
    engine::HipBabyBearPoseidon2Engine, prover_backend::HipBackend, types::SC,
};
use openvm_stark_backend::{
    engine::StarkEngine,
    prover::{
        cpu::CpuBackend,
        hal::DeviceDataTransporter,
        types::{AirProvingContext, ProvingContext},
    },
};
use openvm_stark_sdk::{
    any_rap_arc_vec,
    config::{baby_bear_poseidon2::BabyBearPoseidon2Engine, setup_tracing, FriParameters},
    dummy_airs::fib_air::{air::FibonacciAir, trace::generate_trace_rows},
    engine::StarkFriEngine,
};
use p3_baby_bear::BabyBear;
use p3_field::FieldAlgebra;

const LOG_BLOWUP: usize = 2;
const LOG_TRACE_DEGREE: usize = 3;

// Public inputs:
const A: u32 = 0;
const B: u32 = 1;
const N: usize = 1usize << LOG_TRACE_DEGREE;

type Val = BabyBear;

fn get_fib_number(n: usize) -> u32 {
    let mut a = 0;
    let mut b = 1;
    for _ in 0..n - 1 {
        let c = a + b;
        a = b;
        b = c;
    }
    b
}

fn main() {
    setup_tracing();
    println!("test_single_fib_stark");

    let public_values = [A, B, get_fib_number(N)]
        .map(BabyBear::from_canonical_u32)
        .to_vec();
    let air = FibonacciAir;

    let cpu_trace = Arc::new(generate_trace_rows::<Val>(A, B, N));

    let airs = any_rap_arc_vec![air];

    let hip_engine = HipBabyBearPoseidon2Engine::new(
        FriParameters::standard_with_100_bits_conjectured_security(LOG_BLOWUP),
    );
    let hip_trace = hip_engine.device().transport_matrix_to_device(&cpu_trace);

    let cpu_air_ctx = AirProvingContext::<CpuBackend<SC>>::simple(cpu_trace, public_values.clone());
    let hip_air_ctx = AirProvingContext::<HipBackend>::simple(hip_trace, public_values);

    let mut keygen_builder = hip_engine.keygen_builder();
    let air_ids = hip_engine.set_up_keygen_builder(&mut keygen_builder, &airs);
    let pk_host = keygen_builder.generate_pk();
    let vk = pk_host.get_vk();
    let pk = hip_engine.device().transport_pk_to_device(&pk_host);
    // engine.debug(&airs, &pk.per_air, &air_proof_inputs);
    let cpu_ctx = ProvingContext::new(zip_eq(air_ids.clone(), vec![cpu_air_ctx]).collect());
    let hip_ctx = ProvingContext::new(zip_eq(air_ids, vec![hip_air_ctx]).collect());

    // CPU
    println!("\nStarting CPU proof");
    let cpu_engine = BabyBearPoseidon2Engine::new(
        FriParameters::standard_with_100_bits_conjectured_security(LOG_BLOWUP),
    );
    let cpu_pk = cpu_engine.device().transport_pk_to_device(&pk_host);
    let cpu_proof = cpu_engine.prove(&cpu_pk, cpu_ctx);
    cpu_engine.verify(&vk, &cpu_proof).unwrap();

    // HIP/GPU
    println!("\nStarting HIP/GPU proof");
    let hip_proof = hip_engine.prove(&pk, hip_ctx);
    hip_engine.verify(&vk, &hip_proof).unwrap();

    // Clean shutdown
    openvm_hip_common::hip_runtime_shutdown();
}
