# OpenVM Stark Backend

[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/openvm-org/stark-backend)

[Contributor Docs](./docs)
| [Rustdocs](https://docs.openvm.dev/docs/stark-backend)

A modular proof system backend for proving and verifying multi-chip circuits with inter-chip communication.

The backend is designed to be modular and compatible with different proof systems, with a focus on performance and extensibility. The aim is to support different circuit representations and permutation/lookup arguments.

## Crates

- [`openvm-stark-backend`](crates/stark-backend): General purpose STARK proving system with multi-trace and logup support, built on top of Plonky3.
- [`openvm-stark-sdk`](crates/stark-sdk): Low-level SDK for use with STARK backend to generate proofs for specific STARK configurations.
- [`openvm-cuda-builder`](crates/cuda-builder): Build utilities and CUDA detection crate, meant to be imported as a build dependency in crates that use CUDA.
- [`openvm-cuda-common`](crates/cuda-common): Shared headers (`.cuh/.h` files) and CUDA utilities library.
- [`openvm-cuda-backend`](crates/cuda-backend): CUDA implementation of a STARK prover backend using all of the previous crates.
- [`openvm-hip-builder`](crates/hip-builder): Build utilities and HIP/ROCm detection crate for AMD GPUs.
- [`openvm-hip-common`](crates/hip-common): Shared utilities and HIP memory management library.
- [`openvm-hip-backend`](crates/hip-backend): HIP/ROCm implementation of a STARK prover backend for AMD GPUs.

Contributors should read [Development without CUDA](./docs/README.md#development-without-cuda) and [Development with CUDA](./docs/README.md#development-with-cuda) for instructions on how to set up their development environments.

## HIP/ROCm Backend (AMD GPUs)

The HIP backend supports AMD GPUs (RDNA3, RDNA3.5, CDNA). For best performance on **consumer AMD GPUs** (RX 7000/9000 series), set the `HIP_DISABLE_VPMM=1` environment variable:

```bash
HIP_FORCE_EXIT=1 HIP_ARCH=gfx1151 HIP_DISABLE_VPMM=1 ./your_hip_binary
```

This disables the Virtual Memory Management allocator which has high overhead on consumer GPUs. See [HIP Backend Benchmarks](crates/hip-backend/benches/README.md) for details.

## Status

As of June 2025, STARK Backend v1.1.0 and later are recommended for production use. STARK Backend completed an external [audit](https://github.com/openvm-org/openvm/blob/main/audits/v1-cantina-report.pdf) on [Cantina](https://cantina.xyz/) from January to March 2025 as well as an internal [audit](https://github.com/openvm-org/openvm/blob/main/audits/v1-internal/README.md) by members of the [Axiom](https://axiom.xyz/) team during the same timeframe.

## Security

See [SECURITY.md](./SECURITY.md).

## Acknowledgements

We studied and built upon the work of other teams in our quest to design a modular and performant proving framework.
We would like to thank these teams for sharing their code for open source development:

- [Plonky3](https://github.com/Plonky3/Plonky3): This codebase is built on top of Plonky3, where we have heavily benefited from their modular design at the polynomial IOP level. We extend Plonky3 by providing higher level interfaces for proving multi-chip circuits.
- [Valida](https://github.com/valida-xyz/valida): Valida introduced the exceptionally elegant interactions interface for multi-chip communication via logup permutation arguments. We have found this interface quite well thought out and have built upon and extended it.
- [SP1](https://github.com/succinctlabs/sp1): We learned from SP1's `AirBuilder` designs, and the original design for the `InteractionBuilder` was inspired by them.
- [Risc0](https://github.com/risc0/risc0): We used some of Risc0's open source CUDA kernels as the starting point for our own CUDA kernels.
- [Supranational](https://github.com/supranational/sppark): We ported and modified [sppark](https://github.com/supranational/sppark)'s open source NTT CUDA kernels for use in our CUDA backend.
- [Scroll](https://github.com/scroll-tech/): Members of the Scroll team made foundational contributions to the CUDA backend.
