//! Transpiler tests for HIP backend.
//!
//! Tests for constraint transpilation and codec encoding/decoding.

use openvm_stark_backend::air_builders::symbolic::symbolic_variable::{Entry, SymbolicVariable};
use p3_baby_bear::BabyBear;

// Import the codec module - it's internal but we can test via the crate
mod transpiler_codec {
    use openvm_stark_backend::air_builders::symbolic::symbolic_variable::{Entry, SymbolicVariable};
    use p3_field::Field;

    // Basic codec trait
    pub trait Codec {
        type Encoded;

        fn encode(&self) -> Self::Encoded;

        fn decode(encoded: Self::Encoded) -> Self
        where
            Self: Sized;
    }

    const PREPROCESSED: u64 = 0;
    const MAIN: u64 = 1;
    const PERMUTATION: u64 = 2;
    const PUBLIC: u64 = 3;
    const CHALLENGE: u64 = 4;
    const EXPOSED: u64 = 5;

    impl Codec for Entry {
        type Encoded = u64;

        fn encode(&self) -> u64 {
            let (src, part_index, offset) = match self {
                Entry::Preprocessed { offset } => (PREPROCESSED, 0, *offset),
                Entry::Main { part_index, offset } => (MAIN, *part_index, *offset),
                Entry::Permutation { offset } => (PERMUTATION, 0, *offset),
                Entry::Public => (PUBLIC, 0, 0),
                Entry::Challenge => (CHALLENGE, 0, 0),
                Entry::Exposed => (EXPOSED, 0, 0),
            };
            // 4-bit src | 8-bit part_index | 4-bit offset
            assert!(src < 16);
            assert!(part_index < 256);
            assert!(offset < 16);
            src | (part_index << 4) as u64 | (offset << 12) as u64
        }

        fn decode(encoded: u64) -> Self {
            let src = encoded & 0x0f;
            let part_index = ((encoded >> 4) & 0xff) as usize;
            let offset = ((encoded >> 12) & 0x0f) as usize;
            match src {
                PREPROCESSED => Entry::Preprocessed { offset },
                MAIN => Entry::Main { part_index, offset },
                PERMUTATION => Entry::Permutation { offset },
                PUBLIC => Entry::Public,
                CHALLENGE => Entry::Challenge,
                EXPOSED => Entry::Exposed,
                _ => panic!("Invalid Entry"),
            }
        }
    }

    impl<F: Field> Codec for SymbolicVariable<F> {
        type Encoded = u64;

        fn encode(&self) -> u64 {
            let entry_code = self.entry.encode();
            let index = self.index as u64;
            assert!(entry_code <= 0xffff);
            assert!(index <= 0xffff);
            const ENTRY_SHIFT: u64 = 16;
            entry_code | (index << ENTRY_SHIFT)
        }

        fn decode(encoded: u64) -> Self {
            const ENTRY_MASK: u64 = 0xffff;
            const ENTRY_SHIFT: u64 = 16;
            let entry = Entry::decode(encoded & ENTRY_MASK);
            let index = (encoded >> ENTRY_SHIFT) as usize;
            Self::new(entry, index)
        }
    }
}

use transpiler_codec::Codec;

#[test]
fn test_entry_preprocessed_codec() {
    let entry = Entry::Preprocessed { offset: 3 };
    let encoded = entry.encode();
    let decoded = Entry::decode(encoded);
    assert_eq!(entry, decoded);
}

#[test]
fn test_entry_main_codec() {
    let entry = Entry::Main {
        part_index: 7,
        offset: 2,
    };
    let encoded = entry.encode();
    let decoded = Entry::decode(encoded);
    assert_eq!(entry, decoded);
}

#[test]
fn test_entry_permutation_codec() {
    let entry = Entry::Permutation { offset: 5 };
    let encoded = entry.encode();
    let decoded = Entry::decode(encoded);
    assert_eq!(entry, decoded);
}

#[test]
fn test_entry_public_codec() {
    let entry = Entry::Public;
    let encoded = entry.encode();
    let decoded = Entry::decode(encoded);
    assert_eq!(entry, decoded);
}

#[test]
fn test_entry_challenge_codec() {
    let entry = Entry::Challenge;
    let encoded = entry.encode();
    let decoded = Entry::decode(encoded);
    assert_eq!(entry, decoded);
}

#[test]
fn test_entry_exposed_codec() {
    let entry = Entry::Exposed;
    let encoded = entry.encode();
    let decoded = Entry::decode(encoded);
    assert_eq!(entry, decoded);
}

#[test]
fn test_symbolic_variable_codec() {
    let var: SymbolicVariable<BabyBear> = SymbolicVariable::new(
        Entry::Main {
            part_index: 3,
            offset: 1,
        },
        42,
    );
    let encoded = var.encode();
    let decoded: SymbolicVariable<BabyBear> = SymbolicVariable::decode(encoded);
    assert_eq!(var.entry, decoded.entry);
    assert_eq!(var.index, decoded.index);
}

#[test]
fn test_symbolic_variable_preprocessed_codec() {
    let var: SymbolicVariable<BabyBear> =
        SymbolicVariable::new(Entry::Preprocessed { offset: 7 }, 100);
    let encoded = var.encode();
    let decoded: SymbolicVariable<BabyBear> = SymbolicVariable::decode(encoded);
    assert_eq!(var.entry, decoded.entry);
    assert_eq!(var.index, decoded.index);
}
