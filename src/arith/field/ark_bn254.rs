//! Arkworks BN254 field implementation.
//!
//! This module provides scalar field operations for the BN254 (also known as BN128)
//! curve using the Arkworks library. It implements the [`FieldElement`] trait for
//! the scalar field Fr.
//!
//! # Feature Flag
//!
//! This module is only available when the `ark_bn254` feature is enabled.
//!
//! # About BN254
//!
//! BN254 is a pairing-friendly curve that offers a different security/performance
//! tradeoff compared to BLS12-381. It provides approximately 100 bits of security.

use alloc::vec::Vec;
use ark_bn254::Fr as ArkFr;
use ark_ff::{FftField, Field, One as ArkOne, UniformRand, Zero};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use rand_core::RngCore;

use crate::{BackendError, FieldElement};

pub type Fr = ArkFr;

impl FieldElement for Fr {
    type Repr = Vec<u8>;

    fn zero() -> Self {
        Zero::zero()
    }

    fn one() -> Self {
        ArkOne::one()
    }

    fn random<R: RngCore + ?Sized>(rng: &mut R) -> Self {
        Fr::rand(rng)
    }

    fn invert(&self) -> Option<Self> {
        self.inverse()
    }

    fn pow(&self, exp: &[u64; 4]) -> Self {
        Field::pow(self, exp)
    }

    fn to_repr(&self) -> Self::Repr {
        let mut bytes = Vec::new();
        self.serialize_compressed(&mut bytes)
            .expect("scalar serialization");
        bytes
    }

    fn from_repr(repr: &Self::Repr) -> Result<Self, BackendError> {
        Self::deserialize_compressed(repr.as_slice())
            .map_err(|_| BackendError::Serialization("invalid scalar bytes"))
    }

    fn two_adic_root_of_unity() -> Self {
        Fr::TWO_ADIC_ROOT_OF_UNITY
    }

    fn two_adicity_generator(n: usize) -> Self {
        Fr::get_root_of_unity(n as u64).unwrap()
    }

    fn batch_inversion(elements: &mut [Self]) -> Result<(), BackendError> {
        if elements.is_empty() {
            return Ok(());
        }

        // Check for zero elements before batch inversion
        for elem in elements.iter() {
            if elem.is_zero() {
                return Err(BackendError::Math("cannot invert zero element"));
            }
        }

        // Use ark-ff's batch inversion (Montgomery's trick)
        ark_ff::batch_inversion(elements);

        Ok(())
    }
    fn from_u64(n: u64) -> Self {
        Fr::from(n)
    }
}
