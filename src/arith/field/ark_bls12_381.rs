use ark_bls12_381::Fr as ArkFr;
use ark_ff::{Field, One, Zero};
use ark_poly::EvaluationDomain as ArkEvaluationDomain;
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
        One::one()
    }

    fn random<R: RngCore + ?Sized>(rng: &mut R) -> Self {}

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
}
