//! Arkworks BN254 group operations.
//!
//! This module provides elliptic curve group operations for the BN254 (BN128) curve
//! using the Arkworks library. It implements the [`CurvePoint`] and [`TargetGroup`]
//! traits for G1, G2, and Gt (pairing target group).
//!
//! # Feature Flag
//!
//! This module is only available when the `ark_bn254` feature is enabled.
//!
//! # Groups
//!
//! - **G1**: First source group for pairings (points on E(Fq))
//! - **G2**: Second source group for pairings (points on E'(Fq2))
//! - **Gt**: Target group for pairings (elements in Fq12)

use alloc::vec::Vec;
use ark_bn254::{Bn254, G1Affine, G1Projective, G2Affine, G2Projective};
use ark_ec::PrimeGroup;
use ark_ec::pairing::PairingOutput;
use ark_ec::{AffineRepr, CurveGroup, VariableBaseMSM};
use ark_ff::{PrimeField, Zero};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use tracing::instrument;

use crate::{BackendError, CurvePoint, Fr, TargetGroup};

#[derive(Clone, Copy, Debug)]
/// G1 group element wrapper for the Arkworks BN254 backend.
pub struct G1(pub G1Projective);

#[derive(Clone, Copy, Debug)]
/// G2 group element wrapper for the Arkworks BN254 backend.
pub struct G2(pub G2Projective);

/// Target group type for the Arkworks BN254 backend.
pub type Gt = PairingOutput<Bn254>;

impl CurvePoint<Fr> for G1 {
    type Affine = G1Affine;

    fn identity() -> Self {
        G1(G1Projective::zero())
    }

    fn generator() -> Self {
        G1(<G1Projective as PrimeGroup>::generator())
    }

    fn is_identity(&self) -> bool {
        self.0 == G1Projective::zero()
    }

    fn from_affine(affine: &Self::Affine) -> Self {
        G1(affine.into_group())
    }

    fn to_affine(&self) -> Self::Affine {
        self.0.into_affine()
    }

    fn add(&self, other: &Self) -> Self {
        G1(self.0 + other.0)
    }

    fn sub(&self, other: &Self) -> Self {
        G1(self.0 - other.0)
    }

    fn negate(&self) -> Self {
        G1(-self.0)
    }

    fn mul_scalar(&self, scalar: &Fr) -> Self {
        G1(self.0 * scalar)
    }

    fn batch_normalize(points: &[Self]) -> Vec<Self::Affine> {
        let projective: Vec<G1Projective> = points.iter().map(|p| p.0).collect();
        <G1Projective as CurveGroup>::normalize_batch(&projective)
    }

    #[instrument(level = "trace", skip_all, fields(len = points.len()))]
    fn multi_scalar_multipliation(points: &[Self], scalars: &[Fr]) -> Self {
        assert_eq!(
            points.len(),
            scalars.len(),
            "points and scalars must have the same length"
        );
        let affine_points = Self::batch_normalize(points);
        let result = G1Projective::msm(&affine_points, scalars).unwrap();
        G1(result)
    }
}

impl CurvePoint<Fr> for G2 {
    type Affine = G2Affine;

    fn identity() -> Self {
        G2(G2Projective::zero())
    }

    fn generator() -> Self {
        G2(<G2Projective as PrimeGroup>::generator())
    }

    fn is_identity(&self) -> bool {
        self.0 == G2Projective::zero()
    }

    fn from_affine(affine: &Self::Affine) -> Self {
        G2(affine.into_group())
    }

    fn to_affine(&self) -> Self::Affine {
        self.0.into_affine()
    }

    fn add(&self, other: &Self) -> Self {
        G2(self.0 + other.0)
    }

    fn sub(&self, other: &Self) -> Self {
        G2(self.0 - other.0)
    }

    fn negate(&self) -> Self {
        G2(-self.0)
    }

    fn mul_scalar(&self, scalar: &Fr) -> Self {
        G2(self.0 * scalar)
    }

    fn batch_normalize(points: &[Self]) -> Vec<Self::Affine> {
        let projective: Vec<G2Projective> = points.iter().map(|p| p.0).collect();
        <G2Projective as CurveGroup>::normalize_batch(&projective)
    }

    #[instrument(level = "trace", skip_all, fields(len = points.len()))]
    fn multi_scalar_multipliation(points: &[Self], scalars: &[Fr]) -> Self {
        assert_eq!(
            points.len(),
            scalars.len(),
            "points and scalars must have the same length"
        );
        let affine_points = Self::batch_normalize(points);
        let result = G2Projective::msm(&affine_points, scalars).unwrap();
        G2(result)
    }
}

impl From<&G1> for G1Projective {
    fn from(g1: &G1) -> Self {
        g1.0
    }
}

impl From<G1> for G1Projective {
    fn from(g1: G1) -> Self {
        g1.0
    }
}

impl From<&G2> for G2Projective {
    fn from(g2: &G2) -> Self {
        g2.0
    }
}

impl From<G2> for G2Projective {
    fn from(g2: G2) -> Self {
        g2.0
    }
}

impl TargetGroup for Gt {
    type Scalar = Fr;
    type Repr = Vec<u8>;

    fn identity() -> Self {
        <Gt as Zero>::zero()
    }

    fn mul_scalar(&self, scalar: &Self::Scalar) -> Self {
        let bigint = scalar.into_bigint();
        self.mul_bigint(bigint.as_ref())
    }

    fn combine(&self, other: &Self) -> Self {
        self + other
    }

    fn to_repr(&self) -> Self::Repr {
        let mut bytes = Vec::new();
        self.serialize_compressed(&mut bytes)
            .expect("target serialization");
        bytes
    }

    fn from_repr(bytes: &Self::Repr) -> Result<Self, BackendError> {
        Gt::deserialize_compressed(bytes.as_slice())
            .map_err(|_| BackendError::Serialization("invalid GT bytes"))
    }
}
