use ark_bls12_381::{Bls12_381, G1Affine, G1Projective, G2Affine, G2Projective};
use ark_ec::pairing::PairingOutput;
use ark_ec::{AffineRepr, CurveGroup};

use crate::{CurvePoint, Fr};

pub type G1 = G1Projective;
pub type G2 = G2Projective;
pub type Gt = PairingOutput<Bls12_381>;

impl CurvePoint<Fr> for G1 {
    type Affine = G1Affine;

    fn identity() -> Self {
        <G1Projective as CurveGroup>::zero()
    }

    fn generator() -> Self {
        <G1Projective as CurveGroup>::generator()
    }

    fn is_identity(&self) -> bool {
        <G1Projective as CurveGroup>::is_zero(self)
    }

    fn from_affine(affine: &Self::Affine) -> Self {
        affine.into_group()
    }

    fn to_affine(&self) -> Self::Affine {
        self.into_affine()
    }

    fn add(&self, other: &Self) -> Self {
        self + other
    }

    fn sub(&self, other: &Self) -> Self {
        self - other
    }

    fn negate(&self) -> Self {
        -*self
    }

    fn mul_scalar(&self, scalar: &Fr) -> Self {
        self * scalar
    }

    fn batch_normalize(points: &[Self]) -> Vec<Self::Affine> {
        <G1Projective as CurveGroup>::normalize_batch(points)
    }
}

impl CurvePoint<Fr> for G2 {
    type Affine = G2Affine;

    fn identity() -> Self {
        <G2Projective as CurveGroup>::zero()
    }

    fn generator() -> Self {
        <G2Projective as CurveGroup>::generator()
    }

    fn is_identity(&self) -> bool {
        <G2Projective as CurveGroup>::is_zero(self)
    }

    fn from_affine(affine: &Self::Affine) -> Self {
        affine.into_group()
    }

    fn to_affine(&self) -> Self::Affine {
        self.into_affine()
    }

    fn add(&self, other: &Self) -> Self {
        self + other
    }

    fn sub(&self, other: &Self) -> Self {
        self - other
    }

    fn negate(&self) -> Self {
        -*self
    }

    fn mul_scalar(&self, scalar: &Fr) -> Self {
        self * scalar
    }

    fn batch_normalize(points: &[Self]) -> Vec<Self::Affine> {
        <G2Projective as CurveGroup>::normalize_batch(points)
    }
}
