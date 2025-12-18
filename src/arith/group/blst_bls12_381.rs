//! blst-backed concrete implementation for BLS12-381 operations.
//!
//! This module provides the concrete types and implementations for the
//! `BlstBackend` when the `blst` feature is enabled. It implements the
//! `FieldElement`, `CurvePoint`, `TargetGroup`, `Polynomial`, `EvaluationDomain`,
//! `PolynomialCommitment`, `MsmProvider`, and `PairingBackend` traits defined in
//! `crate::backend::mod` using the `blstrs` crate.
//!
//! Exported types include:
//! - `DensePolynomial` - dense coefficient polynomial type used for KZG
//! - `G1`, `G2`, `Gt` - wrapped group/target types
//! - `Radix2EvaluationDomain` - FFT domain implementation
//! - `BlstKzg` - KZG commitment implementation
//! - `BlstMsm` - MSM provider
//! - `BlstBackend` - top-level backend type
//!
//! # Feature
//!
//! Compiled when the Cargo feature `blst` is enabled.

use std::io::Cursor;

use blstrs::{Compress, G1Affine, G1Projective, G2Affine, G2Projective, Gt as BlstGt, Scalar};
use group::{Curve, Group, prime::PrimeCurveAffine};

use crate::{BackendError, CurvePoint, TargetGroup};

pub type G1 = G1Projective;
pub type G2 = G2Projective;
pub type Gt = BlstGt;

impl CurvePoint<Scalar> for G1 {
    type Affine = G1Affine;

    fn identity() -> Self {
        <G1Projective as Group>::identity()
    }

    fn generator() -> Self {
        <G1Projective as Group>::generator()
    }

    fn is_identity(&self) -> bool {
        <Self as Group>::is_identity(self).into()
    }

    fn from_affine(affine: &Self::Affine) -> Self {
        affine.into()
    }

    fn to_affine(&self) -> Self::Affine {
        self.into()
    }

    fn add(&self, other: &Self) -> Self {
        self + other
    }

    fn sub(&self, other: &Self) -> Self {
        self - other
    }

    fn negate(&self) -> Self {
        -self
    }

    fn mul_scalar(&self, scalar: &Scalar) -> Self {
        self * scalar
    }

    fn batch_normalize(points: &[Self]) -> Vec<Self::Affine> {
        let mut affines = vec![G1Affine::identity(); points.len()];
        <G1Projective as Curve>::batch_normalize(&points, &mut affines);
        affines
    }

    fn multi_scalar_multipliation(points: &[Self], scalars: &[Scalar]) -> Self {
        assert_eq!(
            points.len(),
            scalars.len(),
            "points and scalars must have the same length"
        );
        G1::multi_exp(points, scalars)
    }
}

impl CurvePoint<Scalar> for G2 {
    type Affine = G2Affine;

    fn identity() -> Self {
        <G2Projective as Group>::identity()
    }

    fn generator() -> Self {
        <G2Projective as Group>::generator()
    }

    fn is_identity(&self) -> bool {
        <Self as Group>::is_identity(self).into()
    }

    fn from_affine(affine: &Self::Affine) -> Self {
        affine.into()
    }

    fn to_affine(&self) -> Self::Affine {
        self.into()
    }

    fn add(&self, other: &Self) -> Self {
        self + other
    }

    fn sub(&self, other: &Self) -> Self {
        self - other
    }

    fn negate(&self) -> Self {
        -self
    }

    fn mul_scalar(&self, scalar: &Scalar) -> Self {
        self * scalar
    }

    fn batch_normalize(points: &[Self]) -> Vec<Self::Affine> {
        let mut affines = vec![G2Affine::identity(); points.len()];
        <G2Projective as Curve>::batch_normalize(&points, &mut affines);
        affines
    }

    fn multi_scalar_multipliation(points: &[Self], scalars: &[Scalar]) -> Self {
        assert_eq!(
            points.len(),
            scalars.len(),
            "points and scalars must have the same length"
        );
        G2::multi_exp(points, scalars)
    }
}

impl TargetGroup for Gt {
    type Scalar = Scalar;
    type Repr = Vec<u8>;

    fn identity() -> Self {
        <Gt as Group>::identity()
    }

    fn mul_scalar(&self, scalar: &Self::Scalar) -> Self {
        self * scalar
    }

    fn combine(&self, other: &Self) -> Self {
        self + other
    }

    fn to_repr(&self) -> Self::Repr {
        let mut bytes = Vec::with_capacity(288);
        self.write_compressed(&mut bytes)
            .map_err(|_| BackendError::Serialization("gt serialization failure"))
            .expect("in-memory serialization should not fail");
        bytes
    }

    fn from_repr(bytes: &Self::Repr) -> Result<Self, BackendError> {
        let mut cursor = Cursor::new(bytes.as_slice());
        Gt::read_compressed(&mut cursor)
            .map_err(|_| BackendError::Serialization("invalid GT bytes"))
    }
}
