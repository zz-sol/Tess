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
//!
//! # Example
//!
//! ```rust,no_run
//! # #[cfg(feature = "blst")]
//! # {
//! use tess::backend::BlstBackend;
//! # }
//! ```
use blstrs::Scalar;
use ff::Field;
use rand_core::RngCore;

use crate::{BackendError, FieldElement};

pub type Fr = Scalar;

impl FieldElement for Scalar {
    type Repr = Vec<u8>;

    fn zero() -> Self {
        Scalar::ZERO
    }

    fn one() -> Self {
        Scalar::ONE
    }

    fn random<R: RngCore + ?Sized>(rng: &mut R) -> Self {
        <Scalar as Field>::random(rng)
    }

    fn invert(&self) -> Option<Self> {
        Field::invert(self).into()
    }

    fn pow(&self, exp: &[u64; 4]) -> Self {
        self.pow_vartime(exp)
    }

    fn to_repr(&self) -> Self::Repr {
        self.to_bytes_be().to_vec()
    }

    fn from_repr(repr: &Self::Repr) -> Result<Self, BackendError> {
        let mut bytes = [0u8; 32];
        if repr.len() != 32 {
            return Err(BackendError::Serialization("invalid scalar length"));
        }
        bytes.copy_from_slice(repr);
        Option::<Scalar>::from(Scalar::from_bytes_be(&bytes))
            .ok_or(BackendError::Serialization("invalid scalar bytes"))
    }
}
