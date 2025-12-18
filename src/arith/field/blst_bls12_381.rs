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

use blstrs::Scalar;
use ff::Field;
use ff::PrimeField;
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

    fn two_adic_root_of_unity() -> Self {
        // BLS12-381 has 2-adic root of unity
        // Using the 2-adic root of unity from the ff crate
        Scalar::ROOT_OF_UNITY
    }

    fn two_adicity_generator(n: usize) -> Self {
        // Get a primitive n-th root of unity
        if n == 1 {
            return Scalar::ONE;
        }

        assert!(
            n.is_power_of_two(),
            "domain size must be a power of two for two-adicity generator"
        );
        let log_n = n.trailing_zeros() as usize;
        let two_adicity = Scalar::S as usize;
        assert!(
            log_n <= two_adicity,
            "requested domain exceeds scalar field two-adicity"
        );

        // Compute root^{2^{two_adicity - log_n}} to get an n-th root of unity.
        let exp_power = 1u64 << (two_adicity - log_n);
        let root = Self::two_adic_root_of_unity();

        // Convert to [u64; 4] format for pow
        let mut exp = [0u64; 4];
        exp[0] = exp_power;
        <Self as FieldElement>::pow(&root, &exp)
    }

    fn batch_inversion(elements: &mut [Self]) -> Result<(), BackendError> {
        use ff::BatchInvert;

        if elements.is_empty() {
            return Ok(());
        }

        // Check for zero elements before batch inversion
        for elem in elements.iter() {
            if bool::from(elem.is_zero()) {
                return Err(BackendError::Math("cannot invert zero element"));
            }
        }

        // Use ff crate's batch inversion (Montgomery's trick)
        elements.iter_mut().batch_invert();

        Ok(())
    }

    fn from_u64(n: u64) -> Self {
        Scalar::from(n)
    }
}
