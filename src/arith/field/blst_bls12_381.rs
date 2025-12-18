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
        use ff::Field;
        if n == 1 {
            return Scalar::ONE;
        }

        // Get the 2-adic root of unity and raise it to the power of (2^k / n)
        let root = Self::two_adic_root_of_unity();
        let k = (n - 1).next_power_of_two().trailing_zeros() as usize + 1;
        let exp_power = (1u64 << k) / n as u64;

        // Convert to [u64; 4] format for pow
        let mut exp = [0u64; 4];
        exp[0] = exp_power;
        <Self as FieldElement>::pow(&root, &exp)
    }

    fn batch_inversion(elements: &mut [Self]) -> Result<(), BackendError> {
        use ff::Field;

        if elements.is_empty() {
            return Ok(());
        }

        let mut prod = Scalar::ONE;
        let mut products = Vec::with_capacity(elements.len());

        for elem in elements.iter() {
            if bool::from(elem.is_zero()) {
                return Err(BackendError::Math("cannot invert zero element"));
            }
            products.push(prod);
            prod *= *elem;
        }

        let mut inv_prod = <Self as FieldElement>::invert(&prod)
            .ok_or(BackendError::Math("batch inversion failed"))?;
        for (i, elem) in elements.iter_mut().enumerate().rev() {
            *elem = inv_prod * products[i];
            inv_prod *= *elem;
        }

        Ok(())
    }

    fn from_u64(n: u64) -> Self {
        Scalar::from(n)
    }
}
