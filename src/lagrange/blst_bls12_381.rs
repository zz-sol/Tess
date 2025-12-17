//! blst-specific Lagrange helpers for BLS12-381.
//!
//! This module implements `LagrangeField` helpers using `blstrs::Scalar` for
//! BLS12-381. These helpers provide efficient polynomial interpolation and
//! Lagrange basis generation used by the protocol implementation.
//!
//! # Feature
//!
//! Compiled when the Cargo feature `blst` is enabled.
//!
//! # Exports
//!
//! - `lagrange_poly`, `lagrange_polys`, `interp_mostly_zero`

use blstrs::Scalar;
use ff::{BatchInvert, PrimeField};

use crate::backend::DensePolynomial;
use crate::errors::BackendError;

use super::{LagrangeField, interp_mostly_zero_impl, lagrange_poly_impl, lagrange_polys_impl};

impl LagrangeField for Scalar {
    const TWO_ADICITY: u32 = Scalar::S;

    fn two_adic_root_of_unity() -> Self {
        Scalar::ROOT_OF_UNITY
    }

    fn batch_inversion(values: &mut [Self]) -> Result<(), BackendError> {
        values.iter_mut().batch_invert();
        Ok(())
    }
}

/// Compute the i-th Lagrange polynomial L_i(x) for `n` parties.
///
/// Returns a dense polynomial in coefficient form where `L_i(j) = δ_{i,j}` for
/// j in [0..n). This is used to build KZG commitments to Lagrange basis
/// polynomials during SRS/key-generation phases.
pub fn lagrange_poly(n: usize, index: usize) -> Result<DensePolynomial, BackendError> {
    lagrange_poly_impl::<Scalar, _, _>(n, index, DensePolynomial::from_coefficients_vec)
}

/// Return all Lagrange basis polynomials for `n` parties.
///
/// The returned vector has length `n` and contains `L_0, L_1, ..., L_{n-1}`.
pub fn lagrange_polys(n: usize) -> Result<Vec<DensePolynomial>, BackendError> {
    lagrange_polys_impl::<Scalar, _, _>(n, DensePolynomial::from_coefficients_vec)
}

/// Interpolate a polynomial given evaluations where most points are zero.
///
/// This helper is optimized for the common case where `points` contains
/// mostly zeros and only a small number of non-zero evaluations, returning a
/// DensePolynomial that matches the provided evaluations.
pub fn interp_mostly_zero(
    eval: Scalar,
    points: &[Scalar],
) -> Result<DensePolynomial, BackendError> {
    interp_mostly_zero_impl::<Scalar, _, _>(eval, points, DensePolynomial::from_coefficients_vec)
}
