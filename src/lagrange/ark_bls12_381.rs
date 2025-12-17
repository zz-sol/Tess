//! Arkworks-specific Lagrange helpers for BLS12-381.
//!
//! This module implements `LagrangeField` helpers using arkworks' `Fr` for
//! BLS12-381. These functions provide efficient polynomial interpolation and
//! Lagrange basis generation used by the protocol implementation.
//!
//! # Feature
//!
//! Compiled when the Cargo feature `ark_bls12381` is enabled.
//!
//! # Exports
//!
//! - `lagrange_poly`, `lagrange_polys`, `interp_mostly_zero`

use ark_bls12_381::Fr as BlsFr;
use ark_ff::{FftField, batch_inversion};
use ark_poly::DenseUVPolynomial;
use ark_poly::univariate::DensePolynomial;

use crate::errors::BackendError;

use super::{LagrangeField, interp_mostly_zero_impl, lagrange_poly_impl, lagrange_polys_impl};

impl LagrangeField for BlsFr {
    const TWO_ADICITY: u32 = <BlsFr as FftField>::TWO_ADICITY;

    fn two_adic_root_of_unity() -> Self {
        <BlsFr as FftField>::TWO_ADIC_ROOT_OF_UNITY
    }

    fn batch_inversion(values: &mut [Self]) -> Result<(), BackendError> {
        batch_inversion(values);
        Ok(())
    }
}

/// Compute the i-th Lagrange polynomial L_i(x) for `n` parties (arkworks).
///
/// Returns a dense polynomial in coefficient form where `L_i(j) = δ_{i,j}` for
/// j in [0..n). Used during SRS and key generation.
pub fn lagrange_poly(n: usize, index: usize) -> Result<DensePolynomial<BlsFr>, BackendError> {
    lagrange_poly_impl::<BlsFr, _, _>(n, index, DensePolynomial::from_coefficients_vec)
}

/// Return all Lagrange basis polynomials for `n` parties (arkworks).
pub fn lagrange_polys(n: usize) -> Result<Vec<DensePolynomial<BlsFr>>, BackendError> {
    lagrange_polys_impl::<BlsFr, _, _>(n, DensePolynomial::from_coefficients_vec)
}

/// Interpolate a polynomial optimized for mostly-zero evaluation vectors (arkworks).
pub fn interp_mostly_zero(
    eval: BlsFr,
    points: &[BlsFr],
) -> Result<DensePolynomial<BlsFr>, BackendError> {
    interp_mostly_zero_impl::<BlsFr, _, _>(eval, points, DensePolynomial::from_coefficients_vec)
}
