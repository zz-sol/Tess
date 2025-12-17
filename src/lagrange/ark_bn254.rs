//! Arkworks-specific Lagrange helpers for BN254.
//!
//! This module implements `LagrangeField` helpers using arkworks' `Fr` for
//! BN254. These functions provide efficient polynomial interpolation and
//! Lagrange basis generation used by the protocol implementation.
//!
//! # Feature
//!
//! Compiled when the Cargo feature `ark_bn254` is enabled.
//!
//! # Exports
//!
//! - `lagrange_poly`, `lagrange_polys`, `interp_mostly_zero`

use ark_bn254::Fr as BnFr;
use ark_ff::{FftField, batch_inversion};
use ark_poly::{DenseUVPolynomial, univariate::DensePolynomial};

use crate::errors::BackendError;

use super::{LagrangeField, interp_mostly_zero_impl, lagrange_poly_impl, lagrange_polys_impl};

impl LagrangeField for BnFr {
    const TWO_ADICITY: u32 = <BnFr as FftField>::TWO_ADICITY;

    fn two_adic_root_of_unity() -> Self {
        <BnFr as FftField>::TWO_ADIC_ROOT_OF_UNITY
    }

    fn batch_inversion(values: &mut [Self]) -> Result<(), BackendError> {
        batch_inversion(values);
        Ok(())
    }
}

/// Compute the i-th Lagrange polynomial L_i(x) for `n` parties (arkworks BN254).
pub fn lagrange_poly(n: usize, index: usize) -> Result<DensePolynomial<BnFr>, BackendError> {
    lagrange_poly_impl::<BnFr, _, _>(n, index, DensePolynomial::from_coefficients_vec)
}

/// Return all Lagrange basis polynomials for `n` parties (arkworks BN254).
pub fn lagrange_polys(n: usize) -> Result<Vec<DensePolynomial<BnFr>>, BackendError> {
    lagrange_polys_impl::<BnFr, _, _>(n, DensePolynomial::from_coefficients_vec)
}

/// Interpolate a polynomial optimized for mostly-zero evaluation vectors (arkworks BN254).
pub fn interp_mostly_zero(
    eval: BnFr,
    points: &[BnFr],
) -> Result<DensePolynomial<BnFr>, BackendError> {
    interp_mostly_zero_impl::<BnFr, _, _>(eval, points, DensePolynomial::from_coefficients_vec)
}
