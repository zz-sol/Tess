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

pub fn lagrange_poly(n: usize, index: usize) -> Result<DensePolynomial<BlsFr>, BackendError> {
    lagrange_poly_impl::<BlsFr, _, _>(n, index, DensePolynomial::from_coefficients_vec)
}

pub fn lagrange_polys(n: usize) -> Result<Vec<DensePolynomial<BlsFr>>, BackendError> {
    lagrange_polys_impl::<BlsFr, _, _>(n, DensePolynomial::from_coefficients_vec)
}

pub fn interp_mostly_zero(
    eval: BlsFr,
    points: &[BlsFr],
) -> Result<DensePolynomial<BlsFr>, BackendError> {
    interp_mostly_zero_impl::<BlsFr, _, _>(eval, points, DensePolynomial::from_coefficients_vec)
}
