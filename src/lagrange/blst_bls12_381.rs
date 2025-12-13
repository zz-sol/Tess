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

pub fn lagrange_poly(n: usize, index: usize) -> Result<DensePolynomial, BackendError> {
    lagrange_poly_impl::<Scalar, _, _>(n, index, DensePolynomial::from_coefficients_vec)
}

pub fn lagrange_polys(n: usize) -> Result<Vec<DensePolynomial>, BackendError> {
    lagrange_polys_impl::<Scalar, _, _>(n, DensePolynomial::from_coefficients_vec)
}

pub fn interp_mostly_zero(
    eval: Scalar,
    points: &[Scalar],
) -> Result<DensePolynomial, BackendError> {
    interp_mostly_zero_impl::<Scalar, _, _>(eval, points, DensePolynomial::from_coefficients_vec)
}
