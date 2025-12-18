use ark_bn254::Fr;
use ark_poly::DenseUVPolynomial as ArkDenseUVPolynomial;
use ark_poly::Polynomial as ArkPolynomial;
use ark_poly::univariate::DensePolynomial as DPoly;

use crate::Polynomial;

#[derive(Clone, Debug)]
pub struct DensePolynomial(pub DPoly<Fr>);

impl DensePolynomial {
    pub fn from_coefficients_vec(coeffs: Vec<Fr>) -> Self {
        DensePolynomial(DPoly::from_coefficients_slice(&coeffs))
    }
}

impl Polynomial<Fr> for DensePolynomial {
    fn degree(&self) -> usize {
        ArkPolynomial::degree(&self.0)
    }

    fn coeffs(&self) -> &[Fr] {
        self.0.coeffs()
    }

    fn evaluate(&self, point: &Fr) -> Fr {
        ArkPolynomial::evaluate(&self.0, point)
    }

    fn truncate(&mut self, len: usize) {
        self.0.coeffs.truncate(len);
    }

    fn from_coefficients_vec(coeffs: Vec<Fr>) -> Self {
        DensePolynomial::from_coefficients_vec(coeffs)
    }
}
