use ark_bn254::Fr;
use ark_poly::Polynomial as ArkPolynomial;
use ark_poly::univariate::DensePolynomial as DPoly;

use crate::Polynomial;

pub type DensePolynomial = DPoly<Fr>;

impl Polynomial<Fr> for DensePolynomial {
    fn degree(&self) -> usize {
        ArkPolynomial::degree(self)
    }

    fn coeffs(&self) -> &[Fr] {
        &self.coeffs
    }

    fn evaluate(&self, point: &Fr) -> Fr {
        ArkPolynomial::evaluate(self, point)
    }

    fn truncate(&mut self, len: usize) {
        self.coeffs.truncate(len);
    }
}
