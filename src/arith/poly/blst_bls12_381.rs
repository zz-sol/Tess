use ff::Field;

use crate::{Fr, Polynomial};

#[derive(Clone, Debug)]
pub struct DensePolynomial {
    pub coeffs: Vec<Fr>,
}

impl Polynomial<Fr> for DensePolynomial {
    fn degree(&self) -> usize {
        if self.coeffs.len() == 1 && self.coeffs[0] == Fr::ZERO {
            0
        } else {
            self.coeffs.len().saturating_sub(1)
        }
    }

    fn coeffs(&self) -> &[Fr] {
        &self.coeffs
    }

    fn evaluate(&self, point: &Fr) -> Fr {
        if self.coeffs.is_empty() {
            return Fr::ZERO;
        }
        let mut result = *self.coeffs.last().unwrap();
        for coeff in self.coeffs.iter().rev().skip(1) {
            result = result * point + coeff;
        }
        result
    }

    fn truncate(&mut self, len: usize) {
        self.coeffs.truncate(len);
    }
}

impl DensePolynomial {
    /// Create a dense polynomial from the provided coefficient vector.
    ///
    /// The coefficients are in ascending order (constant term first). The
    /// constructor trims leading zero coefficients to keep the representation
    /// canonical.
    pub fn from_coefficients_vec(coeffs: Vec<Fr>) -> Self {
        let mut poly = DensePolynomial { coeffs };
        poly.truncate_leading_zeros();
        poly
    }

    fn truncate_leading_zeros(&mut self) {
        while self.coeffs.len() > 1 && self.coeffs.last() == Some(&Fr::ZERO) {
            self.coeffs.pop();
        }
        if self.coeffs.is_empty() {
            self.coeffs.push(Fr::ZERO);
        }
    }

    pub fn degree(&self) -> usize {
        if self.coeffs.len() == 1 && self.coeffs[0] == Fr::ZERO {
            0
        } else {
            self.coeffs.len().saturating_sub(1)
        }
    }
    /// Evaluate the polynomial at `point` using Horner's method.
    ///
    /// Returns the field element `p(point)`.
    pub fn evaluate(&self, point: &Fr) -> Fr {
        if self.coeffs.is_empty() {
            return Fr::ZERO;
        }
        let mut result = *self.coeffs.last().unwrap();
        for coeff in self.coeffs.iter().rev().skip(1) {
            result = result * point + coeff;
        }
        result
    }
}
