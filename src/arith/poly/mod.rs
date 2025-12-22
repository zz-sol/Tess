//! Polynomial operations and abstractions.
//!
//! This module provides trait abstractions for univariate polynomials and evaluation domains,
//! which are essential for the KZG commitment scheme and Lagrange interpolation used in TESS.
//!
//! # Overview
//!
//! The module defines the core polynomial operations used across backends:
//!
//! - **[`Polynomial`]**: Interfaces for univariate polynomials (evaluation, coefficient access)
//! - **[`EvaluationDomain`]**: FFT operations over multiplicative subgroups
//! - **[`DensePolynomial`]**: Dense coefficient polynomials with helpers like FFT multiplication
//! - **[`Radix2EvaluationDomain`]**: FFT-friendly evaluation domain implementation
//!
//! # Polynomial Representation
//!
//! Polynomials are stored in **coefficient form** with ascending coefficients:
//! - `p(x) = c_0 + c_1*x + c_2*x^2 + ... + c_n*x^n`
//! - Stored as `[c_0, c_1, c_2, ..., c_n]`
//!
//! # FFT Evaluation Domains
//!
//! Domains are multiplicative subgroups of the scalar field and are used for:
//! - Efficient polynomial interpolation via FFT/IFFT
//! - Lagrange basis generation
//! - Fast polynomial multiplication
//!
//! Domain size must be a power of two for FFT.
//!
//! # Example
//!
//! ```rust
//! use tess::{DensePolynomial, FieldElement, Fr, Polynomial, Radix2EvaluationDomain};
//! use rand::thread_rng;
//!
//! let mut rng = thread_rng();
//! let coeffs = vec![Fr::from_u64(1), Fr::from_u64(2), Fr::from_u64(3)];
//! let poly = DensePolynomial::from_coefficients_vec(coeffs);
//!
//! let domain = Radix2EvaluationDomain::new(4).unwrap();
//! let evals = domain.fft(poly.coeffs());
//!
//! let x = Fr::random(&mut rng);
//! let _ = poly.evaluate(&x);
//! ```

use alloc::vec::Vec;
use core::fmt::Debug;
use core::ops::{Add, Div, Mul, Sub};

use crate::{FieldElement, Fr};

/// Polynomial interface for univariate polynomials.
pub trait Polynomial<F: FieldElement>: Clone + Send + Sync + Debug + 'static {
    /// Returns the degree of this polynomial.
    fn degree(&self) -> usize;

    /// Returns the coefficients in ascending order (constant term first).
    fn coeffs(&self) -> &[F];

    /// Evaluates the polynomial at the given point using Horner's method.
    fn evaluate(&self, point: &F) -> F;

    /// Truncates the polynomial to the specified length.
    fn truncate(&mut self, len: usize);

    /// Constructs a polynomial from its coefficients (ascending order).
    fn from_coefficients_vec(coeffs: Vec<F>) -> Self;
}

/// FFT evaluation domain for polynomial operations.
pub trait EvaluationDomain<F: FieldElement>: Clone + Send + Sync + Debug + 'static {
    /// Returns the size of this evaluation domain (must be power of two).
    fn size(&self) -> usize;

    /// Returns all elements in the domain (roots of unity).
    fn elements(&self) -> Vec<F>;

    /// Forward FFT: converts coefficients to evaluations.
    fn fft(&self, coeffs: &[F]) -> Vec<F>;

    /// Inverse FFT: converts evaluations to coefficients.
    fn ifft(&self, evals: &[F]) -> Vec<F>;
}

/// Helper trait alias that bundles the field arithmetic bounds that polynomials require.
pub trait FieldArithmetic:
    FieldElement + Add<Output = Self> + Sub<Output = Self> + Mul<Output = Self> + PartialEq
{
}

impl<T> FieldArithmetic for T where
    T: FieldElement + Add<Output = T> + Sub<Output = T> + Mul<Output = T> + PartialEq
{
}

/// Dense univariate polynomial represented by ascending coefficients.
#[derive(Clone, Debug, PartialEq)]
pub struct DensePolynomialGeneric<F: FieldArithmetic> {
    /// Coefficients where `coeffs[i]` is the coefficient for x^i
    pub coeffs: Vec<F>,
}

impl<F: FieldArithmetic> DensePolynomialGeneric<F> {
    /// Construct from a coefficient vector and trim leading zeros.
    pub fn from_coefficients_vec(coeffs: Vec<F>) -> Self {
        let mut poly = DensePolynomialGeneric { coeffs };
        poly.truncate_leading_zeros();
        poly
    }

    /// Zero polynomial.
    pub fn zero() -> Self {
        DensePolynomialGeneric {
            coeffs: vec![F::zero()],
        }
    }

    fn truncate_leading_zeros(&mut self) {
        Self::truncate_leading_zeros_internal(&mut self.coeffs);
    }

    fn truncate_leading_zeros_internal(coeffs: &mut Vec<F>) {
        while coeffs.len() > 1 && coeffs.last() == Some(&F::zero()) {
            coeffs.pop();
        }
        if coeffs.is_empty() {
            coeffs.push(F::zero());
        }
    }

    /// Degree of the polynomial.
    pub fn degree(&self) -> usize {
        if self.coeffs.len() == 1 && self.coeffs[0] == F::zero() {
            0
        } else {
            self.coeffs.len().saturating_sub(1)
        }
    }

    /// Evaluate using Horner's method.
    pub fn evaluate(&self, point: &F) -> F {
        if self.coeffs.is_empty() {
            return F::zero();
        }
        let mut result = *self.coeffs.last().unwrap();
        for coeff in self.coeffs.iter().rev().skip(1) {
            result = result * *point + *coeff;
        }
        result
    }

    /// Naive (quadratic) multiplication.
    pub fn naive_mul(&self, other: &DensePolynomialGeneric<F>) -> DensePolynomialGeneric<F> {
        if self.coeffs.is_empty() || other.coeffs.is_empty() {
            return DensePolynomialGeneric::zero();
        }

        let mut result = vec![F::zero(); self.coeffs.len() + other.coeffs.len() - 1];

        for (i, a) in self.coeffs.iter().enumerate() {
            for (j, b) in other.coeffs.iter().enumerate() {
                let product = *a * *b;
                result[i + j] = result[i + j] + product;
            }
        }

        DensePolynomialGeneric::from_coefficients_vec(result)
    }

    /// Synthetic division by (x - root).
    pub fn divide_by_linear(&self, root: F) -> (DensePolynomialGeneric<F>, F) {
        assert!(self.coeffs.len() > 1, "cannot divide constant polynomial");

        let n = self.coeffs.len() - 1;
        let mut quotient = vec![F::zero(); n];
        let mut carry = *self.coeffs.last().unwrap();

        for (idx, coeff) in self.coeffs.iter().rev().skip(1).enumerate() {
            let q_pos = n - 1 - idx;
            quotient[q_pos] = carry;
            carry = *coeff + root * carry;
        }

        DensePolynomialGeneric::truncate_leading_zeros_internal(&mut quotient);
        (
            DensePolynomialGeneric::from_coefficients_vec(quotient),
            carry,
        )
    }

    /// FFT-based multiplication.
    pub fn fft_mul(&self, other: &DensePolynomialGeneric<F>) -> DensePolynomialGeneric<F> {
        if self.coeffs.is_empty() || other.coeffs.is_empty() {
            return DensePolynomialGeneric::zero();
        }

        let result_len = self.coeffs.len() + other.coeffs.len() - 1;
        let size = result_len.next_power_of_two();
        let domain = Radix2EvaluationDomainGeneric::new(size)
            .expect("result length must fit inside the 2-adic subgroup");

        let mut a_eval = domain.fft(&self.coeffs);
        let b_eval = domain.fft(&other.coeffs);

        for (a, b) in a_eval.iter_mut().zip(b_eval.iter()) {
            *a = *a * *b;
        }

        let mut coeffs = domain.ifft(&a_eval);
        coeffs.truncate(result_len);
        DensePolynomialGeneric::from_coefficients_vec(coeffs)
    }

    /// Divide by vanishing polynomial x^n - 1.
    pub fn divide_by_vanishing_poly(
        &self,
        domain: Radix2EvaluationDomainGeneric<F>,
    ) -> (DensePolynomialGeneric<F>, DensePolynomialGeneric<F>) {
        let n = domain.size;

        if self.degree() < n {
            return (DensePolynomialGeneric::zero(), self.clone());
        }

        let mut remainder = self.coeffs.clone();
        let mut quotient = vec![F::zero(); self.coeffs.len().saturating_sub(n)];

        for i in (n..=self.degree()).rev() {
            let coeff = remainder[i];
            quotient[i - n] = coeff;
            remainder[i] = F::zero();
            remainder[i - n] = remainder[i - n] + coeff;
        }

        remainder.truncate(n);

        let mut quot_poly = DensePolynomialGeneric::from_coefficients_vec(quotient);
        let mut rem_poly = DensePolynomialGeneric::from_coefficients_vec(remainder);
        quot_poly.truncate_leading_zeros();
        rem_poly.truncate_leading_zeros();

        (quot_poly, rem_poly)
    }
}

impl<F: FieldArithmetic> Polynomial<F> for DensePolynomialGeneric<F> {
    fn degree(&self) -> usize {
        self.degree()
    }

    fn coeffs(&self) -> &[F] {
        &self.coeffs
    }

    fn evaluate(&self, point: &F) -> F {
        self.evaluate(point)
    }

    fn truncate(&mut self, len: usize) {
        self.coeffs.truncate(len);
        self.truncate_leading_zeros();
    }

    fn from_coefficients_vec(coeffs: Vec<F>) -> Self {
        DensePolynomialGeneric::from_coefficients_vec(coeffs)
    }
}

impl<F: FieldArithmetic> Add for DensePolynomialGeneric<F> {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        let max_len = self.coeffs.len().max(other.coeffs.len());
        let mut result = vec![F::zero(); max_len];

        for (i, coeff) in self.coeffs.iter().enumerate() {
            result[i] = result[i] + *coeff;
        }
        for (i, coeff) in other.coeffs.iter().enumerate() {
            result[i] = result[i] + *coeff;
        }

        DensePolynomialGeneric::from_coefficients_vec(result)
    }
}

impl<F: FieldArithmetic> Sub for DensePolynomialGeneric<F> {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        let max_len = self.coeffs.len().max(other.coeffs.len());
        let mut result = vec![F::zero(); max_len];

        for (i, coeff) in self.coeffs.iter().enumerate() {
            result[i] = result[i] + *coeff;
        }
        for (i, coeff) in other.coeffs.iter().enumerate() {
            result[i] = result[i] - *coeff;
        }

        DensePolynomialGeneric::from_coefficients_vec(result)
    }
}

impl<F: FieldArithmetic> Mul<F> for &DensePolynomialGeneric<F> {
    type Output = DensePolynomialGeneric<F>;

    fn mul(self, scalar: F) -> DensePolynomialGeneric<F> {
        let coeffs = self.coeffs.iter().map(|c| *c * scalar).collect();
        DensePolynomialGeneric::from_coefficients_vec(coeffs)
    }
}

impl<F: FieldArithmetic> Mul<&DensePolynomialGeneric<F>> for &DensePolynomialGeneric<F> {
    type Output = DensePolynomialGeneric<F>;

    fn mul(self, other: &DensePolynomialGeneric<F>) -> DensePolynomialGeneric<F> {
        self.fft_mul(other)
    }
}

impl<F: FieldArithmetic> Div for &DensePolynomialGeneric<F> {
    type Output = DensePolynomialGeneric<F>;

    fn div(self, divisor: &DensePolynomialGeneric<F>) -> DensePolynomialGeneric<F> {
        assert!(
            !(divisor.coeffs.len() == 1 && divisor.coeffs[0] == F::zero()),
            "division by zero polynomial"
        );

        if self.degree() < divisor.degree() {
            return DensePolynomialGeneric::zero();
        }

        let mut remainder = self.clone();
        let mut quotient = vec![F::zero(); self.degree() - divisor.degree() + 1];

        let divisor_leading_inv = divisor.coeffs.last().unwrap().invert().unwrap();

        for i in (0..=self.degree() - divisor.degree()).rev() {
            let pos = i + divisor.degree();
            let coeff = remainder.coeffs[pos] * divisor_leading_inv;
            quotient[i] = coeff;

            for (j, &div_coeff) in divisor.coeffs.iter().enumerate() {
                remainder.coeffs[i + j] = remainder.coeffs[i + j] - coeff * div_coeff;
            }
        }

        DensePolynomialGeneric::from_coefficients_vec(quotient)
    }
}

/// Radix-2 FFT domain over the scalar field.
#[derive(Clone, Debug)]
pub struct Radix2EvaluationDomainGeneric<F: FieldArithmetic> {
    /// Domain size (power of two)
    pub size: usize,
    group_gen: F,
    group_gen_inv: F,
}

impl<F: FieldArithmetic> Radix2EvaluationDomainGeneric<F> {
    /// Create a domain of the specified size.
    pub fn new(size: usize) -> Option<Self> {
        if !size.is_power_of_two() || size == 0 {
            return None;
        }

        let group_gen = F::two_adicity_generator(size);
        let group_gen_inv = group_gen.invert().unwrap();

        Some(Radix2EvaluationDomainGeneric {
            size,
            group_gen,
            group_gen_inv,
        })
    }

    /// Iterator over the domain elements.
    pub fn elements(&self) -> Vec<F> {
        let mut current = F::one();
        let mut elements = Vec::with_capacity(self.size);
        for i in 0..self.size {
            elements.push(current);
            if i < self.size - 1 {
                current = current * self.group_gen;
            }
        }
        elements
    }

    /// Forward FFT: coefficient -> evaluation.
    pub fn fft(&self, coeffs: &[F]) -> Vec<F> {
        let mut a = coeffs.to_vec();
        a.resize(self.size, F::zero());
        self.fft_in_place(&mut a);
        a
    }

    fn fft_in_place(&self, a: &mut [F]) {
        let n = a.len();
        assert_eq!(n, self.size);

        if n == 1 {
            return;
        }

        let mut j = 0;
        for i in 1..n {
            let mut bit = n >> 1;
            while j & bit != 0 {
                j ^= bit;
                bit >>= 1;
            }
            j ^= bit;
            if i < j {
                a.swap(i, j);
            }
        }

        let mut len = 2;
        while len <= n {
            let half_len = len / 2;
            let angle = self.size / len;
            let mut omega_step = F::one();
            for _ in 0..angle {
                omega_step = omega_step * self.group_gen;
            }

            for start in (0..n).step_by(len) {
                let mut omega = F::one();
                for j in 0..half_len {
                    let u = a[start + j];
                    let v = a[start + j + half_len] * omega;
                    a[start + j] = u + v;
                    a[start + j + half_len] = u - v;
                    omega = omega * omega_step;
                }
            }

            len *= 2;
        }
    }

    /// Inverse FFT: evaluation -> coefficient.
    pub fn ifft(&self, evals: &[F]) -> Vec<F> {
        let mut a = evals.to_vec();
        a.resize(self.size, F::zero());
        self.ifft_in_place(&mut a);
        a
    }

    fn ifft_in_place(&self, a: &mut [F]) {
        let mut domain_inv = self.clone();
        domain_inv.group_gen = self.group_gen_inv;
        domain_inv.fft_in_place(a);
        let n_inv = F::from_u64(self.size as u64).invert().unwrap();
        for coeff in a.iter_mut() {
            *coeff = *coeff * n_inv;
        }
    }
}

impl<F: FieldArithmetic> EvaluationDomain<F> for Radix2EvaluationDomainGeneric<F> {
    fn size(&self) -> usize {
        self.size
    }

    fn elements(&self) -> Vec<F> {
        Radix2EvaluationDomainGeneric::elements(self)
    }

    fn fft(&self, coeffs: &[F]) -> Vec<F> {
        Radix2EvaluationDomainGeneric::fft(self, coeffs)
    }

    fn ifft(&self, evals: &[F]) -> Vec<F> {
        Radix2EvaluationDomainGeneric::ifft(self, evals)
    }
}

/// Polynomial evaluations tied to an FFT domain.
#[derive(Clone, Debug)]
pub struct EvaluationsGeneric<F: FieldArithmetic> {
    /// Evaluation values in the domain order.
    pub evals: Vec<F>,
    /// Evaluation domain associated with the values.
    pub domain: Radix2EvaluationDomainGeneric<F>,
}

impl<F: FieldArithmetic> EvaluationsGeneric<F> {
    /// Creates evaluations from a value vector and a domain.
    pub fn from_vec_and_domain(evals: Vec<F>, domain: Radix2EvaluationDomainGeneric<F>) -> Self {
        assert_eq!(evals.len(), domain.size);
        EvaluationsGeneric { evals, domain }
    }

    /// Interpolate coefficients via inverse FFT.
    pub fn interpolate(self) -> DensePolynomialGeneric<F> {
        let coeffs = self.domain.ifft(&self.evals);
        DensePolynomialGeneric::from_coefficients_vec(coeffs)
    }
}

/// Re-export the concrete types for the active scalar field.
pub type DensePolynomial = DensePolynomialGeneric<Fr>;
/// FFT evaluation domain for the active scalar field.
pub type Radix2EvaluationDomain = Radix2EvaluationDomainGeneric<Fr>;
/// Polynomial evaluations for the active scalar field.
pub type Evaluations = EvaluationsGeneric<Fr>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn polynomial_evaluation() {
        let poly = DensePolynomial::from_coefficients_vec(vec![
            Fr::one(),
            Fr::from_u64(2),
            Fr::from_u64(3),
        ]);
        assert_eq!(poly.evaluate(&Fr::zero()), Fr::one());
        assert_eq!(poly.evaluate(&Fr::one()), Fr::from_u64(6));
    }

    #[test]
    fn polynomial_addition() {
        let p1 = DensePolynomial::from_coefficients_vec(vec![Fr::one(), Fr::from_u64(2)]);
        let p2 = DensePolynomial::from_coefficients_vec(vec![Fr::from_u64(3), Fr::from_u64(4)]);
        let sum = p1 + p2;
        assert_eq!(sum.coeffs, vec![Fr::from_u64(4), Fr::from_u64(6)]);
    }

    #[test]
    fn polynomial_fft_domain() {
        let domain = Radix2EvaluationDomain::new(4).unwrap();
        let elements = domain.elements();
        assert_eq!(elements.len(), 4);
        let omega4 = elements[1] * elements[1] * elements[1] * elements[1];
        assert_eq!(omega4, Fr::one());
    }

    #[test]
    fn fft_matches_naive_multiplication() {
        let a = DensePolynomial::from_coefficients_vec(vec![
            Fr::from_u64(1),
            Fr::from_u64(2),
            Fr::from_u64(3),
            Fr::from_u64(4),
        ]);
        let b = DensePolynomial::from_coefficients_vec(vec![
            Fr::from_u64(5),
            Fr::from_u64(6),
            Fr::from_u64(7),
        ]);
        let naive = a.naive_mul(&b);
        let fft = a.fft_mul(&b);
        assert_eq!(naive, fft);
    }

    #[test]
    fn divide_by_linear_test() {
        let poly = DensePolynomial::from_coefficients_vec(vec![-Fr::one(), Fr::zero(), Fr::one()]);
        let (quot, rem) = poly.divide_by_linear(Fr::one());
        assert_eq!(rem, Fr::zero());
        assert_eq!(
            quot,
            DensePolynomial::from_coefficients_vec(vec![Fr::one(), Fr::one()])
        );
    }
}
