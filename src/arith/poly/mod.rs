//! Polynomial operations and abstractions.
//!
//! This module provides trait abstractions for univariate polynomials and evaluation domains,
//! which are essential for the KZG commitment scheme and Lagrange interpolation used in TESS.
//!
//! # Overview
//!
//! The module defines two main traits:
//!
//! - **[`Polynomial`]**: Univariate polynomial operations (evaluation, coefficient access)
//! - **[`EvaluationDomain`]**: FFT operations over multiplicative subgroups
//!
//! # Polynomial Representation
//!
//! Polynomials are represented in **coefficient form** with coefficients in ascending order:
//! - `p(x) = c_0 + c_1*x + c_2*x^2 + ... + c_n*x^n`
//! - Stored as `[c_0, c_1, c_2, ..., c_n]`
//!
//! # FFT Evaluation Domains
//!
//! Evaluation domains are multiplicative subgroups of the scalar field, used for:
//! - Efficient polynomial interpolation via FFT/IFFT
//! - Lagrange polynomial basis generation
//! - Fast polynomial multiplication
//!
//! Domain size must be a power of two for FFT to work correctly.
//!
//! # Example
//!
//! ```rust
//! use tess::{DensePolynomial, FieldElement, Fr, Polynomial};
//! use rand::thread_rng;
//!
//! let mut rng = thread_rng();
//!
//! // Create a polynomial from coefficients: p(x) = 1 + 2x + 3x^2
//! let coeffs = vec![Fr::from_u64(1), Fr::from_u64(2), Fr::from_u64(3)];
//! let poly = DensePolynomial::from_coefficients_vec(coeffs);
//!
//! // Evaluate at a random point
//! let x = Fr::random(&mut rng);
//! let y = poly.evaluate(&x);
//!
//! // Access polynomial properties
//! println!("Degree: {}", poly.degree());
//! println!("Coefficients: {:?}", poly.coeffs());
//! ```

use std::fmt::Debug;

use crate::FieldElement;

#[cfg(feature = "blst")]
mod blst_bls12_381;
#[cfg(feature = "blst")]
pub use blst_bls12_381::DensePolynomial;

#[cfg(feature = "ark_bls12381")]
mod ark_bls12_381;
#[cfg(feature = "ark_bls12381")]
pub use ark_bls12_381::DensePolynomial;

#[cfg(feature = "ark_bn254")]
mod ark_bn254;
#[cfg(feature = "ark_bn254")]
pub use ark_bn254::DensePolynomial;

/// Polynomial interface for univariate polynomials.
///
/// Polynomials are represented in coefficient form and are used extensively
/// in the KZG commitment scheme for Lagrange interpolation and evaluation.
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
///
/// This trait provides FFT/IFFT operations over a multiplicative subgroup,
/// which is used for efficient Lagrange polynomial basis generation and
/// polynomial operations in the threshold scheme.
///
/// The domain size must be a power of two for FFT to work correctly.
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
