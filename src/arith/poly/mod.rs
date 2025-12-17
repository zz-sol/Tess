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
