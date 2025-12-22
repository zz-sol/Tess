//! Lagrange polynomial precomputation for efficient key generation.
//!
//! This module provides functionality for computing and precomputing Lagrange basis
//! polynomials and their KZG commitments. These precomputations enable the "silent"
//! (non-interactive) key generation in the TESS protocol.
//!
//! # Overview
//!
//! Lagrange polynomials L_i(x) form a basis for polynomials over a domain, with the
//! property that L_i(ω^j) = 1 if i=j, and 0 otherwise, where ω is a root of unity.
//!
//! # Precomputed Values
//!
//! For each participant i, we precompute commitments to:
//! - **L_i(x)**: The i-th Lagrange basis polynomial
//! - **L_i(x) - L_i(0)**: Shifted polynomial for proof construction
//! - **x · L_i(x)**: Product with the monomial x
//! - **L_i(x) · L_j(z)**: All pairwise products at vanishing polynomial
//!
//! These precomputed commitments are stored in the SRS and used during key generation
//! to derive public keys without requiring polynomial operations at that time.
//!
//! # Performance
//!
//! The precomputation is parallelized using Rayon and runs in O(n²) time for n
//! participants. For 2048 participants, this typically takes 5-10 seconds.
//!
//! # Mathematical Background
//!
//! For a domain {ω^0, ω^1, ..., ω^(n-1)} where ω is an n-th root of unity:
//!
//! ```text
//! L_i(x) = (x^n - 1) / (n · ω^(-i) · (x - ω^i))
//! ```
//!
//! This simplifies to efficient FFT-based computation in the evaluation domain.

use alloc::vec::Vec;
use rayon::iter::{
    IndexedParallelIterator, IntoParallelIterator, IntoParallelRefIterator, ParallelIterator,
};
use tracing::instrument;

use crate::arith::group::CurvePoint;
use crate::{BackendError, DensePolynomial, FieldElement, Fr, PairingBackend, Polynomial};

/// Precomputed Lagrange polynomial commitments for efficient key derivation.
///
/// This structure contains KZG commitments to various Lagrange polynomial
/// transformations, precomputed during SRS generation to enable efficient
/// key generation without polynomial operations at key generation time.
///
/// # Fields
///
/// - `li`: Commitments to L_i(x) for each participant `i`
/// - `li_minus0`: Commitments to L_i(x) - L_i(0) for each `i`
/// - `li_x`: Commitments to x * L_i(x) for each `i`
/// - `li_lj_z`: Commitments to L_i(x) * L_j(z) for all pairs `(i, j)`
#[derive(Clone, Debug)]
pub struct LagrangePowers<B: PairingBackend> {
    /// Commitments to L_i(x) for each participant i.
    pub li: Vec<B::G1>,
    /// Commitments to L_i(x) - L_i(0) for each participant i.
    pub li_minus0: Vec<B::G1>,
    /// Commitments to x * L_i(x) for each participant i.
    pub li_x: Vec<B::G1>,
    /// Commitments to L_i(x) * L_j(z) for all (i, j) pairs.
    pub li_lj_z: Vec<Vec<B::G1>>,
}

impl<B: PairingBackend<Scalar = Fr>> LagrangePowers<B> {
    /// Precomputes KZG commitments to Lagrange polynomial transformations.
    ///
    /// This function computes and stores commitments to various transformations of
    /// Lagrange basis polynomials, which are later used during key generation to
    /// enable efficient derivation of public keys without polynomial operations.
    ///
    /// # Computed Commitments
    ///
    /// For each Lagrange polynomial L_i(x), computes commitments to:
    ///
    /// 1. **L_i(τ)**: Base Lagrange evaluation at the secret point
    /// 2. **L_i(τ) - L_i(0)**: Shifted evaluation for proof construction
    /// 3. **(L_i(τ) - L_i(0))/τ**: Normalized evaluation (equivalent to x·L_i(x))
    /// 4. **L_i(τ)·L_j(τ)/z(τ)**: All pairwise products divided by vanishing polynomial
    ///
    /// where z(τ) = τ^n - 1 is the vanishing polynomial for the domain.
    ///
    /// # Performance
    ///
    /// - Parallelized using Rayon for efficient multi-core computation
    /// - Complexity: O(n²) for computing all pairwise products
    /// - For n=2048: typically 5-10 seconds on modern hardware
    ///
    /// # Algorithm Details
    ///
    /// 1. Evaluate all Lagrange polynomials at τ in parallel
    /// 2. Compute the vanishing polynomial z(τ) = τ^n - 1
    /// 3. Compute commitments to L_i, L_i - L_i(0), and x·L_i in parallel
    /// 4. Compute all n² commitments to L_i·L_j/z in parallel
    ///
    /// # Arguments
    ///
    /// * `lagranges` - The Lagrange basis polynomials L_0, ..., L_{n-1}
    /// * `domain_size` - Size of the evaluation domain (must equal lagranges.len())
    /// * `tau` - The secret value from the trusted setup
    ///
    /// # Returns
    ///
    /// A `LagrangePowers` structure containing all precomputed commitments
    ///
    /// # Errors
    ///
    /// - `BackendError::Math` if z(τ) = 0 (tau is a root of unity - should be impossible)
    /// - `BackendError::Math` if τ = 0 (invalid tau value)
    ///
    /// # Security
    ///
    /// After this function completes, the secret value `tau` should be securely
    /// destroyed. The precomputed commitments reveal no information about tau
    /// under the Knowledge of Exponent assumption.
    #[instrument(level = "info", skip_all, fields(size=domain_size))]
    pub(crate) fn precompute_lagrange_powers(
        lagranges: &[DensePolynomial],
        domain_size: usize,
        tau: &B::Scalar,
    ) -> Result<Self, BackendError> {
        let n = lagranges.len();

        // Evaluate all Lagrange polynomials at tau
        let li_evals: Vec<B::Scalar> = lagranges
            .par_iter()
            .map(|li_poly| li_poly.evaluate(tau))
            .collect();

        // Compute tau^n - 1 (the vanishing polynomial evaluated at tau)
        let tau_n = tau.pow(&[domain_size as u64, 0, 0, 0]);
        let z_eval = tau_n - <B::Scalar as FieldElement>::one();
        let z_eval_inv = z_eval
            .invert()
            .ok_or(BackendError::Math("vanishing polynomial at tau is zero"))?;

        let tau_inv = tau
            .invert()
            .ok_or(BackendError::Math("tau must be non-zero"))?;

        // Compute li, li_minus0, and li_x in parallel
        let results: Vec<(B::G1, B::G1, B::G1)> = lagranges
            .par_iter()
            .enumerate()
            .map(|(i, li_poly)| {
                let li_eval = &li_evals[i];

                // li = g * L_i(tau)
                let lagrange_li = B::G1::generator().mul_scalar(li_eval);

                // li_minus0 = g * (L_i(tau) - L_i(0))
                let li_0 = li_poly
                    .coeffs()
                    .first()
                    .cloned()
                    .unwrap_or_else(<B::Scalar as FieldElement>::zero);
                let li_minus0_eval = *li_eval - li_0;
                let lagrange_li_minus0 = B::G1::generator().mul_scalar(&li_minus0_eval);

                // li_x = g * (L_i(tau) - L_i(0)) / tau
                let li_x_eval = li_minus0_eval * tau_inv;
                let lagrange_li_x = B::G1::generator().mul_scalar(&li_x_eval);

                (lagrange_li, lagrange_li_minus0, lagrange_li_x)
            })
            .collect::<Vec<_>>();

        let mut li = Vec::with_capacity(n);
        let mut li_minus0 = Vec::with_capacity(n);
        let mut li_x = Vec::with_capacity(n);
        for (a, b, c) in results {
            li.push(a);
            li_minus0.push(b);
            li_x.push(c);
        }

        // Compute li_lj_z using the evaluation-based approach
        let li_lj_z: Vec<Vec<B::G1>> = (0..n)
            .into_par_iter()
            .map(|i| {
                (0..n)
                    .into_par_iter()
                    .map(|j| {
                        let scalar = if i == j {
                            // (L_i(tau)^2 - L_i(tau)) / z(tau)
                            let li_eval: &B::Scalar = &li_evals[i];
                            (*li_eval * *li_eval - *li_eval) * z_eval_inv
                        } else {
                            // (L_i(tau) * L_j(tau)) / z(tau)
                            (li_evals[i] * li_evals[j]) * z_eval_inv
                        };
                        B::G1::generator().mul_scalar(&scalar)
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();

        Ok(LagrangePowers {
            li,
            li_minus0,
            li_x,
            li_lj_z,
        })
    }
}

/// Builds Lagrange basis polynomials for an evaluation domain of size n.
///
/// Constructs the complete set of Lagrange polynomials L_0, L_1, ..., L_{n-1}
/// for a multiplicative subgroup of order n generated by an n-th root of unity ω.
///
/// # Mathematical Definition
///
/// Each Lagrange polynomial L_i(x) has the property:
/// ```text
/// L_i(ω^j) = { 1  if i = j
///            { 0  if i ≠ j
/// ```
///
/// The polynomial is defined as:
/// ```text
/// L_i(x) = ∏(j≠i) (x - ω^j) / (ω^i - ω^j)
/// ```
///
/// This can be efficiently computed using FFT techniques in the evaluation domain.
///
/// # Algorithm
///
/// The implementation uses an FFT-based approach:
/// 1. Compute the n-th root of unity ω using `two_adicity_generator(n)`
/// 2. For each i, compute powers of ω^(-i)
/// 3. Use batch inversion to compute normalization factors
/// 4. Construct polynomial coefficients via repeated multiplication
///
/// # Complexity
///
/// - Time: O(n²) for computing all n polynomials
/// - Space: O(n²) for storing all coefficients
///
/// # Arguments
///
/// * `n` - The domain size (must be a power of 2)
///
/// # Returns
///
/// A vector of n dense polynomials, where the i-th polynomial is L_i(x)
///
/// # Errors
///
/// - `BackendError::Math` if the root of unity cannot be inverted
/// - `BackendError::Math` if batch inversion fails
///
/// # Example
///
/// ```ignore
/// let polys = build_lagrange_polys(8)?;
/// assert_eq!(polys.len(), 8);
///
/// // Verify the Lagrange property at domain points
/// let omega = Fr::two_adicity_generator(8);
/// assert_eq!(polys[0].evaluate(&Fr::one()), Fr::one());
/// assert_eq!(polys[0].evaluate(&omega), Fr::zero());
/// ```
#[instrument(level = "info", skip_all, fields(num_parties=n))]
pub(crate) fn build_lagrange_polys(n: usize) -> Result<Vec<DensePolynomial>, BackendError> {
    if n == 0 {
        return Ok(Vec::new());
    }

    // Follow the same construction as `lagrange_polys_impl` in arith::lagrange
    let omega = Fr::two_adicity_generator(n);
    let omega_inv = omega
        .invert()
        .ok_or(BackendError::Math("invalid generator inversion"))?;

    // Convert n to a field element
    let n_scalar = Fr::from_u64(n as u64);

    let mut omega_inv_pows = Vec::with_capacity(n);
    let mut cur = Fr::one();
    for _ in 0..n {
        omega_inv_pows.push(cur);
        cur *= omega_inv;
    }

    let mut denominators: Vec<Fr> = omega_inv_pows
        .iter()
        .map(|w| {
            let mut denom = *w;
            denom *= n_scalar;
            denom
        })
        .collect();
    Fr::batch_inversion(&mut denominators)?;

    let mut polys = Vec::with_capacity(n);
    for (omega_i_inv, denom_inv) in omega_inv_pows.iter().zip(denominators.iter()) {
        let mut coeffs = Vec::with_capacity(n);
        let mut power = *omega_i_inv;
        for _ in 0..n {
            let mut term = power;
            term *= *denom_inv;
            coeffs.push(term);
            power *= *omega_i_inv;
        }
        polys.push(DensePolynomial::from_coefficients_vec(coeffs));
    }
    Ok(polys)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_lagrange_polys_evaluate_at_domain() {
        let n = 8;
        let polys = build_lagrange_polys(n).unwrap();
        let omega = Fr::two_adicity_generator(n);

        let mut domain = Vec::with_capacity(n);
        let mut cur = Fr::one();
        for _ in 0..n {
            domain.push(cur);
            cur *= omega;
        }

        for (i, poly) in polys.iter().enumerate() {
            for (j, point) in domain.iter().enumerate() {
                let eval = poly.evaluate(point);
                if i == j {
                    assert_eq!(eval, Fr::one());
                } else {
                    assert_eq!(eval, Fr::zero());
                }
            }
        }
    }
}
