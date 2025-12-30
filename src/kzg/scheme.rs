//! KZG polynomial commitment scheme implementation.
//!
//! This module provides the core implementation of Kate-Zaverucha-Goldberg (KZG)
//! polynomial commitments, which are used throughout the TESS protocol for:
//!
//! - Committing to polynomials representing secret shares
//! - Generating proofs of polynomial evaluations
//! - Verifying ciphertext correctness
//!
//! # Overview
//!
//! KZG commitments are a polynomial commitment scheme with the following properties:
//!
//! - **Succinct**: Commitments are single group elements (constant size)
//! - **Efficient**: Evaluation proofs are also single group elements
//! - **Binding**: Computationally infeasible to open to different polynomial
//! - **Hiding**: Commitments reveal no information about the polynomial
//!
//! # Structure
//!
//! The module provides:
//!
//! - [`KZG`]: The main commitment scheme implementing [`PolynomialCommitment`]
//! - [`SRS`]: Structured Reference String containing powers of tau
//!
//! # Security
//!
//! KZG security relies on:
//! - The discrete logarithm problem in the chosen pairing group
//! - The Knowledge of Exponent (KEA) assumption
//! - Secure generation and destruction of the secret tau value
//!
//! # Example
//!
//! ```rust
//! use rand::thread_rng;
//! use tess::{DensePolynomial, Fr, KZG, PairingEngine, Polynomial, PolynomialCommitment, FieldElement};
//!
//! let mut rng = thread_rng();
//!
//! // Generate SRS (trusted setup)
//! let seed = [0u8; 32];
//! let srs = <KZG as PolynomialCommitment<PairingEngine>>::setup(100, &seed).unwrap();
//!
//! // Commit to a polynomial
//! let coeffs = vec![Fr::one(), Fr::from_u64(2), Fr::from_u64(3)];
//! let poly = DensePolynomial::from_coefficients_vec(coeffs);
//! let commitment = <KZG as PolynomialCommitment<PairingEngine>>::commit_g1(&srs, &poly).unwrap();
//! ```

use alloc::string::String;
use alloc::vec::Vec;
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
#[cfg(feature = "parallel")]
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};

use crate::CurvePoint;
use crate::TargetGroup;
use crate::{
    BackendError, DensePolynomial, FieldElement, Fr, PairingBackend, Polynomial,
    PolynomialCommitment,
};

/// KZG polynomial commitment scheme implementation.
///
/// This is a zero-sized type that implements the [`PolynomialCommitment`] trait,
/// providing Kate-Zaverucha-Goldberg commitment functionality.
///
/// # Overview
///
/// KZG commitments allow committing to a polynomial in a way that:
/// - The commitment is succinct (constant size, single group element)
/// - One can efficiently prove evaluations at any point
/// - The scheme is binding and hiding under the Knowledge of Exponent assumption
///
/// # Example
///
/// ```rust
/// use rand_core::RngCore;
/// use tess::{DensePolynomial, Fr, KZG, PairingBackend, PairingEngine, Polynomial, PolynomialCommitment, FieldElement};
///
/// // Setup: generate SRS with max degree 10
/// let mut seed = [0u8; 32];
/// rand::thread_rng().fill_bytes(&mut seed);
/// let srs = <KZG as PolynomialCommitment<PairingEngine>>::setup(10, &seed).unwrap();
///
/// // Create a polynomial
/// let coeffs = vec![Fr::from_u64(1), Fr::from_u64(2), Fr::from_u64(3)];
/// let poly = DensePolynomial::from_coefficients_vec(coeffs);
///
/// // Commit to the polynomial in G1
/// let commitment = <KZG as PolynomialCommitment<PairingEngine>>::commit_g1(&srs, &poly).unwrap();
/// println!("Commitment: {:?}", commitment);
/// ```
#[derive(Debug)]
pub struct KZG;

impl KZG {
    fn ensure_unique_points(points: &[Fr]) -> Result<(), BackendError> {
        for i in 0..points.len() {
            for j in (i + 1)..points.len() {
                if points[i] == points[j] {
                    return Err(BackendError::Math("duplicate points in batch opening"));
                }
            }
        }
        Ok(())
    }

    fn vanishing_polynomial(points: &[Fr]) -> DensePolynomial {
        let mut poly = DensePolynomial::from_coefficients_vec(vec![Fr::one()]);
        for point in points {
            poly = poly.mul_by_linear(*point);
        }
        poly
    }

    fn interpolate_remainder(
        points: &[Fr],
        values: &[Fr],
    ) -> Result<DensePolynomial, BackendError> {
        if points.len() != values.len() {
            return Err(BackendError::Math(
                "batch opening: mismatched points and values",
            ));
        }

        Self::ensure_unique_points(points)?;

        let vanishing = Self::vanishing_polynomial(points);
        let mut remainder = DensePolynomial::from_coefficients_vec(vec![Fr::zero()]);

        for (idx, point) in points.iter().enumerate() {
            let (numerator, rem) = vanishing.divide_by_linear(*point);
            if rem != Fr::zero() {
                return Err(BackendError::Math(
                    "non-zero remainder in vanishing division",
                ));
            }
            let denom = numerator.evaluate(point);
            let denom_inv = denom
                .invert()
                .ok_or(BackendError::Math("duplicate points in batch opening"))?;
            let scaled = &numerator * (values[idx] * denom_inv);
            remainder = remainder + scaled;
        }

        Ok(remainder)
    }
}

/// Structured Reference String (SRS) for KZG commitments.
///
/// The SRS contains precomputed powers of tau in both G1 and G2 groups,
/// which are used for polynomial commitments and proof generation.
///
/// # Fields
///
/// - `powers_of_g`: Powers of tau in G1: `[g, g*τ, g*τ², ..., g*τⁿ]`
/// - `powers_of_h`: Powers of tau in G2: `[h, h*τ, h*τ², ..., h*τⁿ]`
/// - `e_gh`: Precomputed pairing `e(g, h)` for efficient verification
///
/// # Security
///
/// The SRS is generated from a secret tau (τ) which must be securely discarded
/// after setup. In production, use a multi-party computation ceremony to ensure
/// that no single party knows tau.
///
/// # Example
///
/// ```rust
/// use tess::{Fr, PairingEngine, SRS, FieldElement};
/// use rand::thread_rng;
///
/// let mut rng = thread_rng();
/// let tau = Fr::random(&mut rng);
///
/// // UNSAFE: Only for testing! In production, use MPC ceremony
/// let srs = SRS::<PairingEngine>::new_unsafe(&tau, 100).unwrap();
/// println!("SRS size: G1={}, G2={}", srs.powers_of_g.len(), srs.powers_of_h.len());
/// ```
#[derive(Debug)]
pub struct SRS<B: PairingBackend<Scalar = Fr>> {
    /// Powers of tau in G1: g * tau^i for i = 0..max_degree.
    pub powers_of_g: Vec<B::G1>,
    /// Powers of tau in G2: h * tau^i for i = 0..max_degree.
    pub powers_of_h: Vec<B::G2>,
    /// Precomputed pairing e(g, h) for verification.
    pub e_gh: B::Target,
}

impl<B: PairingBackend<Scalar = Fr>> Clone for SRS<B>
where
    B::G1: Clone,
    B::G2: Clone,
    B::Target: Clone,
{
    fn clone(&self) -> Self {
        Self {
            powers_of_g: self.powers_of_g.clone(),
            powers_of_h: self.powers_of_h.clone(),
            e_gh: self.e_gh.clone(),
        }
    }
}

impl<B: PairingBackend<Scalar = Fr>> SRS<B> {
    /// Creates a new SRS from a secret tau value.
    ///
    /// # Security Warning
    ///
    /// This function is marked as `unsafe` in its naming because the secret `tau` value
    /// must be securely discarded after calling this function. In production deployments,
    /// this should only be used within a multi-party computation (MPC) ceremony where no
    /// single party knows the full tau value.
    ///
    /// # Parameters
    ///
    /// - `tau`: The secret trapdoor value (must be non-zero)
    /// - `max_degree`: Maximum polynomial degree supported by this SRS
    ///
    /// # Returns
    ///
    /// Returns an SRS containing:
    /// - `powers_of_g[i] = g * τ^i` for i = 0..max_degree
    /// - `powers_of_h[i] = h * τ^i` for i = 0..max_degree
    /// - `e_gh = e(g, h)` (precomputed pairing)
    ///
    /// # Example
    ///
    /// ```rust
    /// use tess::{Fr, PairingEngine, SRS, FieldElement};
    /// use rand::thread_rng;
    ///
    /// let mut rng = thread_rng();
    /// let tau = Fr::random(&mut rng);
    ///
    /// // TESTING ONLY - do not use in production!
    /// let srs = SRS::<PairingEngine>::new_unsafe(&tau, 100).unwrap();
    ///
    /// // In production, immediately drop tau after SRS generation
    /// // and preferably use MPC to generate it
    /// ```
    pub fn new_unsafe(tau: &B::Scalar, max_degree: usize) -> Result<Self, String> {
        if max_degree < 1 {
            return Err(String::from("SRS setup failed"));
        }

        let g = B::G1::generator();
        let h = B::G2::generator();

        let mut powers_of_tau = Vec::with_capacity(max_degree + 1);
        powers_of_tau.push(<B::Scalar as FieldElement>::one());
        let mut cur = *tau;
        for _ in 0..max_degree {
            powers_of_tau.push(cur);
            cur *= tau;
        }

        let powers_of_g: Vec<B::G1> = {
            #[cfg(feature = "parallel")]
            {
                powers_of_tau
                    .par_iter()
                    .map(|power| g.mul_scalar(power))
                    .collect()
            }
            #[cfg(not(feature = "parallel"))]
            {
                powers_of_tau
                    .iter()
                    .map(|power| g.mul_scalar(power))
                    .collect()
            }
        };

        let powers_of_h: Vec<B::G2> = {
            #[cfg(feature = "parallel")]
            {
                powers_of_tau
                    .par_iter()
                    .map(|power| h.mul_scalar(power))
                    .collect()
            }
            #[cfg(not(feature = "parallel"))]
            {
                powers_of_tau
                    .iter()
                    .map(|power| h.mul_scalar(power))
                    .collect()
            }
        };

        let e_gh = B::pairing(&g, &h);
        wipe_scalars(&mut powers_of_tau);

        Ok(SRS {
            powers_of_g,
            powers_of_h,
            e_gh,
        })
    }
}

impl<B: PairingBackend<Scalar = Fr>> PolynomialCommitment<B> for KZG {
    type Parameters = SRS<B>;
    type Polynomial = DensePolynomial;

    fn setup(max_degree: usize, seed: &[u8; 32]) -> Result<Self::Parameters, BackendError> {
        let mut rng = ChaCha20Rng::from_seed(*seed);
        let tau = Fr::random(&mut rng);
        SRS::new_unsafe(&tau, max_degree).map_err(BackendError::Other)
    }

    fn commit_g1(
        params: &Self::Parameters,
        polynomial: &Self::Polynomial,
    ) -> Result<B::G1, BackendError> {
        let degree = polynomial.degree();
        if degree + 1 > params.powers_of_g.len() {
            return Err(BackendError::Math("polynomial degree too large"));
        }
        let scalars = &polynomial.coeffs()[..=degree];
        let commitment =
            B::G1::multi_scalar_multiplication(&params.powers_of_g[..=degree], scalars);
        Ok(commitment)
    }

    fn commit_g2(
        params: &Self::Parameters,
        polynomial: &Self::Polynomial,
    ) -> Result<B::G2, BackendError> {
        let degree = polynomial.degree();
        if degree + 1 > params.powers_of_h.len() {
            return Err(BackendError::Math("polynomial degree too large"));
        }
        let scalars = &polynomial.coeffs()[..=degree];
        let commitment =
            B::G2::multi_scalar_multiplication(&params.powers_of_h[..=degree], scalars);
        Ok(commitment)
    }

    fn open_g1(
        params: &Self::Parameters,
        polynomial: &Self::Polynomial,
        point: &B::Scalar,
    ) -> Result<(B::Scalar, B::G1), BackendError> {
        if params.powers_of_g.is_empty() {
            return Err(BackendError::Math("missing SRS powers"));
        }

        let value = polynomial.evaluate(point);
        if polynomial.degree() == 0 {
            return Ok((value, B::G1::identity()));
        }

        let mut shifted = polynomial.clone();
        if let Some(constant) = shifted.coeffs.get_mut(0) {
            *constant -= value;
        }
        let (quotient, remainder) = shifted.divide_by_linear(*point);
        if remainder != Fr::zero() {
            return Err(BackendError::Math("non-zero remainder in opening"));
        }
        let proof = Self::commit_g1(params, &quotient)?;
        Ok((value, proof))
    }

    fn verify_g1(
        params: &Self::Parameters,
        commitment: &B::G1,
        point: &B::Scalar,
        value: &B::Scalar,
        proof: &B::G1,
    ) -> Result<bool, BackendError> {
        if params.powers_of_h.len() < 2 {
            return Err(BackendError::Math("insufficient SRS powers"));
        }

        let g = B::G1::generator();
        let h = params.powers_of_h[0];
        let h_tau = params.powers_of_h[1];

        // Verify: e(C - g*v, h) == e(π, h*τ - h*z)
        // Using multi-pairing: e(C - g*v, h) * e(-π, h*τ - h*z) == 1
        let lhs = commitment.sub(&g.mul_scalar(value));
        let neg_proof = proof.negate();
        let rhs = h_tau.sub(&h.mul_scalar(point));

        let result = B::multi_pairing(&[lhs, neg_proof], &[h, rhs])?;
        Ok(result == <B::Target as TargetGroup>::identity())
    }

    fn batch_open_g1(
        params: &Self::Parameters,
        polynomial: &Self::Polynomial,
        points: &[B::Scalar],
    ) -> Result<(Vec<B::Scalar>, B::G1), BackendError> {
        if points.is_empty() {
            return Ok((Vec::new(), B::G1::identity()));
        }

        let mut values = Vec::with_capacity(points.len());
        for point in points {
            values.push(polynomial.evaluate(point));
        }

        let remainder = Self::interpolate_remainder(points, &values)?;
        let vanishing = Self::vanishing_polynomial(points);
        let diff = polynomial.clone() - remainder;
        let quotient = &diff / &vanishing;
        let proof = Self::commit_g1(params, &quotient)?;

        Ok((values, proof))
    }

    fn batch_verify_g1(
        params: &Self::Parameters,
        commitment: &B::G1,
        points: &[B::Scalar],
        values: &[B::Scalar],
        proof: &B::G1,
    ) -> Result<bool, BackendError> {
        if points.len() != values.len() {
            return Err(BackendError::Math(
                "batch verify: mismatched points and values",
            ));
        }

        if points.is_empty() {
            return Ok(true);
        }

        if params.powers_of_h.is_empty() {
            return Err(BackendError::Math("insufficient SRS powers"));
        }

        let h = params.powers_of_h[0];

        let remainder = Self::interpolate_remainder(points, values)?;
        let vanishing = Self::vanishing_polynomial(points);
        let remainder_commitment = Self::commit_g1(params, &remainder)?;
        let vanishing_commitment = Self::commit_g2(params, &vanishing)?;

        let lhs = commitment.sub(&remainder_commitment);
        let neg_proof = proof.negate();

        let result = B::multi_pairing(&[lhs, neg_proof], &[h, vanishing_commitment])?;
        Ok(result == <B::Target as TargetGroup>::identity())
    }
}

fn wipe_scalars<F: FieldElement + Copy>(scalars: &mut [F]) {
    let zero = F::zero();
    for scalar in scalars {
        unsafe { core::ptr::write_volatile(scalar, zero) };
    }
}
