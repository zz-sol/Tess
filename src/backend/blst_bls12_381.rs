//! blst-backed concrete implementation for BLS12-381 operations.
//!
//! This module provides the concrete types and implementations for the
//! `BlstBackend` when the `blst` feature is enabled. It implements the
//! `FieldElement`, `CurvePoint`, `TargetGroup`, `Polynomial`, `EvaluationDomain`,
//! `PolynomialCommitment`, `MsmProvider`, and `PairingBackend` traits defined in
//! `crate::backend::mod` using the `blstrs` crate.
//!
//! Exported types include:
//! - `DensePolynomial` - dense coefficient polynomial type used for KZG
//! - `BlstG1`, `BlstG2`, `BlstGt` - wrapped group/target types
//! - `Radix2EvaluationDomain` - FFT domain implementation
//! - `BlstKzg` - KZG commitment implementation
//! - `BlstMsm` - MSM provider
//! - `BlstBackend` - top-level backend type
//!
//! # Feature
//!
//! Compiled when the Cargo feature `blst` is enabled.
//!
//! # Example
//!
//! ```rust,no_run
//! # #[cfg(feature = "blst")]
//! # {
use blstrs::{
    Bls12, Compress, G1Affine, G1Projective, G2Affine, G2Prepared, G2Projective, Gt, Scalar,
};
use ff::{Field, PrimeField};
use group::{Curve, Group, prime::PrimeCurveAffine};
use pairing::{MillerLoopResult as PairingMillerLoopResult, MultiMillerLoop};
use rand_core::RngCore;
use rayon::prelude::*;
use std::io::Cursor;

use crate::backend::{
    CurvePoint, EvaluationDomain, FieldElement, MsmProvider, PairingBackend, Polynomial,
    PolynomialCommitment, TargetGroup,
};
use crate::errors::BackendError;

#[derive(Clone, Debug)]
pub struct DensePolynomial {
    pub coeffs: Vec<Scalar>,
}

impl DensePolynomial {
    /// Create a dense polynomial from the provided coefficient vector.
    ///
    /// The coefficients are in ascending order (constant term first). The
    /// constructor trims leading zero coefficients to keep the representation
    /// canonical.
    pub fn from_coefficients_vec(coeffs: Vec<Scalar>) -> Self {
        let mut poly = DensePolynomial { coeffs };
        poly.truncate_leading_zeros();
        poly
    }

    fn truncate_leading_zeros(&mut self) {
        while self.coeffs.len() > 1 && self.coeffs.last() == Some(&Scalar::ZERO) {
            self.coeffs.pop();
        }
        if self.coeffs.is_empty() {
            self.coeffs.push(Scalar::ZERO);
        }
    }

    pub fn degree(&self) -> usize {
        if self.coeffs.len() == 1 && self.coeffs[0] == Scalar::ZERO {
            0
        } else {
            self.coeffs.len().saturating_sub(1)
        }
    }
    /// Evaluate the polynomial at `point` using Horner's method.
    ///
    /// Returns the field element `p(point)`.
    pub fn evaluate(&self, point: &Scalar) -> Scalar {
        if self.coeffs.is_empty() {
            return Scalar::ZERO;
        }
        let mut result = *self.coeffs.last().unwrap();
        for coeff in self.coeffs.iter().rev().skip(1) {
            result = result * point + coeff;
        }
        result
    }
}

impl FieldElement for Scalar {
    type Repr = Vec<u8>;

    fn zero() -> Self {
        Scalar::ZERO
    }

    fn one() -> Self {
        Scalar::ONE
    }

    fn random<R: RngCore + ?Sized>(rng: &mut R) -> Self {
        <Scalar as Field>::random(rng)
    }

    fn invert(&self) -> Option<Self> {
        Field::invert(self).into()
    }

    fn pow(&self, exp: &[u64; 4]) -> Self {
        self.pow_vartime(exp)
    }

    fn to_repr(&self) -> Self::Repr {
        self.to_bytes_be().to_vec()
    }

    fn from_repr(repr: &Self::Repr) -> Result<Self, BackendError> {
        let mut bytes = [0u8; 32];
        if repr.len() != 32 {
            return Err(BackendError::Serialization("invalid scalar length"));
        }
        bytes.copy_from_slice(repr);
        Option::<Scalar>::from(Scalar::from_bytes_be(&bytes))
            .ok_or(BackendError::Serialization("invalid scalar bytes"))
    }
}

#[derive(Clone, Debug, Copy)]
pub struct BlstG1(pub G1Projective);

#[derive(Clone, Debug, Copy)]
pub struct BlstG2(pub G2Projective);

#[derive(Clone, Debug)]
pub struct BlstGt(pub Gt);

impl CurvePoint<Scalar> for BlstG1 {
    type Affine = G1Affine;

    fn identity() -> Self {
        BlstG1(G1Projective::identity())
    }

    fn generator() -> Self {
        BlstG1(G1Projective::generator())
    }

    fn is_identity(&self) -> bool {
        bool::from(self.0.is_identity())
    }

    fn from_affine(affine: &Self::Affine) -> Self {
        BlstG1(affine.into())
    }

    fn to_affine(&self) -> Self::Affine {
        self.0.into()
    }

    fn add(&self, other: &Self) -> Self {
        BlstG1(self.0 + other.0)
    }

    fn sub(&self, other: &Self) -> Self {
        BlstG1(self.0 - other.0)
    }

    fn negate(&self) -> Self {
        BlstG1(-self.0)
    }

    fn mul_scalar(&self, scalar: &Scalar) -> Self {
        BlstG1(self.0 * scalar)
    }

    fn batch_normalize(points: &[Self]) -> Vec<Self::Affine> {
        let projectives: Vec<G1Projective> = points.iter().map(|p| p.0).collect();
        let mut affines = vec![G1Affine::identity(); projectives.len()];
        G1Projective::batch_normalize(&projectives, &mut affines);
        affines
    }
}

impl CurvePoint<Scalar> for BlstG2 {
    type Affine = G2Affine;

    fn identity() -> Self {
        BlstG2(G2Projective::identity())
    }

    fn generator() -> Self {
        BlstG2(G2Projective::generator())
    }

    fn is_identity(&self) -> bool {
        bool::from(self.0.is_identity())
    }

    fn from_affine(affine: &Self::Affine) -> Self {
        BlstG2(affine.into())
    }

    fn to_affine(&self) -> Self::Affine {
        self.0.into()
    }

    fn add(&self, other: &Self) -> Self {
        BlstG2(self.0 + other.0)
    }

    fn sub(&self, other: &Self) -> Self {
        BlstG2(self.0 - other.0)
    }

    fn negate(&self) -> Self {
        BlstG2(-self.0)
    }

    fn mul_scalar(&self, scalar: &Scalar) -> Self {
        BlstG2(self.0 * scalar)
    }

    fn batch_normalize(points: &[Self]) -> Vec<Self::Affine> {
        let projectives: Vec<G2Projective> = points.iter().map(|p| p.0).collect();
        let mut affines = vec![G2Affine::identity(); projectives.len()];
        G2Projective::batch_normalize(&projectives, &mut affines);
        affines
    }
}

impl TargetGroup for BlstGt {
    type Scalar = Scalar;
    type Repr = Vec<u8>;

    fn identity() -> Self {
        BlstGt(Gt::identity())
    }

    fn mul_scalar(&self, scalar: &Self::Scalar) -> Self {
        BlstGt(self.0 * scalar)
    }

    fn combine(&self, other: &Self) -> Self {
        let mut tmp = self.0;
        tmp += &other.0;
        BlstGt(tmp)
    }

    fn to_repr(&self) -> Self::Repr {
        let mut bytes = Vec::with_capacity(288);
        self.0
            .write_compressed(&mut bytes)
            .map_err(|_| BackendError::Serialization("gt serialization failure"))
            .expect("in-memory serialization should not fail");
        bytes
    }

    fn from_repr(bytes: &Self::Repr) -> Result<Self, BackendError> {
        let mut cursor = Cursor::new(bytes.as_slice());
        Gt::read_compressed(&mut cursor)
            .map(BlstGt)
            .map_err(|_| BackendError::Serialization("invalid GT bytes"))
    }
}

impl Polynomial<BlstBackend> for DensePolynomial {
    fn degree(&self) -> usize {
        self.degree()
    }

    fn coeffs(&self) -> &[Scalar] {
        &self.coeffs
    }

    fn evaluate(&self, point: &Scalar) -> Scalar {
        self.evaluate(point)
    }

    fn truncate(&mut self, len: usize) {
        self.coeffs.truncate(len);
    }
}

#[derive(Clone, Debug)]
pub struct Radix2EvaluationDomain {
    pub size: usize,
    pub group_gen: Scalar,
    pub group_gen_inv: Scalar,
}

impl Radix2EvaluationDomain {
    pub fn new(size: usize) -> Option<Self> {
        if !size.is_power_of_two() || size == 0 {
            return None;
        }
        let log_size = size.trailing_zeros();
        if log_size > 32 {
            return None;
        }
        let mut exp = [0u64; 4];
        exp[0] = 1u64 << (32 - log_size);
        let group_gen = Scalar::ROOT_OF_UNITY.pow_vartime(exp);
        let group_gen_inv = Option::<Scalar>::from(Field::invert(&group_gen))
            .expect("root of unity must be invertible");
        Some(Radix2EvaluationDomain {
            size,
            group_gen,
            group_gen_inv,
        })
    }

    pub fn element_powers(&self) -> Vec<Scalar> {
        let mut current = Scalar::ONE;
        let mut result = Vec::with_capacity(self.size);
        for _ in 0..self.size {
            result.push(current);
            current *= self.group_gen;
        }
        result
    }

    fn fft_in_place(&self, a: &mut [Scalar], generator: Scalar) {
        let n = a.len();
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
            let half = len / 2;
            let stride = self.size / len;
            let mut exp = [0u64; 4];
            exp[0] = stride as u64;
            let omega_step = generator.pow_vartime(exp);
            for start in (0..n).step_by(len) {
                let mut omega = Scalar::ONE;
                for j in 0..half {
                    let u = a[start + j];
                    let v = a[start + j + half] * omega;
                    a[start + j] = u + v;
                    a[start + j + half] = u - v;
                    omega *= omega_step;
                }
            }
            len *= 2;
        }
    }
}

impl EvaluationDomain<BlstBackend> for Radix2EvaluationDomain {
    fn size(&self) -> usize {
        self.size
    }

    fn elements(&self) -> Vec<Scalar> {
        self.element_powers()
    }

    fn fft(&self, coeffs: &[Scalar]) -> Vec<Scalar> {
        let mut a = coeffs.to_vec();
        a.resize(self.size, Scalar::ZERO);
        self.fft_in_place(&mut a, self.group_gen);
        a
    }

    fn ifft(&self, evals: &[Scalar]) -> Vec<Scalar> {
        let mut a = evals.to_vec();
        a.resize(self.size, Scalar::ZERO);
        self.fft_in_place(&mut a, self.group_gen_inv);
        let n_inv = Option::<Scalar>::from(Field::invert(&Scalar::from(self.size as u64)))
            .expect("domain size invertible in scalar field");
        for coeff in a.iter_mut() {
            *coeff *= n_inv;
        }
        a
    }
}

#[derive(Clone, Debug)]
pub struct BlstPowers {
    pub powers_of_g: Vec<G1Affine>,
    pub powers_of_h: Vec<G2Affine>,
    pub e_gh: Gt,
}
/// Helper to construct KZG powers-of-tau parameters for the blst backend.
///
/// Returns `BlstPowers` containing `τ^i * G1`, `τ^i * G2` and the pairing
/// `e(G, H)` for use in polynomial commitment operations.
fn setup_powers(max_degree: usize, tau: &Scalar) -> Result<BlstPowers, BackendError> {
    if max_degree < 1 {
        return Err(BackendError::Math("degree must be >= 1"));
    }
    let g = G1Projective::generator();
    let h = G2Projective::generator();
    let mut powers_of_tau = vec![Scalar::ONE];
    let mut cur = *tau;
    for _ in 0..max_degree {
        powers_of_tau.push(cur);
        cur *= tau;
    }

    let powers_of_g_proj: Vec<G1Projective> =
        powers_of_tau.par_iter().map(|power| g * power).collect();
    let mut powers_of_g = vec![G1Affine::identity(); max_degree + 1];
    G1Projective::batch_normalize(&powers_of_g_proj, &mut powers_of_g);

    let powers_of_h_proj: Vec<G2Projective> =
        powers_of_tau.par_iter().map(|power| h * power).collect();
    let mut powers_of_h = vec![G2Affine::identity(); max_degree + 1];
    G2Projective::batch_normalize(&powers_of_h_proj, &mut powers_of_h);

    let e_gh = blstrs::pairing(&G1Affine::from(g), &G2Affine::from(h));

    Ok(BlstPowers {
        powers_of_g,
        powers_of_h,
        e_gh,
    })
}

#[derive(Debug)]
pub struct BlstKzg;

impl PolynomialCommitment<BlstBackend> for BlstKzg {
    type Parameters = BlstPowers;
    type Polynomial = DensePolynomial;

    fn setup(max_degree: usize, tau: &Scalar) -> Result<Self::Parameters, BackendError> {
        // Construct powers-of-tau parameters for KZG commitments.
        setup_powers(max_degree, tau)
    }

    fn commit_g1(
        params: &Self::Parameters,
        polynomial: &Self::Polynomial,
    ) -> Result<BlstG1, BackendError> {
        let degree = polynomial.degree();
        if degree + 1 > params.powers_of_g.len() {
            return Err(BackendError::Math("polynomial degree too large"));
        }
        let scalars = &polynomial.coeffs[..=degree];
        let bases: Vec<G1Projective> = params.powers_of_g[..=degree]
            .iter()
            .map(G1Projective::from)
            .collect();
        // Compute commitment as multi-exponentiation over prepared G1 powers.
        Ok(BlstG1(G1Projective::multi_exp(&bases, scalars)))
    }

    fn commit_g2(
        params: &Self::Parameters,
        polynomial: &Self::Polynomial,
    ) -> Result<BlstG2, BackendError> {
        let degree = polynomial.degree();
        if degree + 1 > params.powers_of_h.len() {
            return Err(BackendError::Math("polynomial degree too large"));
        }
        let scalars = &polynomial.coeffs[..=degree];
        let bases: Vec<G2Projective> = params.powers_of_h[..=degree]
            .iter()
            .map(G2Projective::from)
            .collect();
        // Compute commitment in G2 via multi-exponentiation.
        Ok(BlstG2(G2Projective::multi_exp(&bases, scalars)))
    }
}

#[derive(Debug)]
pub struct BlstMsm;

impl MsmProvider<BlstBackend> for BlstMsm {
    fn msm_g1(bases: &[BlstG1], scalars: &[Scalar]) -> Result<BlstG1, BackendError> {
        if bases.len() != scalars.len() {
            return Err(BackendError::Math("msm length mismatch"));
        }
        // Perform multi-exponentiation on provided G1 bases and scalars.
        let projectives: Vec<G1Projective> = bases.iter().map(|p| p.0).collect();
        Ok(BlstG1(G1Projective::multi_exp(&projectives, scalars)))
    }

    fn msm_g2(bases: &[BlstG2], scalars: &[Scalar]) -> Result<BlstG2, BackendError> {
        if bases.len() != scalars.len() {
            return Err(BackendError::Math("msm length mismatch"));
        }
        // Perform multi-exponentiation on provided G2 bases and scalars.
        let projectives: Vec<G2Projective> = bases.iter().map(|p| p.0).collect();
        Ok(BlstG2(G2Projective::multi_exp(&projectives, scalars)))
    }
}

#[derive(Clone, Debug, Default)]
pub struct BlstBackend;
/// blst-backed `PairingBackend` implementation for BLS12-381.
///
/// `BlstBackend` ties together scalar types, curve groups, KZG commitment
/// implementation and MSM provider for the `blstrs` backend.

impl PairingBackend for BlstBackend {
    type Scalar = Scalar;
    type G1 = BlstG1;
    type G2 = BlstG2;
    type Target = BlstGt;
    type PolynomialCommitment = BlstKzg;
    type Domain = Radix2EvaluationDomain;
    type Msm = BlstMsm;

    fn pairing(g1: &Self::G1, g2: &Self::G2) -> Self::Target {
        BlstGt(blstrs::pairing(&g1.to_affine(), &g2.to_affine()))
    }

    fn multi_pairing(g1: &[Self::G1], g2: &[Self::G2]) -> Result<Self::Target, BackendError> {
        if g1.len() != g2.len() {
            return Err(BackendError::Math("pairing length mismatch"));
        }
        let lhs_proj: Vec<G1Projective> = g1.iter().map(|p| p.0).collect();
        let rhs_proj: Vec<G2Projective> = g2.iter().map(|p| p.0).collect();
        let mut g1_affine = vec![G1Affine::identity(); lhs_proj.len()];
        let mut g2_affine = vec![G2Affine::identity(); rhs_proj.len()];
        G1Projective::batch_normalize(&lhs_proj, &mut g1_affine);
        G2Projective::batch_normalize(&rhs_proj, &mut g2_affine);
        let g2_prepared: Vec<G2Prepared> =
            g2_affine.iter().map(|aff| G2Prepared::from(*aff)).collect();
        let terms: Vec<_> = g1_affine.iter().zip(g2_prepared.iter()).collect();
        let result = Bls12::multi_miller_loop(&terms).final_exponentiation();
        Ok(BlstGt(result))
    }
}
