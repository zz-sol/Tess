use ark_bn254::{
    Bn254, Fr as BnFr, G1Affine as BnG1Affine, G1Projective as BnG1, G2Affine as BnG2Affine,
    G2Projective as BnG2,
};
use ark_ec::{
    AffineRepr, CurveGroup, PrimeGroup,
    pairing::{Pairing, PairingOutput},
    scalar_mul::variable_base::VariableBaseMSM,
};
use ark_ff::{Field, One, PrimeField, Zero};
use ark_poly::univariate::DensePolynomial;
use ark_poly::{
    EvaluationDomain as ArkEvaluationDomain, Polynomial as ArkPolynomial, Radix2EvaluationDomain,
};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use rand_core::RngCore;
use std::fmt::Debug;

use super::sample_field;
use crate::backend::{
    CurvePoint, EvaluationDomain, FieldElement, MsmProvider, PairingBackend, Polynomial,
    PolynomialCommitment, TargetGroup,
};
use crate::errors::BackendError;

impl FieldElement for BnFr {
    type Repr = Vec<u8>;

    fn zero() -> Self {
        Zero::zero()
    }

    fn one() -> Self {
        One::one()
    }

    fn random<R: RngCore + ?Sized>(rng: &mut R) -> Self {
        sample_field(rng)
    }

    fn invert(&self) -> Option<Self> {
        self.inverse()
    }

    fn pow(&self, exp: &[u64; 4]) -> Self {
        Field::pow(self, exp)
    }

    fn to_repr(&self) -> Self::Repr {
        let mut bytes = Vec::new();
        self.serialize_compressed(&mut bytes)
            .expect("scalar serialization");
        bytes
    }

    fn from_repr(repr: &Self::Repr) -> Result<Self, BackendError> {
        Self::deserialize_compressed(repr.as_slice())
            .map_err(|_| BackendError::Serialization("invalid scalar bytes"))
    }
}

#[derive(Clone, Debug)]
pub struct ArkBnG1(pub BnG1);

#[derive(Clone, Debug)]
pub struct ArkBnG2(pub BnG2);

#[derive(Clone, Debug)]
pub struct ArkBnG1Affine(pub BnG1Affine);

#[derive(Clone, Debug)]
pub struct ArkBnG2Affine(pub BnG2Affine);

#[derive(Clone, Debug)]
pub struct ArkBnGt(pub PairingOutput<Bn254>);

impl CurvePoint<BnFr> for ArkBnG1 {
    type Affine = ArkBnG1Affine;

    fn identity() -> Self {
        ArkBnG1(BnG1::zero())
    }

    fn generator() -> Self {
        ArkBnG1(BnG1::generator())
    }

    fn is_identity(&self) -> bool {
        self.0.is_zero()
    }

    fn from_affine(affine: &Self::Affine) -> Self {
        ArkBnG1(affine.0.into_group())
    }

    fn to_affine(&self) -> Self::Affine {
        ArkBnG1Affine(self.0.into_affine())
    }

    fn add(&self, other: &Self) -> Self {
        let mut tmp = self.0;
        tmp += other.0;
        ArkBnG1(tmp)
    }

    fn sub(&self, other: &Self) -> Self {
        let mut tmp = self.0;
        tmp -= other.0;
        ArkBnG1(tmp)
    }

    fn negate(&self) -> Self {
        ArkBnG1(-self.0)
    }

    fn mul_scalar(&self, scalar: &BnFr) -> Self {
        ArkBnG1(self.0.mul_bigint(scalar.into_bigint()))
    }

    fn batch_normalize(points: &[Self]) -> Vec<Self::Affine> {
        let projectives: Vec<BnG1> = points.iter().map(|p| p.0).collect();
        BnG1::normalize_batch(&projectives)
            .into_iter()
            .map(ArkBnG1Affine)
            .collect()
    }
}

impl CurvePoint<BnFr> for ArkBnG2 {
    type Affine = ArkBnG2Affine;

    fn identity() -> Self {
        ArkBnG2(BnG2::zero())
    }

    fn generator() -> Self {
        ArkBnG2(BnG2::generator())
    }

    fn is_identity(&self) -> bool {
        self.0.is_zero()
    }

    fn from_affine(affine: &Self::Affine) -> Self {
        ArkBnG2(affine.0.into_group())
    }

    fn to_affine(&self) -> Self::Affine {
        ArkBnG2Affine(self.0.into_affine())
    }

    fn add(&self, other: &Self) -> Self {
        let mut tmp = self.0;
        tmp += other.0;
        ArkBnG2(tmp)
    }

    fn sub(&self, other: &Self) -> Self {
        let mut tmp = self.0;
        tmp -= other.0;
        ArkBnG2(tmp)
    }

    fn negate(&self) -> Self {
        ArkBnG2(-self.0)
    }

    fn mul_scalar(&self, scalar: &BnFr) -> Self {
        ArkBnG2(self.0.mul_bigint(scalar.into_bigint()))
    }

    fn batch_normalize(points: &[Self]) -> Vec<Self::Affine> {
        let projectives: Vec<BnG2> = points.iter().map(|p| p.0).collect();
        BnG2::normalize_batch(&projectives)
            .into_iter()
            .map(ArkBnG2Affine)
            .collect()
    }
}

impl TargetGroup for ArkBnGt {
    type Scalar = BnFr;
    type Repr = Vec<u8>;

    fn identity() -> Self {
        ArkBnGt(PairingOutput::<Bn254>::zero())
    }

    fn mul_scalar(&self, scalar: &Self::Scalar) -> Self {
        ArkBnGt(self.0.mul_bigint(scalar.into_bigint()))
    }

    fn combine(&self, other: &Self) -> Self {
        let mut tmp = self.0;
        tmp += &other.0;
        ArkBnGt(tmp)
    }

    fn to_repr(&self) -> Self::Repr {
        let mut bytes = Vec::new();
        self.0
            .serialize_compressed(&mut bytes)
            .expect("target serialization");
        bytes
    }

    fn from_repr(bytes: &Self::Repr) -> Result<Self, BackendError> {
        PairingOutput::<Bn254>::deserialize_compressed(bytes.as_slice())
            .map(ArkBnGt)
            .map_err(|_| BackendError::Serialization("invalid GT bytes"))
    }
}

impl Polynomial<ArkworksBn254> for DensePolynomial<BnFr> {
    fn degree(&self) -> usize {
        <DensePolynomial<BnFr> as ArkPolynomial<BnFr>>::degree(self)
    }

    fn coeffs(&self) -> &[BnFr] {
        &self.coeffs
    }

    fn evaluate(&self, point: &BnFr) -> BnFr {
        <DensePolynomial<BnFr> as ArkPolynomial<BnFr>>::evaluate(self, point)
    }

    fn truncate(&mut self, len: usize) {
        self.coeffs.truncate(len);
    }
}

impl EvaluationDomain<ArkworksBn254> for Radix2EvaluationDomain<BnFr> {
    fn size(&self) -> usize {
        <Radix2EvaluationDomain<BnFr> as ArkEvaluationDomain<BnFr>>::size(self)
    }

    fn elements(&self) -> Vec<BnFr> {
        <Radix2EvaluationDomain<BnFr> as ArkEvaluationDomain<BnFr>>::elements(self).collect()
    }

    fn fft(&self, coeffs: &[BnFr]) -> Vec<BnFr> {
        <Radix2EvaluationDomain<BnFr> as ArkEvaluationDomain<BnFr>>::fft(self, coeffs)
    }

    fn ifft(&self, evals: &[BnFr]) -> Vec<BnFr> {
        <Radix2EvaluationDomain<BnFr> as ArkEvaluationDomain<BnFr>>::ifft(self, evals)
    }
}

#[derive(Clone, Debug)]
pub struct BnPowers {
    pub powers_of_g: Vec<ArkBnG1Affine>,
    pub powers_of_h: Vec<ArkBnG2Affine>,
    pub e_gh: ArkBnGt,
}

fn setup_powers_bn(max_degree: usize, tau: &BnFr) -> Result<BnPowers, BackendError> {
    if max_degree < 1 {
        return Err(BackendError::Math("degree must be >= 1"));
    }

    let g = BnG1::generator();
    let h = BnG2::generator();

    let mut powers_of_tau = vec![<BnFr as One>::one()];
    let mut cur = *tau;
    for _ in 0..max_degree {
        powers_of_tau.push(cur);
        cur *= tau;
    }

    let g_proj: Vec<BnG1> = powers_of_tau
        .iter()
        .map(|power| g.mul_bigint((*power).into_bigint()))
        .collect();
    let powers_of_g = BnG1::normalize_batch(&g_proj)
        .into_iter()
        .map(ArkBnG1Affine)
        .collect();

    let h_proj: Vec<BnG2> = powers_of_tau
        .iter()
        .map(|power| h.mul_bigint((*power).into_bigint()))
        .collect();
    let powers_of_h = BnG2::normalize_batch(&h_proj)
        .into_iter()
        .map(ArkBnG2Affine)
        .collect();

    let e_gh = ArkBnGt(Bn254::pairing(g.into_affine(), h.into_affine()));

    Ok(BnPowers {
        powers_of_g,
        powers_of_h,
        e_gh,
    })
}

fn convert_bn_scalars(scalars: &[BnFr]) -> Vec<<BnFr as PrimeField>::BigInt> {
    scalars.iter().map(|s| (*s).into_bigint()).collect()
}

#[derive(Debug)]
pub struct BnKzg;

impl PolynomialCommitment<ArkworksBn254> for BnKzg {
    type Parameters = BnPowers;
    type Polynomial = DensePolynomial<BnFr>;

    fn setup(max_degree: usize, tau: &BnFr) -> Result<Self::Parameters, BackendError> {
        setup_powers_bn(max_degree, tau)
    }

    fn commit_g1(
        params: &Self::Parameters,
        polynomial: &Self::Polynomial,
    ) -> Result<ArkBnG1, BackendError> {
        let degree = <DensePolynomial<BnFr> as Polynomial<ArkworksBn254>>::degree(polynomial);
        if degree + 1 > params.powers_of_g.len() {
            return Err(BackendError::Math("polynomial degree too large"));
        }

        let scalars = convert_bn_scalars(&polynomial.coeffs[..=degree]);
        let bases: Vec<BnG1Affine> = params
            .powers_of_g
            .iter()
            .take(degree + 1)
            .map(|p| p.0)
            .collect();

        Ok(ArkBnG1(BnG1::msm_bigint(&bases, &scalars)))
    }

    fn commit_g2(
        params: &Self::Parameters,
        polynomial: &Self::Polynomial,
    ) -> Result<ArkBnG2, BackendError> {
        let degree = <DensePolynomial<BnFr> as Polynomial<ArkworksBn254>>::degree(polynomial);
        if degree + 1 > params.powers_of_h.len() {
            return Err(BackendError::Math("polynomial degree too large"));
        }

        let scalars = convert_bn_scalars(&polynomial.coeffs[..=degree]);
        let bases: Vec<BnG2Affine> = params
            .powers_of_h
            .iter()
            .take(degree + 1)
            .map(|p| p.0)
            .collect();

        Ok(ArkBnG2(BnG2::msm_bigint(&bases, &scalars)))
    }
}

#[derive(Debug)]
pub struct BnMsm;

impl MsmProvider<ArkworksBn254> for BnMsm {
    fn msm_g1(bases: &[ArkBnG1], scalars: &[BnFr]) -> Result<ArkBnG1, BackendError> {
        if bases.len() != scalars.len() {
            return Err(BackendError::Math("msm length mismatch"));
        }
        let projectives: Vec<BnG1> = bases.iter().map(|p| p.0).collect();
        let affines = BnG1::normalize_batch(&projectives);
        let coeffs = convert_bn_scalars(scalars);
        Ok(ArkBnG1(BnG1::msm_bigint(&affines, &coeffs)))
    }

    fn msm_g2(bases: &[ArkBnG2], scalars: &[BnFr]) -> Result<ArkBnG2, BackendError> {
        if bases.len() != scalars.len() {
            return Err(BackendError::Math("msm length mismatch"));
        }
        let projectives: Vec<BnG2> = bases.iter().map(|p| p.0).collect();
        let affines = BnG2::normalize_batch(&projectives);
        let coeffs = convert_bn_scalars(scalars);
        Ok(ArkBnG2(BnG2::msm_bigint(&affines, &coeffs)))
    }
}

#[derive(Clone, Debug, Default)]
pub struct ArkworksBn254;

impl PairingBackend for ArkworksBn254 {
    type Scalar = BnFr;
    type G1 = ArkBnG1;
    type G2 = ArkBnG2;
    type Target = ArkBnGt;
    type PolynomialCommitment = BnKzg;
    type Domain = Radix2EvaluationDomain<BnFr>;
    type Msm = BnMsm;

    fn pairing(g1: &Self::G1, g2: &Self::G2) -> Self::Target {
        ArkBnGt(Bn254::pairing(g1.0.into_affine(), g2.0.into_affine()))
    }

    fn multi_pairing(g1: &[Self::G1], g2: &[Self::G2]) -> Result<Self::Target, BackendError> {
        if g1.len() != g2.len() {
            return Err(BackendError::Math("pairing length mismatch"));
        }
        let lhs = g1.iter().map(|p| p.0.into_affine()).collect::<Vec<_>>();
        let rhs = g2.iter().map(|p| p.0.into_affine()).collect::<Vec<_>>();
        Ok(ArkBnGt(Bn254::multi_pairing(lhs, rhs)))
    }
}
