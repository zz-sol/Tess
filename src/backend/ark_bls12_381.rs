use ark_bls12_381::{
    Bls12_381, Fr as BlsFr, G1Affine as RawG1Affine, G1Projective as RawG1,
    G2Affine as RawG2Affine, G2Projective as RawG2,
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

#[cfg(feature = "ark_bls12381")]
impl FieldElement for BlsFr {
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

#[cfg(feature = "ark_bls12381")]
#[derive(Clone, Debug)]
pub struct ArkG1(pub RawG1);

#[cfg(feature = "ark_bls12381")]
#[derive(Clone, Debug)]
pub struct ArkG2(pub RawG2);

#[cfg(feature = "ark_bls12381")]
#[derive(Clone, Debug)]
pub struct ArkG1Affine(pub RawG1Affine);

#[cfg(feature = "ark_bls12381")]
#[derive(Clone, Debug)]
pub struct ArkG2Affine(pub RawG2Affine);

#[cfg(feature = "ark_bls12381")]
#[derive(Clone, Debug)]
pub struct ArkGt(pub PairingOutput<Bls12_381>);

#[cfg(feature = "ark_bls12381")]
impl CurvePoint<BlsFr> for ArkG1 {
    type Affine = ArkG1Affine;

    fn identity() -> Self {
        ArkG1(RawG1::zero())
    }

    fn generator() -> Self {
        ArkG1(RawG1::generator())
    }

    fn from_affine(affine: &Self::Affine) -> Self {
        ArkG1(affine.0.into_group())
    }

    fn to_affine(&self) -> Self::Affine {
        ArkG1Affine(self.0.into_affine())
    }

    fn add(&self, other: &Self) -> Self {
        let mut tmp = self.0;
        tmp += other.0;
        ArkG1(tmp)
    }

    fn sub(&self, other: &Self) -> Self {
        let mut tmp = self.0;
        tmp -= other.0;
        ArkG1(tmp)
    }

    fn negate(&self) -> Self {
        ArkG1(-self.0)
    }

    fn mul_scalar(&self, scalar: &BlsFr) -> Self {
        ArkG1(self.0.mul_bigint(scalar.into_bigint()))
    }

    fn batch_normalize(points: &[Self]) -> Vec<Self::Affine> {
        let projectives: Vec<RawG1> = points.iter().map(|p| p.0).collect();
        RawG1::normalize_batch(&projectives)
            .into_iter()
            .map(ArkG1Affine)
            .collect()
    }
}

#[cfg(feature = "ark_bls12381")]
impl CurvePoint<BlsFr> for ArkG2 {
    type Affine = ArkG2Affine;

    fn identity() -> Self {
        ArkG2(RawG2::zero())
    }

    fn generator() -> Self {
        ArkG2(RawG2::generator())
    }

    fn from_affine(affine: &Self::Affine) -> Self {
        ArkG2(affine.0.into_group())
    }

    fn to_affine(&self) -> Self::Affine {
        ArkG2Affine(self.0.into_affine())
    }

    fn add(&self, other: &Self) -> Self {
        let mut tmp = self.0;
        tmp += other.0;
        ArkG2(tmp)
    }

    fn sub(&self, other: &Self) -> Self {
        let mut tmp = self.0;
        tmp -= other.0;
        ArkG2(tmp)
    }

    fn negate(&self) -> Self {
        ArkG2(-self.0)
    }

    fn mul_scalar(&self, scalar: &BlsFr) -> Self {
        ArkG2(self.0.mul_bigint(scalar.into_bigint()))
    }

    fn batch_normalize(points: &[Self]) -> Vec<Self::Affine> {
        let projectives: Vec<RawG2> = points.iter().map(|p| p.0).collect();
        RawG2::normalize_batch(&projectives)
            .into_iter()
            .map(ArkG2Affine)
            .collect()
    }
}

#[cfg(feature = "ark_bls12381")]
impl TargetGroup for ArkGt {
    type Scalar = BlsFr;
    type Repr = Vec<u8>;

    fn identity() -> Self {
        ArkGt(PairingOutput::<Bls12_381>::zero())
    }

    fn mul_scalar(&self, scalar: &Self::Scalar) -> Self {
        ArkGt(self.0.mul_bigint(scalar.into_bigint()))
    }

    fn combine(&self, other: &Self) -> Self {
        let mut tmp = self.0;
        tmp += &other.0;
        ArkGt(tmp)
    }

    fn to_repr(&self) -> Self::Repr {
        let mut bytes = Vec::new();
        self.0
            .serialize_compressed(&mut bytes)
            .expect("target serialization");
        bytes
    }

    fn from_repr(bytes: &Self::Repr) -> Result<Self, BackendError> {
        PairingOutput::<Bls12_381>::deserialize_compressed(bytes.as_slice())
            .map(ArkGt)
            .map_err(|_| BackendError::Serialization("invalid GT bytes"))
    }
}

#[cfg(feature = "ark_bls12381")]
impl Polynomial<ArkworksBls12> for DensePolynomial<BlsFr> {
    fn degree(&self) -> usize {
        <DensePolynomial<BlsFr> as ArkPolynomial<BlsFr>>::degree(self)
    }

    fn coeffs(&self) -> &[BlsFr] {
        &self.coeffs
    }

    fn evaluate(&self, point: &BlsFr) -> BlsFr {
        <DensePolynomial<BlsFr> as ArkPolynomial<BlsFr>>::evaluate(self, point)
    }

    fn truncate(&mut self, len: usize) {
        self.coeffs.truncate(len);
    }
}

#[cfg(feature = "ark_bls12381")]
impl EvaluationDomain<ArkworksBls12> for Radix2EvaluationDomain<BlsFr> {
    fn size(&self) -> usize {
        <Radix2EvaluationDomain<BlsFr> as ArkEvaluationDomain<BlsFr>>::size(self)
    }

    fn elements(&self) -> Vec<BlsFr> {
        <Radix2EvaluationDomain<BlsFr> as ArkEvaluationDomain<BlsFr>>::elements(self).collect()
    }

    fn fft(&self, coeffs: &[BlsFr]) -> Vec<BlsFr> {
        <Radix2EvaluationDomain<BlsFr> as ArkEvaluationDomain<BlsFr>>::fft(self, coeffs)
    }

    fn ifft(&self, evals: &[BlsFr]) -> Vec<BlsFr> {
        <Radix2EvaluationDomain<BlsFr> as ArkEvaluationDomain<BlsFr>>::ifft(self, evals)
    }
}

#[cfg(feature = "ark_bls12381")]
#[derive(Clone, Debug)]
pub struct BlsPowers {
    pub powers_of_g: Vec<ArkG1Affine>,
    pub powers_of_h: Vec<ArkG2Affine>,
    pub e_gh: ArkGt,
}

#[cfg(feature = "ark_bls12381")]
fn setup_powers_bls(max_degree: usize, tau: &BlsFr) -> Result<BlsPowers, BackendError> {
    if max_degree < 1 {
        return Err(BackendError::Math("degree must be >= 1"));
    }

    let g = RawG1::generator();
    let h = RawG2::generator();

    let mut powers_of_tau = vec![<BlsFr as One>::one()];
    let mut cur = *tau;
    for _ in 0..max_degree {
        powers_of_tau.push(cur);
        cur *= tau;
    }

    let g_proj: Vec<RawG1> = powers_of_tau
        .iter()
        .map(|power| g.mul_bigint((*power).into_bigint()))
        .collect();
    let powers_of_g = RawG1::normalize_batch(&g_proj)
        .into_iter()
        .map(ArkG1Affine)
        .collect();

    let h_proj: Vec<RawG2> = powers_of_tau
        .iter()
        .map(|power| h.mul_bigint((*power).into_bigint()))
        .collect();
    let powers_of_h = RawG2::normalize_batch(&h_proj)
        .into_iter()
        .map(ArkG2Affine)
        .collect();

    let e_gh = ArkGt(Bls12_381::pairing(g.into_affine(), h.into_affine()));

    Ok(BlsPowers {
        powers_of_g,
        powers_of_h,
        e_gh,
    })
}

#[cfg(feature = "ark_bls12381")]
fn convert_scalars(scalars: &[BlsFr]) -> Vec<<BlsFr as PrimeField>::BigInt> {
    scalars.iter().map(|s| (*s).into_bigint()).collect()
}

#[cfg(feature = "ark_bls12381")]
#[derive(Debug)]
pub struct BlsKzg;

#[cfg(feature = "ark_bls12381")]
impl PolynomialCommitment<ArkworksBls12> for BlsKzg {
    type Parameters = BlsPowers;
    type Polynomial = DensePolynomial<BlsFr>;

    fn setup(max_degree: usize, tau: &BlsFr) -> Result<Self::Parameters, BackendError> {
        setup_powers_bls(max_degree, tau)
    }

    fn commit_g1(
        params: &Self::Parameters,
        polynomial: &Self::Polynomial,
    ) -> Result<ArkG1, BackendError> {
        let degree = <DensePolynomial<BlsFr> as Polynomial<ArkworksBls12>>::degree(polynomial);
        if degree + 1 > params.powers_of_g.len() {
            return Err(BackendError::Math("polynomial degree too large"));
        }

        let scalars = convert_scalars(&polynomial.coeffs[..=degree]);
        let bases: Vec<RawG1Affine> = params
            .powers_of_g
            .iter()
            .take(degree + 1)
            .map(|p| p.0)
            .collect();

        Ok(ArkG1(RawG1::msm_bigint(&bases, &scalars)))
    }

    fn commit_g2(
        params: &Self::Parameters,
        polynomial: &Self::Polynomial,
    ) -> Result<ArkG2, BackendError> {
        let degree = <DensePolynomial<BlsFr> as Polynomial<ArkworksBls12>>::degree(polynomial);
        if degree + 1 > params.powers_of_h.len() {
            return Err(BackendError::Math("polynomial degree too large"));
        }

        let scalars = convert_scalars(&polynomial.coeffs[..=degree]);
        let bases: Vec<RawG2Affine> = params
            .powers_of_h
            .iter()
            .take(degree + 1)
            .map(|p| p.0)
            .collect();

        Ok(ArkG2(RawG2::msm_bigint(&bases, &scalars)))
    }
}

#[cfg(feature = "ark_bls12381")]
#[derive(Debug)]
pub struct BlsMsm;

#[cfg(feature = "ark_bls12381")]
impl MsmProvider<ArkworksBls12> for BlsMsm {
    fn msm_g1(bases: &[ArkG1], scalars: &[BlsFr]) -> Result<ArkG1, BackendError> {
        if bases.len() != scalars.len() {
            return Err(BackendError::Math("msm length mismatch"));
        }
        let projectives: Vec<RawG1> = bases.iter().map(|p| p.0).collect();
        let affines = RawG1::normalize_batch(&projectives);
        let coeffs = convert_scalars(scalars);
        Ok(ArkG1(RawG1::msm_bigint(&affines, &coeffs)))
    }

    fn msm_g2(bases: &[ArkG2], scalars: &[BlsFr]) -> Result<ArkG2, BackendError> {
        if bases.len() != scalars.len() {
            return Err(BackendError::Math("msm length mismatch"));
        }
        let projectives: Vec<RawG2> = bases.iter().map(|p| p.0).collect();
        let affines = RawG2::normalize_batch(&projectives);
        let coeffs = convert_scalars(scalars);
        Ok(ArkG2(RawG2::msm_bigint(&affines, &coeffs)))
    }
}

#[cfg(feature = "ark_bls12381")]
#[derive(Clone, Debug, Default)]
pub struct ArkworksBls12;

#[cfg(feature = "ark_bls12381")]
impl PairingBackend for ArkworksBls12 {
    type Scalar = BlsFr;
    type G1 = ArkG1;
    type G2 = ArkG2;
    type Target = ArkGt;
    type PolynomialCommitment = BlsKzg;
    type Domain = Radix2EvaluationDomain<BlsFr>;
    type Msm = BlsMsm;

    fn pairing(g1: &Self::G1, g2: &Self::G2) -> Self::Target {
        ArkGt(Bls12_381::pairing(g1.0.into_affine(), g2.0.into_affine()))
    }

    fn multi_pairing(g1: &[Self::G1], g2: &[Self::G2]) -> Result<Self::Target, BackendError> {
        if g1.len() != g2.len() {
            return Err(BackendError::Math("pairing length mismatch"));
        }
        let lhs = g1
            .iter()
            .map(|p| p.0.into_affine())
            .collect::<Vec<_>>();
        let rhs = g2
            .iter()
            .map(|p| p.0.into_affine())
            .collect::<Vec<_>>();
        Ok(ArkGt(Bls12_381::multi_pairing(lhs, rhs)))
    }
}
