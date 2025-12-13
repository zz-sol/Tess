#[cfg(feature = "ark_bls12381")]
use ark_bls12_381::{
    Bls12_381, Fr as BlsFr, G1Affine as RawG1Affine, G1Projective as RawG1,
    G2Affine as RawG2Affine, G2Projective as RawG2,
};
#[cfg(feature = "ark_bn254")]
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

use crate::backend::{
    CurvePoint, EvaluationDomain, FieldElement, MsmProvider, PairingBackend, Polynomial,
    PolynomialCommitment, TargetGroup,
};
use crate::errors::BackendError;

fn sample_field<F: PrimeField, R: RngCore + ?Sized>(rng: &mut R) -> F {
    let mut bytes = vec![0u8; ((F::MODULUS_BIT_SIZE + 7) / 8) as usize];
    rng.fill_bytes(&mut bytes);
    F::from_le_bytes_mod_order(&bytes)
}

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

#[cfg(feature = "ark_bn254")]
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
        let mut tmp = self.0.clone();
        tmp += other.0.clone();
        ArkG1(tmp)
    }

    fn sub(&self, other: &Self) -> Self {
        let mut tmp = self.0.clone();
        tmp -= other.0.clone();
        ArkG1(tmp)
    }

    fn negate(&self) -> Self {
        ArkG1(-self.0.clone())
    }

    fn mul_scalar(&self, scalar: &BlsFr) -> Self {
        ArkG1(self.0.mul_bigint(scalar.into_bigint()))
    }

    fn batch_normalize(points: &[Self]) -> Vec<Self::Affine> {
        let projectives: Vec<RawG1> = points.iter().map(|p| p.0.clone()).collect();
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
        let mut tmp = self.0.clone();
        tmp += other.0.clone();
        ArkG2(tmp)
    }

    fn sub(&self, other: &Self) -> Self {
        let mut tmp = self.0.clone();
        tmp -= other.0.clone();
        ArkG2(tmp)
    }

    fn negate(&self) -> Self {
        ArkG2(-self.0.clone())
    }

    fn mul_scalar(&self, scalar: &BlsFr) -> Self {
        ArkG2(self.0.mul_bigint(scalar.into_bigint()))
    }

    fn batch_normalize(points: &[Self]) -> Vec<Self::Affine> {
        let projectives: Vec<RawG2> = points.iter().map(|p| p.0.clone()).collect();
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
        let mut tmp = self.0.clone();
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
    let mut cur = tau.clone();
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
            .map(|p| p.0.clone())
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
            .map(|p| p.0.clone())
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
        let projectives: Vec<RawG1> = bases.iter().map(|p| p.0.clone()).collect();
        let affines = RawG1::normalize_batch(&projectives);
        let coeffs = convert_scalars(scalars);
        Ok(ArkG1(RawG1::msm_bigint(&affines, &coeffs)))
    }

    fn msm_g2(bases: &[ArkG2], scalars: &[BlsFr]) -> Result<ArkG2, BackendError> {
        if bases.len() != scalars.len() {
            return Err(BackendError::Math("msm length mismatch"));
        }
        let projectives: Vec<RawG2> = bases.iter().map(|p| p.0.clone()).collect();
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
        ArkGt(Bls12_381::pairing(
            g1.0.clone().into_affine(),
            g2.0.clone().into_affine(),
        ))
    }

    fn multi_pairing(g1: &[Self::G1], g2: &[Self::G2]) -> Result<Self::Target, BackendError> {
        if g1.len() != g2.len() {
            return Err(BackendError::Math("pairing length mismatch"));
        }
        let lhs = g1
            .iter()
            .map(|p| p.0.clone().into_affine())
            .collect::<Vec<_>>();
        let rhs = g2
            .iter()
            .map(|p| p.0.clone().into_affine())
            .collect::<Vec<_>>();
        Ok(ArkGt(Bls12_381::multi_pairing(lhs, rhs)))
    }
}

#[cfg(feature = "ark_bn254")]
#[derive(Clone, Debug)]
pub struct ArkBnG1(pub BnG1);

#[cfg(feature = "ark_bn254")]
#[derive(Clone, Debug)]
pub struct ArkBnG2(pub BnG2);

#[cfg(feature = "ark_bn254")]
#[derive(Clone, Debug)]
pub struct ArkBnG1Affine(pub BnG1Affine);

#[cfg(feature = "ark_bn254")]
#[derive(Clone, Debug)]
pub struct ArkBnG2Affine(pub BnG2Affine);

#[cfg(feature = "ark_bn254")]
#[derive(Clone, Debug)]
pub struct ArkBnGt(pub PairingOutput<Bn254>);

#[cfg(feature = "ark_bn254")]
impl CurvePoint<BnFr> for ArkBnG1 {
    type Affine = ArkBnG1Affine;

    fn identity() -> Self {
        ArkBnG1(BnG1::zero())
    }

    fn generator() -> Self {
        ArkBnG1(BnG1::generator())
    }

    fn from_affine(affine: &Self::Affine) -> Self {
        ArkBnG1(affine.0.into_group())
    }

    fn to_affine(&self) -> Self::Affine {
        ArkBnG1Affine(self.0.into_affine())
    }

    fn add(&self, other: &Self) -> Self {
        let mut tmp = self.0.clone();
        tmp += other.0.clone();
        ArkBnG1(tmp)
    }

    fn sub(&self, other: &Self) -> Self {
        let mut tmp = self.0.clone();
        tmp -= other.0.clone();
        ArkBnG1(tmp)
    }

    fn negate(&self) -> Self {
        ArkBnG1(-self.0.clone())
    }

    fn mul_scalar(&self, scalar: &BnFr) -> Self {
        ArkBnG1(self.0.mul_bigint(scalar.into_bigint()))
    }

    fn batch_normalize(points: &[Self]) -> Vec<Self::Affine> {
        let projectives: Vec<BnG1> = points.iter().map(|p| p.0.clone()).collect();
        BnG1::normalize_batch(&projectives)
            .into_iter()
            .map(ArkBnG1Affine)
            .collect()
    }
}

#[cfg(feature = "ark_bn254")]
impl CurvePoint<BnFr> for ArkBnG2 {
    type Affine = ArkBnG2Affine;

    fn identity() -> Self {
        ArkBnG2(BnG2::zero())
    }

    fn generator() -> Self {
        ArkBnG2(BnG2::generator())
    }

    fn from_affine(affine: &Self::Affine) -> Self {
        ArkBnG2(affine.0.into_group())
    }

    fn to_affine(&self) -> Self::Affine {
        ArkBnG2Affine(self.0.into_affine())
    }

    fn add(&self, other: &Self) -> Self {
        let mut tmp = self.0.clone();
        tmp += other.0.clone();
        ArkBnG2(tmp)
    }

    fn sub(&self, other: &Self) -> Self {
        let mut tmp = self.0.clone();
        tmp -= other.0.clone();
        ArkBnG2(tmp)
    }

    fn negate(&self) -> Self {
        ArkBnG2(-self.0.clone())
    }

    fn mul_scalar(&self, scalar: &BnFr) -> Self {
        ArkBnG2(self.0.mul_bigint(scalar.into_bigint()))
    }

    fn batch_normalize(points: &[Self]) -> Vec<Self::Affine> {
        let projectives: Vec<BnG2> = points.iter().map(|p| p.0.clone()).collect();
        BnG2::normalize_batch(&projectives)
            .into_iter()
            .map(ArkBnG2Affine)
            .collect()
    }
}

#[cfg(feature = "ark_bn254")]
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
        let mut tmp = self.0.clone();
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

#[cfg(feature = "ark_bn254")]
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

#[cfg(feature = "ark_bn254")]
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

#[cfg(feature = "ark_bn254")]
#[derive(Clone, Debug)]
pub struct BnPowers {
    pub powers_of_g: Vec<ArkBnG1Affine>,
    pub powers_of_h: Vec<ArkBnG2Affine>,
    pub e_gh: ArkBnGt,
}

#[cfg(feature = "ark_bn254")]
fn setup_powers_bn(max_degree: usize, tau: &BnFr) -> Result<BnPowers, BackendError> {
    if max_degree < 1 {
        return Err(BackendError::Math("degree must be >= 1"));
    }

    let g = BnG1::generator();
    let h = BnG2::generator();

    let mut powers_of_tau = vec![<BnFr as One>::one()];
    let mut cur = tau.clone();
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

#[cfg(feature = "ark_bn254")]
fn convert_bn_scalars(scalars: &[BnFr]) -> Vec<<BnFr as PrimeField>::BigInt> {
    scalars.iter().map(|s| (*s).into_bigint()).collect()
}

#[cfg(feature = "ark_bn254")]
#[derive(Debug)]
pub struct BnKzg;

#[cfg(feature = "ark_bn254")]
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
            .map(|p| p.0.clone())
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
            .map(|p| p.0.clone())
            .collect();

        Ok(ArkBnG2(BnG2::msm_bigint(&bases, &scalars)))
    }
}

#[cfg(feature = "ark_bn254")]
#[derive(Debug)]
pub struct BnMsm;

#[cfg(feature = "ark_bn254")]
impl MsmProvider<ArkworksBn254> for BnMsm {
    fn msm_g1(bases: &[ArkBnG1], scalars: &[BnFr]) -> Result<ArkBnG1, BackendError> {
        if bases.len() != scalars.len() {
            return Err(BackendError::Math("msm length mismatch"));
        }
        let projectives: Vec<BnG1> = bases.iter().map(|p| p.0.clone()).collect();
        let affines = BnG1::normalize_batch(&projectives);
        let coeffs = convert_bn_scalars(scalars);
        Ok(ArkBnG1(BnG1::msm_bigint(&affines, &coeffs)))
    }

    fn msm_g2(bases: &[ArkBnG2], scalars: &[BnFr]) -> Result<ArkBnG2, BackendError> {
        if bases.len() != scalars.len() {
            return Err(BackendError::Math("msm length mismatch"));
        }
        let projectives: Vec<BnG2> = bases.iter().map(|p| p.0.clone()).collect();
        let affines = BnG2::normalize_batch(&projectives);
        let coeffs = convert_bn_scalars(scalars);
        Ok(ArkBnG2(BnG2::msm_bigint(&affines, &coeffs)))
    }
}

#[cfg(feature = "ark_bn254")]
#[derive(Clone, Debug, Default)]
pub struct ArkworksBn254;

#[cfg(feature = "ark_bn254")]
impl PairingBackend for ArkworksBn254 {
    type Scalar = BnFr;
    type G1 = ArkBnG1;
    type G2 = ArkBnG2;
    type Target = ArkBnGt;
    type PolynomialCommitment = BnKzg;
    type Domain = Radix2EvaluationDomain<BnFr>;
    type Msm = BnMsm;

    fn pairing(g1: &Self::G1, g2: &Self::G2) -> Self::Target {
        ArkBnGt(Bn254::pairing(
            g1.0.clone().into_affine(),
            g2.0.clone().into_affine(),
        ))
    }

    fn multi_pairing(g1: &[Self::G1], g2: &[Self::G2]) -> Result<Self::Target, BackendError> {
        if g1.len() != g2.len() {
            return Err(BackendError::Math("pairing length mismatch"));
        }
        let lhs = g1
            .iter()
            .map(|p| p.0.clone().into_affine())
            .collect::<Vec<_>>();
        let rhs = g2
            .iter()
            .map(|p| p.0.clone().into_affine())
            .collect::<Vec<_>>();
        Ok(ArkBnGt(Bn254::multi_pairing(lhs, rhs)))
    }
}

#[cfg(all(test, feature = "ark_bn254"))]
mod tests {
    use super::*;
    use ark_ec::AffineRepr;
    use ark_poly::DenseUVPolynomial;
    use ark_std::UniformRand;
    use rand::{SeedableRng, rngs::StdRng};

    #[test]
    fn bn254_kzg_commitment_smoke() {
        let mut rng = StdRng::from_entropy();
        let tau = BnFr::rand(&mut rng);
        let params = BnKzg::setup(8, &tau).expect("setup");
        let coeffs: Vec<BnFr> = (0..4).map(|_| BnFr::rand(&mut rng)).collect();
        let poly = DensePolynomial::from_coefficients_vec(coeffs);
        let commitment = BnKzg::commit_g1(&params, &poly).expect("commit");
        let affine = commitment.to_affine();
        let projective = affine.0.into_group();
        assert!(
            !projective.is_zero(),
            "commitment should not be identity for random polynomial"
        );
    }

    #[test]
    fn bn254_pairing_matches_reference() {
        let g = ArkBnG1::generator();
        let h = ArkBnG2::generator();
        let backend_result = ArkworksBn254::pairing(&g, &h);
        let direct = ArkBnGt(Bn254::pairing(g.to_affine().0, h.to_affine().0));
        assert_eq!(backend_result.0, direct.0, "pairing mismatch");
    }
}
