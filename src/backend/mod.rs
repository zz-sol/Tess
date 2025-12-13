use core::fmt::Debug;

#[cfg(any(feature = "ark_bls12381", feature = "ark_bn254"))]
use ark_ff::PrimeField;
use rand_core::RngCore;

use crate::errors::BackendError;

/// Field element abstraction shared by all supported curves.
pub trait FieldElement: Clone + Send + Sync + Debug + 'static {
    /// Byte representation type (e.g., 32-byte array for bls12-381 scalars).
    type Repr: AsRef<[u8]> + AsMut<[u8]> + Default + Debug + Send + Sync + Clone + 'static;

    fn zero() -> Self;
    fn one() -> Self;
    fn random<R: RngCore + ?Sized>(rng: &mut R) -> Self;
    fn invert(&self) -> Option<Self>;
    fn pow(&self, exp: &[u64; 4]) -> Self;
    fn to_repr(&self) -> Self::Repr;
    fn from_repr(repr: &Self::Repr) -> Result<Self, BackendError>;
}

/// Elliptic curve point abstraction for both G1 and G2 groups.
pub trait CurvePoint<F: FieldElement>: Clone + Send + Sync + Debug + 'static {
    /// Associated affine representation.
    type Affine: Clone + Debug + Send + Sync + 'static;

    fn identity() -> Self;
    fn generator() -> Self;
    fn is_identity(&self) -> bool;
    fn from_affine(affine: &Self::Affine) -> Self;
    fn to_affine(&self) -> Self::Affine;
    fn add(&self, other: &Self) -> Self;
    fn sub(&self, other: &Self) -> Self;
    fn negate(&self) -> Self;
    fn mul_scalar(&self, scalar: &F) -> Self;
    fn batch_normalize(points: &[Self]) -> Vec<Self::Affine>;
}

/// Pairing target group abstraction.
pub trait TargetGroup: Clone + Send + Sync + Debug + 'static {
    type Scalar: FieldElement;
    type Repr: AsRef<[u8]> + AsMut<[u8]> + Default + Debug + Send + Sync + Clone + 'static;

    fn identity() -> Self;
    fn mul_scalar(&self, scalar: &Self::Scalar) -> Self;
    fn combine(&self, other: &Self) -> Self;
    fn to_repr(&self) -> Self::Repr;
    fn from_repr(bytes: &Self::Repr) -> Result<Self, BackendError>;
}

/// Describes the multi-scalar multiplication capabilities of a backend.
pub trait MsmProvider<B: PairingBackend>: Send + Sync + Debug + 'static {
    fn msm_g1(bases: &[B::G1], scalars: &[B::Scalar]) -> Result<B::G1, BackendError>;
    fn msm_g2(bases: &[B::G2], scalars: &[B::Scalar]) -> Result<B::G2, BackendError>;
}

/// Polynomial interface used by KZG commitment implementations.
pub trait Polynomial<B: PairingBackend>: Clone + Send + Sync + Debug + 'static {
    fn degree(&self) -> usize;
    fn coeffs(&self) -> &[B::Scalar];
    fn evaluate(&self, point: &B::Scalar) -> B::Scalar;
    fn truncate(&mut self, len: usize);
}

/// FFT-like evaluation domain used for Lagrange basis generation.
pub trait EvaluationDomain<B: PairingBackend>: Clone + Send + Sync + Debug + 'static {
    fn size(&self) -> usize;
    fn elements(&self) -> Vec<B::Scalar>;
    fn fft(&self, coeffs: &[B::Scalar]) -> Vec<B::Scalar>;
    fn ifft(&self, evals: &[B::Scalar]) -> Vec<B::Scalar>;
}

/// API for polynomial commitment backends (KZG).
pub trait PolynomialCommitment<B: PairingBackend>: Send + Sync + Debug + 'static {
    type Parameters: Clone + Send + Sync + Debug + 'static;
    type Polynomial: Polynomial<B>;

    fn setup(max_degree: usize, tau: &B::Scalar) -> Result<Self::Parameters, BackendError>;
    fn commit_g1(
        params: &Self::Parameters,
        polynomial: &Self::Polynomial,
    ) -> Result<B::G1, BackendError>;
    fn commit_g2(
        params: &Self::Parameters,
        polynomial: &Self::Polynomial,
    ) -> Result<B::G2, BackendError>;
}

/// Trait implemented by every backend (Arkworks, blstrs, GPU-accelerated, etc.).
pub trait PairingBackend: Send + Sync + Debug + Sized + 'static {
    type Scalar: FieldElement;
    type G1: CurvePoint<Self::Scalar>;
    type G2: CurvePoint<Self::Scalar>;
    type Target: TargetGroup<Scalar = Self::Scalar>;
    type PolynomialCommitment: PolynomialCommitment<Self>;
    type Domain: EvaluationDomain<Self>;
    type Msm: MsmProvider<Self>;

    fn pairing(g1: &Self::G1, g2: &Self::G2) -> Self::Target;
    fn multi_pairing(g1: &[Self::G1], g2: &[Self::G2]) -> Result<Self::Target, BackendError>;
}

#[cfg(any(feature = "ark_bls12381", feature = "ark_bn254"))]
fn sample_field<F: PrimeField, R: RngCore + ?Sized>(rng: &mut R) -> F {
    let byte_len = F::MODULUS_BIT_SIZE.div_ceil(8) as usize;
    let mut bytes = vec![0u8; byte_len];
    rng.fill_bytes(&mut bytes);
    F::from_le_bytes_mod_order(&bytes)
}

#[cfg(feature = "ark_bls12381")]
mod ark_bls12_381;
#[cfg(feature = "ark_bn254")]
mod ark_bn254;
#[cfg(feature = "blst")]
mod blst_bls12_381;

#[cfg(feature = "ark_bls12381")]
pub use ark_bls12_381::*;
#[cfg(feature = "ark_bn254")]
pub use ark_bn254::*;
#[cfg(feature = "blst")]
pub use blst_bls12_381::*;

#[cfg(test)]
mod tests {

    use rand::{SeedableRng, rngs::StdRng};

    use super::{CurvePoint, FieldElement, PairingBackend, PolynomialCommitment};
    #[cfg(feature = "ark_bls12381")]
    use crate::backend::ArkworksBls12;
    #[cfg(feature = "ark_bn254")]
    use crate::backend::ArkworksBn254;
    #[cfg(feature = "blst")]
    use crate::backend::{BlstBackend, DensePolynomial as BlstDensePolynomial};
    #[cfg(feature = "ark_bls12381")]
    use ark_bls12_381::Fr as BlsFr;
    #[cfg(feature = "ark_bn254")]
    use ark_bn254::Fr as BnFr;
    #[cfg(any(feature = "ark_bls12381", feature = "ark_bn254"))]
    use ark_poly::univariate::DensePolynomial as ArkDensePolynomial;

    type CommitmentPolynomial<E> =
        <<E as PairingBackend>::PolynomialCommitment as PolynomialCommitment<E>>::Polynomial;

    fn kzg_commitment_helper<E, F>(make_poly: F)
    where
        E: PairingBackend,
        F: Fn(Vec<E::Scalar>) -> CommitmentPolynomial<E>,
    {
        let mut rng = StdRng::from_entropy();
        let tau = E::Scalar::random(&mut rng);
        let params = E::PolynomialCommitment::setup(8, &tau).expect("setup");
        let coeffs: Vec<E::Scalar> = (0..4).map(|_| E::Scalar::random(&mut rng)).collect();
        let poly = make_poly(coeffs);
        let commitment = E::PolynomialCommitment::commit_g1(&params, &poly).expect("commit");
        assert!(
            !commitment.is_identity(),
            "commitment should not be identity for random polynomial"
        );
    }
    #[test]
    fn kzg_commitment() {
        #[cfg(feature = "blst")]
        kzg_commitment_helper::<BlstBackend, _>(BlstDensePolynomial::from_coefficients_vec);
        #[cfg(feature = "ark_bls12381")]
        kzg_commitment_helper::<ArkworksBls12, _>(
            ArkDensePolynomial::<BlsFr>::from_coefficients_vec,
        );
        #[cfg(feature = "ark_bn254")]
        kzg_commitment_helper::<ArkworksBn254, _>(
            ArkDensePolynomial::<BnFr>::from_coefficients_vec,
        );
    }
}
