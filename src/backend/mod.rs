//! Backend trait abstractions for cryptographic operations.
//!
//! This module defines the core trait hierarchy that allows TESS to support multiple
//! cryptographic backends (Arkworks, blstrs) with a unified interface. The traits abstract
//! over elliptic curve operations, pairing operations, polynomial arithmetic, and KZG
//! commitments.
//!
//! # Architecture
//!
//! The backend system is built on several key trait abstractions:
//!
//! - **[`FieldElement`]**: Scalar field operations (zero, one, random, invert, etc.)
//! - **[`CurvePoint`]**: Elliptic curve point operations for G1/G2 groups
//! - **[`TargetGroup`]**: Pairing output group (GT) operations
//! - **[`Polynomial`]**: Polynomial evaluation and manipulation
//! - **[`EvaluationDomain`]**: FFT domain operations for Lagrange basis generation
//! - **[`PolynomialCommitment`]**: KZG commitment scheme operations
//! - **[`MsmProvider`]**: Multi-scalar multiplication (MSM) provider
//! - **[`PairingBackend`]**: Umbrella trait that ties all operations together
//!
//! # Available Backends
//!
//! ## Arkworks Backends
//!
//! - **`ArkworksBls12`** (feature: `ark_bls12381`): BLS12-381 curve using arkworks
//! - **`ArkworksBn254`** (feature: `ark_bn254`): BN254 curve using arkworks
//!
//! ## blstrs Backend
//!
//! - **`BlstBackend`** (feature: `blst`, default): BLS12-381 using blstrs with optimized assembly
//!
//! # Example: Using a Backend
//!
//! ```rust,no_run
//! # #[cfg(feature = "blst")]
//! # {
//! use tess::backend::{BlstBackend, PairingBackend, FieldElement, CurvePoint};
//! use rand::thread_rng;
//!
//! // Generate random scalar
//! let mut rng = thread_rng();
//! let scalar = <BlstBackend as PairingBackend>::Scalar::random(&mut rng);
//!
//! // Get G1 generator and perform scalar multiplication
//! let g1_gen = <BlstBackend as PairingBackend>::G1::generator();
//! let g1_point = g1_gen.mul_scalar(&scalar);
//!
//! // Perform pairing
//! let g2_gen = <BlstBackend as PairingBackend>::G2::generator();
//! let target = BlstBackend::pairing(&g1_point, &g2_gen);
//! # }
//! ```

use core::fmt::Debug;

#[cfg(any(feature = "ark_bls12381", feature = "ark_bn254"))]
use ark_ff::PrimeField;
use rand_core::RngCore;

use crate::errors::BackendError;

/// Field element abstraction for scalar field operations.
///
/// This trait abstracts over the scalar field Fr of the elliptic curve, providing
/// common field operations needed for cryptographic protocols.
///
/// # Type Parameters
///
/// - `Repr`: Byte representation type for serialization (e.g., `[u8; 32]` for BLS12-381 scalars)
///
/// # Example
///
/// ```rust,no_run
/// # #[cfg(feature = "blst")]
/// # {
/// use tess::backend::{BlstBackend, PairingBackend, FieldElement};
/// use rand::thread_rng;
///
/// type Scalar = <BlstBackend as PairingBackend>::Scalar;
///
/// let mut rng = thread_rng();
/// let a = Scalar::random(&mut rng);
/// let b = Scalar::random(&mut rng);
///
/// // Field operations
/// let zero = Scalar::zero();
/// let one = Scalar::one();
/// let inv = a.invert().expect("non-zero element");
///
/// // Serialization
/// let bytes = a.to_repr();
/// let recovered = Scalar::from_repr(&bytes).expect("valid repr");
/// # }
/// ```
pub trait FieldElement: Clone + Send + Sync + Debug + 'static {
    /// Byte representation type (e.g., 32-byte array for bls12-381 scalars).
    type Repr: AsRef<[u8]> + AsMut<[u8]> + Default + Debug + Send + Sync + Clone + 'static;

    /// Returns the additive identity (zero) element.
    fn zero() -> Self;

    /// Returns the multiplicative identity (one) element.
    fn one() -> Self;

    /// Generates a random field element using the provided RNG.
    fn random<R: RngCore + ?Sized>(rng: &mut R) -> Self;

    /// Computes the multiplicative inverse, returning `None` for zero.
    fn invert(&self) -> Option<Self>;

    /// Raises this element to a power represented as a 256-bit little-endian integer.
    fn pow(&self, exp: &[u64; 4]) -> Self;

    /// Serializes this field element to its byte representation.
    fn to_repr(&self) -> Self::Repr;

    /// Deserializes a field element from its byte representation.
    ///
    /// Returns an error if the representation is invalid (e.g., not reduced modulo the field order).
    fn from_repr(repr: &Self::Repr) -> Result<Self, BackendError>;
}

/// Elliptic curve point abstraction for G1 and G2 groups.
///
/// This trait provides operations on elliptic curve points in projective coordinates,
/// with support for conversion to/from affine coordinates for serialization.
///
/// # Type Parameters
///
/// - `F`: The scalar field type used for scalar multiplication
/// - `Affine`: The affine representation of the curve point
///
/// # Example
///
/// ```rust,no_run
/// # #[cfg(feature = "blst")]
/// # {
/// use tess::backend::{BlstBackend, PairingBackend, FieldElement, CurvePoint};
/// use rand::thread_rng;
///
/// type G1 = <BlstBackend as PairingBackend>::G1;
/// type Scalar = <BlstBackend as PairingBackend>::Scalar;
///
/// let mut rng = thread_rng();
/// let scalar = Scalar::random(&mut rng);
///
/// // Point operations
/// let g = G1::generator();
/// let point = g.mul_scalar(&scalar);
/// let doubled = point.add(&point);
/// let neg = point.negate();
/// # }
/// ```
pub trait CurvePoint<F: FieldElement>: Clone + Send + Sync + Debug + 'static {
    /// Associated affine representation.
    type Affine: Clone + Debug + Send + Sync + 'static;

    /// Returns the point at infinity (identity element).
    fn identity() -> Self;

    /// Returns the standard generator for this group.
    fn generator() -> Self;

    /// Checks if this point is the identity element.
    fn is_identity(&self) -> bool;

    /// Converts from affine to projective coordinates.
    fn from_affine(affine: &Self::Affine) -> Self;

    /// Converts from projective to affine coordinates.
    fn to_affine(&self) -> Self::Affine;

    /// Performs elliptic curve point addition.
    fn add(&self, other: &Self) -> Self;

    /// Performs elliptic curve point subtraction.
    fn sub(&self, other: &Self) -> Self;

    /// Returns the additive inverse of this point.
    fn negate(&self) -> Self;

    /// Performs scalar multiplication: returns `scalar * self`.
    fn mul_scalar(&self, scalar: &F) -> Self;

    /// Batch normalizes multiple projective points to affine coordinates.
    ///
    /// This is more efficient than normalizing points individually due to
    /// Montgomery's trick for batch inversion.
    fn batch_normalize(points: &[Self]) -> Vec<Self::Affine>;
}

/// Pairing target group (GT) abstraction.
///
/// This trait represents the target group of the pairing operation, which is
/// a multiplicative subgroup of the extension field.
///
/// # Example
///
/// ```rust,no_run
/// # #[cfg(feature = "blst")]
/// # {
/// use tess::backend::{BlstBackend, PairingBackend, CurvePoint, TargetGroup};
///
/// type G1 = <BlstBackend as PairingBackend>::G1;
/// type G2 = <BlstBackend as PairingBackend>::G2;
///
/// let g1 = G1::generator();
/// let g2 = G2::generator();
///
/// // Compute pairing
/// let gt = BlstBackend::pairing(&g1, &g2);
///
/// // Target group operations
/// let gt_squared = gt.combine(&gt);
/// # }
/// ```
pub trait TargetGroup: Clone + Send + Sync + Debug + 'static {
    /// Scalar field type for scalar multiplication.
    type Scalar: FieldElement;
    /// Byte representation for serialization.
    type Repr: AsRef<[u8]> + AsMut<[u8]> + Default + Debug + Send + Sync + Clone + 'static;

    /// Returns the multiplicative identity element.
    fn identity() -> Self;

    /// Performs scalar multiplication (exponentiation in multiplicative notation).
    fn mul_scalar(&self, scalar: &Self::Scalar) -> Self;

    /// Combines (multiplies) two target group elements.
    fn combine(&self, other: &Self) -> Self;

    /// Serializes this element to its byte representation.
    fn to_repr(&self) -> Self::Repr;

    /// Deserializes an element from its byte representation.
    fn from_repr(bytes: &Self::Repr) -> Result<Self, BackendError>;
}

/// Multi-scalar multiplication (MSM) provider.
///
/// MSM computes the sum of scalar multiplications: `∑ scalars[i] * bases[i]`.
/// This is a performance-critical operation in KZG commitments and is often
/// parallelized or accelerated using specialized algorithms.
///
/// # Example
///
/// ```rust,no_run
/// # #[cfg(feature = "blst")]
/// # {
/// use tess::backend::{BlstBackend, PairingBackend, FieldElement, CurvePoint, MsmProvider};
/// use rand::thread_rng;
///
/// type Msm = <BlstBackend as PairingBackend>::Msm;
/// type G1 = <BlstBackend as PairingBackend>::G1;
/// type Scalar = <BlstBackend as PairingBackend>::Scalar;
///
/// let mut rng = thread_rng();
/// let bases: Vec<G1> = (0..10).map(|_| G1::generator()).collect();
/// let scalars: Vec<Scalar> = (0..10).map(|_| Scalar::random(&mut rng)).collect();
///
/// let result = Msm::msm_g1(&bases, &scalars).expect("msm failed");
/// # }
/// ```
pub trait MsmProvider<B: PairingBackend>: Send + Sync + Debug + 'static {
    /// Computes multi-scalar multiplication in G1: `∑ scalars[i] * bases[i]`.
    fn msm_g1(bases: &[B::G1], scalars: &[B::Scalar]) -> Result<B::G1, BackendError>;

    /// Computes multi-scalar multiplication in G2: `∑ scalars[i] * bases[i]`.
    fn msm_g2(bases: &[B::G2], scalars: &[B::Scalar]) -> Result<B::G2, BackendError>;
}

/// Polynomial interface for univariate polynomials.
///
/// Polynomials are represented in coefficient form and are used extensively
/// in the KZG commitment scheme for Lagrange interpolation and evaluation.
pub trait Polynomial<B: PairingBackend>: Clone + Send + Sync + Debug + 'static {
    /// Returns the degree of this polynomial.
    fn degree(&self) -> usize;

    /// Returns the coefficients in ascending order (constant term first).
    fn coeffs(&self) -> &[B::Scalar];

    /// Evaluates the polynomial at the given point using Horner's method.
    fn evaluate(&self, point: &B::Scalar) -> B::Scalar;

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
pub trait EvaluationDomain<B: PairingBackend>: Clone + Send + Sync + Debug + 'static {
    /// Returns the size of this evaluation domain (must be power of two).
    fn size(&self) -> usize;

    /// Returns all elements in the domain (roots of unity).
    fn elements(&self) -> Vec<B::Scalar>;

    /// Forward FFT: converts coefficients to evaluations.
    fn fft(&self, coeffs: &[B::Scalar]) -> Vec<B::Scalar>;

    /// Inverse FFT: converts evaluations to coefficients.
    fn ifft(&self, evals: &[B::Scalar]) -> Vec<B::Scalar>;
}

/// KZG polynomial commitment scheme interface.
///
/// This trait provides the core operations for Kate-Zaverucha-Goldberg (KZG)
/// commitments, which are used in the silent setup for threshold encryption.
///
/// # KZG Commitments
///
/// KZG commitments allow committing to a polynomial such that:
/// - The commitment is succinct (constant size)
/// - One can prove evaluations at specific points
/// - The scheme is binding and hiding under appropriate assumptions
///
/// # Example
///
/// ```rust,no_run
/// # #[cfg(feature = "blst")]
/// # {
/// use tess::backend::{BlstBackend, PairingBackend, FieldElement, PolynomialCommitment};
/// use rand::thread_rng;
///
/// type PC = <BlstBackend as PairingBackend>::PolynomialCommitment;
/// type Scalar = <BlstBackend as PairingBackend>::Scalar;
///
/// let mut rng = thread_rng();
/// let tau = Scalar::random(&mut rng);
///
/// // Setup commitment parameters (trusted setup)
/// let params = PC::setup(10, &tau).expect("setup failed");
/// # }
/// ```
pub trait PolynomialCommitment<B: PairingBackend>: Send + Sync + Debug + 'static {
    /// Commitment parameters (powers of tau).
    type Parameters: Clone + Send + Sync + Debug + 'static;
    /// Polynomial type used by this commitment scheme.
    type Polynomial: Polynomial<B>;

    /// Performs trusted setup to generate commitment parameters.
    ///
    /// This generates powers of tau: `[τ^0, τ^1, ..., τ^max_degree]` in both G1 and G2.
    /// The secret `tau` must be securely discarded after setup.
    fn setup(max_degree: usize, tau: &B::Scalar) -> Result<Self::Parameters, BackendError>;

    /// Commits to a polynomial in G1.
    ///
    /// For polynomial `p(x) = ∑ c_i * x^i`, returns `∑ c_i * τ^i * G1`.
    fn commit_g1(
        params: &Self::Parameters,
        polynomial: &Self::Polynomial,
    ) -> Result<B::G1, BackendError>;

    /// Commits to a polynomial in G2.
    ///
    /// For polynomial `p(x) = ∑ c_i * x^i`, returns `∑ c_i * τ^i * G2`.
    fn commit_g2(
        params: &Self::Parameters,
        polynomial: &Self::Polynomial,
    ) -> Result<B::G2, BackendError>;
}

/// Main backend trait that ties together all cryptographic operations.
///
/// This is the primary trait that concrete backends (like [`BlstBackend`], [`ArkworksBls12`],
/// [`ArkworksBn254`]) must implement. It aggregates all the specialized traits and provides
/// pairing operations.
///
/// # Type Parameters
///
/// - `Scalar`: The scalar field type (Fr)
/// - `G1`: The first elliptic curve group
/// - `G2`: The second elliptic curve group
/// - `Target`: The pairing target group (GT)
/// - `PolynomialCommitment`: KZG commitment implementation
/// - `Domain`: FFT evaluation domain
/// - `Msm`: Multi-scalar multiplication provider
///
/// # Example
///
/// ```rust,no_run
/// # #[cfg(feature = "blst")]
/// # {
/// use tess::backend::{BlstBackend, PairingBackend, CurvePoint};
///
/// // Access backend types
/// type Scalar = <BlstBackend as PairingBackend>::Scalar;
/// type G1 = <BlstBackend as PairingBackend>::G1;
/// type G2 = <BlstBackend as PairingBackend>::G2;
///
/// let g1 = G1::generator();
/// let g2 = G2::generator();
///
/// // Compute pairing: e(G1, G2) -> GT
/// let result = BlstBackend::pairing(&g1, &g2);
/// # }
/// ```
pub trait PairingBackend: Send + Sync + Debug + Sized + 'static {
    /// Scalar field type (Fr).
    type Scalar: FieldElement;
    /// First curve group (G1).
    type G1: CurvePoint<Self::Scalar>;
    /// Second curve group (G2).
    type G2: CurvePoint<Self::Scalar>;
    /// Pairing target group (GT).
    type Target: TargetGroup<Scalar = Self::Scalar>;
    /// KZG polynomial commitment implementation.
    type PolynomialCommitment: PolynomialCommitment<Self>;
    /// FFT evaluation domain.
    type Domain: EvaluationDomain<Self>;
    /// Multi-scalar multiplication provider.
    type Msm: MsmProvider<Self>;

    /// Computes the bilinear pairing: `e(g1, g2) -> GT`.
    ///
    /// The pairing satisfies bilinearity: `e(a*P, b*Q) = e(P, Q)^(ab)`.
    fn pairing(g1: &Self::G1, g2: &Self::G2) -> Self::Target;

    /// Computes a product of pairings: `∏ e(g1[i], g2[i])`.
    ///
    /// This is more efficient than computing individual pairings and multiplying.
    /// Returns an error if the input arrays have different lengths.
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
    use ark_poly::{DenseUVPolynomial, univariate::DensePolynomial as ArkDensePolynomial};

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
