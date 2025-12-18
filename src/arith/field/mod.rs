use std::fmt::Debug;

use rand_core::RngCore;

use crate::BackendError;

#[cfg(feature = "blst")]
mod blst_bls12_381;
#[cfg(feature = "blst")]
pub use blst_bls12_381::Fr;

#[cfg(feature = "ark_bls12381")]
mod ark_bls12_381;
#[cfg(feature = "ark_bls12381")]
pub use ark_bls12_381::Fr;

#[cfg(feature = "ark_bn254")]
mod ark_bn254;
#[cfg(feature = "ark_bn254")]
pub use ark_bn254::Fr;

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
/// use rand::thread_rng;
/// use tess::{FieldElement, Fr};
///
/// let mut rng = thread_rng();
/// let a = Fr::random(&mut rng);
/// let b = Fr::random(&mut rng);
///
/// // Field operations
/// let zero = Fr::zero();
/// let one = Fr::one();
/// let inv = a.invert().expect("non-zero element");
///
/// // Serialization
/// let bytes = a.to_repr();
/// let recovered = Fr::from_repr(&bytes).expect("valid repr");
/// ```
pub trait FieldElement: Clone + Send + Sync + Debug + 'static + Copy {
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

    /// Returns a 2-adic root of unity (i.e., an element of order 2^k for some k).
    fn two_adic_root_of_unity() -> Self;

    /// Returns a primitive n-th root of unity.
    ///
    /// Assumes n is a power of 2.
    fn two_adicity_generator(n: usize) -> Self;

    /// Performs batch inversion of a slice of field elements.
    fn batch_inversion(elements: &mut [Self]) -> Result<(), crate::BackendError>;

    /// Convert u64 to self.
    fn from_u64(n: u64) -> Self;
}
