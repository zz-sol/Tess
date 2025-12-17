use std::fmt::Debug;

use crate::{BackendError, FieldElement};

#[cfg(feature = "blst")]
mod blst_bls12_381;
#[cfg(feature = "blst")]
pub use blst_bls12_381::{G1, G2, Gt};

#[cfg(feature = "ark_bls12381")]
mod ark_bls12_381;
#[cfg(feature = "ark_bls12381")]
pub use ark_bls12_381::{G1, G2, Gt};

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
pub trait CurvePoint<F: FieldElement>: Clone + Send + Sync + Debug + 'static + Copy {
    /// Associated affine representation.
    type Affine: Clone + Debug + Send + Sync + 'static + Copy;

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
    type Scalar: FieldElement + Copy;
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
