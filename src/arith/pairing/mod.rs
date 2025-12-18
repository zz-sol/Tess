use std::fmt::Debug;

#[cfg(feature = "blst")]
mod blst_bls12_381;
#[cfg(feature = "blst")]
pub use blst_bls12_381::PairingEngine;

#[cfg(feature = "ark_bls12381")]
mod ark_bls12_381;
#[cfg(feature = "ark_bls12381")]
pub use ark_bls12_381::PairingEngine;

#[cfg(feature = "ark_bn254")]
mod ark_bn254;
#[cfg(feature = "ark_bn254")]
pub use ark_bn254::PairingEngine;

use crate::{BackendError, CurvePoint, FieldElement, TargetGroup};

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
/// ```rust
/// use tess::{CurvePoint, PairingBackend, PairingEngine};
///
/// let g1 = <PairingEngine as PairingBackend>::G1::generator();
/// let g2 = <PairingEngine as PairingBackend>::G2::generator();
///
/// let result = PairingEngine::pairing(&g1, &g2);
/// println!("{:?}", result);
/// ```
pub trait PairingBackend: Send + Sync + Debug + Sized + 'static {
    /// Scalar field type (Fr).
    type Scalar: FieldElement;
    /// First curve group (G1).
    type G1: CurvePoint<Self::Scalar>;
    /// Second curve group (G2).
    type G2: CurvePoint<Self::Scalar>;
    /// Pairing target group (GT).
    type Target: TargetGroup<Scalar = Self::Scalar> + PartialEq;

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
