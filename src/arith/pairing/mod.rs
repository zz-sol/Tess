use std::fmt::Debug;

#[cfg(feature = "blst")]
mod blst_bls12_381;
#[cfg(feature = "blst")]
pub use blst_bls12_381::PairingEngine;

#[cfg(feature = "ark_bls12381")]
mod ark_bls12_381;
#[cfg(feature = "ark_bls12381")]
pub use ark_bls12_381::PairingEngine;

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
