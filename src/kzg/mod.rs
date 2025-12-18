mod scheme;
pub use scheme::{KZG, SRS};

use std::fmt::Debug;

use crate::{BackendError, PairingBackend, Polynomial};

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
/// use tess::backend::{BlstBackend, PairingBackend, FieldElement};
/// use tess::{KZG, PolynomialCommitment};
/// use rand::thread_rng;
///
/// type Scalar = <BlstBackend as PairingBackend>::Scalar;
///
/// let mut rng = thread_rng();
/// let tau = Scalar::random(&mut rng);
/// let seed = tau.to_bytes_be();
///
/// // Setup commitment parameters (trusted setup)
/// let params = <KZG as PolynomialCommitment<BlstBackend>>::setup(10, &seed)
///     .expect("setup failed");
/// # }
/// ```
pub trait PolynomialCommitment<B: PairingBackend>: Send + Sync + Debug + 'static {
    /// Commitment parameters (powers of tau).
    type Parameters: Clone + Send + Sync + Debug + 'static;

    /// Polynomial type used by this commitment scheme.
    type Polynomial: Polynomial<B::Scalar>;

    /// Performs trusted setup to generate commitment parameters.
    ///
    /// This generates powers of tau: `[τ^0, τ^1, ..., τ^max_degree]` in both G1 and G2.
    /// The secret `tau` must be securely discarded after setup.
    fn setup(max_degree: usize, seed: &[u8; 32]) -> Result<Self::Parameters, BackendError>;

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
