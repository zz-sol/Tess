use crate::{Fr, LagrangePowers, PairingBackend, SRS};

/// Structured Reference String for the threshold encryption scheme.
///
/// The SRS is generated once during a trusted setup ceremony and can be
/// reused for multiple key generation operations. It contains KZG commitment
/// parameters (powers of tau) and precomputed Lagrange polynomial commitments.
///
/// # Fields
///
/// - `commitment_params`: KZG commitment parameters (powers of tau in G1 and G2)
/// - `lagrange_powers`: Precomputed Lagrange polynomial commitments
/// - `lagrange_polys`: The actual Lagrange basis polynomials for the domain
///
/// # Security
///
/// The security of the scheme depends on the secret tau being securely
/// discarded after SRS generation. In production, this should be generated
/// via a secure multi-party computation ceremony.
#[derive(Clone, Debug)]
pub struct Params<B: PairingBackend<Scalar = Fr>> {
    pub srs: SRS<B>,
    pub lagrange_powers: LagrangePowers<B>,
}
