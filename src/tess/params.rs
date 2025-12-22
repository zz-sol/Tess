//! Parameter structures for the threshold encryption scheme.
//!
//! This module defines the [`Params`] structure that contains all public
//! parameters needed for the threshold encryption scheme.
//!
//! # Overview
//!
//! The parameters are generated once during a trusted setup ceremony and
//! consist of two main components:
//!
//! 1. **Structured Reference String (SRS)**: KZG commitment parameters
//!    containing powers of a secret value tau in both G1 and G2 groups.
//!
//! 2. **Lagrange Powers**: Precomputed commitments to Lagrange polynomials
//!    that enable efficient key generation and verification.
//!
//! # Trusted Setup
//!
//! The parameters require a trusted setup where a secret value `tau` is used
//! to generate the SRS. For security, `tau` must be:
//! - Generated from a cryptographically secure random source
//! - Used only to compute the SRS
//! - Securely destroyed after parameter generation
//!
//! In production deployments, consider using a multi-party computation (MPC)
//! ceremony to generate the SRS, eliminating the need to trust a single party.
//!
//! # Reusability
//!
//! Once generated, the same parameters can be reused for multiple independent
//! instances of the threshold encryption scheme, as long as all instances use
//! the same number of participants.

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
///
/// # Security
///
/// The security of the scheme depends on the secret tau being securely
/// discarded after SRS generation. In production, this should be generated
/// via a secure multi-party computation ceremony.
#[derive(Clone, Debug)]
pub struct Params<B: PairingBackend<Scalar = Fr>> {
    /// KZG structured reference string parameters.
    pub srs: SRS<B>,
    /// Precomputed Lagrange polynomial commitments.
    pub lagrange_powers: LagrangePowers<B>,
}
