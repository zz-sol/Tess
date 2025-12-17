//! # TESS: Threshold Encryption with Silent Setup
//!
//! TESS is a production-grade cryptographic library implementing threshold encryption
//! with a silent (non-interactive) setup based on Knowledge of Exponent (KZG) commitments.
//!
//! ## Overview
//!
//! Threshold encryption allows a message to be encrypted such that it can only be
//! decrypted when at least `t` out of `n` participants cooperate. TESS implements this
//! using a silent setup, meaning the initial setup does not require interactive
//! communication between participants.
//!
//! ## Architecture
//!
//! The crate is organized into several key modules:
//!
//! - **[`backend`]**: Core trait abstractions for cryptographic operations that allow
//!   multiple backends (Arkworks, blstrs) to provide unified interfaces. Includes traits
//!   for field elements, curve points, pairing operations, polynomials, and KZG commitments.
//!
//! - **[`protocol`]**: High-level threshold encryption protocol implementation. Contains
//!   the [`ThresholdScheme`] trait and [`SilentThreshold`](protocol::SilentThreshold)
//!   implementation, along with key structures like [`SecretKey`](protocol::SecretKey),
//!   [`PublicKey`](protocol::PublicKey), [`Ciphertext`](protocol::Ciphertext), etc.
//!
//! - **[`config`]**: Configuration types including [`ThresholdParameters`], [`BackendConfig`],
//!   [`CurveId`], and [`BackendId`] for setting up threshold schemes.
//!
//! - **[`lagrange`]**: Backend-specific Lagrange polynomial helpers for efficient
//!   polynomial interpolation and evaluation.
//!
//! - **[`errors`]**: Error types for backend and protocol operations.
//!
//! ## Quick Example
//!
//! ```rust,no_run
//! use tess::{ThresholdParameters, BackendConfig, CurveId, BackendId};
//! use tess::protocol::{SilentThreshold, ThresholdScheme};
//! # #[cfg(feature = "blst")]
//! use tess::backend::BlstBackend;
//! # #[cfg(feature = "blst")]
//! use rand::rngs::StdRng;
//! # #[cfg(feature = "blst")]
//! use rand::SeedableRng;
//!
//! # #[cfg(feature = "blst")]
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Configure: 5 parties, threshold of 3, using blst backend with BLS12-381
//! let params = ThresholdParameters::new(
//!     5, 3,
//!     BackendConfig { backend: BackendId::Blst, curve: CurveId::Bls12_381 },
//! )?;
//!
//! // Create RNG and scheme instance
//! let mut rng = StdRng::from_entropy();
//! let scheme = SilentThreshold::<BlstBackend>::default();
//!
//! // Generate Structured Reference String (one-time trusted setup)
//! let srs = scheme.srs_gen(&mut rng, &params)?;
//!
//! // Generate key material for all participants
//! let key_material = scheme.keygen(&mut rng, &params, &srs)?;
//!
//! // Encrypt a message
//! let plaintext = b"Secret message";
//! let ciphertext = scheme.encrypt(
//!     &mut rng, &key_material.aggregate_key, &params, plaintext,
//! )?;
//!
//! // Partial decryptions from threshold participants (e.g., first 3 parties)
//! let mut selector = vec![false; params.parties];
//! let mut partial_decryptions = Vec::new();
//! for i in 0..params.threshold {
//!     selector[i] = true;
//!     let partial = scheme.partial_decrypt(
//!         &key_material.secret_keys[i], &ciphertext,
//!     )?;
//!     partial_decryptions.push(partial);
//! }
//!
//! // Aggregate to recover plaintext
//! let result = scheme.aggregate_decrypt(
//!     &ciphertext, &partial_decryptions, &selector, &key_material.aggregate_key,
//! )?;
//!
//! assert_eq!(result.plaintext.as_deref(), Some(plaintext.as_slice()));
//! # Ok(())
//! # }
//! # #[cfg(not(feature = "blst"))]
//! # fn main() {}
//! ```
//!
//! ## Feature Flags
//!
//! TESS supports multiple cryptographic backends via feature flags:
//!
//! - **`blst`** (default): blstrs backend for BLS12-381
//! - **`ark_bls12381`**: Arkworks backend for BLS12-381
//! - **`ark_bn254`**: Arkworks backend for BN254
//!
//! ## Protocol Workflow
//!
//! 1. **SRS Generation**: Generate a Structured Reference String using [`ThresholdScheme::srs_gen`].
//!    This is a one-time trusted setup that produces KZG commitment parameters.
//!
//! 2. **Key Generation**: Each participant generates keys using [`ThresholdScheme::keygen`],
//!    which produces a secret key and public key with Lagrange commitment hints.
//!
//! 3. **Key Aggregation**: Combine public keys using [`ThresholdScheme::aggregate_public_key`]
//!    to create an aggregate key for encryption.
//!
//! 4. **Encryption**: Encrypt messages using [`ThresholdScheme::encrypt`], which produces
//!    a ciphertext with KZG proof and BLAKE3-encapsulated payload.
//!
//! 5. **Partial Decryption**: Each participant creates a decryption share using
//!    [`ThresholdScheme::partial_decrypt`].
//!
//! 6. **Aggregate Decryption**: Combine at least `t` partial decryptions using
//!    [`ThresholdScheme::aggregate_decrypt`] to recover the plaintext.
//!
//! ## Performance
//!
//! TESS leverages parallel processing using Rayon for performance-critical operations:
//! - Multi-scalar multiplication (MSM) operations
//! - Parallel key generation
//! - FFT operations in polynomial arithmetic
//!
//! ## Security Considerations
//!
//! - **Trusted Setup**: The SRS generation requires a trusted setup. The secret tau value
//!   must be securely discarded after generation.
//! - **Threshold Security**: The scheme is secure as long as fewer than `t` participants
//!   are compromised.
//! - **Payload Encryption**: Uses BLAKE3 as a KDF to derive symmetric keys from the shared
//!   secret for payload encapsulation.

mod arith;
mod errors;
mod kzg;

pub use arith::*;
pub use errors::*;
pub use kzg::*;
