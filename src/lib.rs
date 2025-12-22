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
//! - **`arith`**: Core trait abstractions for cryptographic operations that allow
//!   multiple backends (Arkworks, blstrs) to provide unified interfaces. Includes traits
//!   for field elements, curve points, pairing operations, polynomials, and Lagrange polynomials.
//!
//! - **`tess`**: High-level threshold encryption protocol implementation. Contains
//!   the [`ThresholdEncryption`] trait and [`SilentThresholdScheme`] implementation,
//!   along with key structures like [`SecretKey`], [`PublicKey`], [`Ciphertext`], etc.
//!
//! - **`kzg`**: KZG polynomial commitment scheme with [`SRS`] and [`PolynomialCommitment`] trait.
//!
//! - **`sym_enc`**: Symmetric encryption using BLAKE3 for payload encapsulation.
//!
//! - **`errors`**: Error types for backend and protocol operations.
//!
//! ## Quick Example
//!
//! ```rust,no_run
//! use rand::thread_rng;
//! use tess::{PairingEngine, SilentThresholdScheme};
//!
//! let mut rng = thread_rng();
//! let scheme = SilentThresholdScheme::<PairingEngine>::new();
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

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
//! 1. **SRS Generation**: Generate a Structured Reference String using `param_gen`.
//!    This is a one-time trusted setup that produces KZG commitment parameters.
//!
//! 2. **Key Generation**: Each participant generates keys using `keygen`,
//!    which produces a secret key and public key with Lagrange commitment hints.
//!
//! 3. **Key Aggregation**: Combine public keys using `aggregate_public_key`
//!    to create an aggregate key for encryption.
//!
//! 4. **Encryption**: Encrypt messages using `encrypt`, which produces
//!    a ciphertext with KZG proof and BLAKE3-encapsulated payload.
//!
//! 5. **Partial Decryption**: Each participant creates a decryption share using
//!    `partial_decrypt`.
//!
//! 6. **Aggregate Decryption**: Combine at least `t` partial decryptions using
//!    `aggregate_decrypt` to recover the plaintext.
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

#![cfg_attr(not(feature = "std"), no_std)]
#![deny(missing_docs)]

#[macro_use]
extern crate alloc;
#[cfg(feature = "std")]
extern crate std;

mod arith;
mod errors;
mod kzg;
mod sym_enc;
mod tess;

pub use arith::*;
pub use errors::*;
pub use kzg::*;
pub use sym_enc::*;
pub use tess::*;
