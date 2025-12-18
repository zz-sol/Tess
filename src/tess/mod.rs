//! Threshold encryption protocol implementation.
//!
//! This module implements the high-level threshold encryption protocol using KZG commitments
//! and Lagrange interpolation. The protocol allows a message to be encrypted such that it
//! can only be decrypted when at least `t` out of `n` participants cooperate.
//!
//! # Protocol Overview
//!
//! The threshold encryption scheme consists of six main phases:
//!
//! 1. **SRS Generation** ([`ThresholdScheme::srs_gen`]): Generate a Structured Reference String
//!    containing KZG commitment parameters and precomputed Lagrange polynomial commitments.
//!
//! 2. **Key Generation** ([`ThresholdScheme::keygen`]): Each participant generates a secret key
//!    and derives their public key with Lagrange commitments for efficient verification.
//!
//! 3. **Key Aggregation** ([`ThresholdScheme::aggregate_public_key`]): Combine all public keys
//!    into an aggregate key that will be used for encryption.
//!
//! 4. **Encryption** ([`ThresholdScheme::encrypt`]): Encrypt a payload using the aggregate key,
//!    producing a ciphertext with KZG proofs and BLAKE3-encapsulated payload.
//!
//! 5. **Partial Decryption** ([`ThresholdScheme::partial_decrypt`]): Each participant computes
//!    their decryption share using their secret key.
//!
//! 6. **Aggregate Decryption** ([`ThresholdScheme::aggregate_decrypt`]): Combine at least `t`
//!    partial decryptions to recover the shared secret and decrypt the payload.

use core::fmt::Debug;

use rand_core::RngCore;

use crate::{Fr, PairingBackend, errors::Error};

mod scheme;
pub use scheme::SilentThresholdScheme;

mod keys;
pub use keys::{AggregateKey, KeyMaterial, PublicKey, SecretKey};

mod params;
pub use params::Params;

mod ciphertext;
pub use ciphertext::{Ciphertext, DecryptionResult, PartialDecryption};

/// High-level threshold encryption scheme interface.
///
/// This trait defines the complete API for a threshold encryption scheme,
/// from setup through key generation to encryption and decryption.
pub trait ThresholdEncryption<B: PairingBackend<Scalar = Fr>>:
    Debug + Send + Sync + 'static
{
    /// Generates the Structured Reference String (SRS) for the scheme.
    ///
    /// This is a one-time trusted setup that generates KZG commitment parameters
    /// and precomputes Lagrange polynomial commitments.
    fn param_gen<R: RngCore + ?Sized>(
        &self,
        rng: &mut R,
        parties: usize,
        threshold: usize,
    ) -> Result<Params<B>, Error>;

    /// Generates key material for all participants.
    ///
    /// This generates secret keys for all `n` participants and derives their
    /// corresponding public keys with Lagrange commitment hints.
    fn keygen<R: RngCore + ?Sized>(
        &self,
        rng: &mut R,
        parties: usize,
        srs: &Params<B>,
    ) -> Result<KeyMaterial<B>, Error>;

    /// Recomputes the aggregate key from public keys using precomputed Lagrange powers.
    fn aggregate_public_key(
        &self,
        public_keys: &[PublicKey<B>],
        params: &Params<B>,
        parties: usize,
    ) -> Result<AggregateKey<B>, Error>;

    /// Encrypts a payload using the aggregate key.
    fn encrypt<R: RngCore + ?Sized>(
        &self,
        rng: &mut R,
        agg_key: &AggregateKey<B>,
        threshold: usize,
        payload: &[u8],
    ) -> Result<Ciphertext<B>, Error>;

    /// Computes a partial decryption share.
    fn partial_decrypt(
        &self,
        secret_key: &SecretKey<B>,
        ciphertext: &Ciphertext<B>,
    ) -> Result<PartialDecryption<B>, Error>;

    /// Aggregates partial decryptions to recover the plaintext.
    fn aggregate_decrypt(
        &self,
        ciphertext: &Ciphertext<B>,
        partials: &[PartialDecryption<B>],
        selector: &[bool],
        agg_key: &AggregateKey<B>,
    ) -> Result<DecryptionResult, Error>;
}
