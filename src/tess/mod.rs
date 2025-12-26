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
//! 1. **SRS Generation** ([`ThresholdEncryption::param_gen`]): Generate a Structured Reference String
//!    containing KZG commitment parameters and precomputed Lagrange polynomial commitments.
//!
//! 2. **Key Generation** ([`ThresholdEncryption::keygen`]): Each participant generates a secret key
//!    and derives their public key with Lagrange commitments for efficient verification.
//!
//! 3. **Key Aggregation** ([`ThresholdEncryption::aggregate_public_key`]): Combine all public keys
//!    into an aggregate key that will be used for encryption.
//!
//! 4. **Encryption** ([`ThresholdEncryption::encrypt`]): Encrypt a payload using the aggregate key,
//!    producing a ciphertext with KZG proofs and BLAKE3-encapsulated payload.
//!
//! 5. **Partial Decryption** ([`ThresholdEncryption::partial_decrypt`]): Each participant computes
//!    their decryption share using their secret key.
//!
//! 6. **Aggregate Decryption** ([`ThresholdEncryption::aggregate_decrypt`]): Combine at least `t`
//!    partial decryptions to recover the shared secret and decrypt the payload.

use core::fmt::Debug;

use rand_core::RngCore;

use crate::{Fr, PairingBackend, errors::Error};

mod scheme;
pub use scheme::{SilentThreshold, SilentThresholdScheme};

mod keys;
pub use keys::{AggregateKey, PublicKey, SecretKey, UnsafeKeyMaterial};

mod params;
pub use params::Params;

mod ciphertext;
pub use ciphertext::{Ciphertext, DecryptionResult, PartialDecryption};

/// High-level threshold scheme interface.
///
/// This trait defines the complete API for a threshold scheme, from setup
/// through key generation to encryption and decryption.
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
    /// Unsafe: this generates secret keys for all `n` participants and derives their
    /// corresponding public keys with Lagrange commitment hints.
    ///
    /// You should only use this method for testing scenarios.
    /// For real-world usage, each participant should generate their own key pair
    /// independently using `keygen_single_validator()`. The aggregate public key
    /// can then be computed using `aggregate_public_key()`.
    fn keygen_unsafe<R: RngCore + ?Sized>(
        &self,
        rng: &mut R,
        parties: usize,
        srs: &Params<B>,
    ) -> Result<UnsafeKeyMaterial<B>, Error>;

    /// Generates a key pair for a single validator (silent setup).
    ///
    /// This allows each validator to independently generate their own key pair
    /// without coordination. The validator samples a random secret key and derives
    /// their public key using the precomputed Lagrange commitments in `params`.
    ///
    /// # Arguments
    ///
    /// * `rng` - Cryptographically secure random number generator
    /// * `validator_id` - Index of this validator (must be in range [0, parties))
    /// * `params` - Precomputed parameters from `param_gen()`
    ///
    /// # Returns
    ///
    /// A tuple of `(SecretKey, PublicKey)` for this validator.
    ///
    /// # Errors
    ///
    /// Returns error if `validator_id >= parties` (where parties is from params).
    fn keygen_single_validator<R: RngCore + ?Sized>(
        &self,
        rng: &mut R,
        validator_id: usize,
        params: &Params<B>,
    ) -> Result<(SecretKey<B>, PublicKey<B>), Error>;

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
        params: &Params<B>,
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
