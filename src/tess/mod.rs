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
pub use keys::{AggregateKey, KeyMaterial, PublicKey, SecretKey};

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

#[cfg(test)]
mod tests {
    use rand::{SeedableRng, rngs::StdRng};

    use crate::{Error, Fr, PairingBackend, SilentThreshold, ThresholdEncryption};

    const PARTIES: usize = 8;
    const THRESHOLD: usize = 4;
    const PAYLOAD: &[u8] = b"payload";

    fn run_roundtrip<B: PairingBackend<Scalar = Fr>>() {
        let mut rng = StdRng::from_entropy();
        let scheme = SilentThreshold::<B>::new();
        let params = scheme
            .param_gen(&mut rng, PARTIES, THRESHOLD)
            .expect("param generation");
        let key_material = scheme
            .keygen(&mut rng, PARTIES, &params)
            .expect("key generation");

        let ciphertext = scheme
            .encrypt(
                &mut rng,
                &key_material.aggregate_key,
                &params,
                THRESHOLD,
                PAYLOAD,
            )
            .expect("encryption");

        let mut selector = vec![false; PARTIES];
        let mut partials = Vec::new();

        for idx in 0..(THRESHOLD + 1) {
            selector[idx] = true;
            let partial = scheme
                .partial_decrypt(&key_material.secret_keys[idx], &ciphertext)
                .expect("partial decrypt");
            partials.push(partial);
        }

        let result = scheme
            .aggregate_decrypt(
                &ciphertext,
                &partials,
                &selector,
                &key_material.aggregate_key,
            )
            .expect("aggregate decrypt");

        assert!(result.plaintext.is_some(), "plaintext not recovered");
    }

    fn run_not_enough_shares<B: PairingBackend<Scalar = Fr>>() {
        let mut rng = StdRng::from_entropy();
        let scheme = SilentThreshold::<B>::new();
        let params = scheme
            .param_gen(&mut rng, PARTIES, THRESHOLD)
            .expect("param generation");
        let key_material = scheme
            .keygen(&mut rng, PARTIES, &params)
            .expect("key generation");

        let ciphertext = scheme
            .encrypt(
                &mut rng,
                &key_material.aggregate_key,
                &params,
                THRESHOLD,
                PAYLOAD,
            )
            .expect("encryption");

        let mut selector = vec![false; PARTIES];
        let mut partials = Vec::new();

        for idx in 0..THRESHOLD {
            selector[idx] = true;
            if idx < THRESHOLD - 1 {
                let partial = scheme
                    .partial_decrypt(&key_material.secret_keys[idx], &ciphertext)
                    .expect("partial decrypt");
                partials.push(partial);
            }
        }

        let result = scheme.aggregate_decrypt(
            &ciphertext,
            &partials,
            &selector,
            &key_material.aggregate_key,
        );

        assert!(
            matches!(
                result,
                Err(Error::NotEnoughShares {
                    required,
                    provided
                }) if required == ciphertext.threshold && provided == partials.len()
            ),
            "expected not enough shares"
        );
    }

    fn run_selector_mismatch<B: PairingBackend<Scalar = Fr>>() {
        let mut rng = StdRng::from_entropy();
        let scheme = SilentThreshold::<B>::new();
        let params = scheme
            .param_gen(&mut rng, PARTIES, THRESHOLD)
            .expect("param generation");
        let key_material = scheme
            .keygen(&mut rng, PARTIES, &params)
            .expect("key generation");

        let ciphertext = scheme
            .encrypt(
                &mut rng,
                &key_material.aggregate_key,
                &params,
                THRESHOLD,
                PAYLOAD,
            )
            .expect("encryption");

        let mut partials = Vec::new();
        for idx in 0..(THRESHOLD + 1) {
            let partial = scheme
                .partial_decrypt(&key_material.secret_keys[idx], &ciphertext)
                .expect("partial decrypt");
            partials.push(partial);
        }

        let mut selector = vec![false; PARTIES - 1];
        for idx in 0..selector.len() {
            selector[idx] = true;
        }

        let result = scheme.aggregate_decrypt(
            &ciphertext,
            &partials,
            &selector,
            &key_material.aggregate_key,
        );

        assert!(
            matches!(
                result,
                Err(Error::SelectorMismatch { expected, actual })
                    if expected == PARTIES && actual == PARTIES - 1
            ),
            "expected selector mismatch"
        );
    }

    #[test]
    fn encrypt_decrypt_roundtrip() {
        run_roundtrip::<crate::PairingEngine>();
    }

    #[test]
    fn decrypt_not_enough_shares() {
        run_not_enough_shares::<crate::PairingEngine>();
    }

    #[test]
    fn selector_mismatch() {
        run_selector_mismatch::<crate::PairingEngine>();
    }
}
