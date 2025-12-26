//! Ciphertext and decryption result structures.
//!
//! This module defines the output types for encryption and decryption operations:
//!
//! - [`Ciphertext`]: The encrypted message with KZG proofs
//! - [`PartialDecryption`]: A participant's decryption share
//! - [`DecryptionResult`]: The final decrypted plaintext
//!
//! # Ciphertext Structure
//!
//! A ciphertext contains:
//! - **Encrypted Payload**: The message encrypted using BLAKE3-based XOR encryption
//! - **KZG Proofs**: Zero-knowledge proofs in G1 and G2 that enable verification
//! - **Shared Secret**: Precomputed pairing result for efficient verification
//! - **Threshold**: The minimum number of shares required for decryption
//!
//! # Decryption Protocol
//!
//! 1. Each participant computes a partial decryption using their secret key
//! 2. At least `t` partial decryptions are collected
//! 3. The partial decryptions are aggregated using Lagrange interpolation
//! 4. The ciphertext is verified using the KZG proofs
//! 5. If verification succeeds, the payload is decrypted
//!
//! # Security
//!
//! The KZG proofs ensure that:
//! - The ciphertext was properly formed
//! - The encryption used the correct aggregate public key
//! - Partial decryptions cannot be forged
//! - The threshold requirement is enforced
//!
//! # Important
//!
//! The ciphertext includes the derived `shared_secret`, so the payload encryption
//! is not confidential against anyone who can read the ciphertext. This is
//! intended behavior in this reference-based implementation and is only suitable
//! when confidentiality of the payload is not required.

use alloc::vec::Vec;
use core::fmt::Debug;

use crate::PairingBackend;

/// Ciphertext output from threshold encryption.
///
/// This structure contains the encrypted payload along with KZG proofs
/// that enable threshold decryption and verification.
///
/// # Fields
///
/// - `gamma_g2`: Random group element in G2 used for encryption
/// - `proof_g1`: KZG proof elements in G1 for verification
/// - `proof_g2`: KZG proof elements in G2 for verification
/// - `shared_secret`: Precomputed pairing result for efficiency
/// - `threshold`: Minimum number of partial decryptions required
/// - `payload`: Encrypted message bytes
///
/// # Example
///
/// ```rust
/// use rand::thread_rng;
/// use tess::{PairingEngine, SilentThresholdScheme, ThresholdEncryption};
///
/// let mut rng = thread_rng();
/// let scheme = SilentThresholdScheme::<PairingEngine>::new();
///
/// let params = scheme.param_gen(&mut rng, 8, 4).unwrap();
/// let keys = scheme.keygen_unsafe(&mut rng, 8, &params).unwrap();
///
/// // Encrypt a message
/// let message = b"Secret threshold message";
/// let ciphertext = scheme.encrypt(
///     &mut rng,
///     &keys.aggregate_key,
///     &params,
///     4,
///     message
/// ).unwrap();
///
/// // Ciphertext contains encrypted payload and proofs
/// assert_eq!(ciphertext.threshold, 4);
/// assert!(!ciphertext.payload.is_empty());
/// ```
#[derive(Clone, Debug)]
pub struct Ciphertext<B: PairingBackend> {
    /// Random G2 element used during encryption.
    pub gamma_g2: B::G2,
    /// KZG proof elements in G1.
    pub proof_g1: Vec<B::G1>,
    /// KZG proof elements in G2.
    pub proof_g2: Vec<B::G2>,
    /// Precomputed pairing result for verification.
    pub shared_secret: B::Target,
    /// Threshold required for decryption.
    pub threshold: usize,
    /// Encrypted payload bytes.
    pub payload: Vec<u8>,
}

/// Partial decryption share from a single participant.
///
/// Each participant uses their secret key to compute a partial decryption.
/// At least `t` partial decryptions are required to recover the plaintext.
///
/// # Fields
///
/// - `participant_id`: The unique identifier of the participant (0-indexed)
/// - `response`: The G2 group element representing this participant's decryption share
///
/// # Example
///
/// ```rust
/// use rand::thread_rng;
/// use tess::{PairingEngine, SilentThresholdScheme, ThresholdEncryption};
///
/// let mut rng = thread_rng();
/// let scheme = SilentThresholdScheme::<PairingEngine>::new();
///
/// let params = scheme.param_gen(&mut rng, 8, 4).unwrap();
/// let keys = scheme.keygen_unsafe(&mut rng, 8, &params).unwrap();
/// let ciphertext = scheme.encrypt(&mut rng, &keys.aggregate_key,&params, 4, b"message").unwrap();
///
/// // Each participant creates a partial decryption
/// let partial = scheme.partial_decrypt(&keys.secret_keys[0], &ciphertext).unwrap();
/// assert_eq!(partial.participant_id, 0);
/// ```
#[derive(Debug)]
pub struct PartialDecryption<B: PairingBackend> {
    /// Participant identifier (0-indexed).
    pub participant_id: usize,
    /// Partial decryption share in G2.
    pub response: B::G2,
}

impl<B: PairingBackend> Clone for PartialDecryption<B> {
    fn clone(&self) -> Self {
        Self {
            participant_id: self.participant_id,
            response: self.response,
        }
    }
}

/// Decryption result containing the recovered plaintext.
///
/// This structure is returned after successfully aggregating at least `t`
/// partial decryptions.
///
/// # Fields
///
/// - `plaintext`: The recovered plaintext bytes, or `None` if decryption failed
///
/// # Example
///
/// ```rust
/// use rand::thread_rng;
/// use tess::{PairingEngine, SilentThresholdScheme, ThresholdEncryption};
///
/// let mut rng = thread_rng();
/// let scheme = SilentThresholdScheme::<PairingEngine>::new();
///
/// let params = scheme.param_gen(&mut rng, 8, 4).unwrap();
/// let keys = scheme.keygen_unsafe(&mut rng, 8, &params).unwrap();
///
/// let message = b"Threshold encrypted message";
/// let ciphertext = scheme.encrypt(&mut rng, &keys.aggregate_key, &params, 4, message).unwrap();
///
/// // Collect 5 partial decryptions (t + 1)
/// let mut selector = vec![false; 8];
/// let mut partials = Vec::new();
/// for i in 0..5 {
///     selector[i] = true;
///     partials.push(scheme.partial_decrypt(&keys.secret_keys[i], &ciphertext).unwrap());
/// }
///
/// // Aggregate to recover plaintext
/// let result = scheme.aggregate_decrypt(
///     &ciphertext,
///     &partials,
///     &selector,
///     &keys.aggregate_key
/// ).unwrap();
///
/// assert!(result.plaintext.is_some());
/// assert_eq!(result.plaintext.as_ref().unwrap(), message);
/// ```
#[derive(Clone, Debug)]
pub struct DecryptionResult {
    /// Decrypted plaintext if verification succeeded.
    pub plaintext: Option<Vec<u8>>,
}
