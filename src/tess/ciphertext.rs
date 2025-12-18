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
/// let params = scheme.param_gen(&mut rng, 5, 3).unwrap();
/// let keys = scheme.keygen(&mut rng, 5, &params).unwrap();
///
/// // Encrypt a message
/// let message = b"Secret threshold message";
/// let ciphertext = scheme.encrypt(
///     &mut rng,
///     &keys.aggregate_key,
///     3,
///     message
/// ).unwrap();
///
/// // Ciphertext contains encrypted payload and proofs
/// assert_eq!(ciphertext.threshold, 3);
/// assert!(!ciphertext.payload.is_empty());
/// ```
#[derive(Clone, Debug)]
pub struct Ciphertext<B: PairingBackend> {
    pub gamma_g2: B::G2,
    pub proof_g1: Vec<B::G1>,
    pub proof_g2: Vec<B::G2>,
    pub shared_secret: B::Target,
    pub threshold: usize,
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
/// let params = scheme.param_gen(&mut rng, 5, 3).unwrap();
/// let keys = scheme.keygen(&mut rng, 5, &params).unwrap();
/// let ciphertext = scheme.encrypt(&mut rng, &keys.aggregate_key, 3, b"message").unwrap();
///
/// // Each participant creates a partial decryption
/// let partial = scheme.partial_decrypt(&keys.secret_keys[0], &ciphertext).unwrap();
/// assert_eq!(partial.participant_id, 0);
/// ```
#[derive(Debug)]
pub struct PartialDecryption<B: PairingBackend> {
    pub participant_id: usize,
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
/// let params = scheme.param_gen(&mut rng, 5, 3).unwrap();
/// let keys = scheme.keygen(&mut rng, 5, &params).unwrap();
///
/// let message = b"Threshold encrypted message";
/// let ciphertext = scheme.encrypt(&mut rng, &keys.aggregate_key, 3, message).unwrap();
///
/// // Collect 3 partial decryptions
/// let mut selector = vec![false; 5];
/// let mut partials = Vec::new();
/// for i in 0..3 {
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
    pub plaintext: Option<Vec<u8>>,
}
