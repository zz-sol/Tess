//! Symmetric encryption primitives for payload encapsulation.
//!
//! This module provides symmetric encryption used to encrypt payloads in the
//! threshold encryption scheme. The shared secret derived from the threshold
//! protocol is used to encrypt the actual message payload.
//!
//! # Overview
//!
//! After threshold decryption recovers the shared secret, this module's
//! encryption is used to:
//! - Encrypt the payload during encryption (using shared secret as key)
//! - Decrypt the payload during aggregated decryption
//!
//! # Implementations
//!
//! Currently provides:
//! - **[`Blake3XorEncryption`]**: XOR-based encryption using BLAKE3 in XOF mode
//!
//! # Example
//!
//! ```rust
//! use tess::{Blake3XorEncryption, SymmetricEncryption};
//!
//! let encryption = Blake3XorEncryption::default();
//! let secret = b"my-secret-key-32-bytes-long-min!";
//! let plaintext = b"Hello, threshold encryption!";
//!
//! // Encrypt
//! let ciphertext = encryption.encrypt(secret, plaintext).unwrap();
//!
//! // Decrypt
//! let recovered = encryption.decrypt(secret, &ciphertext).unwrap();
//! assert_eq!(plaintext, &recovered[..]);
//! ```

use alloc::vec::Vec;
use core::fmt::Debug;

use blake3::Hasher;

use crate::Error;

/// Trait for symmetric encryption/decryption operations.
///
/// This trait abstracts away the details of symmetric encryption,
/// allowing for flexible implementations (e.g., BLAKE3 XOR, AES-GCM, ChaCha20-Poly1305).
///
/// # Security
///
/// Implementations should provide authenticated encryption when possible.
/// The current XOR-based implementation is suitable for demonstration but
/// production systems should consider authenticated encryption schemes.
///
/// # Example
///
/// ```rust
/// use tess::{Blake3XorEncryption, SymmetricEncryption};
///
/// let enc = Blake3XorEncryption::default();
/// let key = b"secret-key";
/// let msg = b"secret message";
///
/// let ct = enc.encrypt(key, msg).unwrap();
/// let pt = enc.decrypt(key, &ct).unwrap();
/// assert_eq!(msg, &pt[..]);
/// ```
pub trait SymmetricEncryption: Debug + Send + Sync {
    /// Encrypts plaintext with the given secret.
    ///
    /// # Parameters
    ///
    /// - `secret`: Secret key bytes (length depends on implementation)
    /// - `plaintext`: Data to encrypt
    ///
    /// # Returns
    ///
    /// The encrypted ciphertext, or an error if encryption fails.
    fn encrypt(&self, secret: &[u8], plaintext: &[u8]) -> Result<Vec<u8>, Error>;

    /// Decrypts ciphertext with the given secret.
    ///
    /// # Parameters
    ///
    /// - `secret`: Secret key bytes (must match encryption key)
    /// - `ciphertext`: Encrypted data
    ///
    /// # Returns
    ///
    /// The decrypted plaintext, or an error if decryption fails.
    fn decrypt(&self, secret: &[u8], ciphertext: &[u8]) -> Result<Vec<u8>, Error>;
}

/// BLAKE3-based symmetric encryption using XOR with extended output function (XOF).
///
/// This implementation uses BLAKE3 in XOF mode to derive a keystream from the secret,
/// then XORs it with the plaintext/ciphertext. This provides confidentiality but not
/// authentication.
///
/// # Security Considerations
///
/// - **Confidentiality**: Provides semantic security assuming BLAKE3 XOF is a secure PRF
/// - **Authentication**: Does NOT provide authentication or integrity protection
/// - **Recommended for**: Demonstration and non-critical applications
/// - **Production alternative**: Consider AES-GCM or ChaCha20-Poly1305 for authenticated encryption
///
/// # Example
///
/// ```rust
/// use tess::{Blake3XorEncryption, SymmetricEncryption};
///
/// // Create with custom domain separation
/// let enc = Blake3XorEncryption::new(b"my-app::encryption");
///
/// let secret = b"32-byte-secret-key-here-please!";
/// let message = b"Sensitive data to encrypt";
///
/// let ciphertext = enc.encrypt(secret, message).unwrap();
/// let plaintext = enc.decrypt(secret, &ciphertext).unwrap();
///
/// assert_eq!(message, &plaintext[..]);
/// ```
#[derive(Debug, Clone)]
pub struct Blake3XorEncryption {
    /// Domain separation tag for BLAKE3 KDF.
    domain: &'static [u8],
}

impl Blake3XorEncryption {
    /// Creates a new BLAKE3-based encryption with the given domain.
    pub fn new(domain: &'static [u8]) -> Self {
        Self { domain }
    }
}

impl Default for Blake3XorEncryption {
    fn default() -> Self {
        Self::new(b"tess::payload")
    }
}

impl SymmetricEncryption for Blake3XorEncryption {
    fn encrypt(&self, secret: &[u8], plaintext: &[u8]) -> Result<Vec<u8>, Error> {
        if plaintext.is_empty() {
            return Ok(Vec::new());
        }
        let keystream = self.derive_keystream(secret, plaintext.len());
        Ok(xor_bytes(&keystream, plaintext))
    }

    fn decrypt(&self, secret: &[u8], ciphertext: &[u8]) -> Result<Vec<u8>, Error> {
        if ciphertext.is_empty() {
            return Ok(Vec::new());
        }
        let keystream = self.derive_keystream(secret, ciphertext.len());
        Ok(xor_bytes(&keystream, ciphertext))
    }
}

impl Blake3XorEncryption {
    fn derive_keystream(&self, secret: &[u8], len: usize) -> Vec<u8> {
        if len == 0 {
            return Vec::new();
        }
        let mut hasher = Hasher::new();
        hasher.update(self.domain);
        hasher.update(secret);
        hasher.update(&(len as u64).to_le_bytes());
        let mut reader = hasher.finalize_xof();
        let mut keystream = vec![0u8; len];
        reader.fill(&mut keystream);
        keystream
    }
}

/// XORs two byte slices together.
fn xor_bytes(a: &[u8], b: &[u8]) -> Vec<u8> {
    a.iter().zip(b.iter()).map(|(x, y)| x ^ y).collect()
}
