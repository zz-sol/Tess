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
//!
//! # Example
//!
//! ```rust,no_run
//! # #[cfg(feature = "blst")]
//! # {
//! use tess::{ThresholdParameters, BackendConfig, CurveId, BackendId};
//! use tess::protocol::{SilentThreshold, ThresholdScheme};
//! use tess::backend::BlstBackend;
//! use rand::thread_rng;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let mut rng = thread_rng();
//! let params = ThresholdParameters::new(
//!     5, 3,
//!     BackendConfig { backend: BackendId::Blst, curve: CurveId::Bls12_381 },
//!     None,
//! )?;
//!
//! // Create scheme instance
//! let scheme = SilentThreshold::<BlstBackend>::default();
//!
//! // Generate SRS
//! let (srs, tau) = scheme.srs_gen(&mut rng, &params)?;
//!
//! // Generate keys
//! let key_material = scheme.keygen(&mut rng, &params, &srs)?;
//!
//! // Encrypt
//! let plaintext = b"Secret message";
//! let ciphertext = scheme.encrypt(
//!     &mut rng,
//!     &key_material.aggregate_key,
//!     &params,
//!     plaintext,
//! )?;
//!
//! // Partial decryptions from first 3 participants
//! let mut partials = Vec::new();
//! for i in 0..3 {
//!     let partial = scheme.partial_decrypt(&key_material.secret_keys[i], &ciphertext)?;
//!     partials.push(partial);
//! }
//!
//! // Aggregate to recover plaintext
//! let selector = vec![true, true, true, false, false];
//! let result = scheme.aggregate_decrypt(
//!     &ciphertext,
//!     &partials,
//!     &selector,
//!     &key_material.aggregate_key,
//! )?;
//!
//! assert_eq!(result.plaintext.unwrap(), plaintext);
//! # Ok(())
//! # }
//! # }
//! # #[cfg(not(feature = "blst"))]
//! # fn main() {}
//! ```
//!
//! # Security
//!
//! - **Trusted Setup**: The SRS generation requires a trusted setup. The secret tau must be
//!   securely discarded after generation.
//! - **Threshold Security**: The scheme remains secure as long as fewer than `t` participants
//!   are compromised.
//! - **Payload Encryption**: Uses BLAKE3 as a KDF with domain separation to derive symmetric
//!   encryption keys from the shared secret.

use core::{
    fmt::Debug,
    marker::PhantomData,
    ops::{Sub, SubAssign},
};

use blake3::Hasher;
use rand_core::RngCore;
use rayon::iter::{
    IndexedParallelIterator, IntoParallelIterator, IntoParallelRefIterator, ParallelIterator,
};
use tracing::{instrument, trace};

use crate::{
    backend::{
        CurvePoint, EvaluationDomain, FieldElement, MsmProvider, PairingBackend, Polynomial,
        PolynomialCommitment, TargetGroup,
    },
    config::{BackendId, CurveId, ThresholdParameters},
    errors::{BackendError, Error},
    lagrange::LagrangeField,
};

type CommitmentPolynomial<B> =
    <<B as PairingBackend>::PolynomialCommitment as PolynomialCommitment<B>>::Polynomial;
type CommitmentParams<B> =
    <<B as PairingBackend>::PolynomialCommitment as PolynomialCommitment<B>>::Parameters;
type BackendScalar<B> = <B as PairingBackend>::Scalar;

/// Domain separation tag for BLAKE3 KDF used in payload encryption.
const PAYLOAD_KDF_DOMAIN: &[u8] = b"tess::threshold::payload";

/// Helper trait combining requirements for protocol scalar fields.
pub trait ProtocolScalar: LagrangeField + SubAssign + Sub<Output = Self> {}

impl<T> ProtocolScalar for T where T: LagrangeField + SubAssign + Sub<Output = T> {}

/// Secret key owned by a single participant.
///
/// This represents a participant's secret share in the threshold scheme.
/// The secret scalar is used to compute partial decryptions.
///
/// # Security
///
/// The secret key must be kept confidential. Exposing fewer than `t` secret keys
/// does not compromise the security of the scheme.
#[derive(Clone, Debug)]
pub struct SecretKey<B: PairingBackend> {
    pub participant_id: usize,
    pub scalar: B::Scalar,
}

/// Public key with Lagrange commitment hints for efficient verification.
///
/// This structure contains a participant's public key along with precomputed
/// Lagrange polynomial commitments that enable efficient verification of
/// partial decryptions without requiring polynomial interpolation during
/// the decryption phase.
///
/// # Fields
///
/// - `participant_id`: The participant's unique identifier (0-indexed)
/// - `bls_key`: The participant's BLS public key (scalar * G1)
/// - `lagrange_li`: Commitment to the i-th Lagrange polynomial L_i(x)
/// - `lagrange_li_minus0`: Commitment to L_i(x) - L_i(0)
/// - `lagrange_li_x`: Commitment to x * L_i(x)
/// - `lagrange_li_lj_z`: Commitments to L_i(x) * L_j(z) for all j
#[derive(Debug)]
pub struct PublicKey<B: PairingBackend> {
    pub participant_id: usize,
    pub bls_key: B::G1,
    pub lagrange_li: B::G1,
    pub lagrange_li_minus0: B::G1,
    pub lagrange_li_x: B::G1,
    pub lagrange_li_lj_z: Vec<B::G1>,
}

impl<B: PairingBackend> Clone for PublicKey<B> {
    fn clone(&self) -> Self {
        Self {
            participant_id: self.participant_id,
            bls_key: self.bls_key.clone(),
            lagrange_li: self.lagrange_li.clone(),
            lagrange_li_minus0: self.lagrange_li_minus0.clone(),
            lagrange_li_x: self.lagrange_li_x.clone(),
            lagrange_li_lj_z: self.lagrange_li_lj_z.clone(),
        }
    }
}

/// Aggregate public key for encryption and verification.
///
/// This structure contains the aggregated public keys and precomputed values
/// needed for efficient encryption and verification of partial decryptions.
///
/// # Fields
///
/// - `public_keys`: All participants' public keys
/// - `ask`: Aggregated secret key commitment (sum of all BLS keys)
/// - `z_g2`: Commitment to the vanishing polynomial in G2
/// - `lagrange_row_sums`: Precomputed sums of Lagrange commitments for verification
/// - `precomputed_pairing`: Precomputed pairing for efficient verification
/// - `commitment_params`: KZG commitment parameters from SRS
#[derive(Clone, Debug)]
pub struct AggregateKey<B: PairingBackend> {
    pub public_keys: Vec<PublicKey<B>>,
    pub ask: B::G1,
    pub z_g2: B::G2,
    pub lagrange_row_sums: Vec<B::G1>,
    pub precomputed_pairing: B::Target,
    pub commitment_params: CommitmentParams<B>,
}

/// Ciphertext output from threshold encryption.
///
/// This structure contains the encrypted payload along with KZG proofs
/// that enable threshold decryption and verification.
///
/// # Fields
///
/// - `gamma_g2`: Random point in G2 used for encryption (r * G2)
/// - `proof_g1`: KZG proof elements in G1 for verification
/// - `proof_g2`: KZG proof elements in G2 for verification
/// - `shared_secret`: The encrypted shared secret in GT
/// - `threshold`: The threshold value (t) required for decryption
/// - `payload`: The encrypted payload (BLAKE3-encapsulated)
///
/// # Security
///
/// The ciphertext is computationally hiding under the Decisional Diffie-Hellman
/// assumption in the pairing groups. The KZG proofs ensure that partial
/// decryptions can be verified without trusted interaction.
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
/// - `participant_id`: The participant's unique identifier
/// - `response`: The participant's decryption share (secret * gamma_g2)
#[derive(Debug)]
pub struct PartialDecryption<B: PairingBackend> {
    pub participant_id: usize,
    pub response: B::G2,
}

impl<B: PairingBackend> Clone for PartialDecryption<B> {
    fn clone(&self) -> Self {
        Self {
            participant_id: self.participant_id,
            response: self.response.clone(),
        }
    }
}

/// Precomputed Lagrange polynomial commitments for efficient key derivation.
///
/// This structure contains KZG commitments to various Lagrange polynomial
/// transformations, precomputed during SRS generation to enable efficient
/// key generation without polynomial operations at key generation time.
///
/// # Fields
///
/// - `li`: Commitments to L_i(x) for each participant i
/// - `li_minus0`: Commitments to L_i(x) - L_i(0) for each i
/// - `li_x`: Commitments to x * L_i(x) for each i
/// - `li_lj_z`: Commitments to L_i(x) * L_j(z) for all pairs (i, j)
#[derive(Clone, Debug)]
pub struct LagrangePowers<B: PairingBackend> {
    pub li: Vec<B::G1>,
    pub li_minus0: Vec<B::G1>,
    pub li_x: Vec<B::G1>,
    pub li_lj_z: Vec<Vec<B::G1>>,
}

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
pub struct SRS<B: PairingBackend> {
    pub commitment_params: CommitmentParams<B>,
    pub lagrange_powers: LagrangePowers<B>,
}

/// Complete key material bundle from key generation.
///
/// This structure contains all keys generated during the key generation phase,
/// including secret keys for all participants, their corresponding public keys,
/// and the aggregated key for encryption.
///
/// # Fields
///
/// - `secret_keys`: Secret keys for all participants
/// - `public_keys`: Public keys for all participants
/// - `aggregate_key`: Aggregated key for encryption and verification
/// - `kzg_params`: KZG commitment parameters (same as in SRS)
///
/// # Usage
///
/// In a real deployment, each participant would only receive their own secret key,
/// while public keys and the aggregate key would be distributed to all parties.
#[derive(Clone, Debug)]
pub struct KeyMaterial<B: PairingBackend> {
    pub secret_keys: Vec<SecretKey<B>>,
    pub public_keys: Vec<PublicKey<B>>,
    pub aggregate_key: AggregateKey<B>,
    pub kzg_params: CommitmentParams<B>,
}

/// Decryption result containing the recovered plaintext.
///
/// This structure is returned after successfully aggregating at least `t`
/// partial decryptions. It contains the shared secret and the decrypted payload.
///
/// # Fields
///
/// - `shared_secret`: The recovered shared secret in GT
/// - `opening_proof`: Optional KZG opening proof (reserved for future use)
/// - `plaintext`: The decrypted plaintext payload
#[derive(Clone, Debug)]
pub struct DecryptionResult<B: PairingBackend> {
    pub shared_secret: B::Target,
    pub opening_proof: Option<Vec<u8>>,
    pub plaintext: Option<Vec<u8>>,
}

/// High-level threshold encryption scheme interface.
///
/// This trait defines the complete API for a threshold encryption scheme,
/// from setup through key generation to encryption and decryption.
///
/// # Type Parameters
///
/// - `B`: The protocol backend providing cryptographic operations
///
/// # Example
///
/// See the [module-level documentation](self) for a complete example.
pub trait ThresholdScheme<B: ProtocolBackend>: Debug + Send + Sync + 'static {
    /// Generates the Structured Reference String (SRS) for the scheme.
    ///
    /// This is a one-time trusted setup that generates KZG commitment parameters
    /// (powers of tau) and precomputes Lagrange polynomial commitments for
    /// efficient key generation.
    ///
    /// # Arguments
    ///
    /// - `rng`: Cryptographically secure random number generator
    /// - `params`: Threshold scheme parameters (n, t, backend, optional tau)
    ///
    /// # Returns
    ///
    /// A tuple containing:
    /// - `SRS<B>`: The structured reference string with commitment parameters
    /// - `Vec<u8>`: The serialized tau value (must be securely discarded!)
    ///
    /// # Security
    ///
    /// The secret tau value **must be securely discarded** after generation.
    /// Knowledge of tau would allow forging proofs and breaking the scheme.
    /// In production, consider using an MPC ceremony for trusted setup.
    fn srs_gen<R: RngCore + ?Sized>(
        &self,
        rng: &mut R,
        params: &ThresholdParameters,
    ) -> Result<(SRS<B>, Vec<u8>), Error>;

    /// Generates key material for all participants.
    ///
    /// This generates secret keys for all `n` participants and derives their
    /// corresponding public keys with Lagrange commitment hints. It also
    /// constructs the aggregate key used for encryption.
    ///
    /// # Arguments
    ///
    /// - `rng`: Cryptographically secure random number generator
    /// - `params`: Threshold scheme parameters
    /// - `srs`: The structured reference string from `srs_gen`
    ///
    /// # Returns
    ///
    /// Complete key material including secret keys, public keys, and aggregate key.
    ///
    /// # Distribution
    ///
    /// In a real deployment:
    /// - Each participant receives only their own secret key
    /// - Public keys and aggregate key are distributed to all parties
    /// - Secret keys must be kept confidential
    fn keygen<R: RngCore + ?Sized>(
        &self,
        rng: &mut R,
        params: &ThresholdParameters,
        srs: &SRS<B>,
    ) -> Result<KeyMaterial<B>, Error>;

    /// Recomputes the aggregate key from public keys.
    ///
    /// This allows reconstructing the aggregate key from a set of public keys,
    /// which is useful when:
    /// - Rotating participants
    /// - Recovering from partial key loss
    /// - Updating the key set without regenerating from scratch
    ///
    /// # Arguments
    ///
    /// - `params`: Threshold scheme parameters
    /// - `public_keys`: Slice of public keys to aggregate
    /// - `commitment_params`: KZG commitment parameters from SRS
    ///
    /// # Returns
    ///
    /// The reconstructed aggregate key for encryption and verification.
    fn aggregate_public_key(
        &self,
        params: &ThresholdParameters,
        public_keys: &[PublicKey<B>],
        commitment_params: &CommitmentParams<B>,
    ) -> Result<AggregateKey<B>, Error>;

    /// Encrypts a payload using the aggregate key.
    ///
    /// This encrypts the payload such that it can only be decrypted when at
    /// least `t` participants cooperate. The ciphertext includes KZG proofs
    /// for verification and a BLAKE3-encapsulated payload.
    ///
    /// # Arguments
    ///
    /// - `rng`: Cryptographically secure random number generator
    /// - `agg_key`: The aggregate public key from key generation
    /// - `params`: Threshold scheme parameters
    /// - `payload`: The plaintext data to encrypt
    ///
    /// # Returns
    ///
    /// A ciphertext that can be decrypted with threshold partial decryptions.
    ///
    /// # Security
    ///
    /// The encryption is semantically secure under the DDH assumption in the
    /// pairing groups. Each encryption uses fresh randomness for security.
    fn encrypt<R: RngCore + ?Sized>(
        &self,
        rng: &mut R,
        agg_key: &AggregateKey<B>,
        params: &ThresholdParameters,
        payload: &[u8],
    ) -> Result<Ciphertext<B>, Error>;

    /// Computes a partial decryption share.
    ///
    /// Each participant uses their secret key to compute a decryption share.
    /// The share can be verified against the participant's public key and the
    /// ciphertext without revealing the secret key.
    ///
    /// # Arguments
    ///
    /// - `secret_key`: The participant's secret key
    /// - `ciphertext`: The ciphertext to partially decrypt
    ///
    /// # Returns
    ///
    /// A partial decryption that can be aggregated with others to recover
    /// the plaintext.
    fn partial_decrypt(
        &self,
        secret_key: &SecretKey<B>,
        ciphertext: &Ciphertext<B>,
    ) -> Result<PartialDecryption<B>, Error>;

    /// Aggregates partial decryptions to recover the plaintext.
    ///
    /// This combines at least `t` partial decryptions to recover the shared
    /// secret and decrypt the payload. It verifies each partial decryption
    /// before aggregation to detect malicious participants.
    ///
    /// # Arguments
    ///
    /// - `ciphertext`: The ciphertext to decrypt
    /// - `partials`: The partial decryptions from participants
    /// - `selector`: Boolean array indicating which participants contributed
    /// - `agg_key`: The aggregate key for verification
    ///
    /// # Returns
    ///
    /// The decryption result containing the recovered plaintext.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Fewer than `t` valid partial decryptions are provided
    /// - Any partial decryption fails verification
    /// - The selector doesn't match the provided partials
    fn aggregate_decrypt(
        &self,
        ciphertext: &Ciphertext<B>,
        partials: &[PartialDecryption<B>],
        selector: &[bool],
        agg_key: &AggregateKey<B>,
    ) -> Result<DecryptionResult<B>, Error>;
}

/// Additional capabilities required from backends to support the threshold protocol.
pub trait ProtocolBackend: PairingBackend {
    fn backend_id() -> BackendId;
    fn curve_id() -> CurveId;
    fn parse_tau(bytes: &[u8]) -> Result<Self::Scalar, Error>;
    fn sample_tau<R: RngCore + ?Sized>(rng: &mut R) -> Self::Scalar;
    fn lagrange_polynomials(
        parties: usize,
    ) -> Result<Vec<CommitmentPolynomial<Self>>, BackendError>;
    fn interp_mostly_zero(
        eval: Self::Scalar,
        points: &[Self::Scalar],
    ) -> Result<CommitmentPolynomial<Self>, BackendError>;
    fn polynomial_from_coeffs(coeffs: Vec<Self::Scalar>) -> CommitmentPolynomial<Self>;
    fn domain_new(size: usize) -> Result<Self::Domain, BackendError>;
    fn g_powers(
        params: &CommitmentParams<Self>,
    ) -> &[<Self::G1 as CurvePoint<Self::Scalar>>::Affine];
    fn h_powers(
        params: &CommitmentParams<Self>,
    ) -> &[<Self::G2 as CurvePoint<Self::Scalar>>::Affine];
    fn pairing_generator(params: &CommitmentParams<Self>) -> Self::Target;
}

#[derive(Debug)]
pub struct SilentThreshold<B: ProtocolBackend>(PhantomData<B>);

impl<B: ProtocolBackend> Default for SilentThreshold<B> {
    fn default() -> Self {
        Self(PhantomData)
    }
}

impl<B> SilentThreshold<B>
where
    B: ProtocolBackend,
    BackendScalar<B>: ProtocolScalar,
{
    fn ensure_backend(params: &ThresholdParameters) -> Result<(), Error> {
        if params.backend.backend != B::backend_id() {
            return Err(Error::Backend(BackendError::UnsupportedFeature(
                "backend mismatch for SilentThreshold",
            )));
        }
        if params.backend.curve != B::curve_id() {
            return Err(Error::Backend(BackendError::UnsupportedCurve(
                "curve mismatch for SilentThreshold",
            )));
        }
        Ok(())
    }

    fn generate_secret_keys<R: RngCore + ?Sized>(rng: &mut R, parties: usize) -> Vec<SecretKey<B>> {
        (0..parties)
            .map(|participant_id| SecretKey {
                participant_id,
                scalar: <B::Scalar as FieldElement>::random(rng),
            })
            .collect()
    }
}

impl<B> ThresholdScheme<B> for SilentThreshold<B>
where
    B: ProtocolBackend,
    BackendScalar<B>: ProtocolScalar,
{
    #[instrument(level = "info", skip(self, rng, params))]
    fn srs_gen<R: RngCore + ?Sized>(
        &self,
        rng: &mut R,
        params: &ThresholdParameters,
    ) -> Result<(SRS<B>, Vec<u8>), Error> {
        params.validate()?;
        Self::ensure_backend(params)?;
        let parties = params.parties;
        let tau = if let Some(bytes) = params.kzg_tau.as_ref() {
            B::parse_tau(bytes)?
        } else {
            B::sample_tau(rng)
        };
        let commitment_params =
            B::PolynomialCommitment::setup(parties, &tau).map_err(Error::Backend)?;

        // Precompute Lagrange polynomials and their powers
        let lagranges = B::lagrange_polynomials(parties).map_err(Error::Backend)?;
        let lagrange_powers =
            precompute_lagrange_powers::<B>(&lagranges, parties, &tau, &commitment_params)
                .map_err(Error::Backend)?;

        let srs = SRS {
            commitment_params,
            lagrange_powers,
        };

        let tau_bytes = tau.to_repr().as_ref().to_vec();
        Ok((srs, tau_bytes))
    }

    #[instrument(level = "info", skip(self, rng, params, srs))]
    fn keygen<R: RngCore + ?Sized>(
        &self,
        rng: &mut R,
        params: &ThresholdParameters,
        srs: &SRS<B>,
    ) -> Result<KeyMaterial<B>, Error> {
        params.validate()?;
        Self::ensure_backend(params)?;
        let parties = params.parties;
        B::domain_new(parties).map_err(Error::Backend)?;
        let secret_keys = Self::generate_secret_keys(rng, parties);

        let public_keys = secret_keys
            .par_iter()
            .map(|sk| {
                derive_public_key_from_powers::<B>(sk.participant_id, sk, &srs.lagrange_powers)
            })
            .collect::<Result<Vec<_>, BackendError>>()
            .map_err(Error::Backend)?;
        let aggregate_key =
            aggregate_public_key::<B>(&public_keys, &srs.commitment_params, parties)?;
        Ok(KeyMaterial {
            secret_keys,
            public_keys,
            aggregate_key,
            kzg_params: srs.commitment_params.clone(),
        })
    }

    #[instrument(level = "info", skip(self, params, public_keys, commitment_params))]
    fn aggregate_public_key(
        &self,
        params: &ThresholdParameters,
        public_keys: &[PublicKey<B>],
        commitment_params: &CommitmentParams<B>,
    ) -> Result<AggregateKey<B>, Error> {
        params.validate()?;
        Self::ensure_backend(params)?;
        aggregate_public_key::<B>(public_keys, commitment_params, params.parties)
    }

    #[instrument(level = "info", skip(self, rng, agg_key, params, payload))]
    fn encrypt<R: RngCore + ?Sized>(
        &self,
        rng: &mut R,
        agg_key: &AggregateKey<B>,
        params: &ThresholdParameters,
        payload: &[u8],
    ) -> Result<Ciphertext<B>, Error> {
        params.validate()?;
        Self::ensure_backend(params)?;
        let threshold = params.threshold;
        let kzg_params = &agg_key.commitment_params;
        let g_powers = B::g_powers(kzg_params);
        let h_powers = B::h_powers(kzg_params);
        if threshold + 1 >= g_powers.len() {
            return Err(Error::Backend(BackendError::Math(
                "threshold exceeds supported commitment degree",
            )));
        }
        if h_powers.len() < 2 {
            return Err(Error::Backend(BackendError::Math(
                "not enough G2 powers for encryption",
            )));
        }
        let g = B::G1::from_affine(
            g_powers
                .first()
                .ok_or(BackendError::Math("missing g generator"))?,
        );
        let g_tau_t = B::G1::from_affine(
            g_powers
                .get(threshold + 1)
                .ok_or(BackendError::Math("missing g^{tau^{t+1}}"))?,
        );
        let h = B::G2::from_affine(
            h_powers
                .first()
                .ok_or(BackendError::Math("missing h generator"))?,
        );
        let h_tau = B::G2::from_affine(h_powers.get(1).ok_or(BackendError::Math("missing h^tau"))?);
        let h_minus_one = B::G2::generator().negate();

        let gamma = <B::Scalar as FieldElement>::random(rng);
        let gamma_g2 = h.mul_scalar(&gamma);

        let s0 = <B::Scalar as FieldElement>::random(rng);
        let s1 = <B::Scalar as FieldElement>::random(rng);
        let s2 = <B::Scalar as FieldElement>::random(rng);
        let s3 = <B::Scalar as FieldElement>::random(rng);
        let s4 = <B::Scalar as FieldElement>::random(rng);

        let sa1_0 = agg_key
            .ask
            .mul_scalar(&s0)
            .add(&g_tau_t.mul_scalar(&s3))
            .add(&g.mul_scalar(&s4));
        let sa1_1 = g.mul_scalar(&s2);

        let sa2_0 = h.mul_scalar(&s0).add(&gamma_g2.mul_scalar(&s2));
        let sa2_1 = agg_key.z_g2.mul_scalar(&s0);
        let sa2_2 = h_tau.mul_scalar(&(s0.clone() + s1.clone()));
        let sa2_3 = h.mul_scalar(&s1);
        let sa2_4 = h.mul_scalar(&s3);
        let sa2_5 = h_tau.add(&h_minus_one).mul_scalar(&s4);

        let shared_secret = agg_key.precomputed_pairing.mul_scalar(&s4);

        let proof_g1 = vec![sa1_0, sa1_1];
        let proof_g2 = vec![sa2_0, sa2_1, sa2_2, sa2_3, sa2_4, sa2_5];

        let payload_ct = encrypt_payload::<B>(&shared_secret, payload);

        Ok(Ciphertext {
            gamma_g2,
            proof_g1,
            proof_g2,
            shared_secret,
            threshold,
            payload: payload_ct,
        })
    }

    fn partial_decrypt(
        &self,
        secret_key: &SecretKey<B>,
        ciphertext: &Ciphertext<B>,
    ) -> Result<PartialDecryption<B>, Error> {
        let response = ciphertext.gamma_g2.mul_scalar(&secret_key.scalar);
        Ok(PartialDecryption {
            participant_id: secret_key.participant_id,
            response,
        })
    }

    #[instrument(
        level = "info",
        skip(self, ciphertext, partials, selector, agg_key),
        fields(num_partials = partials.len())
    )]
    fn aggregate_decrypt(
        &self,
        ciphertext: &Ciphertext<B>,
        partials: &[PartialDecryption<B>],
        selector: &[bool],
        agg_key: &AggregateKey<B>,
    ) -> Result<DecryptionResult<B>, Error> {
        aggregate_decrypt::<B>(ciphertext, partials, selector, agg_key)
    }
}

fn derive_keystream<B: PairingBackend>(secret: &B::Target, len: usize) -> Vec<u8> {
    if len == 0 {
        return Vec::new();
    }
    let mut hasher = Hasher::new();
    hasher.update(PAYLOAD_KDF_DOMAIN);
    let repr = secret.to_repr();
    hasher.update(repr.as_ref());
    hasher.update(&(len as u64).to_le_bytes());
    let mut reader = hasher.finalize_xof();
    let mut keystream = vec![0u8; len];
    reader.fill(&mut keystream);
    keystream
}

fn xor_with_keystream(data: &[u8], keystream: &[u8]) -> Vec<u8> {
    data.iter()
        .zip(keystream.iter())
        .map(|(byte, key)| byte ^ key)
        .collect()
}

fn encrypt_payload<B: PairingBackend>(secret: &B::Target, payload: &[u8]) -> Vec<u8> {
    let keystream = derive_keystream::<B>(secret, payload.len());
    xor_with_keystream(payload, &keystream)
}

fn decrypt_payload<B: PairingBackend>(secret: &B::Target, ciphertext: &[u8]) -> Vec<u8> {
    let keystream = derive_keystream::<B>(secret, ciphertext.len());
    xor_with_keystream(ciphertext, &keystream)
}

fn divide_by_linear<B: ProtocolBackend>(
    poly: &CommitmentPolynomial<B>,
    root: B::Scalar,
) -> (CommitmentPolynomial<B>, B::Scalar)
where
    BackendScalar<B>: ProtocolScalar,
{
    assert!(poly.coeffs().len() > 1, "cannot divide constant polynomial");
    let mut quotient = vec![<B::Scalar as FieldElement>::zero(); poly.coeffs().len() - 1];
    let mut carry = poly.coeffs().last().cloned().unwrap();
    for (idx, coeff) in poly.coeffs().iter().rev().skip(1).enumerate() {
        let pos = quotient.len() - 1 - idx;
        quotient[pos] = carry.clone();
        carry = coeff.clone() + root.clone() * carry;
    }
    (B::polynomial_from_coeffs(quotient), carry)
}

#[allow(dead_code)]
fn scale_poly<B: ProtocolBackend>(
    poly: &CommitmentPolynomial<B>,
    scalar: &B::Scalar,
) -> CommitmentPolynomial<B>
where
    BackendScalar<B>: ProtocolScalar,
{
    let coeffs = poly
        .coeffs()
        .iter()
        .map(|c| c.clone() * scalar.clone())
        .collect();
    B::polynomial_from_coeffs(coeffs)
}

#[allow(dead_code)]
fn sub_poly<B: ProtocolBackend>(
    a: &CommitmentPolynomial<B>,
    b: &CommitmentPolynomial<B>,
) -> CommitmentPolynomial<B>
where
    BackendScalar<B>: ProtocolScalar,
{
    let len = a.coeffs().len().max(b.coeffs().len());
    let mut coeffs = vec![<B::Scalar as FieldElement>::zero(); len];
    for (i, coeff) in a.coeffs().iter().enumerate() {
        coeffs[i] += coeff.clone();
    }
    for (i, coeff) in b.coeffs().iter().enumerate() {
        coeffs[i] -= coeff.clone();
    }
    B::polynomial_from_coeffs(coeffs)
}

#[allow(dead_code)]
fn mul_poly<B: ProtocolBackend>(
    a: &CommitmentPolynomial<B>,
    b: &CommitmentPolynomial<B>,
) -> CommitmentPolynomial<B>
where
    BackendScalar<B>: ProtocolScalar,
{
    if a.coeffs().is_empty() || b.coeffs().is_empty() {
        return B::polynomial_from_coeffs(vec![<B::Scalar as FieldElement>::zero()]);
    }
    let mut coeffs = vec![
        <B::Scalar as FieldElement>::zero();
        a.coeffs()
            .len()
            .saturating_add(b.coeffs().len())
            .saturating_sub(1)
    ];
    for (i, coeff_a) in a.coeffs().iter().enumerate() {
        for (j, coeff_b) in b.coeffs().iter().enumerate() {
            coeffs[i + j] += coeff_a.clone() * coeff_b.clone();
        }
    }
    B::polynomial_from_coeffs(coeffs)
}

#[allow(dead_code)]
fn divide_by_vanishing<B: ProtocolBackend>(
    poly: &CommitmentPolynomial<B>,
    domain_size: usize,
) -> (CommitmentPolynomial<B>, CommitmentPolynomial<B>)
where
    BackendScalar<B>: ProtocolScalar,
{
    if poly.coeffs().len() <= domain_size {
        return (
            B::polynomial_from_coeffs(vec![<B::Scalar as FieldElement>::zero()]),
            poly.clone(),
        );
    }
    let mut coeffs = poly.coeffs().to_vec();
    let mut quotient = vec![<B::Scalar as FieldElement>::zero(); coeffs.len() - domain_size];
    while coeffs.len() > domain_size {
        let d = coeffs.len() - 1;
        let lead = coeffs[d].clone();
        let q_idx = d - domain_size;
        quotient[q_idx] = lead.clone();
        coeffs.pop();
        coeffs[q_idx] += lead;
        while coeffs
            .last()
            .map(|c| c == &<B::Scalar as FieldElement>::zero())
            .unwrap_or(false)
            && !coeffs.is_empty()
        {
            coeffs.pop();
        }
    }
    if coeffs.is_empty() {
        coeffs.push(<B::Scalar as FieldElement>::zero());
    }
    (
        B::polynomial_from_coeffs(quotient),
        B::polynomial_from_coeffs(coeffs),
    )
}

fn precompute_lagrange_powers<B: ProtocolBackend>(
    lagranges: &[CommitmentPolynomial<B>],
    domain_size: usize,
    tau: &B::Scalar,
    _params: &CommitmentParams<B>,
) -> Result<LagrangePowers<B>, BackendError>
where
    BackendScalar<B>: ProtocolScalar,
{
    let n = lagranges.len();

    // Evaluate all Lagrange polynomials at tau
    let li_evals: Vec<B::Scalar> = lagranges
        .iter()
        .map(|li_poly| li_poly.evaluate(tau))
        .collect();

    // Compute tau^n - 1 (the vanishing polynomial evaluated at tau)
    let tau_n = tau.pow(&[domain_size as u64, 0, 0, 0]);
    let z_eval = tau_n - <B::Scalar as FieldElement>::one();
    let z_eval_inv = z_eval
        .invert()
        .ok_or(BackendError::Math("vanishing polynomial at tau is zero"))?;

    let tau_inv = tau
        .invert()
        .ok_or(BackendError::Math("tau must be non-zero"))?;

    // Compute li, li_minus0, and li_x in parallel
    let results: Vec<(B::G1, B::G1, B::G1)> = lagranges
        .par_iter()
        .enumerate()
        .map(|(i, li_poly)| {
            let li_eval = &li_evals[i];

            // li = g * L_i(tau)
            let lagrange_li = B::G1::generator().mul_scalar(li_eval);

            // li_minus0 = g * (L_i(tau) - L_i(0))
            let li_0 = li_poly
                .coeffs()
                .first()
                .cloned()
                .unwrap_or_else(<B::Scalar as FieldElement>::zero);
            let li_minus0_eval = li_eval.clone() - li_0;
            let lagrange_li_minus0 = B::G1::generator().mul_scalar(&li_minus0_eval);

            // li_x = g * (L_i(tau) - L_i(0)) / tau
            let li_x_eval = li_minus0_eval * tau_inv.clone();
            let lagrange_li_x = B::G1::generator().mul_scalar(&li_x_eval);

            Ok((lagrange_li, lagrange_li_minus0, lagrange_li_x))
        })
        .collect::<Result<Vec<_>, BackendError>>()?;

    let mut li = Vec::with_capacity(n);
    let mut li_minus0 = Vec::with_capacity(n);
    let mut li_x = Vec::with_capacity(n);
    for (a, b, c) in results {
        li.push(a);
        li_minus0.push(b);
        li_x.push(c);
    }

    // Compute li_lj_z using the evaluation-based approach
    let li_lj_z: Vec<Vec<B::G1>> = (0..n)
        .into_par_iter()
        .map(|i| {
            (0..n)
                .into_par_iter()
                .map(|j| {
                    let scalar = if i == j {
                        // (L_i(tau)^2 - L_i(tau)) / z(tau)
                        let li_eval: &B::Scalar = &li_evals[i];
                        (li_eval.clone() * li_eval.clone() - li_eval.clone()) * z_eval_inv.clone()
                    } else {
                        // (L_i(tau) * L_j(tau)) / z(tau)
                        (li_evals[i].clone() * li_evals[j].clone()) * z_eval_inv.clone()
                    };
                    Ok(B::G1::generator().mul_scalar(&scalar))
                })
                .collect::<Result<Vec<_>, BackendError>>()
        })
        .collect::<Result<Vec<_>, BackendError>>()?;

    Ok(LagrangePowers {
        li,
        li_minus0,
        li_x,
        li_lj_z,
    })
}

#[allow(dead_code)]
fn derive_public_key<B: ProtocolBackend>(
    participant_id: usize,
    sk: &SecretKey<B>,
    lagranges: &[CommitmentPolynomial<B>],
    domain_size: usize,
    params: &CommitmentParams<B>,
) -> Result<PublicKey<B>, BackendError>
where
    BackendScalar<B>: ProtocolScalar,
{
    let li = lagranges
        .get(participant_id)
        .ok_or(BackendError::Math("missing lagrange polynomial"))?;

    let li_poly = li.clone();
    let sk_li_poly = scale_poly::<B>(&li_poly, &sk.scalar);
    let lagrange_li = B::PolynomialCommitment::commit_g1(params, &sk_li_poly)?;

    let mut minus0_coeffs = sk_li_poly.coeffs().to_vec();
    if let Some(constant) = minus0_coeffs.get_mut(0) {
        *constant = <B::Scalar as FieldElement>::zero();
    }
    let lagrange_li_minus0 =
        B::PolynomialCommitment::commit_g1(params, &B::polynomial_from_coeffs(minus0_coeffs))?;

    let shift_coeffs = if li_poly.coeffs().len() > 1 {
        li_poly.coeffs()[1..].to_vec()
    } else {
        vec![<B::Scalar as FieldElement>::zero()]
    };
    let shift_poly = B::polynomial_from_coeffs(shift_coeffs);
    let sk_shift_poly = scale_poly::<B>(&shift_poly, &sk.scalar);
    let lagrange_li_x = B::PolynomialCommitment::commit_g1(params, &sk_shift_poly)?;

    let mut lagrange_li_lj_z = Vec::with_capacity(lagranges.len());
    for (idx, lj) in lagranges.iter().enumerate() {
        let numerator = if idx == participant_id {
            sub_poly::<B>(&mul_poly::<B>(&li_poly, &li_poly), &li_poly)
        } else {
            mul_poly::<B>(lj, &li_poly)
        };
        let (f, remainder) = divide_by_vanishing::<B>(&numerator, domain_size);
        if remainder
            .coeffs()
            .iter()
            .any(|c| *c != <B::Scalar as FieldElement>::zero())
        {
            return Err(BackendError::Math(
                "division by vanishing polynomial failed",
            ));
        }
        let scaled = scale_poly::<B>(&f, &sk.scalar);
        let commitment = B::PolynomialCommitment::commit_g1(params, &scaled)?;
        lagrange_li_lj_z.push(commitment);
    }

    Ok(PublicKey {
        participant_id,
        bls_key: B::G1::generator().mul_scalar(&sk.scalar),
        lagrange_li,
        lagrange_li_minus0,
        lagrange_li_x,
        lagrange_li_lj_z,
    })
}

fn derive_public_key_from_powers<B: ProtocolBackend>(
    participant_id: usize,
    sk: &SecretKey<B>,
    powers: &LagrangePowers<B>,
) -> Result<PublicKey<B>, BackendError>
where
    BackendScalar<B>: ProtocolScalar,
{
    let lagrange_li = powers
        .li
        .get(participant_id)
        .ok_or(BackendError::Math("missing lagrange power"))?
        .mul_scalar(&sk.scalar);

    let lagrange_li_minus0 = powers
        .li_minus0
        .get(participant_id)
        .ok_or(BackendError::Math("missing lagrange power minus0"))?
        .mul_scalar(&sk.scalar);

    let lagrange_li_x = powers
        .li_x
        .get(participant_id)
        .ok_or(BackendError::Math("missing lagrange power x"))?
        .mul_scalar(&sk.scalar);

    let lagrange_li_lj_z = powers
        .li_lj_z
        .get(participant_id)
        .ok_or(BackendError::Math("missing lagrange powers li_lj_z"))?
        .iter()
        .map(|val| val.mul_scalar(&sk.scalar))
        .collect();

    Ok(PublicKey {
        participant_id,
        bls_key: B::G1::generator().mul_scalar(&sk.scalar),
        lagrange_li,
        lagrange_li_minus0,
        lagrange_li_x,
        lagrange_li_lj_z,
    })
}

fn aggregate_public_key<B: ProtocolBackend>(
    public_keys: &[PublicKey<B>],
    params: &CommitmentParams<B>,
    parties: usize,
) -> Result<AggregateKey<B>, Error>
where
    BackendScalar<B>: ProtocolScalar,
{
    if public_keys.is_empty() {
        return Err(Error::InvalidConfig(
            "cannot aggregate empty public key set".into(),
        ));
    }
    if public_keys.len() != parties {
        return Err(Error::InvalidConfig("public key length mismatch".into()));
    }

    let mut ask = B::G1::identity();
    for pk in public_keys {
        ask = ask.add(&pk.lagrange_li);
    }

    let mut lagrange_row_sums = vec![B::G1::identity(); parties];
    for (idx, row) in lagrange_row_sums.iter_mut().enumerate() {
        for pk in public_keys {
            if let Some(val) = pk.lagrange_li_lj_z.get(idx) {
                *row = row.add(val);
            }
        }
    }

    let h_powers = B::h_powers(params);
    let g2_tau_n = h_powers
        .get(parties)
        .ok_or(Error::Backend(BackendError::Math("missing h^tau^n")))?;
    let z_g2 = B::G2::from_affine(g2_tau_n).sub(&B::G2::generator());

    Ok(AggregateKey {
        public_keys: public_keys.to_vec(),
        ask,
        z_g2,
        lagrange_row_sums,
        precomputed_pairing: B::pairing_generator(params),
        commitment_params: params.clone(),
    })
}

fn aggregate_decrypt<B: ProtocolBackend>(
    ciphertext: &Ciphertext<B>,
    partials: &[PartialDecryption<B>],
    selector: &[bool],
    agg_key: &AggregateKey<B>,
) -> Result<DecryptionResult<B>, Error>
where
    BackendScalar<B>: ProtocolScalar,
{
    let n = agg_key.public_keys.len();
    if selector.len() != n {
        return Err(Error::SelectorMismatch {
            expected: n,
            actual: selector.len(),
        });
    }

    let mut responses = vec![B::G2::identity(); n];
    let mut seen = vec![false; n];
    for partial in partials {
        if partial.participant_id >= n {
            return Err(Error::MalformedInput("partial id out of range".into()));
        }
        if seen[partial.participant_id] {
            return Err(Error::MalformedInput("duplicate partial id".into()));
        }
        responses[partial.participant_id] = partial.response.clone();
        seen[partial.participant_id] = true;
    }

    let provided = selector
        .iter()
        .enumerate()
        .filter(|(idx, selected)| **selected && seen[*idx])
        .count();
    let required = ciphertext.threshold + 1;
    if provided < required {
        return Err(Error::NotEnoughShares { required, provided });
    }

    let domain = B::domain_new(n).map_err(Error::Backend)?;
    let domain_elements = domain.elements();
    let omega_zero = domain_elements
        .first()
        .cloned()
        .ok_or(Error::Backend(BackendError::Math(
            "invalid evaluation domain",
        )))?;

    let mut points = vec![omega_zero.clone()];
    let mut parties = Vec::new();
    for (i, (&selected, omega)) in selector.iter().zip(domain_elements.iter()).enumerate() {
        if selected {
            if !seen[i] {
                return Err(Error::NotEnoughShares { required, provided });
            }
            parties.push(i);
        } else if i != 0 {
            points.push(omega.clone());
        }
    }
    let nonselected_count = selector.iter().filter(|&&selected| !selected).count();
    trace!(
        interpolation_points = points.len(),
        selected_count = parties.len(),
        nonselected_count,
        zero_selected = selector[0],
        "interpolation setup"
    );

    let b = B::interp_mostly_zero(<B::Scalar as FieldElement>::one(), &points)
        .map_err(Error::Backend)?;
    let b_evals = domain.fft(b.coeffs());

    let b_g2 = B::PolynomialCommitment::commit_g2(&agg_key.commitment_params, &b)
        .map_err(Error::Backend)?;

    let mut minus_one_coeffs = b.coeffs().to_vec();
    if let Some(constant) = minus_one_coeffs.get_mut(0) {
        *constant -= <B::Scalar as FieldElement>::one();
    }
    let b_minus_one = B::polynomial_from_coeffs(minus_one_coeffs);
    let (q0, remainder) = divide_by_linear::<B>(&b_minus_one, omega_zero);
    if remainder != <B::Scalar as FieldElement>::zero() {
        return Err(Error::Backend(BackendError::Math(
            "division by linear failed",
        )));
    }
    let q0_g1 = B::PolynomialCommitment::commit_g1(&agg_key.commitment_params, &q0)
        .map_err(Error::Backend)?;

    let mut bhat_coeffs = vec![<B::Scalar as FieldElement>::zero(); ciphertext.threshold + 1];
    bhat_coeffs.extend(b.coeffs().iter().cloned());
    let bhat = B::polynomial_from_coeffs(bhat_coeffs);
    let bhat_g1 = B::PolynomialCommitment::commit_g1(&agg_key.commitment_params, &bhat)
        .map_err(Error::Backend)?;

    let n_inv = <B::Scalar as From<u64>>::from(n as u64)
        .invert()
        .ok_or(Error::Backend(BackendError::Math(
            "domain size inversion failed",
        )))?;

    let scalars: Vec<B::Scalar> = parties.iter().map(|&i| b_evals[i].clone()).collect();
    trace!(
        selected_indices = ?parties,
        scalars = ?scalars,
        "selection scalars",
    );

    let apk = if scalars.is_empty() {
        B::G1::identity()
    } else {
        let bases: Vec<B::G1> = parties
            .iter()
            .map(|&i| agg_key.public_keys[i].bls_key.clone())
            .collect();
        B::Msm::msm_g1(&bases, &scalars)
            .map_err(Error::Backend)?
            .mul_scalar(&n_inv)
    };

    let sigma = if scalars.is_empty() {
        B::G2::identity()
    } else {
        let bases: Vec<B::G2> = parties.iter().map(|&i| responses[i].clone()).collect();
        B::Msm::msm_g2(&bases, &scalars)
            .map_err(Error::Backend)?
            .mul_scalar(&n_inv)
    };

    let qx = if scalars.is_empty() {
        B::G1::identity()
    } else {
        let bases: Vec<B::G1> = parties
            .iter()
            .map(|&i| agg_key.public_keys[i].lagrange_li_x.clone())
            .collect();
        B::Msm::msm_g1(&bases, &scalars).map_err(Error::Backend)?
    };

    let qz = if scalars.is_empty() {
        B::G1::identity()
    } else {
        let bases: Vec<B::G1> = parties
            .iter()
            .map(|&i| agg_key.lagrange_row_sums[i].clone())
            .collect();
        B::Msm::msm_g1(&bases, &scalars).map_err(Error::Backend)?
    };

    let qhatx = if scalars.is_empty() {
        B::G1::identity()
    } else {
        let bases: Vec<B::G1> = parties
            .iter()
            .map(|&i| agg_key.public_keys[i].lagrange_li_minus0.clone())
            .collect();
        B::Msm::msm_g1(&bases, &scalars).map_err(Error::Backend)?
    };

    let mut lhs = vec![
        apk.negate(),
        qz.negate(),
        qx.negate(),
        qhatx,
        bhat_g1.negate(),
        q0_g1.negate(),
    ];
    lhs.extend(ciphertext.proof_g1.iter().cloned());

    let mut rhs = Vec::with_capacity(ciphertext.proof_g2.len() + 2);
    rhs.extend(ciphertext.proof_g2.iter().cloned());
    rhs.push(b_g2);
    rhs.push(sigma);

    let shared_secret = B::multi_pairing(&lhs, &rhs).map_err(Error::Backend)?;
    let plaintext = if ciphertext.payload.is_empty() {
        None
    } else {
        Some(decrypt_payload::<B>(&shared_secret, &ciphertext.payload))
    };

    Ok(DecryptionResult {
        shared_secret,
        opening_proof: None,
        plaintext,
    })
}

#[cfg(feature = "ark_bls12381")]
pub mod ark_bls12_381;
#[cfg(feature = "ark_bn254")]
pub mod ark_bn254;
#[cfg(feature = "blst")]
pub mod blst_bls12_381;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{BackendConfig, BackendId, CurveId};
    use rand::{SeedableRng, rngs::StdRng};

    fn sample_params(backend: BackendConfig) -> ThresholdParameters {
        ThresholdParameters {
            parties: 8,
            threshold: 4,
            chunk_size: 32,
            backend,
            kzg_tau: None,
        }
    }

    fn run_roundtrip<B: ProtocolBackend>(backend: BackendConfig)
    where
        BackendScalar<B>: ProtocolScalar,
        <<B as PairingBackend>::Target as TargetGroup>::Repr: PartialEq,
    {
        let mut rng = StdRng::from_entropy();
        let scheme = SilentThreshold::<B>::default();
        let mut params = sample_params(backend);
        let (srs, tau_bytes) = scheme.srs_gen(&mut rng, &params).expect("srs gen");
        params.kzg_tau = Some(tau_bytes);
        let km = scheme.keygen(&mut rng, &params, &srs).expect("keygen");
        let ct = scheme
            .encrypt(&mut rng, &km.aggregate_key, &params, b"payload")
            .expect("encrypt");

        let mut selector = vec![false; params.parties];
        let mut partials = Vec::new();
        for (idx, selected) in selector.iter_mut().enumerate().take(params.threshold + 1) {
            *selected = true;
            partials.push(
                scheme
                    .partial_decrypt(&km.secret_keys[idx], &ct)
                    .expect("partial decrypt"),
            );
        }

        let result = scheme
            .aggregate_decrypt(&ct, &partials, &selector, &km.aggregate_key)
            .expect("aggregate decrypt");

        assert_eq!(
            result.shared_secret.to_repr(),
            ct.shared_secret.to_repr(),
            "shared secret mismatch"
        );
        assert_eq!(
            result.plaintext.as_deref(),
            Some(b"payload".as_slice()),
            "unexpected plaintext"
        );
    }

    fn run_not_enough_shares<B: ProtocolBackend>(backend: BackendConfig)
    where
        BackendScalar<B>: ProtocolScalar,
    {
        let mut rng = StdRng::from_entropy();
        let scheme = SilentThreshold::<B>::default();
        let mut params = sample_params(backend);
        let (srs, tau_bytes) = scheme.srs_gen(&mut rng, &params).expect("srs gen");
        params.kzg_tau = Some(tau_bytes);
        let km = scheme.keygen(&mut rng, &params, &srs).expect("keygen");
        let ct = scheme
            .encrypt(&mut rng, &km.aggregate_key, &params, b"payload")
            .expect("encrypt");

        let mut selector = vec![false; params.parties];
        let mut partials = Vec::new();
        for (idx, selected) in selector.iter_mut().enumerate().take(params.threshold) {
            *selected = true;
            partials.push(
                scheme
                    .partial_decrypt(&km.secret_keys[idx], &ct)
                    .expect("partial decrypt"),
            );
        }

        let result = scheme.aggregate_decrypt(&ct, &partials, &selector, &km.aggregate_key);
        assert!(
            matches!(
                result,
                Err(Error::NotEnoughShares { required, provided })
                    if required == params.threshold + 1 && provided == params.threshold
            ),
            "unexpected result: {:?}",
            result
        );
    }

    fn run_selector_mismatch<B: ProtocolBackend>(backend: BackendConfig)
    where
        BackendScalar<B>: ProtocolScalar,
    {
        let mut rng = StdRng::from_entropy();
        let scheme = SilentThreshold::<B>::default();
        let mut params = sample_params(backend);
        let (srs, tau_bytes) = scheme.srs_gen(&mut rng, &params).expect("srs gen");
        params.kzg_tau = Some(tau_bytes);
        let km = scheme.keygen(&mut rng, &params, &srs).expect("keygen");
        let ct = scheme
            .encrypt(&mut rng, &km.aggregate_key, &params, b"payload")
            .expect("encrypt");

        let mut selector = vec![false; params.parties];
        let mut partials = Vec::new();
        for (idx, selected) in selector.iter_mut().enumerate().take(params.threshold + 1) {
            *selected = true;
            partials.push(
                scheme
                    .partial_decrypt(&km.secret_keys[idx], &ct)
                    .expect("partial decrypt"),
            );
        }

        let mismatched_selector = selector[..selector.len() - 1].to_vec();
        let err = scheme.aggregate_decrypt(&ct, &partials, &mismatched_selector, &km.aggregate_key);
        assert!(matches!(err, Err(Error::SelectorMismatch { .. })));
    }

    fn run_duplicate_partial<B: ProtocolBackend>(backend: BackendConfig)
    where
        BackendScalar<B>: ProtocolScalar,
    {
        let mut rng = StdRng::from_entropy();
        let scheme = SilentThreshold::<B>::default();
        let mut params = sample_params(backend);
        let (srs, tau_bytes) = scheme.srs_gen(&mut rng, &params).expect("srs gen");
        params.kzg_tau = Some(tau_bytes);
        let km = scheme.keygen(&mut rng, &params, &srs).expect("keygen");
        let ct = scheme
            .encrypt(&mut rng, &km.aggregate_key, &params, b"payload")
            .expect("encrypt");

        let mut selector = vec![false; params.parties];
        let mut partials = Vec::new();
        for (idx, selected) in selector.iter_mut().enumerate().take(params.threshold + 1) {
            *selected = true;
            let share = scheme
                .partial_decrypt(&km.secret_keys[idx], &ct)
                .expect("partial decrypt");
            if idx == 0 {
                partials.push(share.clone());
            }
            partials.push(share);
        }

        let err = scheme.aggregate_decrypt(&ct, &partials, &selector, &km.aggregate_key);
        assert!(matches!(err, Err(Error::MalformedInput(_))));
    }

    #[cfg(feature = "blst")]
    #[test]
    fn blst_encrypt_decrypt_roundtrip() {
        use crate::backend::BlstBackend;
        run_roundtrip::<BlstBackend>(BackendConfig::new(BackendId::Blst, CurveId::Bls12_381));
    }

    #[cfg(feature = "blst")]
    #[test]
    fn blst_decrypt_not_enough_shares() {
        use crate::backend::BlstBackend;
        run_not_enough_shares::<BlstBackend>(BackendConfig::new(
            BackendId::Blst,
            CurveId::Bls12_381,
        ));
    }

    #[cfg(feature = "blst")]
    #[test]
    fn blst_decrypt_selector_mismatch() {
        use crate::backend::BlstBackend;
        run_selector_mismatch::<BlstBackend>(BackendConfig::new(
            BackendId::Blst,
            CurveId::Bls12_381,
        ));
    }

    #[cfg(feature = "blst")]
    #[test]
    fn blst_decrypt_duplicate_partial() {
        use crate::backend::BlstBackend;
        run_duplicate_partial::<BlstBackend>(BackendConfig::new(
            BackendId::Blst,
            CurveId::Bls12_381,
        ));
    }

    #[cfg(feature = "ark_bls12381")]
    #[test]
    fn ark_bls_encrypt_decrypt_roundtrip() {
        use crate::backend::ArkworksBls12;
        run_roundtrip::<ArkworksBls12>(BackendConfig::new(BackendId::Arkworks, CurveId::Bls12_381));
    }

    #[cfg(feature = "ark_bls12381")]
    #[test]
    fn ark_bls_not_enough_shares() {
        use crate::backend::ArkworksBls12;
        run_not_enough_shares::<ArkworksBls12>(BackendConfig::new(
            BackendId::Arkworks,
            CurveId::Bls12_381,
        ));
    }

    #[cfg(feature = "ark_bls12381")]
    #[test]
    fn ark_bls_selector_mismatch() {
        use crate::backend::ArkworksBls12;
        run_selector_mismatch::<ArkworksBls12>(BackendConfig::new(
            BackendId::Arkworks,
            CurveId::Bls12_381,
        ));
    }

    #[cfg(feature = "ark_bls12381")]
    #[test]
    fn ark_bls_duplicate_partial() {
        use crate::backend::ArkworksBls12;
        run_duplicate_partial::<ArkworksBls12>(BackendConfig::new(
            BackendId::Arkworks,
            CurveId::Bls12_381,
        ));
    }

    #[cfg(feature = "ark_bn254")]
    #[test]
    fn ark_bn_roundtrip() {
        use crate::backend::ArkworksBn254;
        run_roundtrip::<ArkworksBn254>(BackendConfig::new(BackendId::Arkworks, CurveId::Bn254));
    }

    #[cfg(feature = "ark_bn254")]
    #[test]
    fn ark_bn_not_enough_shares() {
        use crate::backend::ArkworksBn254;
        run_not_enough_shares::<ArkworksBn254>(BackendConfig::new(
            BackendId::Arkworks,
            CurveId::Bn254,
        ));
    }

    #[cfg(feature = "ark_bn254")]
    #[test]
    fn ark_bn_selector_mismatch() {
        use crate::backend::ArkworksBn254;
        run_selector_mismatch::<ArkworksBn254>(BackendConfig::new(
            BackendId::Arkworks,
            CurveId::Bn254,
        ));
    }

    #[cfg(feature = "ark_bn254")]
    #[test]
    fn ark_bn_duplicate_partial() {
        use crate::backend::ArkworksBn254;
        run_duplicate_partial::<ArkworksBn254>(BackendConfig::new(
            BackendId::Arkworks,
            CurveId::Bn254,
        ));
    }
}
