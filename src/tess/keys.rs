//! Key structures for the threshold encryption scheme.
//!
//! This module defines the key types used in the TESS protocol:
//!
//! - [`SecretKey`]: A participant's secret share
//! - [`PublicKey`]: A participant's public key with Lagrange commitment hints
//! - [`AggregateKey`]: The combined public key used for encryption
//! - [`UnsafeKeyMaterial`]: Complete bundle of keys from key generation
//!
//! # Key Generation Flow
//!
//! 1. Each participant receives a random secret scalar
//! 2. Public keys are derived from secret keys using the SRS
//! 3. Lagrange polynomial commitments are computed for efficient verification
//! 4. Public keys are aggregated to create the encryption key
//!
//! # Silent Setup
//!
//! The key generation is "silent" because participants do not need to interact
//! with each other. All keys can be generated independently using the shared
//! SRS from the trusted setup.
//!
//! # Lagrange Commitment Hints
//!
//! Public keys include precomputed commitments to Lagrange polynomials:
//! - `lagrange_li`: Commitment to L_i(x)
//! - `lagrange_li_minus0`: Commitment to L_i(x) - L_i(0)
//! - `lagrange_li_x`: Commitment to x * L_i(x)
//! - `lagrange_li_lj_z`: Commitments to L_i(x) * L_j(z) for all j
//!
//! These precomputed values eliminate the need for polynomial interpolation
//! during decryption, significantly improving performance.

use alloc::vec::Vec;
use core::fmt::Debug;

#[cfg(feature = "parallel")]
use rayon::prelude::*;
use tracing::instrument;
use zeroize::Zeroize;

use crate::{
    Fr, PairingBackend, Params, SRS,
    arith::{CurvePoint, FieldElement},
    errors::{BackendError, Error},
};

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
    /// Participant identifier (0-indexed).
    pub participant_id: usize,
    /// Secret scalar share for this participant.
    pub scalar: B::Scalar,
}

impl<B: PairingBackend> Zeroize for SecretKey<B> {
    fn zeroize(&mut self) {
        self.scalar = B::Scalar::zero();
    }
}

impl<B: PairingBackend> Drop for SecretKey<B> {
    fn drop(&mut self) {
        self.zeroize();
    }
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
    /// Participant identifier (0-indexed).
    pub participant_id: usize,
    /// Standard BLS public key for the participant.
    pub bls_key: B::G1,
    /// Commitment to the participant's L_i(x).
    pub lagrange_li: B::G1,
    /// Commitment to L_i(x) - L_i(0).
    pub lagrange_li_minus0: B::G1,
    /// Commitment to x * L_i(x).
    pub lagrange_li_x: B::G1,
    /// Commitments to L_i(x) * L_j(z) for all j.
    pub lagrange_li_lj_z: Vec<B::G1>,
}

impl<B: PairingBackend> Clone for PublicKey<B> {
    fn clone(&self) -> Self {
        Self {
            participant_id: self.participant_id,
            bls_key: self.bls_key,
            lagrange_li: self.lagrange_li,
            lagrange_li_minus0: self.lagrange_li_minus0,
            lagrange_li_x: self.lagrange_li_x,
            lagrange_li_lj_z: self.lagrange_li_lj_z.clone(),
        }
    }
}

impl<B: PairingBackend<Scalar = Fr>> SecretKey<B> {
    /// Derives a public key from a secret key using precomputed Lagrange commitments.
    ///
    /// This function computes the participant's public key by multiplying the precomputed
    /// Lagrange polynomial commitments from the SRS with the participant's secret scalar.
    /// This enables efficient key generation without requiring polynomial operations.
    ///
    /// # Silent Key Generation
    ///
    /// The key derivation is "silent" because it only requires:
    /// - The participant's secret scalar
    /// - The public SRS parameters
    ///
    /// No interaction with other participants is needed, allowing fully independent
    /// key generation.
    ///
    /// # Computed Values
    ///
    /// The public key contains:
    /// - **bls_key**: g^s where s is the secret scalar (standard BLS public key)
    /// - **lagrange_li**: Commitment to s·L_i(x)
    /// - **lagrange_li_minus0**: Commitment to s·(L_i(x) - L_i(0))
    /// - **lagrange_li_x**: Commitment to s·x·L_i(x)
    /// - **lagrange_li_lj_z**: Commitments to s·L_i(x)·L_j(z) for all j
    ///
    /// These precomputed values enable O(1) verification during decryption instead
    /// of O(n) polynomial interpolation.
    ///
    /// # Arguments
    ///
    /// * `participant_id` - The participant's unique identifier (0-indexed)
    /// * `sk` - The participant's secret key
    /// * `params` - Public parameters containing precomputed Lagrange commitments
    ///
    /// # Returns
    ///
    /// The derived public key, or an error if the participant ID is out of bounds
    ///
    /// # Errors
    ///
    /// Returns `BackendError::Math` if `participant_id >= params.lagrange_powers.li.len()`
    #[instrument(level = "trace", skip_all, fields(participant_id))]
    pub(crate) fn derive_public_key(
        &self,
        params: &Params<B>,
    ) -> Result<PublicKey<B>, BackendError> {
        let powers = &params.lagrange_powers;
        if self.participant_id >= powers.li.len() {
            return Err(BackendError::Math("participant id out of bounds"));
        }

        let bls_key = B::G1::generator().mul_scalar(&self.scalar);
        let lagrange_li = powers.li[self.participant_id].mul_scalar(&self.scalar);
        let lagrange_li_minus0 = powers.li_minus0[self.participant_id].mul_scalar(&self.scalar);
        let lagrange_li_x = powers.li_x[self.participant_id].mul_scalar(&self.scalar);
        let lagrange_li_lj_z = powers.li_lj_z[self.participant_id]
            .iter()
            .map(|elem| elem.mul_scalar(&self.scalar))
            .collect::<Vec<_>>();

        Ok(PublicKey {
            participant_id: self.participant_id,
            bls_key,
            lagrange_li,
            lagrange_li_minus0,
            lagrange_li_x,
            lagrange_li_lj_z,
        })
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
#[derive(Clone, Debug)]
pub struct AggregateKey<B: PairingBackend<Scalar = Fr>> {
    /// Public keys for all participants.
    pub public_keys: Vec<PublicKey<B>>,
    /// Aggregated commitment to the secret keys in G1.
    pub ask: B::G1,
    /// Commitment to the vanishing polynomial in G2.
    pub z_g2: B::G2,
    /// Precomputed Lagrange row sums for verification.
    pub lagrange_row_sums: Vec<B::G1>,
    /// Precomputed pairing used for verification.
    pub precomputed_pairing: B::Target,
    /// KZG parameters used to derive commitments.
    pub kzg_params: SRS<B>,
}

impl<B: PairingBackend<Scalar = Fr>> AggregateKey<B> {
    /// Aggregates multiple public keys into a single aggregate key.
    ///
    /// This function combines the public keys of all participants to create
    /// an aggregate key used for encryption and verification.
    #[instrument(level = "info", skip_all, fields(parties, num_keys = public_keys.len()))]
    pub fn aggregate_keys(
        public_keys: &[PublicKey<B>],
        params: &Params<B>,
        parties: usize,
    ) -> Result<AggregateKey<B>, Error> {
        if public_keys.is_empty() {
            return Err(Error::InvalidConfig(
                "cannot aggregate empty public key set".into(),
            ));
        }
        if public_keys.len() != parties {
            return Err(Error::InvalidConfig("public key length mismatch".into()));
        }

        let ask = {
            #[cfg(feature = "parallel")]
            {
                public_keys
                    .par_iter()
                    .map(|pk| pk.lagrange_li)
                    .reduce(B::G1::identity, |acc, val| acc.add(&val))
            }
            #[cfg(not(feature = "parallel"))]
            {
                public_keys
                    .iter()
                    .fold(B::G1::identity(), |acc, pk| acc.add(&pk.lagrange_li))
            }
        };

        let lagrange_row_sums: Vec<B::G1> = {
            #[cfg(feature = "parallel")]
            {
                (0..parties)
                    .into_par_iter()
                    .map(|idx| {
                        let mut row = B::G1::identity();
                        for pk in public_keys {
                            if let Some(val) = pk.lagrange_li_lj_z.get(idx) {
                                row = row.add(val);
                            }
                        }
                        row
                    })
                    .collect()
            }
            #[cfg(not(feature = "parallel"))]
            {
                (0..parties)
                    .map(|idx| {
                        let mut row = B::G1::identity();
                        for pk in public_keys {
                            if let Some(val) = pk.lagrange_li_lj_z.get(idx) {
                                row = row.add(val);
                            }
                        }
                        row
                    })
                    .collect()
            }
        };

        let g2_gen = params.srs.powers_of_h[0];
        // h * tau^n is available at index `parties` in the SRS
        let h_tau_n = params.srs.powers_of_h[parties];
        let z_g2 = h_tau_n.sub(&g2_gen);

        Ok(AggregateKey {
            public_keys: public_keys.to_vec(),
            ask,
            z_g2,
            lagrange_row_sums,
            precomputed_pairing: params.srs.e_gh.clone(),
            kzg_params: params.srs.clone(),
        })
    }
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
pub struct UnsafeKeyMaterial<B: PairingBackend<Scalar = Fr>> {
    /// Secret keys for all participants.
    pub secret_keys: Vec<SecretKey<B>>,
    /// Public keys for all participants.
    pub public_keys: Vec<PublicKey<B>>,
    /// Aggregated public key for encryption.
    pub aggregate_key: AggregateKey<B>,
    /// KZG commitment parameters.
    pub kzg_params: SRS<B>,
}
