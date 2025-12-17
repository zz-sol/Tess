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

use crate::arith::CurvePoint;
use crate::{
    Fr, KZG, LagrangePowers, PairingBackend, SRS,
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
            bls_key: self.bls_key,
            lagrange_li: self.lagrange_li,
            lagrange_li_minus0: self.lagrange_li_minus0,
            lagrange_li_x: self.lagrange_li_x,
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
pub struct AggregateKey<B: PairingBackend<Scalar = Fr>> {
    pub public_keys: Vec<PublicKey<B>>,
    pub ask: B::G1,
    pub z_g2: B::G2,
    pub lagrange_row_sums: Vec<B::G1>,
    pub precomputed_pairing: B::Target,
    pub srs: SRS<B>,
}

impl<B: PairingBackend<Scalar = Fr>> AggregateKey<B> {
    #[instrument(level = "info", skip_all, fields(parties, num_keys = public_keys.len()))]
    pub(crate) fn aggregate(
        public_keys: &[PublicKey<B>],
        params: &SRS<B>,
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

        let h_powers = &params.powers_of_h;
        let g2_tau_n = h_powers
            .get(parties)
            .ok_or(Error::Backend(BackendError::Math("missing h^tau^n")))?;
        let z_g2 = g2_tau_n.sub(&B::G2::generator());

        Ok(AggregateKey {
            public_keys: public_keys.to_vec(),
            ask,
            z_g2,
            lagrange_row_sums,
            precomputed_pairing: params.e_gh.clone(),
            srs: params.clone(),
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
pub struct KeyMaterial<B: PairingBackend<Scalar = Fr>> {
    pub secret_keys: Vec<SecretKey<B>>,
    pub public_keys: Vec<PublicKey<B>>,
    pub aggregate_key: AggregateKey<B>,
    pub kzg_params: SRS<B>,
}

#[instrument(level = "trace", skip_all, fields(participant_id))]
fn derive_public_key_from_powers<B: PairingBackend<Scalar = Fr>>(
    participant_id: usize,
    sk: &SecretKey<B>,
    powers: &LagrangePowers<B>,
) -> Result<PublicKey<B>, BackendError> {
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
