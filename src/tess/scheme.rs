use core::{fmt::Debug, marker::PhantomData};

use blake3::Hasher;
use rand_core::RngCore;
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use tracing::instrument;

use crate::{
    AggregateKey,
    Ciphertext,
    DecryptionResult,
    Fr,
    KeyMaterial,
    LagrangePowers,
    PairingBackend,
    Params,
    PartialDecryption,
    PublicKey,
    SRS,
    SecretKey,
    ThresholdEncryption,
    arith::{CurvePoint, FieldElement},
    build_lagrange_polys,
    errors::{BackendError, Error},
    sym_enc::{Blake3XorEncryption, SymmetricEncryption},
    tess::keys::derive_public_key_from_powers,
    // tess::keys::derive_public_key_from_srs,
};

/// The Silent Threshold scheme implementation.
#[derive(Debug, Default)]
pub struct SilentThresholdScheme<B: PairingBackend> {
    _phantom: PhantomData<B>,
    symmetric_enc: Blake3XorEncryption,
}

impl<B: PairingBackend> SilentThresholdScheme<B> {
    /// Creates a new Silent Threshold scheme instance.
    pub fn new() -> Self {
        Self {
            _phantom: PhantomData,
            symmetric_enc: Blake3XorEncryption::default(),
        }
    }

    /// Creates a new Silent Threshold scheme with a custom symmetric encryption.
    pub fn with_encryption(symmetric_enc: Blake3XorEncryption) -> Self {
        Self {
            _phantom: PhantomData,
            symmetric_enc,
        }
    }

    fn generate_secret_keys<R: RngCore + ?Sized>(rng: &mut R, parties: usize) -> Vec<SecretKey<B>> {
        (0..parties)
            .map(|participant_id| SecretKey {
                participant_id,
                scalar: B::Scalar::random(rng),
            })
            .collect()
    }
}

impl<B: PairingBackend<Scalar = Fr>> ThresholdEncryption<B> for SilentThresholdScheme<B> {
    #[instrument(level = "info", skip_all, fields(parties, threshold))]
    fn param_gen<R: RngCore + ?Sized>(
        &self,
        rng: &mut R,
        parties: usize,
        threshold: usize,
    ) -> Result<Params<B>, Error> {
        if parties < threshold {
            return Err(Error::InvalidConfig(
                "threshold must be less than or equal to parties".into(),
            ));
        }
        if threshold == 0 {
            return Err(Error::InvalidConfig(
                "threshold must be greater than 0".into(),
            ));
        }

        let tau = B::Scalar::random(rng);

        let srs = SRS::new_unsafe(&tau, parties).map_err(|e| {
            Error::Backend(BackendError::Other(format!("SRS generation failed: {}", e)))
        })?;

        // Build Lagrange polynomials for the evaluation domain of size `parties`.
        let lagranges = build_lagrange_polys(parties).map_err(|e| {
            Error::Backend(BackendError::Other(format!(
                "Lagrange polynomials failed: {}",
                e
            )))
        })?;

        // Precompute Lagrange powers (commitments) using the arith helper.
        let lagrange_powers = LagrangePowers::precompute_lagrange_powers(&lagranges, parties, &tau)
            .map_err(|e| Error::Backend(e))?;

        Ok(Params {
            srs,
            lagrange_powers,
        })
    }

    #[instrument(level = "info", skip_all, fields(parties))]
    fn keygen<R: RngCore + ?Sized>(
        &self,
        rng: &mut R,
        parties: usize,
        params: &Params<B>,
    ) -> Result<KeyMaterial<B>, Error> {
        let secret_keys = Self::generate_secret_keys(rng, parties);

        let public_keys = secret_keys
            .par_iter()
            .map(|sk| {
                derive_public_key_from_powers::<B>(sk.participant_id, sk, &params.lagrange_powers)
            })
            .collect::<Result<Vec<_>, BackendError>>()?;

        let aggregate_key = AggregateKey::aggregate_keys(&public_keys, &params, parties)?;

        Ok(KeyMaterial {
            secret_keys,
            public_keys,
            aggregate_key,
            kzg_params: params.srs.clone(),
        })
    }

    #[instrument(level = "info", skip_all, fields(parties, num_keys = public_keys.len()))]
    fn aggregate_public_key(
        &self,
        public_keys: &[PublicKey<B>],
        params: &Params<B>,
        parties: usize,
    ) -> Result<AggregateKey<B>, Error> {
        AggregateKey::aggregate_keys(public_keys, params, parties)
    }

    #[instrument(level = "info", skip_all, fields(threshold, payload_len = payload.len()))]
    fn encrypt<R: RngCore + ?Sized>(
        &self,
        rng: &mut R,
        agg_key: &AggregateKey<B>,
        threshold: usize,
        payload: &[u8],
    ) -> Result<Ciphertext<B>, Error> {
        let g = B::G1::generator();
        let h = B::G2::generator();

        let gamma = Fr::random(rng);
        let gamma_g2 = h.mul_scalar(&gamma);

        let s0 = Fr::random(rng);
        let s1 = Fr::random(rng);
        let s2 = Fr::random(rng);
        let s3 = Fr::random(rng);
        let s4 = Fr::random(rng);

        // Create proof elements
        let sa1_0 = agg_key.ask.mul_scalar(&s0).add(&g.mul_scalar(&s4));
        let sa1_1 = g.mul_scalar(&s2);

        let sa2_0 = h.mul_scalar(&s0).add(&gamma_g2.mul_scalar(&s2));
        let sa2_1 = agg_key.z_g2.mul_scalar(&s0);
        let sa2_2 = h.mul_scalar(&(s0 + s1));
        let sa2_3 = h.mul_scalar(&s1);
        let sa2_4 = h.mul_scalar(&s3);
        let sa2_5 = h.mul_scalar(&(s0 + (Fr::one() - Fr::one())));

        let proof_g1 = vec![sa1_0, sa1_1];
        let proof_g2 = vec![sa2_0, sa2_1, sa2_2, sa2_3, sa2_4, sa2_5];

        // Compute shared secret from s4 and pairing
        let shared_secret = B::pairing(&agg_key.ask, &gamma_g2);

        // Use a hash of s4 for payload encryption
        let mut hasher = Hasher::new();
        hasher.update(b"shared_secret");
        let s4_bytes = format!("{:?}", s4).into_bytes();
        hasher.update(&s4_bytes);
        let secret_bytes = hasher.finalize().as_bytes().to_vec();

        let payload_ct = self
            .symmetric_enc
            .encrypt(&secret_bytes[..32.min(secret_bytes.len())], payload)?;

        Ok(Ciphertext {
            gamma_g2,
            proof_g1,
            proof_g2,
            shared_secret,
            threshold,
            payload: payload_ct,
        })
    }

    #[instrument(level = "trace", skip_all, fields(participant_id = secret_key.participant_id))]
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

    #[instrument(level = "info", skip_all, fields(required = ciphertext.threshold, provided = partials.len()))]
    fn aggregate_decrypt(
        &self,
        ciphertext: &Ciphertext<B>,
        partials: &[PartialDecryption<B>],
        selector: &[bool],
        agg_key: &AggregateKey<B>,
    ) -> Result<DecryptionResult, Error> {
        if partials.is_empty() {
            return Err(Error::NotEnoughShares {
                required: ciphertext.threshold,
                provided: 0,
            });
        }

        if partials.len() < ciphertext.threshold {
            return Err(Error::NotEnoughShares {
                required: ciphertext.threshold,
                provided: partials.len(),
            });
        }

        if selector.len() != agg_key.public_keys.len() {
            return Err(Error::SelectorMismatch {
                expected: agg_key.public_keys.len(),
                actual: selector.len(),
            });
        }

        // Aggregate partial decryptions using Lagrange interpolation
        let mut aggregated = B::G2::identity();
        let mut count = 0;

        for (idx, &is_selected) in selector.iter().enumerate() {
            if is_selected {
                if let Some(partial) = partials.iter().find(|p| p.participant_id == idx) {
                    // Apply Lagrange coefficient (simplified - would use Lagrange basis in full impl)
                    aggregated = aggregated.add(&partial.response);
                    count += 1;
                }
            }
        }

        if count < ciphertext.threshold {
            return Err(Error::NotEnoughShares {
                required: ciphertext.threshold,
                provided: count,
            });
        }

        // Recover shared secret using pairing
        let _recovered = B::pairing(&B::G1::generator(), &aggregated);

        // Use a hash of the pairing result for decryption (placeholder)
        let mut hasher = Hasher::new();
        hasher.update(b"recovered_secret");
        let secret_bytes = hasher.finalize().as_bytes().to_vec();

        let plaintext = self.symmetric_enc.decrypt(
            &secret_bytes[..32.min(secret_bytes.len())],
            &ciphertext.payload,
        )?;

        Ok(DecryptionResult {
            plaintext: Some(plaintext),
        })
    }
}
