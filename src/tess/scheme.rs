use core::{fmt::Debug, marker::PhantomData};

use blake3::Hasher;
use rand_core::RngCore;
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use tracing::instrument;

use crate::{
    AggregateKey,
    Ciphertext,
    DecryptionResult,
    DensePolynomial,
    Fr,
    KeyMaterial,
    LagrangePowers,
    PairingBackend,
    Params,
    PartialDecryption,
    PublicKey,
    SRS,
    SecretKey,
    TargetGroup,
    ThresholdEncryption,
    arith::{CurvePoint, FieldElement},
    build_lagrange_polys,
    errors::{BackendError, Error},
    sym_enc::{Blake3XorEncryption, SymmetricEncryption},
    tess::keys::derive_public_key, // tess::keys::derive_public_key_from_srs,
};

/// The Silent Threshold scheme implementation.
#[derive(Debug)]
pub struct SilentThresholdScheme<B: PairingBackend> {
    _phantom: PhantomData<B>,
    symmetric_enc: Blake3XorEncryption,
}

pub type SilentThreshold<B> = SilentThresholdScheme<B>;

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

impl<B: PairingBackend> Default for SilentThresholdScheme<B> {
    fn default() -> Self {
        Self::new()
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
            .map_err(Error::Backend)?;

        Ok(Params {
            srs,
            lagrange_powers,
            lagrange_polys: lagranges,
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
            .map(|sk| derive_public_key::<B>(sk.participant_id, sk, params))
            .collect::<Result<Vec<_>, BackendError>>()?;

        let aggregate_key = AggregateKey::aggregate_keys(&public_keys, params, parties)?;
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
        params: &Params<B>,
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

        // sa1[0] = s0*ask + s3*g^{tau^{t+1}} + s4*g
        // sa1[0] = (apk.ask * s[0]) + (params.powers_of_g[t + 1] * s[3]) + (params.powers_of_g[0] * s[4]);
        let sa1_0 = agg_key
            .ask
            .mul_scalar(&s0)
            .add(&params.srs.powers_of_g[threshold + 1].mul_scalar(&s3))
            .add(&g.mul_scalar(&s4));

        // sa1[1] = s2*g
        let sa1_1 = g.mul_scalar(&s2);

        // sa2[0] = s0*h + s2*gamma_g2
        let sa2_0 = h.mul_scalar(&s0).add(&gamma_g2.mul_scalar(&s2));

        // sa2[1] = s0*z_g2
        let sa2_1 = agg_key.z_g2.mul_scalar(&s0);

        // sa2[2] = s0*h^tau + s1*h^tau
        let sa2_2 = params.srs.powers_of_h[1].mul_scalar(&(s0 + s1));

        // sa2[3] = s1*h
        let sa2_3 = h.mul_scalar(&s1);

        // sa2[4] = s3*h
        let sa2_4 = h.mul_scalar(&s3);

        // sa2[5] = s4*h^{tau - omega^0}
        let sa2_5 = params.srs.powers_of_h[1]
            .sub(&params.srs.powers_of_h[0])
            .mul_scalar(&s4);

        let proof_g1 = vec![sa1_0, sa1_1];
        let proof_g2 = vec![sa2_0, sa2_1, sa2_2, sa2_3, sa2_4, sa2_5];

        // Compute shared secret from s4 and pairing
        // enc_key = e_gh^s4
        let shared_secret = agg_key.precomputed_pairing.mul_scalar(&s4);

        // Use a hash of s4 for payload encryption
        let mut hasher = Hasher::new();
        hasher.update(b"shared_secret");
        let s4_bytes = format!("{:?}", s4).into_bytes(); // todo: serialization
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

        let parties = agg_key.public_keys.len();

        let mut partial_map: Vec<Option<&PartialDecryption<B>>> = vec![None; parties];
        for partial in partials {
            if partial.participant_id < parties {
                partial_map[partial.participant_id] = Some(partial);
            }
        }

        let domain_elements = build_domain_elements(parties)?;
        let mut points = Vec::with_capacity(parties);
        points.push(domain_elements[0]);

        let mut selected_indices = Vec::new();
        for (idx, &is_selected) in selector.iter().enumerate().take(parties) {
            if is_selected && partial_map[idx].is_some() {
                selected_indices.push(idx);
            } else if !is_selected {
                points.push(domain_elements[idx]);
            }
        }

        if selected_indices.len() < ciphertext.threshold {
            return Err(Error::NotEnoughShares {
                required: ciphertext.threshold,
                provided: selected_indices.len(),
            });
        }

        let b_polynomial = interp_mostly_zero(Fr::one(), &points)?;
        let b_evals: Vec<Fr> = domain_elements
            .iter()
            .map(|point| b_polynomial.evaluate(point))
            .collect();

        let scalars: Vec<Fr> = selected_indices.iter().map(|&idx| b_evals[idx]).collect();

        let mut aggregated_response = B::G2::identity();
        for (&idx, scalar) in selected_indices.iter().zip(scalars.iter()) {
            if let Some(partial) = partial_map[idx] {
                aggregated_response = aggregated_response.add(&partial.response.mul_scalar(scalar));
            }
        }

        let party_inv = Fr::from_u64(parties as u64).invert().ok_or_else(|| {
            Error::Backend(BackendError::Math("failed to invert party count".into()))
        })?;
        let aggregated_response = aggregated_response.mul_scalar(&party_inv);

        // let matches_gamma = aggregated_response == ciphertext.gamma_g2;
        // dbg!(matches_gamma);

        let recovered = B::pairing(&agg_key.ask, &aggregated_response);
        assert_eq!(recovered, ciphertext.shared_secret);

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

fn build_domain_elements(parties: usize) -> Result<Vec<Fr>, Error> {
    if parties == 0 {
        return Err(Error::InvalidConfig("require at least one party".into()));
    }

    let omega = Fr::two_adicity_generator(parties);
    let mut elements = Vec::with_capacity(parties);
    let mut current = Fr::one();
    for _ in 0..parties {
        elements.push(current);
        current = current * omega;
    }

    Ok(elements)
}

fn interp_mostly_zero(eval: Fr, points: &[Fr]) -> Result<DensePolynomial, Error> {
    if points.is_empty() {
        return Ok(DensePolynomial::from_coefficients_vec(vec![Fr::one()]));
    }

    let mut coeffs = vec![Fr::one()];
    for point in points.iter().skip(1) {
        let neg_point = -*point;
        coeffs.push(Fr::zero());
        for i in (0..coeffs.len() - 1).rev() {
            let (left, right) = coeffs.split_at_mut(i + 1);
            let coef = &mut left[i];
            let next = &mut right[0];
            *next += *coef;
            *coef *= neg_point;
        }
    }

    let mut scale = *coeffs.last().unwrap();
    let anchor = points[0];
    for coef in coeffs.iter().rev().skip(1) {
        scale = scale * anchor + *coef;
    }

    let scale_inv = scale.invert().ok_or_else(|| {
        Error::Backend(BackendError::Math(
            "failed to invert interpolation anchor".into(),
        ))
    })?;
    let multiplier = eval * scale_inv;

    for coeff in coeffs.iter_mut() {
        *coeff = *coeff * multiplier;
    }

    Ok(DensePolynomial::from_coefficients_vec(coeffs))
}
