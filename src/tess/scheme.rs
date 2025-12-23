//! Silent threshold encryption scheme implementation.
//!
//! This module provides the core implementation of the threshold encryption protocol
//! with silent setup. The scheme allows a message to be encrypted such that it can
//! only be decrypted when at least `t` out of `n` participants cooperate.
//!
//! # Overview
//!
//! The [`SilentThresholdScheme`] implements the [`ThresholdEncryption`] trait and
//! provides the following operations:
//!
//! 1. **Parameter Generation** ([`param_gen`](ThresholdEncryption::param_gen)):
//!    Generates the Structured Reference String (SRS) and precomputed Lagrange
//!    polynomial commitments during a one-time trusted setup.
//!
//! 2. **Key Generation** ([`keygen`](ThresholdEncryption::keygen)):
//!    Generates secret and public keys for all participants. Each participant
//!    receives a secret share and corresponding public key with Lagrange hints.
//!
//! 3. **Encryption** ([`encrypt`](ThresholdEncryption::encrypt)):
//!    Encrypts a message using the aggregate public key, producing a ciphertext
//!    with KZG proofs for verification.
//!
//! 4. **Partial Decryption** ([`partial_decrypt`](ThresholdEncryption::partial_decrypt)):
//!    Each participant creates a decryption share using their secret key.
//!
//! 5. **Aggregate Decryption** ([`aggregate_decrypt`](ThresholdEncryption::aggregate_decrypt)):
//!    Combines at least `t` partial decryptions to recover the plaintext, with
//!    verification of the ciphertext's validity using KZG proofs.
//!
//! # Mathematical Background
//!
//! The scheme uses:
//! - **KZG Commitments**: For polynomial commitments and proofs
//! - **Lagrange Interpolation**: For reconstructing secrets from shares
//! - **Pairing-based Cryptography**: For verification using bilinear maps
//! - **BLAKE3**: For symmetric payload encryption via KDF
//!
//! # Example
//!
//! ```rust
//! use rand::thread_rng;
//! use tess::{PairingEngine, SilentThresholdScheme, ThresholdEncryption};
//!
//! let mut rng = thread_rng();
//! let scheme = SilentThresholdScheme::<PairingEngine>::new();
//!
//! // Generate parameters for 8 parties with threshold 5
//! let params = scheme.param_gen(&mut rng, 8, 5).unwrap();
//! let keys = scheme.keygen(&mut rng, 8, &params).unwrap();
//!
//! // Encrypt a message
//! let message = b"Secret message";
//! let ciphertext = scheme.encrypt(
//!     &mut rng,
//!     &keys.aggregate_key,
//!     &params,
//!     5,
//!     message
//! ).unwrap();
//!
//! // Collect partial decryptions from 6 participants
//! let mut selector = vec![false; 8];
//! let mut partials = Vec::new();
//! for i in 0..6 {
//!     selector[i] = true;
//!     partials.push(scheme.partial_decrypt(&keys.secret_keys[i], &ciphertext).unwrap());
//! }
//!
//! // Aggregate and decrypt
//! let result = scheme.aggregate_decrypt(
//!     &ciphertext,
//!     &partials,
//!     &selector,
//!     &keys.aggregate_key
//! ).unwrap();
//!
//! assert_eq!(result.plaintext.unwrap(), message);
//! ```

use alloc::vec::Vec;
use core::{fmt::Debug, marker::PhantomData};

use blake3::Hasher;
use rand_core::RngCore;
#[cfg(feature = "parallel")]
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use tracing::instrument;

use crate::{
    AggregateKey, Ciphertext, DecryptionResult, DensePolynomial, Fr, KZG, KeyMaterial,
    LagrangePowers, PairingBackend, Params, PartialDecryption, Polynomial, PolynomialCommitment,
    PublicKey, Radix2EvaluationDomain, SRS, SecretKey, TargetGroup, ThresholdEncryption,
    arith::{CurvePoint, FieldElement},
    build_lagrange_polys,
    errors::{BackendError, Error},
    sym_enc::{Blake3XorEncryption, SymmetricEncryption},
};

/// The Silent Threshold scheme implementation.
#[derive(Debug)]
pub struct SilentThresholdScheme<B: PairingBackend> {
    _phantom: PhantomData<B>,
    symmetric_enc: Blake3XorEncryption,
}

/// Type alias for the silent threshold scheme implementation.
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

    /// Generates random secret keys for all participants.
    ///
    /// Each participant receives a uniformly random scalar from the field,
    /// which serves as their secret share in the threshold scheme.
    ///
    /// # Arguments
    ///
    /// * `rng` - Cryptographically secure random number generator
    /// * `parties` - Number of participants in the scheme
    ///
    /// # Returns
    ///
    /// A vector of secret keys, one per participant, with IDs 0..parties-1
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
        if threshold >= parties {
            return Err(Error::InvalidConfig(
                "threshold must be less than parties".into(),
            ));
        }
        if threshold == 0 {
            return Err(Error::InvalidConfig(
                "threshold must be greater than 0".into(),
            ));
        }
        if !parties.is_power_of_two() {
            return Err(Error::InvalidConfig(
                "parties must be a power of two".into(),
            ));
        }

        let tau = B::Scalar::random(rng);
        (|| {
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
            let lagrange_powers =
                LagrangePowers::precompute_lagrange_powers(&lagranges, parties, &tau)
                    .map_err(Error::Backend)?;

            Ok(Params {
                srs,
                lagrange_powers,
            })
        })()
    }

    #[instrument(level = "info", skip_all, fields(parties))]
    fn keygen<R: RngCore + ?Sized>(
        &self,
        rng: &mut R,
        parties: usize,
        params: &Params<B>,
    ) -> Result<KeyMaterial<B>, Error> {
        let secret_keys = Self::generate_secret_keys(rng, parties);

        let public_keys = {
            #[cfg(feature = "parallel")]
            {
                secret_keys
                    .par_iter()
                    .map(|sk| sk.derive_public_key(params))
                    .collect::<Result<Vec<_>, BackendError>>()?
            }
            #[cfg(not(feature = "parallel"))]
            {
                secret_keys
                    .iter()
                    .map(|sk| sk.derive_public_key(params))
                    .collect::<Result<Vec<_>, BackendError>>()?
            }
        };

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
        if threshold == 0 {
            return Err(Error::InvalidConfig(
                "threshold must be greater than 0".into(),
            ));
        }
        if threshold + 1 >= params.srs.powers_of_g.len() {
            return Err(Error::InvalidConfig(
                "threshold exceeds available SRS powers".into(),
            ));
        }

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
        let payload_key = derive_payload_key::<B>(&shared_secret);

        let payload_ct = self.symmetric_enc.encrypt(&payload_key, payload)?;

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

        let parties = agg_key.public_keys.len();
        if parties == 0 {
            return Err(Error::InvalidConfig("require at least one party".into()));
        }
        if !parties.is_power_of_two() {
            return Err(Error::InvalidConfig(
                "parties must be a power of two".into(),
            ));
        }
        if selector.len() != parties {
            return Err(Error::SelectorMismatch {
                expected: parties,
                actual: selector.len(),
            });
        }
        if !selector[0] {
            return Err(Error::MalformedInput(
                "selector[0] must be true to anchor interpolation".into(),
            ));
        }
        if ciphertext.proof_g1.len() != 2 || ciphertext.proof_g2.len() != 6 {
            return Err(Error::MalformedInput(
                "ciphertext proof sizes are invalid".into(),
            ));
        }

        let mut partial_map: Vec<Option<&PartialDecryption<B>>> = vec![None; parties];
        for partial in partials {
            if partial.participant_id < parties {
                partial_map[partial.participant_id] = Some(partial);
            }
        }

        let domain = Radix2EvaluationDomain::new(parties)
            .ok_or_else(|| Error::InvalidConfig("invalid evaluation domain size".into()))?;
        let domain_elements = domain.elements();

        let mut points = vec![domain_elements[0]];
        let mut selected_indices = Vec::new();
        for (idx, &is_selected) in selector.iter().enumerate() {
            if is_selected {
                if partial_map[idx].is_none() {
                    return Err(Error::MalformedInput(
                        "missing partial decryption for selected party".into(),
                    ));
                }
                selected_indices.push(idx);
            } else {
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
        let b_evals: Vec<Fr> = domain.fft(b_polynomial.coeffs());

        let scalars: Vec<Fr> = selected_indices.iter().map(|&idx| b_evals[idx]).collect();

        let b_g2 = <KZG as PolynomialCommitment<B>>::commit_g2(&agg_key.kzg_params, &b_polynomial)
            .map_err(Error::Backend)?;

        let mut bminus1 = b_polynomial.clone();
        if let Some(constant) = bminus1.coeffs.get_mut(0) {
            *constant -= Fr::one();
        }
        let (q0, remainder) = bminus1.divide_by_linear(domain_elements[0]);
        if remainder != Fr::zero() {
            return Err(Error::MalformedInput(
                "b polynomial division by anchor has non-zero remainder".into(),
            ));
        }
        let q0_g1 = <KZG as PolynomialCommitment<B>>::commit_g1(&agg_key.kzg_params, &q0)
            .map_err(Error::Backend)?;

        let mut bhat_coeffs = vec![Fr::zero(); ciphertext.threshold + 1];
        bhat_coeffs.extend_from_slice(b_polynomial.coeffs());
        let bhat = DensePolynomial::from_coefficients_vec(bhat_coeffs);
        let bhat_g1 = <KZG as PolynomialCommitment<B>>::commit_g1(&agg_key.kzg_params, &bhat)
            .map_err(Error::Backend)?;

        let party_inv =
            Fr::from_u64(parties as u64)
                .invert()
                .ok_or(Error::Backend(BackendError::Math(
                    "failed to invert party count",
                )))?;
        let scaled_scalars: Vec<Fr> = scalars.iter().map(|scalar| *scalar * party_inv).collect();

        let apk = if scalars.is_empty() {
            B::G1::identity()
        } else {
            let bases: Vec<B::G1> = selected_indices
                .iter()
                .map(|&idx| agg_key.public_keys[idx].bls_key)
                .collect();
            B::G1::multi_scalar_multiplication(&bases, &scaled_scalars)
        };

        let sigma = if scalars.is_empty() {
            B::G2::identity()
        } else {
            let bases: Vec<B::G2> = selected_indices
                .iter()
                .map(|&idx| partial_map[idx].unwrap().response)
                .collect();
            B::G2::multi_scalar_multiplication(&bases, &scaled_scalars)
        };

        let qx = if scalars.is_empty() {
            B::G1::identity()
        } else {
            let points: Vec<B::G1> = selected_indices
                .iter()
                .map(|&idx| agg_key.public_keys[idx].lagrange_li_x)
                .collect();
            B::G1::multi_scalar_multiplication(&points, &scalars)
        };

        let qz = if scalars.is_empty() {
            B::G1::identity()
        } else {
            let points: Vec<B::G1> = selected_indices
                .iter()
                .map(|&idx| agg_key.lagrange_row_sums[idx])
                .collect();
            B::G1::multi_scalar_multiplication(&points, &scalars)
        };

        let qhatx = if scalars.is_empty() {
            B::G1::identity()
        } else {
            let points: Vec<B::G1> = selected_indices
                .iter()
                .map(|&idx| agg_key.public_keys[idx].lagrange_li_minus0)
                .collect();
            B::G1::multi_scalar_multiplication(&points, &scalars)
        };

        let w1 = [
            apk.negate(),
            qz.negate(),
            qx.negate(),
            qhatx,
            bhat_g1.negate(),
            q0_g1.negate(),
        ];
        let w2 = [b_g2, sigma];

        let mut enc_key_lhs = w1.to_vec();
        enc_key_lhs.extend_from_slice(&ciphertext.proof_g1);
        let mut enc_key_rhs = ciphertext.proof_g2.clone();
        enc_key_rhs.extend_from_slice(&w2);

        let enc_key = B::multi_pairing(&enc_key_lhs, &enc_key_rhs).map_err(Error::Backend)?;
        if enc_key != ciphertext.shared_secret {
            return Err(Error::MalformedInput(
                "ciphertext verification failed".into(),
            ));
        }

        let payload_key = derive_payload_key::<B>(&enc_key);
        let plaintext = self
            .symmetric_enc
            .decrypt(&payload_key, &ciphertext.payload)?;

        Ok(DecryptionResult {
            plaintext: Some(plaintext),
        })
    }
}

/// Constructs a polynomial that evaluates to `eval` at the first point and zero at all others.
///
/// This is a specialized Lagrange interpolation that efficiently constructs a polynomial
/// b(x) such that:
/// - b(points[0]) = eval
/// - b(points[i]) = 0 for i > 0
///
/// This is used during aggregate decryption to construct the interpolation polynomial
/// for combining partial decryption shares.
///
/// # Algorithm
///
/// The polynomial is constructed by:
/// 1. Building the product (x - points[1])(x - points[2])...(x - points[n-1])
/// 2. Evaluating this product at points[0] to get the normalization factor
/// 3. Scaling the coefficients so that b(points[0]) = eval
///
/// # Arguments
///
/// * `eval` - The desired evaluation at the first point (typically 1)
/// * `points` - The evaluation points, where points[0] is the "anchor" point
///
/// # Returns
///
/// A polynomial satisfying the interpolation constraints, or an error if the
/// anchor point is a root of the vanishing polynomial.
///
/// # Errors
///
/// Returns `Error::Backend` if the interpolation anchor cannot be inverted,
/// which would indicate that points[0] is equal to one of the other points.
#[instrument(level = "info", skip_all)]
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

    let scale_inv = scale.invert().ok_or(Error::Backend(BackendError::Math(
        "failed to invert interpolation anchor",
    )))?;
    let multiplier = eval * scale_inv;

    for coeff in coeffs.iter_mut() {
        *coeff *= multiplier;
    }

    Ok(DensePolynomial::from_coefficients_vec(coeffs))
}

/// Derives a symmetric encryption key from a pairing target group element.
///
/// Uses BLAKE3 as a key derivation function (KDF) to convert the shared secret
/// from the pairing operation into a 32-byte symmetric key suitable for payload
/// encryption.
///
/// # Domain Separation
///
/// The derivation uses the domain separator "tess::payload-key" to ensure
/// cryptographic independence from other uses of BLAKE3 in the system.
///
/// # Arguments
///
/// * `enc_key` - The shared secret from the pairing operation e(g,h)^s
///
/// # Returns
///
/// A 32-byte symmetric key derived deterministically from the input
///
/// # Security
///
/// The derived key is computationally indistinguishable from random under
/// the assumption that BLAKE3 is a secure hash function and the input
/// has sufficient entropy.
fn derive_payload_key<B: PairingBackend>(enc_key: &B::Target) -> [u8; 32] {
    let mut hasher = Hasher::new();
    hasher.update(b"tess::payload-key");
    let repr = enc_key.to_repr();
    hasher.update(repr.as_ref());
    let digest = hasher.finalize();
    let mut key = [0u8; 32];
    key.copy_from_slice(digest.as_bytes());
    key
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::thread_rng;

    use crate::PairingEngine;

    #[test]
    fn e2e_negative_tampered_ciphertext() {
        let mut rng = thread_rng();
        let scheme = SilentThresholdScheme::<PairingEngine>::new();

        let parties = 8;
        let threshold = 4;
        let params = scheme.param_gen(&mut rng, parties, threshold).unwrap();
        let keys = scheme.keygen(&mut rng, parties, &params).unwrap();

        let payload = b"e2e negative test payload";
        let mut ct = scheme
            .encrypt(&mut rng, &keys.aggregate_key, &params, threshold, payload)
            .unwrap();

        let share_count = threshold + 1;
        let mut selector = vec![false; parties];
        let mut partials = Vec::with_capacity(share_count);
        for (i, selected) in selector.iter_mut().enumerate().take(share_count) {
            *selected = true;
            partials.push(scheme.partial_decrypt(&keys.secret_keys[i], &ct).unwrap());
        }

        ct.proof_g1[0] = <PairingEngine as PairingBackend>::G1::identity();

        let res = scheme.aggregate_decrypt(&ct, &partials, &selector, &keys.aggregate_key);
        assert!(matches!(res, Err(Error::MalformedInput(_))));
    }

    #[test]
    fn interp_mostly_zero_respects_constraints() {
        let points = vec![Fr::one(), Fr::from_u64(3), Fr::from_u64(5)];
        let poly = interp_mostly_zero(Fr::one(), &points).unwrap();

        assert_eq!(poly.evaluate(&points[0]), Fr::one());
        for point in points.iter().skip(1) {
            assert_eq!(poly.evaluate(point), Fr::zero());
        }
    }

    #[test]
    fn derive_payload_key_deterministic() {
        let g1 = <PairingEngine as PairingBackend>::G1::generator();
        let g2 = <PairingEngine as PairingBackend>::G2::generator();
        let enc_key = <PairingEngine as PairingBackend>::pairing(&g1, &g2);

        let key_a = derive_payload_key::<PairingEngine>(&enc_key);
        let key_b = derive_payload_key::<PairingEngine>(&enc_key);
        assert_eq!(key_a, key_b);
    }
}
