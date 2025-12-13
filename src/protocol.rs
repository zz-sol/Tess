#[cfg(feature = "ark_bls12381")]
use blake3::Hasher;
use core::fmt::Debug;

use rand_core::RngCore;

use crate::{
    backend::{PairingBackend, PolynomialCommitment},
    config::ThresholdParameters,
    errors::Error,
};

/// Secret key owned by a participant.
#[derive(Clone, Debug)]
pub struct SecretKey<B: PairingBackend> {
    pub participant_id: usize,
    pub scalar: B::Scalar,
}

/// Public metadata used to verify shares and construct the aggregate key.
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

/// Aggregated key required for encryption and verification of responses.
#[derive(Clone, Debug)]
pub struct AggregateKey<B: PairingBackend> {
    pub public_keys: Vec<PublicKey<B>>,
    pub ask: B::G1,
    pub z_g2: B::G2,
    pub lagrange_row_sums: Vec<B::G1>,
    pub precomputed_pairing: B::Target,
    pub commitment_params: <B::PolynomialCommitment as PolynomialCommitment<B>>::Parameters,
}

/// Ciphertext produced by the silent threshold encryption scheme.
#[derive(Clone, Debug)]
pub struct Ciphertext<B: PairingBackend> {
    pub gamma_g2: B::G2,
    pub proof_g1: Vec<B::G1>,
    pub proof_g2: Vec<B::G2>,
    pub shared_secret: B::Target,
    pub threshold: usize,
    pub payload: Vec<u8>,
}

/// Output of a participant's partial decryption.
#[derive(Clone, Debug)]
pub struct PartialDecryption<B: PairingBackend> {
    pub participant_id: usize,
    pub response: B::G2,
}

/// Bundle returned by key generation.
#[derive(Clone, Debug)]
pub struct KeyMaterial<B: PairingBackend> {
    pub secret_keys: Vec<SecretKey<B>>,
    pub public_keys: Vec<PublicKey<B>>,
    pub aggregate_key: AggregateKey<B>,
    pub kzg_params: <B::PolynomialCommitment as PolynomialCommitment<B>>::Parameters,
}

/// Result produced after aggregation of enough partial decryptions.
#[derive(Clone, Debug)]
pub struct DecryptionResult<B: PairingBackend> {
    pub shared_secret: B::Target,
    pub opening_proof: Option<Vec<u8>>,
    pub plaintext: Option<Vec<u8>>,
}

/// High-level API required by consumers of the scheme.
pub trait ThresholdScheme<B: PairingBackend>: Debug + Send + Sync + 'static {
    /// Generates key material for all parties using the selected backend.
    fn keygen<R: RngCore + ?Sized>(
        &self,
        rng: &mut R,
        params: &ThresholdParameters,
    ) -> Result<KeyMaterial<B>, Error>;

    /// Recomputes the aggregated key from a slice of public keys (e.g. when members are rotated).
    fn aggregate_public_key(
        &self,
        params: &ThresholdParameters,
        public_keys: &[PublicKey<B>],
    ) -> Result<AggregateKey<B>, Error>;

    /// Encrypts a payload with the aggregated key.
    fn encrypt<R: RngCore + ?Sized>(
        &self,
        rng: &mut R,
        agg_key: &AggregateKey<B>,
        params: &ThresholdParameters,
        payload: &[u8],
    ) -> Result<Ciphertext<B>, Error>;

    /// Computes a participant's contribution to the threshold decryption.
    fn partial_decrypt(
        &self,
        secret_key: &SecretKey<B>,
        ciphertext: &Ciphertext<B>,
    ) -> Result<PartialDecryption<B>, Error>;

    /// Aggregates partial decryptions and recovers the shared secret.
    fn aggregate_decrypt(
        &self,
        ciphertext: &Ciphertext<B>,
        partials: &[PartialDecryption<B>],
        selector: &[bool],
        agg_key: &AggregateKey<B>,
    ) -> Result<DecryptionResult<B>, Error>;
}

#[cfg(feature = "ark_bls12381")]
const PAYLOAD_KDF_DOMAIN: &[u8] = b"TESS::threshold::payload";

#[cfg(feature = "ark_bls12381")]
use crate::backend::TargetGroup;

#[cfg(feature = "ark_bls12381")]
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

#[cfg(feature = "ark_bls12381")]
fn xor_with_keystream(data: &[u8], keystream: &[u8]) -> Vec<u8> {
    data.iter()
        .zip(keystream.iter())
        .map(|(byte, key)| byte ^ key)
        .collect()
}

#[cfg(feature = "ark_bls12381")]
fn encrypt_payload<B: PairingBackend>(secret: &B::Target, payload: &[u8]) -> Vec<u8> {
    let keystream = derive_keystream::<B>(secret, payload.len());
    xor_with_keystream(payload, &keystream)
}

#[cfg(feature = "ark_bls12381")]
fn decrypt_payload<B: PairingBackend>(secret: &B::Target, ciphertext: &[u8]) -> Vec<u8> {
    let keystream = derive_keystream::<B>(secret, ciphertext.len());
    xor_with_keystream(ciphertext, &keystream)
}

#[cfg(feature = "ark_bls12381")]
pub mod arkworks {
    use super::*;
    use ark_bls12_381::Fr as BlsFr;
    use ark_poly::univariate::DensePolynomial;
    use ark_poly::{DenseUVPolynomial, EvaluationDomain, Radix2EvaluationDomain};
    use rand_core::RngCore;

    use crate::arkworks_backend::{ArkG1, ArkG2, ArkworksBls12, BlsKzg, BlsMsm, BlsPowers};
    use crate::backend::{CurvePoint, MsmProvider, TargetGroup};
    use crate::config::{BackendId, CurveId};
    use crate::errors::{BackendError, Error};
    use crate::lagrange::{interp_mostly_zero, lagrange_polys};
    use ark_ff::{Field, One, Zero};
    use ark_serialize::CanonicalDeserialize;
    use ark_std::UniformRand;

    #[derive(Debug, Default)]
    pub struct SilentThresholdScheme;

    impl SilentThresholdScheme {
        fn sample_tau<R: RngCore + ?Sized>(&self, rng: &mut R) -> BlsFr {
            BlsFr::rand(rng)
        }

        fn generate_secret_keys<R: RngCore + ?Sized>(
            &self,
            rng: &mut R,
            parties: usize,
        ) -> Vec<SecretKey<ArkworksBls12>> {
            (0..parties)
                .map(|participant_id| SecretKey {
                    participant_id,
                    scalar: BlsFr::rand(rng),
                })
                .collect()
        }
    }

    impl ThresholdScheme<ArkworksBls12> for SilentThresholdScheme {
        fn keygen<R: RngCore + ?Sized>(
            &self,
            rng: &mut R,
            params: &ThresholdParameters,
        ) -> Result<KeyMaterial<ArkworksBls12>, Error> {
            params.validate()?;
            ensure_supported_config(params)?;
            let parties = params.parties;

            let tau = if let Some(bytes) = params.kzg_tau.as_ref() {
                parse_tau(bytes)?
            } else {
                self.sample_tau(rng)
            };
            let kzg_params = BlsKzg::setup(parties, &tau).map_err(|err| Error::Backend(err))?;

            let lagranges = lagrange_polys::<BlsFr>(parties).map_err(Error::Backend)?;
            let domain = Radix2EvaluationDomain::new(parties)
                .ok_or_else(|| Error::Backend(BackendError::Math("invalid evaluation domain")))?;

            let secret_keys = self.generate_secret_keys(rng, parties);
            let public_keys = secret_keys
                .iter()
                .map(|sk| derive_public_key(sk.participant_id, sk, &lagranges, domain, &kzg_params))
                .collect::<Result<Vec<_>, BackendError>>()
                .map_err(Error::Backend)?;

            let aggregate_key = aggregate_public_key(&public_keys, &kzg_params, parties)?;

            Ok(KeyMaterial {
                secret_keys,
                public_keys,
                aggregate_key,
                kzg_params,
            })
        }

        fn aggregate_public_key(
            &self,
            params_cfg: &ThresholdParameters,
            public_keys: &[PublicKey<ArkworksBls12>],
        ) -> Result<AggregateKey<ArkworksBls12>, Error> {
            params_cfg.validate()?;
            ensure_supported_config(params_cfg)?;
            let tau = load_tau_from_params(params_cfg)?;
            let kzg_params =
                BlsKzg::setup(params_cfg.parties, &tau).map_err(|err| Error::Backend(err))?;
            aggregate_public_key(public_keys, &kzg_params, params_cfg.parties)
        }

        fn encrypt<R: RngCore + ?Sized>(
            &self,
            rng: &mut R,
            agg_key: &AggregateKey<ArkworksBls12>,
            params_cfg: &ThresholdParameters,
            payload: &[u8],
        ) -> Result<Ciphertext<ArkworksBls12>, Error> {
            params_cfg.validate()?;
            ensure_supported_config(params_cfg)?;
            let threshold = params_cfg.threshold;
            let kzg_params = &agg_key.commitment_params;

            if threshold + 1 >= kzg_params.powers_of_g.len() {
                return Err(Error::Backend(BackendError::Math(
                    "threshold exceeds supported commitment degree",
                )));
            }
            if kzg_params.powers_of_h.len() < 2 {
                return Err(Error::Backend(BackendError::Math(
                    "not enough G2 powers for encryption",
                )));
            }

            let g = ArkG1::from_affine(
                kzg_params
                    .powers_of_g
                    .get(0)
                    .ok_or(BackendError::Math("missing g generator"))?,
            );
            let g_tau_t = ArkG1::from_affine(
                kzg_params
                    .powers_of_g
                    .get(threshold + 1)
                    .ok_or(BackendError::Math("missing g^{tau^{t+1}}"))?,
            );

            let h = ArkG2::from_affine(
                kzg_params
                    .powers_of_h
                    .get(0)
                    .ok_or(BackendError::Math("missing h generator"))?,
            );
            let h_tau = ArkG2::from_affine(
                kzg_params
                    .powers_of_h
                    .get(1)
                    .ok_or(BackendError::Math("missing h^tau"))?,
            );
            let h_minus_one = ArkG2::generator().negate();

            let gamma = BlsFr::rand(rng);
            let gamma_g2 = h.mul_scalar(&gamma);

            let s0 = BlsFr::rand(rng);
            let s1 = BlsFr::rand(rng);
            let s2 = BlsFr::rand(rng);
            let s3 = BlsFr::rand(rng);
            let s4 = BlsFr::rand(rng);

            let sa1_0 = agg_key
                .ask
                .mul_scalar(&s0)
                .add(&g_tau_t.mul_scalar(&s3))
                .add(&g.mul_scalar(&s4));
            let sa1_1 = g.mul_scalar(&s2);

            let sa2_0 = h.mul_scalar(&s0).add(&gamma_g2.mul_scalar(&s2));
            let sa2_1 = agg_key.z_g2.mul_scalar(&s0);
            let sa2_2 = h_tau.mul_scalar(&(s0 + s1));
            let sa2_3 = h.mul_scalar(&s1);
            let sa2_4 = h.mul_scalar(&s3);
            let sa2_5 = h_tau.add(&h_minus_one).mul_scalar(&s4);

            let shared_secret = agg_key.precomputed_pairing.mul_scalar(&s4);

            let mut proof_g1 = Vec::with_capacity(2);
            proof_g1.push(sa1_0);
            proof_g1.push(sa1_1);

            let mut proof_g2 = Vec::with_capacity(6);
            proof_g2.extend([sa2_0, sa2_1, sa2_2, sa2_3, sa2_4, sa2_5]);

            let payload_ct = encrypt_payload::<ArkworksBls12>(&shared_secret, payload);

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
            secret_key: &SecretKey<ArkworksBls12>,
            ciphertext: &Ciphertext<ArkworksBls12>,
        ) -> Result<PartialDecryption<ArkworksBls12>, Error> {
            let response = ciphertext.gamma_g2.mul_scalar(&secret_key.scalar);
            Ok(PartialDecryption {
                participant_id: secret_key.participant_id,
                response,
            })
        }

        fn aggregate_decrypt(
            &self,
            ciphertext: &Ciphertext<ArkworksBls12>,
            partials: &[PartialDecryption<ArkworksBls12>],
            selector: &[bool],
            agg_key: &AggregateKey<ArkworksBls12>,
        ) -> Result<DecryptionResult<ArkworksBls12>, Error> {
            aggregate_decrypt(ciphertext, partials, selector, agg_key)
        }
    }

    fn ensure_supported_config(params: &ThresholdParameters) -> Result<(), Error> {
        if params.backend.backend != BackendId::Arkworks {
            return Err(Error::Backend(BackendError::UnsupportedFeature(
                "SilentThresholdScheme is only implemented for the Arkworks backend",
            )));
        }
        if params.backend.curve != CurveId::Bls12_381 {
            return Err(Error::Backend(BackendError::UnsupportedCurve(
                "SilentThresholdScheme currently targets BLS12-381",
            )));
        }
        Ok(())
    }

    fn parse_tau(bytes: &[u8]) -> Result<BlsFr, Error> {
        BlsFr::deserialize_compressed(bytes)
            .map_err(|_| Error::InvalidConfig("invalid trusted tau encoding".into()))
    }

    fn load_tau_from_params(params: &ThresholdParameters) -> Result<BlsFr, Error> {
        let bytes = params
            .kzg_tau
            .as_ref()
            .ok_or_else(|| Error::InvalidConfig("missing trusted tau in parameters".into()))?;
        parse_tau(bytes)
    }

    fn derive_public_key(
        participant_id: usize,
        sk: &SecretKey<ArkworksBls12>,
        lagranges: &[DensePolynomial<BlsFr>],
        domain: Radix2EvaluationDomain<BlsFr>,
        params: &BlsPowers,
    ) -> Result<PublicKey<ArkworksBls12>, BackendError> {
        let li = lagranges
            .get(participant_id)
            .ok_or(BackendError::Math("missing lagrange polynomial"))?;

        let li_poly = li.clone();
        let sk_li_poly = (&li_poly) * sk.scalar;
        let lagrange_li = BlsKzg::commit_g1(params, &sk_li_poly)?;

        let mut minus0_poly = sk_li_poly.clone();
        if let Some(constant) = minus0_poly.coeffs.get_mut(0) {
            *constant = BlsFr::zero();
        }
        let lagrange_li_minus0 = BlsKzg::commit_g1(params, &minus0_poly)?;

        let shift_coeffs = if li.coeffs.len() > 1 {
            li.coeffs[1..].to_vec()
        } else {
            vec![BlsFr::zero()]
        };
        let shift_poly = DensePolynomial::from_coefficients_vec(shift_coeffs);
        let sk_shift_poly = (&shift_poly) * sk.scalar;
        let lagrange_li_x = BlsKzg::commit_g1(params, &sk_shift_poly)?;

        let mut lagrange_li_lj_z = Vec::with_capacity(lagranges.len());
        for (idx, lj) in lagranges.iter().enumerate() {
            let numerator = if idx == participant_id {
                ((&li_poly) * (&li_poly)) - li_poly.clone()
            } else {
                lj.clone() * li_poly.clone()
            };

            let (f, _) = numerator.divide_by_vanishing_poly(domain);
            let scaled = (&f) * sk.scalar;
            let commitment = BlsKzg::commit_g1(params, &scaled)?;
            lagrange_li_lj_z.push(commitment);
        }

        Ok(PublicKey {
            participant_id,
            bls_key: ArkG1::generator().mul_scalar(&sk.scalar),
            lagrange_li,
            lagrange_li_minus0,
            lagrange_li_x,
            lagrange_li_lj_z,
        })
    }

    fn aggregate_public_key(
        public_keys: &[PublicKey<ArkworksBls12>],
        params: &BlsPowers,
        parties: usize,
    ) -> Result<AggregateKey<ArkworksBls12>, Error> {
        if public_keys.is_empty() {
            return Err(Error::InvalidConfig(
                "cannot aggregate empty public key set".into(),
            ));
        }
        if public_keys.len() != parties {
            return Err(Error::InvalidConfig("public key length mismatch".into()));
        }

        let mut ask = ArkG1::identity();
        for pk in public_keys {
            ask = ask.add(&pk.lagrange_li);
        }

        let mut lagrange_row_sums = vec![ArkG1::identity(); parties];
        for (idx, row) in lagrange_row_sums.iter_mut().enumerate() {
            for pk in public_keys {
                if let Some(val) = pk.lagrange_li_lj_z.get(idx) {
                    *row = row.add(val);
                }
            }
        }

        let g2_tau_n = params
            .powers_of_h
            .get(parties)
            .ok_or_else(|| Error::Backend(BackendError::Math("missing h^tau^n")))?;
        let z_g2 = ArkG2::from_affine(g2_tau_n).sub(&ArkG2::generator());

        Ok(AggregateKey {
            public_keys: public_keys.to_vec(),
            ask,
            z_g2,
            lagrange_row_sums,
            precomputed_pairing: params.e_gh.clone(),
            commitment_params: params.clone(),
        })
    }

    fn divide_by_linear(
        poly: &DensePolynomial<BlsFr>,
        root: BlsFr,
    ) -> (DensePolynomial<BlsFr>, BlsFr) {
        assert!(poly.coeffs.len() > 1, "cannot divide constant polynomial");
        let mut quotient = vec![BlsFr::zero(); poly.coeffs.len() - 1];
        let mut carry = *poly.coeffs.last().unwrap();
        for (idx, coeff) in poly.coeffs.iter().rev().skip(1).enumerate() {
            let pos = quotient.len() - 1 - idx;
            quotient[pos] = carry;
            carry = *coeff + root * carry;
        }
        (DensePolynomial::from_coefficients_vec(quotient), carry)
    }

    fn aggregate_decrypt(
        ciphertext: &Ciphertext<ArkworksBls12>,
        partials: &[PartialDecryption<ArkworksBls12>],
        selector: &[bool],
        agg_key: &AggregateKey<ArkworksBls12>,
    ) -> Result<DecryptionResult<ArkworksBls12>, Error> {
        let n = agg_key.public_keys.len();
        if selector.len() != n {
            return Err(Error::SelectorMismatch {
                expected: n,
                actual: selector.len(),
            });
        }

        let mut responses = vec![ArkG2::identity(); n];
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

        let domain = Radix2EvaluationDomain::new(n)
            .ok_or_else(|| Error::Backend(BackendError::Math("invalid evaluation domain")))?;
        let domain_elements: Vec<BlsFr> = domain.elements().collect();

        let mut points = vec![domain_elements[0]];
        let mut parties = Vec::new();
        for (i, (&selected, &omega)) in selector.iter().zip(domain_elements.iter()).enumerate() {
            if selected {
                if !seen[i] {
                    return Err(Error::NotEnoughShares { required, provided });
                }
                parties.push(i);
            } else {
                points.push(omega);
            }
        }

        let b = interp_mostly_zero(BlsFr::one(), &points).map_err(Error::Backend)?;
        let b_evals = domain.fft(&b.coeffs);

        let b_g2 = BlsKzg::commit_g2(&agg_key.commitment_params, &b).map_err(Error::Backend)?;

        let mut b_minus_one = b.clone();
        if let Some(constant) = b_minus_one.coeffs.get_mut(0) {
            *constant -= BlsFr::one();
        }
        let (q0, remainder) = divide_by_linear(&b_minus_one, domain_elements[0]);
        if !remainder.is_zero() {
            return Err(Error::Backend(BackendError::Math(
                "division by linear failed",
            )));
        }
        let q0_g1 = BlsKzg::commit_g1(&agg_key.commitment_params, &q0).map_err(Error::Backend)?;

        let mut bhat_coeffs = vec![BlsFr::zero(); ciphertext.threshold + 1];
        bhat_coeffs.extend_from_slice(&b.coeffs);
        let bhat = DensePolynomial::from_coefficients_vec(bhat_coeffs);
        let bhat_g1 =
            BlsKzg::commit_g1(&agg_key.commitment_params, &bhat).map_err(Error::Backend)?;

        let n_inv = BlsFr::from(n as u64)
            .inverse()
            .ok_or_else(|| Error::Backend(BackendError::Math("domain size inversion failed")))?;

        let scalars: Vec<BlsFr> = parties.iter().map(|&i| b_evals[i]).collect();

        let apk = if scalars.is_empty() {
            ArkG1::identity()
        } else {
            let bases: Vec<ArkG1> = parties
                .iter()
                .map(|&i| agg_key.public_keys[i].bls_key.clone())
                .collect();
            BlsMsm::msm_g1(&bases, &scalars)
                .map_err(Error::Backend)?
                .mul_scalar(&n_inv)
        };

        let sigma = if scalars.is_empty() {
            ArkG2::identity()
        } else {
            let bases: Vec<ArkG2> = parties.iter().map(|&i| responses[i].clone()).collect();
            BlsMsm::msm_g2(&bases, &scalars)
                .map_err(Error::Backend)?
                .mul_scalar(&n_inv)
        };

        let qx = if scalars.is_empty() {
            ArkG1::identity()
        } else {
            let bases: Vec<ArkG1> = parties
                .iter()
                .map(|&i| agg_key.public_keys[i].lagrange_li_x.clone())
                .collect();
            BlsMsm::msm_g1(&bases, &scalars).map_err(Error::Backend)?
        };

        let qz = if scalars.is_empty() {
            ArkG1::identity()
        } else {
            let bases: Vec<ArkG1> = parties
                .iter()
                .map(|&i| agg_key.lagrange_row_sums[i].clone())
                .collect();
            BlsMsm::msm_g1(&bases, &scalars).map_err(Error::Backend)?
        };

        let qhatx = if scalars.is_empty() {
            ArkG1::identity()
        } else {
            let bases: Vec<ArkG1> = parties
                .iter()
                .map(|&i| agg_key.public_keys[i].lagrange_li_minus0.clone())
                .collect();
            BlsMsm::msm_g1(&bases, &scalars).map_err(Error::Backend)?
        };

        let mut lhs = Vec::new();
        lhs.push(apk.negate());
        lhs.push(qz.negate());
        lhs.push(qx.negate());
        lhs.push(qhatx);
        lhs.push(bhat_g1.negate());
        lhs.push(q0_g1.negate());
        lhs.extend(ciphertext.proof_g1.iter().cloned());

        let mut rhs = Vec::new();
        rhs.extend(ciphertext.proof_g2.iter().cloned());
        rhs.push(b_g2);
        rhs.push(sigma);

        let shared_secret = ArkworksBls12::multi_pairing(&lhs, &rhs).map_err(Error::Backend)?;
        let plaintext = if ciphertext.payload.is_empty() {
            None
        } else {
            Some(decrypt_payload::<ArkworksBls12>(
                &shared_secret,
                &ciphertext.payload,
            ))
        };

        Ok(DecryptionResult {
            shared_secret,
            opening_proof: None,
            plaintext,
        })
    }
}

#[cfg(all(test, feature = "ark_bls12381"))]
mod tests {
    use super::arkworks::SilentThresholdScheme;
    use super::*;
    use crate::{
        backend::TargetGroup,
        config::{BackendConfig, BackendId, CurveId},
    };
    use rand::{SeedableRng, rngs::StdRng};

    fn sample_params() -> ThresholdParameters {
        ThresholdParameters {
            parties: 8,
            threshold: 4,
            chunk_size: 32,
            backend: BackendConfig::new(BackendId::Arkworks, CurveId::Bls12_381),
            kzg_tau: None,
        }
    }

    #[test]
    fn arkworks_encrypt_decrypt_roundtrip() {
        let mut rng = StdRng::from_entropy();
        let scheme = SilentThresholdScheme::default();
        let params = sample_params();
        let km = scheme.keygen(&mut rng, &params).expect("keygen");
        let ct = scheme
            .encrypt(&mut rng, &km.aggregate_key, &params, b"payload")
            .expect("encrypt");

        let mut selector = vec![false; params.parties];
        let mut partials = Vec::new();
        for idx in 0..=params.threshold {
            selector[idx] = true;
            let share = scheme
                .partial_decrypt(&km.secret_keys[idx], &ct)
                .expect("partial decrypt");
            partials.push(share);
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

    #[test]
    fn arkworks_decrypt_not_enough_shares() {
        let mut rng = StdRng::from_entropy();
        let scheme = SilentThresholdScheme::default();
        let params = sample_params();
        let km = scheme.keygen(&mut rng, &params).expect("keygen");
        let ct = scheme
            .encrypt(&mut rng, &km.aggregate_key, &params, b"payload")
            .expect("encrypt");

        let mut selector = vec![false; params.parties];
        let mut partials = Vec::new();
        for idx in 0..params.threshold {
            selector[idx] = true;
            let share = scheme
                .partial_decrypt(&km.secret_keys[idx], &ct)
                .expect("partial decrypt");
            partials.push(share);
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

    #[test]
    fn arkworks_decrypt_selector_mismatch() {
        let mut rng = StdRng::from_entropy();
        let scheme = SilentThresholdScheme::default();
        let params = sample_params();
        let km = scheme.keygen(&mut rng, &params).expect("keygen");
        let ct = scheme
            .encrypt(&mut rng, &km.aggregate_key, &params, b"payload")
            .expect("encrypt");

        let mut selector = vec![false; params.parties];
        let mut partials = Vec::new();
        for idx in 0..=params.threshold {
            selector[idx] = true;
            let share = scheme
                .partial_decrypt(&km.secret_keys[idx], &ct)
                .expect("partial decrypt");
            partials.push(share);
        }
        let mismatched_selector = selector[..selector.len() - 1].to_vec();
        let err = scheme.aggregate_decrypt(&ct, &partials, &mismatched_selector, &km.aggregate_key);
        assert!(matches!(err, Err(Error::SelectorMismatch { .. })));
    }

    #[test]
    fn arkworks_decrypt_duplicate_partial() {
        let mut rng = StdRng::from_entropy();
        let scheme = SilentThresholdScheme::default();
        let params = sample_params();
        let km = scheme.keygen(&mut rng, &params).expect("keygen");
        let ct = scheme
            .encrypt(&mut rng, &km.aggregate_key, &params, b"payload")
            .expect("encrypt");

        let mut selector = vec![false; params.parties];
        let mut partials = Vec::new();
        for idx in 0..=params.threshold {
            selector[idx] = true;
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
}
