use super::*;
use ::ark_bn254::Fr as BnFr;
use ark_poly::univariate::DensePolynomial;
use ark_poly::{DenseUVPolynomial, EvaluationDomain, Radix2EvaluationDomain};
use rand_core::RngCore;

use crate::backend::{
    ArkBnG1, ArkBnG2, ArkworksBn254, BnKzg, BnMsm, BnPowers, CurvePoint, MsmProvider, TargetGroup,
};
use crate::config::{BackendId, CurveId};
use crate::errors::{BackendError, Error};
use crate::lagrange::ark_bn254::{interp_mostly_zero, lagrange_polys};
use ark_ff::{Field, One, Zero};
use ark_serialize::CanonicalDeserialize;
use ark_std::UniformRand;
use blake3::Hasher;

const PAYLOAD_KDF_DOMAIN: &[u8] = b"TESS::threshold::payload";

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

#[derive(Debug, Default)]
pub struct SilentThresholdBn;

impl SilentThresholdBn {
    fn sample_tau<R: RngCore + ?Sized>(&self, rng: &mut R) -> BnFr {
        BnFr::rand(rng)
    }

    fn generate_secret_keys<R: RngCore + ?Sized>(
        &self,
        rng: &mut R,
        parties: usize,
    ) -> Vec<SecretKey<ArkworksBn254>> {
        (0..parties)
            .map(|participant_id| SecretKey {
                participant_id,
                scalar: BnFr::rand(rng),
            })
            .collect()
    }
}

impl ThresholdScheme<ArkworksBn254> for SilentThresholdBn {
    fn keygen<R: RngCore + ?Sized>(
        &self,
        rng: &mut R,
        params: &ThresholdParameters,
    ) -> Result<KeyMaterial<ArkworksBn254>, Error> {
        params.validate()?;
        ensure_supported_bn(params)?;
        let parties = params.parties;
        let tau = if let Some(bytes) = params.kzg_tau.as_ref() {
            parse_tau(bytes)?
        } else {
            self.sample_tau(rng)
        };
        let kzg_params = BnKzg::setup(parties, &tau).map_err(Error::Backend)?;
        let lagranges = lagrange_polys(parties).map_err(Error::Backend)?;
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
        params: &ThresholdParameters,
        public_keys: &[PublicKey<ArkworksBn254>],
    ) -> Result<AggregateKey<ArkworksBn254>, Error> {
        params.validate()?;
        ensure_supported_bn(params)?;
        let tau = load_tau_from_params(params)?;
        let kzg_params = BnKzg::setup(params.parties, &tau).map_err(Error::Backend)?;
        aggregate_public_key(public_keys, &kzg_params, params.parties)
    }

    fn encrypt<R: RngCore + ?Sized>(
        &self,
        rng: &mut R,
        agg_key: &AggregateKey<ArkworksBn254>,
        params: &ThresholdParameters,
        payload: &[u8],
    ) -> Result<Ciphertext<ArkworksBn254>, Error> {
        params.validate()?;
        ensure_supported_bn(params)?;
        let threshold = params.threshold;
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
        let g = ArkBnG1::from_affine(
            kzg_params
                .powers_of_g
                .get(0)
                .ok_or(BackendError::Math("missing g generator"))?,
        );
        let g_tau_t = ArkBnG1::from_affine(
            kzg_params
                .powers_of_g
                .get(threshold + 1)
                .ok_or(BackendError::Math("missing g^{tau^{t+1}}"))?,
        );
        let h = ArkBnG2::from_affine(
            kzg_params
                .powers_of_h
                .get(0)
                .ok_or(BackendError::Math("missing h generator"))?,
        );
        let h_tau = ArkBnG2::from_affine(
            kzg_params
                .powers_of_h
                .get(1)
                .ok_or(BackendError::Math("missing h^tau"))?,
        );
        let h_minus_one = ArkBnG2::generator().negate();

        let gamma = BnFr::rand(rng);
        let gamma_g2 = h.mul_scalar(&gamma);

        let s0 = BnFr::rand(rng);
        let s1 = BnFr::rand(rng);
        let s2 = BnFr::rand(rng);
        let s3 = BnFr::rand(rng);
        let s4 = BnFr::rand(rng);

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

        let proof_g1 = vec![sa1_0, sa1_1];
        let proof_g2 = vec![sa2_0, sa2_1, sa2_2, sa2_3, sa2_4, sa2_5];

        let payload_ct = encrypt_payload::<ArkworksBn254>(&shared_secret, payload);

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
        secret_key: &SecretKey<ArkworksBn254>,
        ciphertext: &Ciphertext<ArkworksBn254>,
    ) -> Result<PartialDecryption<ArkworksBn254>, Error> {
        let response = ciphertext.gamma_g2.mul_scalar(&secret_key.scalar);
        Ok(PartialDecryption {
            participant_id: secret_key.participant_id,
            response,
        })
    }

    fn aggregate_decrypt(
        &self,
        ciphertext: &Ciphertext<ArkworksBn254>,
        partials: &[PartialDecryption<ArkworksBn254>],
        selector: &[bool],
        agg_key: &AggregateKey<ArkworksBn254>,
    ) -> Result<DecryptionResult<ArkworksBn254>, Error> {
        aggregate_decrypt(ciphertext, partials, selector, agg_key)
    }
}

fn ensure_supported_bn(params: &ThresholdParameters) -> Result<(), Error> {
    if params.backend.backend != BackendId::Arkworks {
        return Err(Error::Backend(BackendError::UnsupportedFeature(
            "SilentThresholdBn is only available for Arkworks backend",
        )));
    }
    if params.backend.curve != CurveId::Bn254 {
        return Err(Error::Backend(BackendError::UnsupportedCurve(
            "SilentThresholdBn targets BN254",
        )));
    }
    Ok(())
}

fn parse_tau(bytes: &[u8]) -> Result<BnFr, Error> {
    BnFr::deserialize_compressed(bytes)
        .map_err(|_| Error::InvalidConfig("invalid trusted tau encoding".into()))
}

fn load_tau_from_params(params: &ThresholdParameters) -> Result<BnFr, Error> {
    let bytes = params
        .kzg_tau
        .as_ref()
        .ok_or_else(|| Error::InvalidConfig("missing trusted tau in parameters".into()))?;
    parse_tau(bytes)
}

fn derive_public_key(
    participant_id: usize,
    sk: &SecretKey<ArkworksBn254>,
    lagranges: &[DensePolynomial<BnFr>],
    domain: Radix2EvaluationDomain<BnFr>,
    params: &BnPowers,
) -> Result<PublicKey<ArkworksBn254>, BackendError> {
    let li = lagranges
        .get(participant_id)
        .ok_or(BackendError::Math("missing lagrange polynomial"))?;

    let li_poly = li.clone();
    let sk_li_poly = (&li_poly) * sk.scalar;
    let lagrange_li = BnKzg::commit_g1(params, &sk_li_poly)?;

    let mut minus0_poly = sk_li_poly.clone();
    if let Some(constant) = minus0_poly.coeffs.get_mut(0) {
        *constant = BnFr::zero();
    }
    let lagrange_li_minus0 = BnKzg::commit_g1(params, &minus0_poly)?;

    let shift_coeffs = if li.coeffs.len() > 1 {
        li.coeffs[1..].to_vec()
    } else {
        vec![BnFr::zero()]
    };
    let shift_poly = DensePolynomial::from_coefficients_vec(shift_coeffs);
    let sk_shift_poly = (&shift_poly) * sk.scalar;
    let lagrange_li_x = BnKzg::commit_g1(params, &sk_shift_poly)?;

    let mut lagrange_li_lj_z = Vec::with_capacity(lagranges.len());
    for (idx, lj) in lagranges.iter().enumerate() {
        let numerator = if idx == participant_id {
            ((&li_poly) * (&li_poly)) - li_poly.clone()
        } else {
            lj.clone() * li_poly.clone()
        };
        let (f, _) = numerator.divide_by_vanishing_poly(domain);
        let scaled = (&f) * sk.scalar;
        let commitment = BnKzg::commit_g1(params, &scaled)?;
        lagrange_li_lj_z.push(commitment);
    }

    Ok(PublicKey {
        participant_id,
        bls_key: ArkBnG1::generator().mul_scalar(&sk.scalar),
        lagrange_li,
        lagrange_li_minus0,
        lagrange_li_x,
        lagrange_li_lj_z,
    })
}

fn aggregate_public_key(
    public_keys: &[PublicKey<ArkworksBn254>],
    params: &BnPowers,
    parties: usize,
) -> Result<AggregateKey<ArkworksBn254>, Error> {
    if public_keys.is_empty() {
        return Err(Error::InvalidConfig(
            "cannot aggregate empty public key set".into(),
        ));
    }
    if public_keys.len() != parties {
        return Err(Error::InvalidConfig("public key length mismatch".into()));
    }

    let mut ask = ArkBnG1::identity();
    for pk in public_keys {
        ask = ask.add(&pk.lagrange_li);
    }

    let mut lagrange_row_sums = vec![ArkBnG1::identity(); parties];
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
    let z_g2 = ArkBnG2::from_affine(g2_tau_n).sub(&ArkBnG2::generator());

    Ok(AggregateKey {
        public_keys: public_keys.to_vec(),
        ask,
        z_g2,
        lagrange_row_sums,
        precomputed_pairing: params.e_gh.clone(),
        commitment_params: params.clone(),
    })
}

fn divide_by_linear(poly: &DensePolynomial<BnFr>, root: BnFr) -> (DensePolynomial<BnFr>, BnFr) {
    assert!(poly.coeffs.len() > 1, "cannot divide constant polynomial");
    let mut quotient = vec![BnFr::zero(); poly.coeffs.len() - 1];
    let mut carry = *poly.coeffs.last().unwrap();
    for (idx, coeff) in poly.coeffs.iter().rev().skip(1).enumerate() {
        let pos = quotient.len() - 1 - idx;
        quotient[pos] = carry;
        carry = *coeff + root * carry;
    }
    (DensePolynomial::from_coefficients_vec(quotient), carry)
}

fn aggregate_decrypt(
    ciphertext: &Ciphertext<ArkworksBn254>,
    partials: &[PartialDecryption<ArkworksBn254>],
    selector: &[bool],
    agg_key: &AggregateKey<ArkworksBn254>,
) -> Result<DecryptionResult<ArkworksBn254>, Error> {
    let n = agg_key.public_keys.len();
    if selector.len() != n {
        return Err(Error::SelectorMismatch {
            expected: n,
            actual: selector.len(),
        });
    }

    let mut responses = vec![ArkBnG2::identity(); n];
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
    let domain_elements: Vec<BnFr> = domain.elements().collect();

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

    let b = interp_mostly_zero(BnFr::one(), &points).map_err(Error::Backend)?;
    let b_evals = domain.fft(&b.coeffs);

    let b_g2 = BnKzg::commit_g2(&agg_key.commitment_params, &b).map_err(Error::Backend)?;

    let mut b_minus_one = b.clone();
    if let Some(constant) = b_minus_one.coeffs.get_mut(0) {
        *constant -= BnFr::one();
    }
    let (q0, remainder) = divide_by_linear(&b_minus_one, domain_elements[0]);
    if !remainder.is_zero() {
        return Err(Error::Backend(BackendError::Math(
            "division by linear failed",
        )));
    }
    let q0_g1 = BnKzg::commit_g1(&agg_key.commitment_params, &q0).map_err(Error::Backend)?;

    let mut bhat_coeffs = vec![BnFr::zero(); ciphertext.threshold + 1];
    bhat_coeffs.extend_from_slice(&b.coeffs);
    let bhat = DensePolynomial::from_coefficients_vec(bhat_coeffs);
    let bhat_g1 = BnKzg::commit_g1(&agg_key.commitment_params, &bhat).map_err(Error::Backend)?;

    let n_inv = BnFr::from(n as u64)
        .inverse()
        .ok_or_else(|| Error::Backend(BackendError::Math("domain size inversion failed")))?;

    let scalars: Vec<BnFr> = parties.iter().map(|&i| b_evals[i]).collect();

    let apk = if scalars.is_empty() {
        ArkBnG1::identity()
    } else {
        let bases: Vec<ArkBnG1> = parties
            .iter()
            .map(|&i| agg_key.public_keys[i].bls_key.clone())
            .collect();
        BnMsm::msm_g1(&bases, &scalars)
            .map_err(Error::Backend)?
            .mul_scalar(&n_inv)
    };

    let sigma = if scalars.is_empty() {
        ArkBnG2::identity()
    } else {
        let bases: Vec<ArkBnG2> = parties.iter().map(|&i| responses[i].clone()).collect();
        BnMsm::msm_g2(&bases, &scalars)
            .map_err(Error::Backend)?
            .mul_scalar(&n_inv)
    };

    let qx = if scalars.is_empty() {
        ArkBnG1::identity()
    } else {
        let bases: Vec<ArkBnG1> = parties
            .iter()
            .map(|&i| agg_key.public_keys[i].lagrange_li_x.clone())
            .collect();
        BnMsm::msm_g1(&bases, &scalars).map_err(Error::Backend)?
    };

    let qz = if scalars.is_empty() {
        ArkBnG1::identity()
    } else {
        let bases: Vec<ArkBnG1> = parties
            .iter()
            .map(|&i| agg_key.lagrange_row_sums[i].clone())
            .collect();
        BnMsm::msm_g1(&bases, &scalars).map_err(Error::Backend)?
    };

    let qhatx = if scalars.is_empty() {
        ArkBnG1::identity()
    } else {
        let bases: Vec<ArkBnG1> = parties
            .iter()
            .map(|&i| agg_key.public_keys[i].lagrange_li_minus0.clone())
            .collect();
        BnMsm::msm_g1(&bases, &scalars).map_err(Error::Backend)?
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

    let shared_secret = ArkworksBn254::multi_pairing(&lhs, &rhs).map_err(Error::Backend)?;
    let plaintext = if ciphertext.payload.is_empty() {
        None
    } else {
        Some(decrypt_payload::<ArkworksBn254>(
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{BackendConfig, BackendId, CurveId};
    use rand::{SeedableRng, rngs::StdRng};

    fn sample_params() -> ThresholdParameters {
        ThresholdParameters {
            parties: 8,
            threshold: 4,
            chunk_size: 32,
            backend: BackendConfig::new(BackendId::Arkworks, CurveId::Bn254),
            kzg_tau: None,
        }
    }

    #[test]
    fn bn254_encrypt_decrypt_roundtrip() {
        let mut rng = StdRng::from_entropy();
        let scheme = SilentThresholdBn::default();
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
    fn bn254_decrypt_not_enough_shares() {
        let mut rng = StdRng::from_entropy();
        let scheme = SilentThresholdBn::default();
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
    fn bn254_decrypt_selector_mismatch() {
        let mut rng = StdRng::from_entropy();
        let scheme = SilentThresholdBn::default();
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
    fn bn254_decrypt_duplicate_partial() {
        let mut rng = StdRng::from_entropy();
        let scheme = SilentThresholdBn::default();
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
