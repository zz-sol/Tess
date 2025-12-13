use super::*;
use blake3::Hasher;
use blstrs::Scalar;
use rand_core::RngCore;

use crate::backend::{
    BlstBackend, BlstG1, BlstG2, BlstGt, BlstKzg, BlstMsm, BlstPowers, CurvePoint, DensePolynomial,
    EvaluationDomain, MsmProvider, Radix2EvaluationDomain, TargetGroup,
};
use crate::config::{BackendId, CurveId};
use crate::errors::{BackendError, Error};
use crate::lagrange::blst_bls12_381::{interp_mostly_zero, lagrange_polys};
use ff::Field;

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
pub struct SilentThresholdBlst;

impl SilentThresholdBlst {
    fn sample_tau<R: RngCore + ?Sized>(&self, rng: &mut R) -> Scalar {
        Scalar::random(&mut *rng)
    }

    fn generate_secret_keys<R: RngCore + ?Sized>(
        &self,
        rng: &mut R,
        parties: usize,
    ) -> Vec<SecretKey<BlstBackend>> {
        let mut keys = Vec::with_capacity(parties);
        for participant_id in 0..parties {
            keys.push(SecretKey {
                participant_id,
                scalar: Scalar::random(&mut *rng),
            });
        }
        keys
    }
}

impl ThresholdScheme<BlstBackend> for SilentThresholdBlst {
    fn keygen<R: RngCore + ?Sized>(
        &self,
        rng: &mut R,
        params: &ThresholdParameters,
    ) -> Result<KeyMaterial<BlstBackend>, Error> {
        params.validate()?;
        ensure_supported_blst(params)?;
        let parties = params.parties;
        let tau = if let Some(bytes) = params.kzg_tau.as_ref() {
            parse_tau(bytes)?
        } else {
            self.sample_tau(rng)
        };
        let kzg_params = BlstKzg::setup(parties, &tau).map_err(Error::Backend)?;
        let lagranges = lagrange_polys(parties).map_err(Error::Backend)?;
        let domain = Radix2EvaluationDomain::new(parties)
            .ok_or_else(|| Error::Backend(BackendError::Math("invalid evaluation domain")))?;
        let secret_keys = self.generate_secret_keys(rng, parties);
        let public_keys = secret_keys
            .iter()
            .map(|sk| {
                derive_public_key(
                    sk.participant_id,
                    sk,
                    &lagranges,
                    domain.clone(),
                    &kzg_params,
                )
            })
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
        public_keys: &[PublicKey<BlstBackend>],
    ) -> Result<AggregateKey<BlstBackend>, Error> {
        params.validate()?;
        ensure_supported_blst(params)?;
        let tau = load_tau_from_params(params)?;
        let kzg_params = BlstKzg::setup(params.parties, &tau).map_err(Error::Backend)?;
        aggregate_public_key(public_keys, &kzg_params, params.parties)
    }

    fn encrypt<R: RngCore + ?Sized>(
        &self,
        rng: &mut R,
        agg_key: &AggregateKey<BlstBackend>,
        params: &ThresholdParameters,
        payload: &[u8],
    ) -> Result<Ciphertext<BlstBackend>, Error> {
        params.validate()?;
        ensure_supported_blst(params)?;
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
        let g = BlstG1::from_affine(
            kzg_params
                .powers_of_g
                .get(0)
                .ok_or(BackendError::Math("missing g generator"))?,
        );
        let g_tau_t = BlstG1::from_affine(
            kzg_params
                .powers_of_g
                .get(threshold + 1)
                .ok_or(BackendError::Math("missing g^{tau^{t+1}}"))?,
        );
        let h = BlstG2::from_affine(
            kzg_params
                .powers_of_h
                .get(0)
                .ok_or(BackendError::Math("missing h generator"))?,
        );
        let h_tau = BlstG2::from_affine(
            kzg_params
                .powers_of_h
                .get(1)
                .ok_or(BackendError::Math("missing h^tau"))?,
        );
        let h_minus_one = BlstG2::generator().negate();

        let gamma = Scalar::random(&mut *rng);
        let gamma_g2 = h.mul_scalar(&gamma);

        let s0 = Scalar::random(&mut *rng);
        let s1 = Scalar::random(&mut *rng);
        let s2 = Scalar::random(&mut *rng);
        let s3 = Scalar::random(&mut *rng);
        let s4 = Scalar::random(&mut *rng);

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

        let payload_ct = encrypt_payload::<BlstBackend>(&shared_secret, payload);

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
        secret_key: &SecretKey<BlstBackend>,
        ciphertext: &Ciphertext<BlstBackend>,
    ) -> Result<PartialDecryption<BlstBackend>, Error> {
        let response = ciphertext.gamma_g2.mul_scalar(&secret_key.scalar);
        Ok(PartialDecryption {
            participant_id: secret_key.participant_id,
            response,
        })
    }

    fn aggregate_decrypt(
        &self,
        ciphertext: &Ciphertext<BlstBackend>,
        partials: &[PartialDecryption<BlstBackend>],
        selector: &[bool],
        agg_key: &AggregateKey<BlstBackend>,
    ) -> Result<DecryptionResult<BlstBackend>, Error> {
        aggregate_decrypt(ciphertext, partials, selector, agg_key)
    }
}

fn ensure_supported_blst(params: &ThresholdParameters) -> Result<(), Error> {
    if params.backend.backend != BackendId::Blst {
        return Err(Error::Backend(BackendError::UnsupportedFeature(
            "SilentThresholdBlst is only available for the blstrs backend",
        )));
    }
    if params.backend.curve != CurveId::Bls12_381 {
        return Err(Error::Backend(BackendError::UnsupportedCurve(
            "SilentThresholdBlst targets BLS12-381",
        )));
    }
    Ok(())
}

fn parse_tau(bytes: &[u8]) -> Result<Scalar, Error> {
    if bytes.len() != 32 {
        return Err(Error::InvalidConfig(
            "trusted tau must be 32 bytes for blst scalar".into(),
        ));
    }
    let mut arr = [0u8; 32];
    arr.copy_from_slice(bytes);
    Option::<Scalar>::from(Scalar::from_bytes_be(&arr))
        .ok_or_else(|| Error::InvalidConfig("invalid trusted tau encoding".into()))
}

fn load_tau_from_params(params: &ThresholdParameters) -> Result<Scalar, Error> {
    let bytes = params
        .kzg_tau
        .as_ref()
        .ok_or_else(|| Error::InvalidConfig("missing trusted tau in parameters".into()))?;
    parse_tau(bytes)
}

fn scale_poly(poly: &DensePolynomial, scalar: Scalar) -> DensePolynomial {
    DensePolynomial::from_coefficients_vec(poly.coeffs.iter().map(|c| *c * scalar).collect())
}

fn sub_poly(a: &DensePolynomial, b: &DensePolynomial) -> DensePolynomial {
    let len = a.coeffs.len().max(b.coeffs.len());
    let mut coeffs = vec![Scalar::ZERO; len];
    for (i, coeff) in a.coeffs.iter().enumerate() {
        coeffs[i] += *coeff;
    }
    for (i, coeff) in b.coeffs.iter().enumerate() {
        coeffs[i] -= *coeff;
    }
    DensePolynomial::from_coefficients_vec(coeffs)
}

fn mul_poly(a: &DensePolynomial, b: &DensePolynomial) -> DensePolynomial {
    if a.coeffs.is_empty() || b.coeffs.is_empty() {
        return DensePolynomial::from_coefficients_vec(vec![Scalar::ZERO]);
    }
    let mut coeffs = vec![Scalar::ZERO; a.coeffs.len() + b.coeffs.len() - 1];
    for (i, coeff_a) in a.coeffs.iter().enumerate() {
        for (j, coeff_b) in b.coeffs.iter().enumerate() {
            coeffs[i + j] += *coeff_a * *coeff_b;
        }
    }
    DensePolynomial::from_coefficients_vec(coeffs)
}

fn divide_by_linear(poly: &DensePolynomial, root: Scalar) -> (DensePolynomial, Scalar) {
    assert!(poly.coeffs.len() > 1, "cannot divide constant polynomial");
    let mut quotient = vec![Scalar::ZERO; poly.coeffs.len() - 1];
    let mut carry = *poly.coeffs.last().unwrap();
    for (idx, coeff) in poly.coeffs.iter().rev().skip(1).enumerate() {
        let pos = quotient.len() - 1 - idx;
        quotient[pos] = carry;
        carry = *coeff + root * carry;
    }
    (DensePolynomial::from_coefficients_vec(quotient), carry)
}

fn divide_by_vanishing(
    poly: &DensePolynomial,
    domain_size: usize,
) -> (DensePolynomial, DensePolynomial) {
    if poly.coeffs.len() <= domain_size {
        return (
            DensePolynomial::from_coefficients_vec(vec![Scalar::ZERO]),
            poly.clone(),
        );
    }
    let mut coeffs = poly.coeffs.clone();
    let mut quotient = vec![Scalar::ZERO; coeffs.len() - domain_size];
    while coeffs.len() > domain_size {
        let d = coeffs.len() - 1;
        let lead = coeffs[d];
        let q_idx = d - domain_size;
        quotient[q_idx] = lead;
        coeffs.pop();
        coeffs[q_idx] += lead;
        while coeffs.last() == Some(&Scalar::ZERO) && coeffs.len() > 0 {
            coeffs.pop();
        }
    }
    if coeffs.is_empty() {
        coeffs.push(Scalar::ZERO);
    }
    (
        DensePolynomial::from_coefficients_vec(quotient),
        DensePolynomial::from_coefficients_vec(coeffs),
    )
}

fn derive_public_key(
    participant_id: usize,
    sk: &SecretKey<BlstBackend>,
    lagranges: &[DensePolynomial],
    domain: Radix2EvaluationDomain,
    params: &BlstPowers,
) -> Result<PublicKey<BlstBackend>, BackendError> {
    let li = lagranges
        .get(participant_id)
        .ok_or(BackendError::Math("missing lagrange polynomial"))?;

    let li_poly = li.clone();
    let sk_li_poly = scale_poly(&li_poly, sk.scalar);
    let lagrange_li = BlstKzg::commit_g1(params, &sk_li_poly)?;

    let mut minus0_poly = sk_li_poly.clone();
    if let Some(constant) = minus0_poly.coeffs.get_mut(0) {
        *constant = Scalar::ZERO;
    }
    let lagrange_li_minus0 = BlstKzg::commit_g1(params, &minus0_poly)?;

    let shift_coeffs = if li.coeffs.len() > 1 {
        li.coeffs[1..].to_vec()
    } else {
        vec![Scalar::ZERO]
    };
    let shift_poly = DensePolynomial::from_coefficients_vec(shift_coeffs);
    let sk_shift_poly = scale_poly(&shift_poly, sk.scalar);
    let lagrange_li_x = BlstKzg::commit_g1(params, &sk_shift_poly)?;

    let mut lagrange_li_lj_z = Vec::with_capacity(lagranges.len());
    for (idx, lj) in lagranges.iter().enumerate() {
        let numerator = if idx == participant_id {
            sub_poly(&mul_poly(&li_poly, &li_poly), &li_poly)
        } else {
            mul_poly(lj, &li_poly)
        };
        let (f, remainder) = divide_by_vanishing(&numerator, domain.size);
        if remainder.coeffs.iter().any(|c| *c != Scalar::ZERO) {
            return Err(BackendError::Math(
                "division by vanishing polynomial failed",
            ));
        }
        let scaled = scale_poly(&f, sk.scalar);
        let commitment = BlstKzg::commit_g1(params, &scaled)?;
        lagrange_li_lj_z.push(commitment);
    }

    Ok(PublicKey {
        participant_id,
        bls_key: BlstG1::generator().mul_scalar(&sk.scalar),
        lagrange_li,
        lagrange_li_minus0,
        lagrange_li_x,
        lagrange_li_lj_z,
    })
}

fn aggregate_public_key(
    public_keys: &[PublicKey<BlstBackend>],
    params: &BlstPowers,
    parties: usize,
) -> Result<AggregateKey<BlstBackend>, Error> {
    if public_keys.is_empty() {
        return Err(Error::InvalidConfig(
            "cannot aggregate empty public key set".into(),
        ));
    }
    if public_keys.len() != parties {
        return Err(Error::InvalidConfig("public key length mismatch".into()));
    }

    let mut ask = BlstG1::identity();
    for pk in public_keys {
        ask = ask.add(&pk.lagrange_li);
    }

    let mut lagrange_row_sums = vec![BlstG1::identity(); parties];
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
    let z_g2 = BlstG2::from_affine(g2_tau_n).sub(&BlstG2::generator());

    Ok(AggregateKey {
        public_keys: public_keys.to_vec(),
        ask,
        z_g2,
        lagrange_row_sums,
        precomputed_pairing: BlstGt(params.e_gh.clone()),
        commitment_params: params.clone(),
    })
}

fn aggregate_decrypt(
    ciphertext: &Ciphertext<BlstBackend>,
    partials: &[PartialDecryption<BlstBackend>],
    selector: &[bool],
    agg_key: &AggregateKey<BlstBackend>,
) -> Result<DecryptionResult<BlstBackend>, Error> {
    let n = agg_key.public_keys.len();
    if selector.len() != n {
        return Err(Error::SelectorMismatch {
            expected: n,
            actual: selector.len(),
        });
    }

    let mut responses = vec![BlstG2::identity(); n];
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
    let domain_elements = domain.elements();

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

    let b = interp_mostly_zero(Scalar::ONE, &points).map_err(Error::Backend)?;
    let b_evals = domain.fft(&b.coeffs);

    let b_g2 = BlstKzg::commit_g2(&agg_key.commitment_params, &b).map_err(Error::Backend)?;

    let mut b_minus_one = b.clone();
    if let Some(constant) = b_minus_one.coeffs.get_mut(0) {
        *constant -= Scalar::ONE;
    }
    let (q0, remainder) = divide_by_linear(&b_minus_one, domain_elements[0]);
    if remainder != Scalar::ZERO {
        return Err(Error::Backend(BackendError::Math(
            "division by linear failed",
        )));
    }
    let q0_g1 = BlstKzg::commit_g1(&agg_key.commitment_params, &q0).map_err(Error::Backend)?;

    let mut bhat_coeffs = vec![Scalar::ZERO; ciphertext.threshold + 1];
    bhat_coeffs.extend_from_slice(&b.coeffs);
    let bhat = DensePolynomial::from_coefficients_vec(bhat_coeffs);
    let bhat_g1 = BlstKzg::commit_g1(&agg_key.commitment_params, &bhat).map_err(Error::Backend)?;

    let n_inv = Option::<Scalar>::from(Scalar::from(n as u64).invert())
        .ok_or_else(|| Error::Backend(BackendError::Math("domain size inversion failed")))?;

    let scalars: Vec<Scalar> = parties.iter().map(|&i| b_evals[i]).collect();

    let apk = if scalars.is_empty() {
        BlstG1::identity()
    } else {
        let bases: Vec<BlstG1> = parties
            .iter()
            .map(|&i| agg_key.public_keys[i].bls_key.clone())
            .collect();
        BlstMsm::msm_g1(&bases, &scalars)
            .map_err(Error::Backend)?
            .mul_scalar(&n_inv)
    };

    let sigma = if scalars.is_empty() {
        BlstG2::identity()
    } else {
        let bases: Vec<BlstG2> = parties.iter().map(|&i| responses[i].clone()).collect();
        BlstMsm::msm_g2(&bases, &scalars)
            .map_err(Error::Backend)?
            .mul_scalar(&n_inv)
    };

    let qx = if scalars.is_empty() {
        BlstG1::identity()
    } else {
        let bases: Vec<BlstG1> = parties
            .iter()
            .map(|&i| agg_key.public_keys[i].lagrange_li_x.clone())
            .collect();
        BlstMsm::msm_g1(&bases, &scalars).map_err(Error::Backend)?
    };

    let qz = if scalars.is_empty() {
        BlstG1::identity()
    } else {
        let bases: Vec<BlstG1> = parties
            .iter()
            .map(|&i| agg_key.lagrange_row_sums[i].clone())
            .collect();
        BlstMsm::msm_g1(&bases, &scalars).map_err(Error::Backend)?
    };

    let qhatx = if scalars.is_empty() {
        BlstG1::identity()
    } else {
        let bases: Vec<BlstG1> = parties
            .iter()
            .map(|&i| agg_key.public_keys[i].lagrange_li_minus0.clone())
            .collect();
        BlstMsm::msm_g1(&bases, &scalars).map_err(Error::Backend)?
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

    let shared_secret = BlstBackend::multi_pairing(&lhs, &rhs).map_err(Error::Backend)?;
    let plaintext = if ciphertext.payload.is_empty() {
        None
    } else {
        Some(decrypt_payload::<BlstBackend>(
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
            backend: BackendConfig::new(BackendId::Blst, CurveId::Bls12_381),
            kzg_tau: None,
        }
    }

    #[test]
    fn blst_encrypt_decrypt_roundtrip() {
        let mut rng = StdRng::from_entropy();
        let scheme = SilentThresholdBlst::default();
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
    fn blst_decrypt_not_enough_shares() {
        let mut rng = StdRng::from_entropy();
        let scheme = SilentThresholdBlst::default();
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
    fn blst_decrypt_selector_mismatch() {
        let mut rng = StdRng::from_entropy();
        let scheme = SilentThresholdBlst::default();
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
    fn blst_decrypt_duplicate_partial() {
        let mut rng = StdRng::from_entropy();
        let scheme = SilentThresholdBlst::default();
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
