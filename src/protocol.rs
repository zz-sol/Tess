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
}

/// Ciphertext produced by the silent threshold encryption scheme.
#[derive(Clone, Debug)]
pub struct Ciphertext<B: PairingBackend> {
    pub gamma_g2: B::G2,
    pub proof_g1: Vec<B::G1>,
    pub proof_g2: Vec<B::G2>,
    pub shared_secret: B::Target,
    pub threshold: usize,
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

pub mod arkworks {
    use super::*;
    use ark_bls12_381::Fr as BlsFr;
    use ark_poly::univariate::DensePolynomial;
    use ark_poly::{DenseUVPolynomial, EvaluationDomain, Radix2EvaluationDomain};
    use rand_core::RngCore;

    use crate::arkworks_backend::{ArkG1, ArkG2, ArkworksBls12, BlsKzg, BlsPowers};
    use crate::backend::CurvePoint;
    use crate::errors::{BackendError, Error};
    use crate::lagrange::lagrange_polys;
    use ark_std::{UniformRand, Zero};

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
                .map(|_| SecretKey {
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
            let parties = params.parties;

            let tau = self.sample_tau(rng);
            let kzg_params = BlsKzg::setup(parties, &tau).map_err(|err| Error::Backend(err))?;

            let lagranges = lagrange_polys::<BlsFr>(parties).map_err(Error::Backend)?;
            let domain = Radix2EvaluationDomain::new(parties)
                .ok_or_else(|| Error::Backend(BackendError::Math("invalid evaluation domain")))?;

            let secret_keys = self.generate_secret_keys(rng, parties);
            let public_keys = secret_keys
                .iter()
                .enumerate()
                .map(|(idx, sk)| derive_public_key(idx, sk, &lagranges, domain, &kzg_params))
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
            if params_cfg.parties != public_keys.len() {
                return Err(Error::InvalidConfig(
                    "public key list must match parties".into(),
                ));
            }
            Err(Error::Backend(BackendError::UnsupportedFeature(
                "aggregate_public_key without KZG params not yet supported",
            )))
        }

        fn encrypt<R: RngCore + ?Sized>(
            &self,
            _rng: &mut R,
            _agg_key: &AggregateKey<ArkworksBls12>,
            _params: &ThresholdParameters,
            _payload: &[u8],
        ) -> Result<Ciphertext<ArkworksBls12>, Error> {
            Err(Error::Backend(BackendError::UnsupportedFeature(
                "encryption not implemented",
            )))
        }

        fn partial_decrypt(
            &self,
            _secret_key: &SecretKey<ArkworksBls12>,
            _ciphertext: &Ciphertext<ArkworksBls12>,
        ) -> Result<PartialDecryption<ArkworksBls12>, Error> {
            Err(Error::Backend(BackendError::UnsupportedFeature(
                "partial decryption not implemented",
            )))
        }

        fn aggregate_decrypt(
            &self,
            _ciphertext: &Ciphertext<ArkworksBls12>,
            _partials: &[PartialDecryption<ArkworksBls12>],
            _selector: &[bool],
            _agg_key: &AggregateKey<ArkworksBls12>,
        ) -> Result<DecryptionResult<ArkworksBls12>, Error> {
            Err(Error::Backend(BackendError::UnsupportedFeature(
                "aggregate decryption not implemented",
            )))
        }
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
        })
    }
}
