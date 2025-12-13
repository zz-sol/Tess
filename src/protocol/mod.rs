use core::{
    fmt::Debug,
    marker::PhantomData,
    ops::{Sub, SubAssign},
};

use blake3::Hasher;
use rand_core::RngCore;

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

const PAYLOAD_KDF_DOMAIN: &[u8] = b"tess::threshold::payload";

pub trait ProtocolScalar: LagrangeField + SubAssign + Sub<Output = Self> {}

impl<T> ProtocolScalar for T where T: LagrangeField + SubAssign + Sub<Output = T> {}

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
    pub commitment_params: CommitmentParams<B>,
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

/// Bundle returned by key generation.
#[derive(Clone, Debug)]
pub struct KeyMaterial<B: PairingBackend> {
    pub secret_keys: Vec<SecretKey<B>>,
    pub public_keys: Vec<PublicKey<B>>,
    pub aggregate_key: AggregateKey<B>,
    pub kzg_params: CommitmentParams<B>,
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
    fn keygen<R: RngCore + ?Sized>(
        &self,
        rng: &mut R,
        params: &ThresholdParameters,
    ) -> Result<KeyMaterial<B>, Error> {
        params.validate()?;
        Self::ensure_backend(params)?;
        let parties = params.parties;
        let tau = if let Some(bytes) = params.kzg_tau.as_ref() {
            B::parse_tau(bytes)?
        } else {
            B::sample_tau(rng)
        };
        let kzg_params = B::PolynomialCommitment::setup(parties, &tau).map_err(Error::Backend)?;
        let lagranges = B::lagrange_polynomials(parties).map_err(Error::Backend)?;
        // Ensure domain availability early to surface errors.
        B::domain_new(parties).map_err(Error::Backend)?;
        let secret_keys = Self::generate_secret_keys(rng, parties);
        let public_keys = secret_keys
            .iter()
            .map(|sk| {
                derive_public_key::<B>(sk.participant_id, sk, &lagranges, parties, &kzg_params)
            })
            .collect::<Result<Vec<_>, BackendError>>()
            .map_err(Error::Backend)?;
        let aggregate_key = aggregate_public_key::<B>(&public_keys, &kzg_params, parties)?;
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
        public_keys: &[PublicKey<B>],
    ) -> Result<AggregateKey<B>, Error> {
        params.validate()?;
        Self::ensure_backend(params)?;
        let tau = load_tau_from_params::<B>(params)?;
        let kzg_params =
            B::PolynomialCommitment::setup(params.parties, &tau).map_err(Error::Backend)?;
        aggregate_public_key::<B>(public_keys, &kzg_params, params.parties)
    }

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

fn load_tau_from_params<B: ProtocolBackend>(
    params: &ThresholdParameters,
) -> Result<B::Scalar, Error> {
    let bytes = params
        .kzg_tau
        .as_ref()
        .ok_or_else(|| Error::InvalidConfig("missing trusted tau in parameters".into()))?;
    B::parse_tau(bytes)
}

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
        } else {
            points.push(omega.clone());
        }
    }

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
        let params = sample_params(backend);
        let km = scheme.keygen(&mut rng, &params).expect("keygen");
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
        let params = sample_params(backend);
        let km = scheme.keygen(&mut rng, &params).expect("keygen");
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
        let params = sample_params(backend);
        let km = scheme.keygen(&mut rng, &params).expect("keygen");
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
        let params = sample_params(backend);
        let km = scheme.keygen(&mut rng, &params).expect("keygen");
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
