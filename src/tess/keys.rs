use core::fmt::Debug;

use tracing::instrument;

use crate::{
    DensePolynomial, FieldElement, Fr, KZG, PairingBackend, Params, Polynomial, SRS,
    arith::CurvePoint,
    errors::{BackendError, Error},
    kzg::PolynomialCommitment,
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
#[derive(Clone, Debug)]
pub struct AggregateKey<B: PairingBackend<Scalar = Fr>> {
    pub public_keys: Vec<PublicKey<B>>,
    pub ask: B::G1,
    pub z_g2: B::G2,
    pub lagrange_row_sums: Vec<B::G1>,
    pub precomputed_pairing: B::Target,
}

impl<B: PairingBackend<Scalar = Fr>> AggregateKey<B> {
    #[instrument(level = "trace", skip_all, fields(parties, num_keys = public_keys.len()))]
    pub(crate) fn aggregate_keys(
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
pub(crate) fn derive_public_key<B: PairingBackend<Scalar = Fr>>(
    participant_id: usize,
    sk: &SecretKey<B>,
    params: &Params<B>,
) -> Result<PublicKey<B>, BackendError> {
    let lagrange_li = params
        .lagrange_powers
        .li
        .get(participant_id)
        .ok_or(BackendError::Math("missing lagrange power"))?
        .mul_scalar(&sk.scalar);

    let lagrange_li_minus0 = params
        .lagrange_powers
        .li_minus0
        .get(participant_id)
        .ok_or(BackendError::Math("missing lagrange power minus0"))?
        .mul_scalar(&sk.scalar);

    let lagrange_li_x = params
        .lagrange_powers
        .li_x
        .get(participant_id)
        .ok_or(BackendError::Math("missing lagrange power x"))?
        .mul_scalar(&sk.scalar);

    let domain_size = params.lagrange_polys.len();
    if domain_size == 0 {
        return Err(BackendError::Math("missing lagrange polynomials"));
    }

    let li_poly = params
        .lagrange_polys
        .get(participant_id)
        .ok_or(BackendError::Math("missing lagrange polynomial"))?
        .clone();

    let mut sk_li_lj_z = Vec::with_capacity(domain_size);
    for j in 0..domain_size {
        let other_poly = params
            .lagrange_polys
            .get(j)
            .ok_or(BackendError::Math("missing lagrange polynomial"))?;
        let num = if participant_id == j {
            poly_sub(&naive_poly_mul(&li_poly, &li_poly), &li_poly)
        } else {
            naive_poly_mul(&li_poly, other_poly)
        };

        let quotient = divide_by_vanishing_poly(&num, domain_size);
        let sk_times_f = scale_poly(&quotient, &sk.scalar);
        let commitment = KZG::commit_g1(&params.srs, &sk_times_f)?;
        sk_li_lj_z.push(commitment);
    }

    let bls_pk = B::G1::generator().mul_scalar(&sk.scalar);

    Ok(PublicKey {
        participant_id,
        bls_key: bls_pk,
        lagrange_li,
        lagrange_li_minus0,
        lagrange_li_x,
        lagrange_li_lj_z: sk_li_lj_z,
    })
}

fn naive_poly_mul(lhs: &DensePolynomial, rhs: &DensePolynomial) -> DensePolynomial {
    let left = lhs.coeffs();
    let right = rhs.coeffs();
    if left.is_empty() || right.is_empty() {
        return DensePolynomial::from_coefficients_vec(vec![Fr::zero()]);
    }

    let mut product = vec![Fr::zero(); left.len() + right.len() - 1];
    for (i, a) in left.iter().enumerate() {
        for (j, b) in right.iter().enumerate() {
            let term = (*a) * (*b);
            product[i + j] += term;
        }
    }

    DensePolynomial::from_coefficients_vec(product)
}

fn poly_sub(lhs: &DensePolynomial, rhs: &DensePolynomial) -> DensePolynomial {
    let left = lhs.coeffs();
    let right = rhs.coeffs();
    let max_len = usize::max(left.len(), right.len());
    let mut coeffs = Vec::with_capacity(max_len);
    for i in 0..max_len {
        let l = left.get(i).cloned().unwrap_or_else(Fr::zero);
        let r = right.get(i).cloned().unwrap_or_else(Fr::zero);
        coeffs.push(l - r);
    }
    DensePolynomial::from_coefficients_vec(coeffs)
}

fn divide_by_vanishing_poly(poly: &DensePolynomial, domain_size: usize) -> DensePolynomial {
    let coeffs = poly.coeffs();
    if coeffs.len() <= domain_size {
        return DensePolynomial::from_coefficients_vec(vec![Fr::zero()]);
    }

    let mut quotient = coeffs[domain_size..].to_vec();
    let len = coeffs.len();
    for i in 1..(len / domain_size) {
        let start = domain_size * (i + 1);
        if start >= len {
            break;
        }

        let remainder = &coeffs[start..];
        let limit = quotient.len().min(remainder.len());
        for idx in 0..limit {
            quotient[idx] += remainder[idx];
        }
    }

    DensePolynomial::from_coefficients_vec(quotient)
}

fn scale_poly(poly: &DensePolynomial, scalar: &Fr) -> DensePolynomial {
    let scaled = poly
        .coeffs()
        .iter()
        .map(|coeff| (*coeff) * (*scalar))
        .collect();
    DensePolynomial::from_coefficients_vec(scaled)
}
