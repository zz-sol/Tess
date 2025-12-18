use rayon::iter::{IntoParallelRefIterator, ParallelIterator};

use crate::CurvePoint;
use crate::{
    BackendError, DensePolynomial, FieldElement, Fr, PairingBackend, PolynomialCommitment,
};

#[derive(Debug)]
pub struct KZG;

#[derive(Debug)]
pub struct SRS<B: PairingBackend<Scalar = Fr>> {
    pub powers_of_g: Vec<B::G1>,
    pub powers_of_h: Vec<B::G2>,
    pub e_gh: B::Target,
}

impl<B: PairingBackend<Scalar = Fr>> Clone for SRS<B>
where
    B::G1: Clone,
    B::G2: Clone,
    B::Target: Clone,
{
    fn clone(&self) -> Self {
        Self {
            powers_of_g: self.powers_of_g.clone(),
            powers_of_h: self.powers_of_h.clone(),
            e_gh: self.e_gh.clone(),
        }
    }
}

impl<B: PairingBackend<Scalar = Fr>> SRS<B> {
    /// Creates a new SRS with precomputed Lagrange commitments
    pub fn new(tau: &B::Scalar, parties: usize) -> Result<Self, String> {
        KZG::setup(parties, tau).map_err(|e| format!("SRS setup failed: {:?}", e))
    }
}

impl<B: PairingBackend<Scalar = Fr>> PolynomialCommitment<B> for KZG {
    type Parameters = SRS<B>;
    type Polynomial = DensePolynomial;

    fn setup(max_degree: usize, tau: &Fr) -> Result<Self::Parameters, BackendError> {
        // Construct powers-of-tau parameters for KZG commitments.
        setup_powers_bls::<B>(max_degree, tau)
    }

    fn commit_g1(
        params: &Self::Parameters,
        polynomial: &Self::Polynomial,
    ) -> Result<B::G1, BackendError> {
        let degree = polynomial.degree();
        if degree + 1 > params.powers_of_g.len() {
            return Err(BackendError::Math("polynomial degree too large"));
        }
        let scalars = &polynomial.coeffs[..=degree];
        let mut acc = B::G1::identity();
        for (base, scalar) in params.powers_of_g[..=degree].iter().zip(scalars.iter()) {
            acc = acc.add(&base.mul_scalar(scalar));
        }
        Ok(acc)
    }

    fn commit_g2(
        params: &Self::Parameters,
        polynomial: &Self::Polynomial,
    ) -> Result<B::G2, BackendError> {
        let degree = polynomial.degree();
        if degree + 1 > params.powers_of_h.len() {
            return Err(BackendError::Math("polynomial degree too large"));
        }
        let scalars = &polynomial.coeffs[..=degree];
        let mut acc = B::G2::identity();
        for (base, scalar) in params.powers_of_h[..=degree].iter().zip(scalars.iter()) {
            acc = acc.add(&base.mul_scalar(scalar));
        }
        Ok(acc)
    }
}

fn setup_powers_bls<B: PairingBackend<Scalar = Fr>>(
    max_degree: usize,
    tau: &B::Scalar,
) -> Result<SRS<B>, BackendError> {
    if max_degree < 1 {
        return Err(BackendError::Math("degree must be >= 1"));
    }

    let g = B::G1::generator();
    let h = B::G2::generator();

    let mut powers_of_tau = vec![<B::Scalar as FieldElement>::one()];
    let mut cur = *tau;
    for _ in 0..max_degree {
        powers_of_tau.push(cur);
        cur *= tau;
    }

    let powers_of_g: Vec<B::G1> = powers_of_tau
        .par_iter()
        .map(|power| g.mul_scalar(power))
        .collect();

    let powers_of_h: Vec<B::G2> = powers_of_tau
        .par_iter()
        .map(|power| h.mul_scalar(power))
        .collect();

    let e_gh = B::pairing(&g, &h);

    Ok(SRS {
        powers_of_g,
        powers_of_h,
        e_gh,
    })
}
