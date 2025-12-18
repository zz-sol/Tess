use rayon::iter::{
    IndexedParallelIterator, IntoParallelIterator, IntoParallelRefIterator, ParallelIterator,
};
use tracing::instrument;

use crate::arith::group::CurvePoint;
use crate::{BackendError, DensePolynomial, FieldElement, Fr, PairingBackend, Polynomial};

/// Precomputed Lagrange polynomial commitments for efficient key derivation.
///
/// This structure contains KZG commitments to various Lagrange polynomial
/// transformations, precomputed during SRS generation to enable efficient
/// key generation without polynomial operations at key generation time.
///
/// # Fields
///
/// - `li`: Commitments to L_i(x) for each participant i
/// - `li_minus0`: Commitments to L_i(x) - L_i(0) for each i
/// - `li_x`: Commitments to x * L_i(x) for each i
/// - `li_lj_z`: Commitments to L_i(x) * L_j(z) for all pairs (i, j)
#[derive(Clone, Debug)]
pub struct LagrangePowers<B: PairingBackend> {
    pub li: Vec<B::G1>,
    pub li_minus0: Vec<B::G1>,
    pub li_x: Vec<B::G1>,
    pub li_lj_z: Vec<Vec<B::G1>>,
}

impl<B: PairingBackend<Scalar = Fr>> LagrangePowers<B> {
    #[instrument(level = "info", skip_all, fields(size=domain_size))]
    pub(crate) fn precompute_lagrange_powers(
        lagranges: &[DensePolynomial],
        domain_size: usize,
        tau: &B::Scalar,
    ) -> Result<Self, BackendError> {
        let n = lagranges.len();

        // Evaluate all Lagrange polynomials at tau
        let li_evals: Vec<B::Scalar> = lagranges
            .iter()
            .map(|li_poly| li_poly.evaluate(tau))
            .collect();

        // Compute tau^n - 1 (the vanishing polynomial evaluated at tau)
        let tau_n = tau.pow(&[domain_size as u64, 0, 0, 0]);
        let z_eval = tau_n - <B::Scalar as FieldElement>::one();
        let z_eval_inv = z_eval
            .invert()
            .ok_or(BackendError::Math("vanishing polynomial at tau is zero"))?;

        let tau_inv = tau
            .invert()
            .ok_or(BackendError::Math("tau must be non-zero"))?;

        // Compute li, li_minus0, and li_x in parallel
        let results: Vec<(B::G1, B::G1, B::G1)> = lagranges
            .par_iter()
            .enumerate()
            .map(|(i, li_poly)| {
                let li_eval = &li_evals[i];

                // li = g * L_i(tau)
                let lagrange_li = B::G1::generator().mul_scalar(li_eval);

                // li_minus0 = g * (L_i(tau) - L_i(0))
                let li_0 = li_poly
                    .coeffs()
                    .first()
                    .cloned()
                    .unwrap_or_else(<B::Scalar as FieldElement>::zero);
                let li_minus0_eval = *li_eval - li_0;
                let lagrange_li_minus0 = B::G1::generator().mul_scalar(&li_minus0_eval);

                // li_x = g * (L_i(tau) - L_i(0)) / tau
                let li_x_eval = li_minus0_eval * tau_inv;
                let lagrange_li_x = B::G1::generator().mul_scalar(&li_x_eval);

                Ok((lagrange_li, lagrange_li_minus0, lagrange_li_x))
            })
            .collect::<Result<Vec<_>, BackendError>>()?;

        let mut li = Vec::with_capacity(n);
        let mut li_minus0 = Vec::with_capacity(n);
        let mut li_x = Vec::with_capacity(n);
        for (a, b, c) in results {
            li.push(a);
            li_minus0.push(b);
            li_x.push(c);
        }

        // Compute li_lj_z using the evaluation-based approach
        let li_lj_z: Vec<Vec<B::G1>> = (0..n)
            .into_par_iter()
            .map(|i| {
                (0..n)
                    .into_par_iter()
                    .map(|j| {
                        let scalar = if i == j {
                            // (L_i(tau)^2 - L_i(tau)) / z(tau)
                            let li_eval: &B::Scalar = &li_evals[i];
                            (*li_eval * *li_eval - *li_eval) * z_eval_inv
                        } else {
                            // (L_i(tau) * L_j(tau)) / z(tau)
                            (li_evals[i] * li_evals[j]) * z_eval_inv
                        };
                        Ok(B::G1::generator().mul_scalar(&scalar))
                    })
                    .collect::<Result<Vec<_>, BackendError>>()
            })
            .collect::<Result<Vec<_>, BackendError>>()?;

        Ok(LagrangePowers {
            li,
            li_minus0,
            li_x,
            li_lj_z,
        })
    }
}

/// Build Lagrange polynomials L_0 ... L_{n-1} for a domain of size `n`.
#[instrument(level = "info", skip_all, fields(num_parties=n))]
pub(crate) fn build_lagrange_polys(n: usize) -> Result<Vec<DensePolynomial>, BackendError> {
    if n == 0 {
        return Ok(Vec::new());
    }

    // Follow the same construction as `lagrange_polys_impl` in arith::lagrange
    let omega = Fr::two_adicity_generator(n);
    let omega_inv = omega
        .invert()
        .ok_or(BackendError::Math("invalid generator inversion"))?;

    // Convert n to a field element
    let n_scalar = Fr::from_u64(n as u64);

    let mut omega_inv_pows = Vec::with_capacity(n);
    let mut cur = Fr::one();
    for _ in 0..n {
        omega_inv_pows.push(cur);
        cur *= omega_inv;
    }

    let mut denominators: Vec<Fr> = omega_inv_pows
        .iter()
        .map(|w| {
            let mut denom = *w;
            denom *= n_scalar;
            denom
        })
        .collect();
    Fr::batch_inversion(&mut denominators)?;

    let mut polys = Vec::with_capacity(n);
    for (omega_i_inv, denom_inv) in omega_inv_pows.iter().zip(denominators.iter()) {
        let mut coeffs = Vec::with_capacity(n);
        let mut power = *omega_i_inv;
        for _ in 0..n {
            let mut term = power;
            term *= *denom_inv;
            coeffs.push(term);
            power *= *omega_i_inv;
        }
        polys.push(DensePolynomial::from_coefficients_vec(coeffs));
    }
    Ok(polys)
}
