use ark_ff::FftField;
use ark_poly::{
    EvaluationDomain, Evaluations, Radix2EvaluationDomain, univariate::DensePolynomial,
};

use crate::errors::BackendError;

/// Return the i-th Lagrange basis polynomial over a radix-2 domain of size `n`.
pub fn lagrange_poly<F: FftField>(
    n: usize,
    index: usize,
) -> Result<DensePolynomial<F>, BackendError> {
    if index >= n {
        return Err(BackendError::Math("lagrange index out of range"));
    }
    if !n.is_power_of_two() {
        return Err(BackendError::Math("domain size must be a power of two"));
    }

    let domain =
        Radix2EvaluationDomain::new(n).ok_or(BackendError::Math("invalid evaluation domain"))?;
    let mut evals = vec![F::zero(); n];
    evals[index] = F::one();

    let evaluations = Evaluations::from_vec_and_domain(evals, domain);
    Ok(evaluations.interpolate())
}

/// Compute every Lagrange basis polynomial on an n-point radix-2 domain.
pub fn lagrange_polys<F: FftField>(n: usize) -> Result<Vec<DensePolynomial<F>>, BackendError> {
    (0..n).map(|i| lagrange_poly(n, i)).collect()
}
