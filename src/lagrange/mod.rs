#[cfg(feature = "ark_bls12381")]
pub mod ark_bls12_381;
#[cfg(feature = "ark_bn254")]
pub mod ark_bn254;
#[cfg(feature = "blst")]
pub mod blst_bls12_381;

#[cfg(any(feature = "ark_bls12381", feature = "ark_bn254"))]
mod shared {
    use ark_ff::{FftField, Field, batch_inversion};
    use ark_poly::{
        DenseUVPolynomial, EvaluationDomain, Evaluations, Radix2EvaluationDomain,
        univariate::DensePolynomial,
    };

    use crate::errors::BackendError;

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
        let domain: Radix2EvaluationDomain<F> = Radix2EvaluationDomain::new(n)
            .ok_or(BackendError::Math("invalid evaluation domain"))?;
        let mut evals = vec![F::zero(); n];
        evals[index] = F::one();
        let evaluations = Evaluations::from_vec_and_domain(evals, domain);
        Ok(evaluations.interpolate())
    }

    pub fn lagrange_polys<F: FftField>(n: usize) -> Result<Vec<DensePolynomial<F>>, BackendError> {
        if !n.is_power_of_two() {
            return Err(BackendError::Math("domain size must be a power of two"));
        }
        let domain: Radix2EvaluationDomain<F> = Radix2EvaluationDomain::new(n)
            .ok_or(BackendError::Math("invalid evaluation domain"))?;
        let omega_inv = domain
            .group_gen
            .inverse()
            .ok_or(BackendError::Math("invalid group generator"))?;
        let n_scalar = F::from(n as u64);

        let mut omega_inv_pows = Vec::with_capacity(n);
        let mut cur = F::one();
        for _ in 0..n {
            omega_inv_pows.push(cur);
            cur *= omega_inv;
        }

        let mut denominators: Vec<F> = omega_inv_pows.iter().map(|w| *w * n_scalar).collect();
        batch_inversion(&mut denominators);

        let mut polys = Vec::with_capacity(n);
        for (omega_i_inv, denom_inv) in omega_inv_pows.iter().zip(denominators.iter()) {
            let mut coeffs = Vec::with_capacity(n);
            let mut power = *omega_i_inv;
            for _ in 0..n {
                coeffs.push(power * denom_inv);
                power *= *omega_i_inv;
            }
            polys.push(DensePolynomial::from_coefficients_vec(coeffs));
        }

        Ok(polys)
    }

    pub fn interp_mostly_zero<F: Field>(
        eval: F,
        points: &[F],
    ) -> Result<DensePolynomial<F>, BackendError> {
        if points.is_empty() {
            return Ok(DensePolynomial::from_coefficients_vec(vec![F::one()]));
        }

        let mut coeffs = vec![F::one()];
        for &point in points.iter().skip(1) {
            let neg_point = -point;
            coeffs.push(F::zero());
            for i in (0..coeffs.len() - 1).rev() {
                let (head, tail) = coeffs.split_at_mut(i + 1);
                let coef = &mut head[i];
                let next = &mut tail[0];
                *next += *coef;
                *coef *= neg_point;
            }
        }

        let mut scale = *coeffs.last().unwrap();
        for coeff in coeffs.iter().rev().skip(1) {
            scale = scale * points[0] + coeff;
        }
        let scale_inv = scale
            .inverse()
            .ok_or_else(|| BackendError::Math("interpolation scale inversion failed"))?;

        for coeff in coeffs.iter_mut() {
            *coeff *= eval * scale_inv;
        }

        Ok(DensePolynomial::from_coefficients_vec(coeffs))
    }
}

#[cfg(any(feature = "ark_bls12381", feature = "ark_bn254"))]
macro_rules! impl_lagrange_backend {
    ($field:ty) => {
        use ark_poly::univariate::DensePolynomial;

        use crate::errors::BackendError;

        pub fn lagrange_poly(
            n: usize,
            index: usize,
        ) -> Result<DensePolynomial<$field>, BackendError> {
            super::shared::lagrange_poly::<$field>(n, index)
        }

        pub fn lagrange_polys(n: usize) -> Result<Vec<DensePolynomial<$field>>, BackendError> {
            super::shared::lagrange_polys::<$field>(n)
        }

        pub fn interp_mostly_zero(
            eval: $field,
            points: &[$field],
        ) -> Result<DensePolynomial<$field>, BackendError> {
            super::shared::interp_mostly_zero(eval, points)
        }
    };
}

#[cfg(any(feature = "ark_bls12381", feature = "ark_bn254"))]
pub(super) use impl_lagrange_backend;
