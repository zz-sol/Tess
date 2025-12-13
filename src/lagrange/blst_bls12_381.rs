use blstrs::Scalar;
use ff::{Field, PrimeField};

use crate::backend::DensePolynomial;
use crate::errors::BackendError;

const ROOT_OF_UNITY: Scalar = Scalar::ROOT_OF_UNITY;
const TWO_ADICITY: u32 = Scalar::S;

fn ensure_domain_size(n: usize) -> Result<(), BackendError> {
    if n == 0 {
        return Err(BackendError::Math("domain size must be non-zero"));
    }
    if !n.is_power_of_two() {
        return Err(BackendError::Math("domain size must be a power of two"));
    }
    if n.trailing_zeros() > TWO_ADICITY {
        return Err(BackendError::Math("domain size exceeds field two-adicity"));
    }
    Ok(())
}

fn generator_for_size(n: usize) -> Result<Scalar, BackendError> {
    ensure_domain_size(n)?;
    let log_n = n.trailing_zeros();
    let shift = TWO_ADICITY - log_n;
    let exp = 1u64 << shift;
    Ok(ROOT_OF_UNITY.pow_vartime(&[exp, 0, 0, 0]))
}

fn batch_inversion(values: &mut [Scalar]) -> Result<(), BackendError> {
    if values.is_empty() {
        return Ok(());
    }
    let mut prefix = vec![Scalar::ONE; values.len()];
    let mut acc = Scalar::ONE;
    for (i, value) in values.iter().enumerate() {
        prefix[i] = acc;
        acc *= value;
    }
    let inv = Option::<Scalar>::from(acc.invert())
        .ok_or_else(|| BackendError::Math("batch inversion failed"))?;
    let mut suffix = inv;
    for (value, pref) in values.iter_mut().zip(prefix.into_iter()).rev() {
        let tmp = *value;
        *value = pref * suffix;
        suffix *= tmp;
    }
    Ok(())
}

pub fn lagrange_poly(n: usize, index: usize) -> Result<DensePolynomial, BackendError> {
    if index >= n {
        return Err(BackendError::Math("lagrange index out of range"));
    }
    let polys = lagrange_polys(n)?;
    polys
        .into_iter()
        .nth(index)
        .ok_or_else(|| BackendError::Math("lagrange polynomial missing"))
}

pub fn lagrange_polys(n: usize) -> Result<Vec<DensePolynomial>, BackendError> {
    ensure_domain_size(n)?;
    let omega = generator_for_size(n)?;
    let omega_inv = Option::<Scalar>::from(omega.invert())
        .ok_or_else(|| BackendError::Math("invalid generator inversion"))?;
    let n_scalar = Scalar::from(n as u64);

    let mut omega_inv_pows = Vec::with_capacity(n);
    let mut cur = Scalar::ONE;
    for _ in 0..n {
        omega_inv_pows.push(cur);
        cur *= omega_inv;
    }

    let mut denominators: Vec<Scalar> = omega_inv_pows.iter().map(|w| *w * n_scalar).collect();
    batch_inversion(&mut denominators)?;

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

pub fn interp_mostly_zero(
    eval: Scalar,
    points: &[Scalar],
) -> Result<DensePolynomial, BackendError> {
    if points.is_empty() {
        return Ok(DensePolynomial::from_coefficients_vec(vec![Scalar::ONE]));
    }

    let mut coeffs = vec![Scalar::ONE];
    for &point in points.iter().skip(1) {
        let neg_point = -point;
        coeffs.push(Scalar::ZERO);
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
    let scale_inv = Option::<Scalar>::from(scale.invert())
        .ok_or_else(|| BackendError::Math("interpolation scale inversion failed"))?;

    for coeff in coeffs.iter_mut() {
        *coeff *= eval * scale_inv;
    }

    Ok(DensePolynomial::from_coefficients_vec(coeffs))
}
