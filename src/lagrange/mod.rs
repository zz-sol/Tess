#[cfg(feature = "ark_bls12381")]
pub mod ark_bls12_381;
#[cfg(feature = "ark_bn254")]
pub mod ark_bn254;
#[cfg(feature = "blst")]
pub mod blst_bls12_381;

use core::ops::{Add, AddAssign, Mul, MulAssign, Neg};

use crate::backend::FieldElement;
use crate::errors::BackendError;

/// Trait capturing the minimal field functionality required by the Lagrange helpers.
pub trait LagrangeField:
    FieldElement
    + From<u64>
    + Add<Output = Self>
    + AddAssign
    + Mul<Output = Self>
    + MulAssign
    + Neg<Output = Self>
    + PartialEq
{
    /// Two-adicity of the field multiplicative group.
    const TWO_ADICITY: u32;

    /// Generator of the 2^TWO_ADICITY subgroup.
    fn two_adic_root_of_unity() -> Self;

    /// Performs in-place batch inversion for the provided values.
    fn batch_inversion(values: &mut [Self]) -> Result<(), BackendError>;
}

fn ensure_domain_size<F: LagrangeField>(n: usize) -> Result<(), BackendError> {
    if n == 0 {
        return Err(BackendError::Math("domain size must be non-zero"));
    }
    if !n.is_power_of_two() {
        return Err(BackendError::Math("domain size must be a power of two"));
    }
    if n.trailing_zeros() > F::TWO_ADICITY {
        return Err(BackendError::Math("domain size exceeds field two-adicity"));
    }
    Ok(())
}

fn generator_for_size<F: LagrangeField>(n: usize) -> Result<F, BackendError> {
    ensure_domain_size::<F>(n)?;
    let log_n = n.trailing_zeros();
    let shift = F::TWO_ADICITY - log_n;
    let mut exp = [0u64; 4];
    exp[0] = 1u64 << shift;
    Ok(F::two_adic_root_of_unity().pow(&exp))
}

pub(super) fn lagrange_poly_impl<F, P, PF>(
    n: usize,
    index: usize,
    poly_from_coeffs: PF,
) -> Result<P, BackendError>
where
    F: LagrangeField,
    PF: Fn(Vec<F>) -> P + Copy,
{
    if index >= n {
        return Err(BackendError::Math("lagrange index out of range"));
    }
    let polys = lagrange_polys_impl::<F, P, PF>(n, poly_from_coeffs)?;
    polys
        .into_iter()
        .nth(index)
        .ok_or(BackendError::Math("lagrange polynomial missing"))
}

pub(super) fn lagrange_polys_impl<F, P, PF>(
    n: usize,
    poly_from_coeffs: PF,
) -> Result<Vec<P>, BackendError>
where
    F: LagrangeField,
    PF: Fn(Vec<F>) -> P + Copy,
{
    ensure_domain_size::<F>(n)?;
    let omega = generator_for_size::<F>(n)?;
    let omega_inv = omega
        .invert()
        .ok_or(BackendError::Math("invalid generator inversion"))?;
    let n_scalar = F::from(n as u64);

    let mut omega_inv_pows = Vec::with_capacity(n);
    let mut cur = F::one();
    for _ in 0..n {
        omega_inv_pows.push(cur.clone());
        cur *= omega_inv.clone();
    }

    let mut denominators: Vec<F> = omega_inv_pows
        .iter()
        .map(|w| {
            let mut denom = w.clone();
            denom *= n_scalar.clone();
            denom
        })
        .collect();
    F::batch_inversion(&mut denominators)?;

    let mut polys = Vec::with_capacity(n);
    for (omega_i_inv, denom_inv) in omega_inv_pows.iter().zip(denominators.iter()) {
        let mut coeffs = Vec::with_capacity(n);
        let mut power = omega_i_inv.clone();
        for _ in 0..n {
            let mut term = power.clone();
            term *= denom_inv.clone();
            coeffs.push(term);
            power *= omega_i_inv.clone();
        }
        polys.push(poly_from_coeffs(coeffs));
    }
    Ok(polys)
}

pub(super) fn interp_mostly_zero_impl<F, P, PF>(
    eval: F,
    points: &[F],
    poly_from_coeffs: PF,
) -> Result<P, BackendError>
where
    F: LagrangeField,
    PF: Fn(Vec<F>) -> P,
{
    if points.is_empty() {
        return Ok(poly_from_coeffs(vec![F::one()]));
    }

    let mut coeffs = vec![F::one()];
    for point in points.iter().skip(1) {
        let neg_point = -point.clone();
        coeffs.push(F::zero());
        for i in (0..coeffs.len() - 1).rev() {
            let (head, tail) = coeffs.split_at_mut(i + 1);
            let coef = &mut head[i];
            let next = &mut tail[0];
            let coef_clone = coef.clone();
            *next += coef_clone;
            *coef *= neg_point.clone();
        }
    }

    let mut scale = coeffs
        .last()
        .cloned()
        .ok_or(BackendError::Math("interpolation scale missing"))?;
    let first_point = points[0].clone();
    for coeff in coeffs.iter().rev().skip(1) {
        scale = scale * first_point.clone() + coeff.clone();
    }
    let scale_inv = scale
        .invert()
        .ok_or(BackendError::Math("interpolation scale inversion failed"))?;

    let mut factor = eval;
    factor *= scale_inv;
    for coeff in coeffs.iter_mut() {
        *coeff *= factor.clone();
    }

    Ok(poly_from_coeffs(coeffs))
}
