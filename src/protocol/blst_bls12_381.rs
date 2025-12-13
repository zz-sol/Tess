use blstrs::Scalar;
use ff::Field;
use rand_core::RngCore;

use crate::{
    backend::{BlstBackend, BlstGt, BlstPowers, DensePolynomial, Radix2EvaluationDomain},
    config::{BackendId, CurveId},
    errors::{BackendError, Error},
    lagrange::blst_bls12_381::{interp_mostly_zero, lagrange_polys},
};

use super::{ProtocolBackend, SilentThreshold};

pub type SilentThresholdBlst = SilentThreshold<BlstBackend>;

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

impl ProtocolBackend for BlstBackend {
    fn backend_id() -> BackendId {
        BackendId::Blst
    }

    fn curve_id() -> CurveId {
        CurveId::Bls12_381
    }

    fn parse_tau(bytes: &[u8]) -> Result<Self::Scalar, Error> {
        parse_tau(bytes)
    }

    fn sample_tau<R: RngCore + ?Sized>(rng: &mut R) -> Self::Scalar {
        Scalar::random(rng)
    }

    fn lagrange_polynomials(parties: usize) -> Result<Vec<DensePolynomial>, BackendError> {
        lagrange_polys(parties)
    }

    fn interp_mostly_zero(
        eval: Self::Scalar,
        points: &[Self::Scalar],
    ) -> Result<DensePolynomial, BackendError> {
        interp_mostly_zero(eval, points)
    }

    fn polynomial_from_coeffs(coeffs: Vec<Self::Scalar>) -> DensePolynomial {
        DensePolynomial::from_coefficients_vec(coeffs)
    }

    fn domain_new(size: usize) -> Result<Self::Domain, BackendError> {
        Radix2EvaluationDomain::new(size).ok_or(BackendError::Math("invalid evaluation domain"))
    }

    fn g_powers(
        params: &BlstPowers,
    ) -> &[<Self::G1 as crate::backend::CurvePoint<Self::Scalar>>::Affine] {
        &params.powers_of_g
    }

    fn h_powers(
        params: &BlstPowers,
    ) -> &[<Self::G2 as crate::backend::CurvePoint<Self::Scalar>>::Affine] {
        &params.powers_of_h
    }

    fn pairing_generator(params: &BlstPowers) -> Self::Target {
        BlstGt(params.e_gh.clone())
    }
}
