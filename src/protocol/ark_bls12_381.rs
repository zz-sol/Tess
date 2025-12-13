use ark_bls12_381::Fr as BlsFr;
use ark_poly::{Radix2EvaluationDomain, univariate::DensePolynomial};
use ark_serialize::CanonicalDeserialize;
use ark_std::UniformRand;
use rand_core::RngCore;

use crate::{
    backend::{ArkGt, ArkworksBls12, BlsPowers},
    config::{BackendId, CurveId},
    errors::{BackendError, Error},
    lagrange::ark_bls12_381::{interp_mostly_zero, lagrange_polys},
};

use super::{ProtocolBackend, SilentThreshold};

pub type SilentThresholdScheme = SilentThreshold<ArkworksBls12>;

impl ProtocolBackend for ArkworksBls12 {
    fn backend_id() -> BackendId {
        BackendId::Arkworks
    }

    fn curve_id() -> CurveId {
        CurveId::Bls12_381
    }

    fn parse_tau(bytes: &[u8]) -> Result<Self::Scalar, Error> {
        BlsFr::deserialize_compressed(bytes)
            .map_err(|_| Error::InvalidConfig("invalid trusted tau encoding".into()))
    }

    fn sample_tau<R: RngCore + ?Sized>(rng: &mut R) -> Self::Scalar {
        BlsFr::rand(rng)
    }

    fn lagrange_polynomials(parties: usize) -> Result<Vec<DensePolynomial<BlsFr>>, BackendError> {
        lagrange_polys(parties)
    }

    fn interp_mostly_zero(
        eval: Self::Scalar,
        points: &[Self::Scalar],
    ) -> Result<DensePolynomial<BlsFr>, BackendError> {
        interp_mostly_zero(eval, points)
    }

    fn polynomial_from_coeffs(coeffs: Vec<Self::Scalar>) -> DensePolynomial<BlsFr> {
        DensePolynomial::from_coefficients_vec(coeffs)
    }

    fn domain_new(size: usize) -> Result<Self::Domain, BackendError> {
        Radix2EvaluationDomain::<BlsFr>::new(size)
            .ok_or(BackendError::Math("invalid evaluation domain"))
    }

    fn g_powers(
        params: &BlsPowers,
    ) -> &[<Self::G1 as crate::backend::CurvePoint<Self::Scalar>>::Affine] {
        &params.powers_of_g
    }

    fn h_powers(
        params: &BlsPowers,
    ) -> &[<Self::G2 as crate::backend::CurvePoint<Self::Scalar>>::Affine] {
        &params.powers_of_h
    }

    fn pairing_generator(params: &BlsPowers) -> Self::Target {
        ArkGt(params.e_gh.clone())
    }
}
