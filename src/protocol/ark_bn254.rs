//! Arkworks-backed protocol glue for BN254.
//!
//! This file contains the `ProtocolBackend` implementation for the `ArkworksBn254`
//! backend. It adapts arkworks-specific types (scalars, polynomials, domains, and
//! SRS powers) to the generic protocol interface used by `protocol::mod`.
//!
//! # Feature
//!
//! Compiled when the Cargo feature `ark_bn254` is enabled.
//!
//! # Example
//!
//! ```rust,no_run
//! # #[cfg(feature = "ark_bn254")]
//! # {
//! use tess::backend::ArkworksBn254;
//! # }
//! ```

use ark_bn254::Fr as BnFr;
use ark_poly::{
    DenseUVPolynomial, EvaluationDomain, Radix2EvaluationDomain, univariate::DensePolynomial,
};
use ark_serialize::CanonicalDeserialize;
use ark_std::UniformRand;
use rand_core::RngCore;

use crate::{
    backend::{ArkworksBn254, BnPowers},
    config::{BackendId, CurveId},
    errors::{BackendError, Error},
    lagrange::ark_bn254::{interp_mostly_zero, lagrange_polys},
};

use super::{ProtocolBackend, SilentThreshold};

pub type SilentThresholdBn = SilentThreshold<ArkworksBn254>;

impl ProtocolBackend for ArkworksBn254 {
    fn backend_id() -> BackendId {
        BackendId::Arkworks
    }

    fn curve_id() -> CurveId {
        CurveId::Bn254
    }

    fn parse_tau(bytes: &[u8]) -> Result<Self::Scalar, Error> {
        BnFr::deserialize_compressed(bytes)
            .map_err(|_| Error::InvalidConfig("invalid trusted tau encoding".into()))
    }

    fn sample_tau<R: RngCore + ?Sized>(rng: &mut R) -> Self::Scalar {
        BnFr::rand(rng)
    }

    fn lagrange_polynomials(parties: usize) -> Result<Vec<DensePolynomial<BnFr>>, BackendError> {
        lagrange_polys(parties)
    }

    fn interp_mostly_zero(
        eval: Self::Scalar,
        points: &[Self::Scalar],
    ) -> Result<DensePolynomial<BnFr>, BackendError> {
        interp_mostly_zero(eval, points)
    }

    fn polynomial_from_coeffs(coeffs: Vec<Self::Scalar>) -> DensePolynomial<BnFr> {
        DensePolynomial::from_coefficients_vec(coeffs)
    }

    fn domain_new(size: usize) -> Result<Self::Domain, BackendError> {
        Radix2EvaluationDomain::<BnFr>::new(size)
            .ok_or(BackendError::Math("invalid evaluation domain"))
    }

    fn g_powers(
        params: &BnPowers,
    ) -> &[<Self::G1 as crate::backend::CurvePoint<Self::Scalar>>::Affine] {
        &params.powers_of_g
    }

    fn h_powers(
        params: &BnPowers,
    ) -> &[<Self::G2 as crate::backend::CurvePoint<Self::Scalar>>::Affine] {
        &params.powers_of_h
    }

    fn pairing_generator(params: &BnPowers) -> Self::Target {
        params.e_gh.clone()
    }
}
