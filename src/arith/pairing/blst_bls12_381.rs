use alloc::vec::Vec;
use blstrs::Bls12;
use blstrs::G1Affine;
use blstrs::G2Affine;
use blstrs::G2Prepared;
use group::Curve;
use group::prime::PrimeCurveAffine;
use pairing::MillerLoopResult;
use pairing::MultiMillerLoop;

use crate::{BackendError, Fr, G1, G2, Gt, PairingBackend};

/// Pairing engine implementation for the blst BLS12-381 backend.
#[derive(Debug)]
pub struct PairingEngine;

/// blst-backed `PairingBackend` implementation for BLS12-381.
///
/// This implementation ties together scalar types, curve groups, and pairing
/// operations for the blstrs backend, providing high-performance operations
/// for the BLS12-381 curve.
impl PairingBackend for PairingEngine {
    type Scalar = Fr;
    type G1 = G1;
    type G2 = G2;
    type Target = Gt;

    fn pairing(g1: &Self::G1, g2: &Self::G2) -> Self::Target {
        blstrs::pairing(&g1.to_affine(), &g2.to_affine())
    }

    fn multi_pairing(g1: &[Self::G1], g2: &[Self::G2]) -> Result<Self::Target, BackendError> {
        if g1.len() != g2.len() {
            return Err(BackendError::Math("pairing length mismatch"));
        }
        let mut g1_affine = vec![G1Affine::identity(); g1.len()];
        let mut g2_affine = vec![G2Affine::identity(); g2.len()];
        G1::batch_normalize(g1, &mut g1_affine);
        G2::batch_normalize(g2, &mut g2_affine);
        let g2_prepared: Vec<G2Prepared> =
            g2_affine.iter().map(|aff| G2Prepared::from(*aff)).collect();
        let terms: Vec<_> = g1_affine.iter().zip(g2_prepared.iter()).collect();
        let result = Bls12::multi_miller_loop(&terms).final_exponentiation();
        Ok(result)
    }
}
