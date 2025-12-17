use ark_bn254::Bn254;
use ark_ec::pairing::Pairing;

use crate::{BackendError, Fr, G1, G2, Gt, PairingBackend};

#[derive(Debug)]
pub struct PairingEngine;

impl PairingBackend for PairingEngine {
    type Scalar = Fr;
    type G1 = G1;
    type G2 = G2;
    type Target = Gt;

    fn pairing(g1: &Self::G1, g2: &Self::G2) -> Self::Target {
        Bn254::pairing(&g1.0, &g2.0)
    }

    fn multi_pairing(g1: &[Self::G1], g2: &[Self::G2]) -> Result<Self::Target, BackendError> {
        if g1.len() != g2.len() {
            return Err(BackendError::Math("pairing length mismatch"));
        }
        let g1_proj: Vec<_> = g1.iter().map(|p| p.0).collect();
        let g2_proj: Vec<_> = g2.iter().map(|p| p.0).collect();
        Ok(Bn254::multi_pairing(&g1_proj, &g2_proj))
    }
}
