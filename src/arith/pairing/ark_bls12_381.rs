use crate::{Fr, PairingBackend, TargetGroup};

#[derive(Debug)]
pub struct PairingEngine;

impl PairingBackend for PairingEngine {
    type Scalar = Fr;
}
