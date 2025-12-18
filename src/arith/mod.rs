//! Cryptographic backend abstractions and implementations.
//!
//! This module provides trait abstractions for cryptographic primitives used in TESS,
//! allowing multiple backend implementations (blstrs, Arkworks) to provide unified interfaces.
//!
//! # Architecture
//!
//! The module is organized into several submodules, each providing a specific abstraction layer:
//!
//! - **[`field`]**: Scalar field operations (Fr) - addition, multiplication, inversion
//! - **[`group`]**: Elliptic curve point operations (G1, G2, GT) - addition, scalar multiplication
//! - **[`pairing`]**: Bilinear pairing operations - `e(G1, G2) -> GT`
//! - **[`poly`]**: Polynomial operations - evaluation, interpolation, FFT
//! - **[`lagrange`]**: Lagrange polynomial helpers - precomputed commitments for efficient key generation
//!
//! # Backend Support
//!
//! TESS supports multiple cryptographic backends via feature flags:
//!
//! | Feature | Backend | Curve | Status |
//! |---------|---------|-------|--------|
//! | `blst` (default) | blstrs | BLS12-381 | Stable |
//! | `ark_bls12381` | Arkworks | BLS12-381 | Stable |
//! | `ark_bn254` | Arkworks | BN254 | Stable |
//!
//! # Example
//!
//! ```rust
//! use rand::thread_rng;
//! use tess::{CurvePoint, FieldElement, PairingBackend, PairingEngine};
//!
//! // Field operations
//! let mut rng = thread_rng();
//! let scalar = <PairingEngine as PairingBackend>::Scalar::random(&mut rng);
//! let inv = scalar.invert().expect("non-zero scalar");
//!
//! // Curve operations
//! let g1 = <PairingEngine as PairingBackend>::G1::generator();
//! let point = g1.mul_scalar(&scalar);
//!
//! // Pairing operation
//! let g2 = <PairingEngine as PairingBackend>::G2::generator();
//! let gt = PairingEngine::pairing(&g1, &g2);
//! println!("{:?}", gt);
//! ```

mod field;
pub use field::*;

mod group;
pub use group::*;

mod pairing;
pub use pairing::*;

mod poly;
pub use poly::*;

mod lagrange;
pub use lagrange::*;
