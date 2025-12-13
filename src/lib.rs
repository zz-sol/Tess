//! Core traits, configuration, and API definitions for the TESS threshold encryption project.
//!
//! The crate currently focuses on the abstraction layer that allows multiple
//! cryptographic backends (Arkworks, blstrs, future GPU-enhanced MSM engines)
//! to expose a unified interface to the higher-level protocol logic.

pub mod backend;

pub mod config;
pub mod errors;
pub mod lagrange;
pub mod protocol;

pub use backend::*;
pub use config::*;
pub use errors::*;
pub use protocol::ThresholdScheme;
