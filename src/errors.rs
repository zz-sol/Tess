//! Error types for the crate.
//!
//! This module defines low-level backend errors returned by concrete
//! backend implementations (Arkworks, blstrs) as well as the high-level
//! protocol-facing `Error` type used across the crate.
//!
//! The errors implement `core::fmt::Display` and `core::error::Error` so they are
//! easy to convert and debug in higher-level code.
//!
//! # Examples
//!
//! ```rust
//! use tess::Error;
//! ```

use alloc::string::String;
use core::error::Error as CoreError;
use core::fmt;

/// Errors bubbled up from backend implementations (Arkworks, blstrs, etc.).
#[derive(Debug)]
pub enum BackendError {
    /// The selected curve is not supported by this backend.
    UnsupportedCurve(&'static str),
    /// The selected backend feature is not supported.
    UnsupportedFeature(&'static str),
    /// A serialization step failed.
    Serialization(&'static str),
    /// A backend math operation failed.
    Math(&'static str),
    /// A backend error not covered by more specific variants.
    Other(String),
}

/// High-level errors returned by the threshold encryption API.
#[derive(Debug)]
pub enum Error {
    /// Configuration parameters are invalid or inconsistent.
    InvalidConfig(String),
    /// Errors surfaced from the cryptographic backend.
    Backend(BackendError),
    /// Input data is malformed or fails validation.
    MalformedInput(String),
    /// Not enough shares were provided to meet the threshold.
    NotEnoughShares {
        /// Minimum number of shares required.
        required: usize,
        /// Number of shares provided.
        provided: usize,
    },
    /// The selector vector length does not match the expected size.
    SelectorMismatch {
        /// Expected selector length.
        expected: usize,
        /// Actual selector length.
        actual: usize,
    },
}

impl fmt::Display for BackendError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BackendError::UnsupportedCurve(curve) => write!(f, "unsupported curve: {curve}"),
            BackendError::UnsupportedFeature(feature) => {
                write!(f, "unsupported backend feature: {feature}")
            }
            BackendError::Serialization(msg) => write!(f, "serialization failure: {msg}"),
            BackendError::Math(msg) => write!(f, "math error: {msg}"),
            BackendError::Other(msg) => write!(f, "{msg}"),
        }
    }
}

impl From<BackendError> for Error {
    fn from(err: BackendError) -> Self {
        Self::Backend(err)
    }
}

impl CoreError for BackendError {}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::InvalidConfig(msg) => write!(f, "invalid configuration: {msg}"),
            Error::Backend(err) => write!(f, "backend error: {err}"),
            Error::MalformedInput(msg) => write!(f, "malformed input: {msg}"),
            Error::NotEnoughShares { required, provided } => write!(
                f,
                "insufficient shares: required {required}, provided {provided}"
            ),
            Error::SelectorMismatch { expected, actual } => {
                write!(
                    f,
                    "selector length mismatch: expected {expected}, got {actual}"
                )
            }
        }
    }
}

impl CoreError for Error {
    fn source(&self) -> Option<&(dyn CoreError + 'static)> {
        match self {
            Error::Backend(err) => Some(err),
            _ => None,
        }
    }
}
