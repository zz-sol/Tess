//! Error types for the crate.
//!
//! This module defines low-level backend errors returned by concrete
//! backend implementations (Arkworks, blstrs) as well as the high-level
//! protocol-facing `Error` type used across the crate.
//!
//! The errors are implemented with `thiserror` so they are easy to convert
//! and debug in higher-level code.
//!
//! # Examples
//!
//! ```rust
//! use tess::errors::Error;
//! ```

use thiserror::Error;

/// Errors bubbled up from backend implementations (Arkworks, blstrs, etc.).
#[derive(Debug, Error)]
pub enum BackendError {
    #[error("unsupported curve: {0}")]
    UnsupportedCurve(&'static str),
    #[error("unsupported backend feature: {0}")]
    UnsupportedFeature(&'static str),
    #[error("serialization failure: {0}")]
    Serialization(&'static str),
    #[error("math error: {0}")]
    Math(&'static str),
    #[error("{0}")]
    Other(String),
}

/// High-level errors returned by the threshold encryption API.
#[derive(Debug, Error)]
pub enum Error {
    #[error("invalid configuration: {0}")]
    InvalidConfig(String),
    #[error("backend error: {0}")]
    Backend(#[from] BackendError),
    #[error("malformed input: {0}")]
    MalformedInput(String),
    #[error("insufficient shares: required {required}, provided {provided}")]
    NotEnoughShares { required: usize, provided: usize },
    #[error("selector length mismatch: expected {expected}, got {actual}")]
    SelectorMismatch { expected: usize, actual: usize },
}
