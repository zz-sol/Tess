//! Configuration types for threshold encryption parameters.
//!
//! This module provides configuration types that control the threshold encryption
//! scheme, including backend selection, curve selection, and threshold parameters.
//!
//! # Example
//!
//! ```rust
//! use tess::{ThresholdParameters, BackendConfig, BackendId, CurveId};
//!
//! // Configure 5-of-8 threshold with blstrs backend and BLS12-381 curve
//! let config = BackendConfig::new(BackendId::Blst, CurveId::Bls12_381);
//! let params = ThresholdParameters::new(8, 5, config, None).expect("valid params");
//! ```

use serde::{Deserialize, Serialize};

use crate::errors::{BackendError, Error};

/// Supported pairing-friendly elliptic curves.
///
/// TESS supports multiple pairing-friendly curves depending on the backend:
///
/// - **BLS12-381**: A 381-bit curve providing ~128 bits of security. Supported by
///   both Arkworks and blstrs backends.
/// - **BN254**: A 254-bit curve providing ~100 bits of security. Supported only by
///   the Arkworks backend.
///
/// # Security Levels
///
/// - `Bls12_381`: ~128-bit security (recommended for most applications)
/// - `Bn254`: ~100-bit security (faster but lower security margin)
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum CurveId {
    /// BN254 curve (~100-bit security)
    Bn254,
    /// BLS12-381 curve (~128-bit security, recommended)
    Bls12_381,
}

/// Cryptographic backend implementations.
///
/// TESS supports multiple backend implementations for flexibility:
///
/// - **Arkworks**: Pure Rust implementation supporting both BLS12-381 and BN254 curves.
///   Provides good portability and flexibility.
/// - **blst**: Optimized implementation for BLS12-381 using assembly. Provides best
///   performance for BLS12-381 operations.
///
/// # Feature Flags
///
/// Backend support is controlled via Cargo features:
/// - `ark_bls12381`: Enable Arkworks with BLS12-381
/// - `ark_bn254`: Enable Arkworks with BN254
/// - `blst` (default): Enable blstrs with BLS12-381
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum BackendId {
    /// Arkworks backend (pure Rust, supports BLS12-381 and BN254)
    Arkworks,
    /// blstrs backend (optimized assembly, BLS12-381 only)
    Blst,
}

/// Backend and curve configuration.
///
/// This structure pairs a backend implementation with a specific curve.
/// Not all combinations are supported - use [`ensure_supported`](BackendConfig::ensure_supported)
/// to validate.
///
/// # Supported Combinations
///
/// | Backend    | BLS12-381 | BN254 |
/// |------------|-----------|-------|
/// | Arkworks   | ✓         | ✓     |
/// | blst       | ✓         | ✗     |
///
/// # Example
///
/// ```rust
/// use tess::{BackendConfig, BackendId, CurveId};
///
/// // Create a config for blstrs with BLS12-381
/// let config = BackendConfig::new(BackendId::Blst, CurveId::Bls12_381);
/// config.ensure_supported().expect("supported combination");
/// ```
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BackendConfig {
    /// The cryptographic backend to use
    pub backend: BackendId,
    /// The pairing-friendly curve to use
    pub curve: CurveId,
}

impl BackendConfig {
    /// Creates a new backend configuration.
    ///
    /// # Example
    ///
    /// ```rust
    /// use tess::{BackendConfig, BackendId, CurveId};
    ///
    /// let config = BackendConfig::new(BackendId::Arkworks, CurveId::Bn254);
    /// ```
    pub fn new(backend: BackendId, curve: CurveId) -> Self {
        Self { backend, curve }
    }

    /// Validates that this backend/curve combination is supported.
    ///
    /// This checks both that the combination is valid (e.g., blst only supports
    /// BLS12-381) and that the required feature flag is enabled at compile time.
    ///
    /// # Returns
    ///
    /// - `Ok(())` if the configuration is supported and enabled
    /// - `Err(BackendError)` with a descriptive message if unsupported
    ///
    /// # Example
    ///
    /// ```rust
    /// use tess::{BackendConfig, BackendId, CurveId};
    ///
    /// let config = BackendConfig::new(BackendId::Blst, CurveId::Bn254);
    /// // This will return an error because blst doesn't support BN254
    /// assert!(config.ensure_supported().is_err());
    /// ```
    pub fn ensure_supported(&self) -> Result<(), BackendError> {
        match (self.backend, self.curve) {
            (BackendId::Arkworks, CurveId::Bls12_381) => {
                if cfg!(feature = "ark_bls12381") {
                    Ok(())
                } else {
                    Err(BackendError::UnsupportedFeature(
                        "compile with `ark_bls12381` feature to use Arkworks BLS12-381",
                    ))
                }
            }
            (BackendId::Arkworks, CurveId::Bn254) => {
                if cfg!(feature = "ark_bn254") {
                    Ok(())
                } else {
                    Err(BackendError::UnsupportedFeature(
                        "compile with `ark_bn254` feature to use Arkworks BN254",
                    ))
                }
            }
            (BackendId::Blst, CurveId::Bls12_381) => {
                if cfg!(feature = "blst") {
                    Ok(())
                } else {
                    Err(BackendError::UnsupportedFeature(
                        "compile with `blst` feature to use the blstrs backend",
                    ))
                }
            }
            (BackendId::Blst, CurveId::Bn254) => Err(BackendError::UnsupportedCurve(
                "bn254 is not yet supported by the blstrs backend",
            )),
        }
    }
}

/// Complete threshold encryption scheme parameters.
///
/// This structure contains all parameters needed to configure a threshold
/// encryption scheme, including the number of participants, threshold value,
/// backend configuration, and optional pre-generated tau for SRS.
///
/// # Fields
///
/// - `parties`: Total number of participants (n). Must be a power of two.
/// - `threshold`: Minimum number of participants needed to decrypt (t). Must satisfy 1 ≤ t ≤ n.
/// - `chunk_size`: Reserved for future use (batch processing)
/// - `backend`: Backend and curve configuration
/// - `kzg_tau`: Optional pre-generated tau for SRS (for deterministic testing)
///
/// # Constraints
///
/// - `parties` must be at least 2
/// - `parties` must be a power of two (required for FFT operations)
/// - `threshold` must be between 1 and `parties` (inclusive)
/// - `backend` must be a supported combination (checked via `validate`)
///
/// # Example
///
/// ```rust
/// use tess::{ThresholdParameters, BackendConfig, BackendId, CurveId};
///
/// // 3-of-4 threshold encryption using blstrs
/// let params = ThresholdParameters::new(
///     4,    // 4 participants (power of two)
///     3,    // need 3 to decrypt
///     BackendConfig::new(BackendId::Blst, CurveId::Bls12_381),
///     None, // generate fresh tau
/// ).expect("valid parameters");
/// ```
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ThresholdParameters {
    /// Total number of participants (must be power of two)
    pub parties: usize,
    /// Threshold number of participants needed to decrypt
    pub threshold: usize,
    /// Reserved: chunk size for batch processing
    pub chunk_size: usize,
    /// Backend and curve configuration
    pub backend: BackendConfig,
    /// Optional pre-generated tau (for testing only)
    pub kzg_tau: Option<Vec<u8>>,
}

impl ThresholdParameters {
    /// Creates and validates threshold parameters.
    ///
    /// This is a convenience constructor that creates parameters and validates
    /// them in one step.
    ///
    /// # Arguments
    ///
    /// - `parties`: Total number of participants (must be power of two)
    /// - `threshold`: Minimum participants needed to decrypt
    /// - `backend`: Backend and curve configuration
    /// - `kzg_tau`: Optional pre-generated tau (use `None` for production)
    ///
    /// # Returns
    ///
    /// Validated parameters, or an error if validation fails.
    ///
    /// # Example
    ///
    /// ```rust
    /// use tess::{ThresholdParameters, BackendConfig, BackendId, CurveId};
    ///
    /// let params = ThresholdParameters::new(
    ///     8, 5,
    ///     BackendConfig::new(BackendId::Blst, CurveId::Bls12_381),
    ///     None,
    /// )?;
    /// # Ok::<(), tess::Error>(())
    /// ```
    pub fn new(
        parties: usize,
        threshold: usize,
        backend: BackendConfig,
        kzg_tau: Option<Vec<u8>>,
    ) -> Result<Self, Error> {
        let params = Self {
            parties,
            threshold,
            chunk_size: parties, // Default to parties for now
            backend,
            kzg_tau,
        };
        params.validate()?;
        Ok(params)
    }

    /// Validates the threshold parameters.
    ///
    /// This checks that:
    /// - The backend/curve combination is supported
    /// - There are at least 2 participants
    /// - The threshold is valid (1 ≤ t ≤ n)
    /// - The number of parties is a power of two
    ///
    /// # Returns
    ///
    /// - `Ok(())` if parameters are valid
    /// - `Err(Error)` with a descriptive error message if invalid
    ///
    /// # Example
    ///
    /// ```rust
    /// use tess::{ThresholdParameters, BackendConfig, BackendId, CurveId};
    ///
    /// let mut params = ThresholdParameters {
    ///     parties: 3, // Invalid: not power of two!
    ///     threshold: 2,
    ///     chunk_size: 3,
    ///     backend: BackendConfig::new(BackendId::Blst, CurveId::Bls12_381),
    ///     kzg_tau: None,
    /// };
    ///
    /// assert!(params.validate().is_err());
    /// ```
    pub fn validate(&self) -> Result<(), Error> {
        self.backend.ensure_supported().map_err(Error::Backend)?;
        if self.parties < 2 {
            return Err(Error::InvalidConfig(
                "need at least two parties for threshold encryption".into(),
            ));
        }
        if self.threshold == 0 || self.threshold > self.parties {
            return Err(Error::InvalidConfig(
                "threshold must be within [1, parties]".into(),
            ));
        }
        if !self.parties.is_power_of_two() {
            return Err(Error::InvalidConfig(
                "current protocol assumes power-of-two domain size".into(),
            ));
        }
        Ok(())
    }
}
