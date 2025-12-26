mod scheme;
pub use scheme::{KZG, SRS};

use core::fmt::Debug;

use crate::{BackendError, PairingBackend, Polynomial};

/// KZG polynomial commitment scheme interface.
///
/// This trait provides the core operations for Kate-Zaverucha-Goldberg (KZG)
/// commitments, which are used in the silent setup for threshold encryption.
///
/// # KZG Commitments
///
/// KZG commitments allow committing to a polynomial such that:
/// - The commitment is succinct (constant size)
/// - One can prove evaluations at specific points
/// - The scheme is binding and hiding under appropriate assumptions
///
/// # Example
///
/// ```rust
/// use rand_core::RngCore;
/// use tess::{BackendError, Fr, KZG, PairingBackend, PolynomialCommitment};
///
/// fn setup<B: PairingBackend<Scalar = Fr>>(rng: &mut impl RngCore) -> Result<<KZG as PolynomialCommitment<B>>::Parameters, BackendError> {
///     let mut seed = [0u8; 32];
///     rng.fill_bytes(&mut seed);
///     <KZG as PolynomialCommitment<B>>::setup(10, &seed)
/// }
/// ```
pub trait PolynomialCommitment<B: PairingBackend>: Send + Sync + Debug + 'static {
    /// Commitment parameters (powers of tau).
    type Parameters: Clone + Send + Sync + Debug + 'static;

    /// Polynomial type used by this commitment scheme.
    type Polynomial: Polynomial<B::Scalar>;

    /// Performs trusted setup to generate commitment parameters.
    ///
    /// This generates powers of tau: `[τ^0, τ^1, ..., τ^max_degree]` in both G1 and G2.
    /// The secret `tau` must be securely discarded after setup.
    fn setup(max_degree: usize, seed: &[u8; 32]) -> Result<Self::Parameters, BackendError>;

    /// Commits to a polynomial in G1.
    ///
    /// For polynomial `p(x) = ∑ c_i * x^i`, returns `∑ c_i * τ^i * G1`.
    fn commit_g1(
        params: &Self::Parameters,
        polynomial: &Self::Polynomial,
    ) -> Result<B::G1, BackendError>;

    /// Commits to a polynomial in G2.
    ///
    /// For polynomial `p(x) = ∑ c_i * x^i`, returns `∑ c_i * τ^i * G2`.
    fn commit_g2(
        params: &Self::Parameters,
        polynomial: &Self::Polynomial,
    ) -> Result<B::G2, BackendError>;

    /// Opens a commitment at a point in G1, returning the evaluation and proof.
    fn open_g1(
        params: &Self::Parameters,
        polynomial: &Self::Polynomial,
        point: &B::Scalar,
    ) -> Result<(B::Scalar, B::G1), BackendError>;

    /// Verifies a commitment opening in G1.
    fn verify_g1(
        params: &Self::Parameters,
        commitment: &B::G1,
        point: &B::Scalar,
        value: &B::Scalar,
        proof: &B::G1,
    ) -> Result<bool, BackendError>;

    /// Opens a commitment at multiple points in G1, returning evaluations and proofs.
    ///
    /// This is more efficient than calling `open_g1` multiple times for the same polynomial.
    #[allow(clippy::type_complexity)]
    fn batch_open_g1(
        params: &Self::Parameters,
        polynomial: &Self::Polynomial,
        points: &[B::Scalar],
    ) -> Result<(Vec<B::Scalar>, Vec<B::G1>), BackendError>;

    /// Verifies multiple commitment openings in G1 for the same commitment.
    ///
    /// This is more efficient than calling `verify_g1` multiple times.
    fn batch_verify_g1(
        params: &Self::Parameters,
        commitment: &B::G1,
        points: &[B::Scalar],
        values: &[B::Scalar],
        proofs: &[B::G1],
    ) -> Result<bool, BackendError>;
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;
    use rand::{SeedableRng, rngs::StdRng};
    use rand_core::RngCore;

    use crate::{
        CurvePoint, DensePolynomial, FieldElement, Fr, KZG, PairingBackend, PolynomialCommitment,
        SRS,
    };

    fn kzg_commitment_helper<B: PairingBackend<Scalar = Fr>>(rng: &mut StdRng) {
        let mut seed = [0u8; 32];
        rng.fill_bytes(&mut seed);
        let params: SRS<B> = KZG::setup(8, &seed).expect("setup");
        let coeffs: Vec<Fr> = (0..4).map(|_| Fr::random(rng)).collect();
        let poly = DensePolynomial::from_coefficients_vec(coeffs);
        let commitment: B::G1 = KZG::commit_g1(&params, &poly).expect("commit");
        assert!(
            !commitment.is_identity(),
            "commitment should not be identity for random polynomial"
        );
    }

    #[test]
    fn kzg_commitment() {
        kzg_commitment_helper::<crate::PairingEngine>(&mut StdRng::from_entropy());
    }

    #[test]
    fn kzg_open_verify() {
        let mut rng = StdRng::from_entropy();
        let mut seed = [0u8; 32];
        rng.fill_bytes(&mut seed);
        let params: SRS<crate::PairingEngine> = KZG::setup(8, &seed).expect("setup");
        let coeffs: Vec<Fr> = (0..4).map(|_| Fr::random(&mut rng)).collect();
        let poly = DensePolynomial::from_coefficients_vec(coeffs);
        let commitment = KZG::commit_g1(&params, &poly).expect("commit");
        let point = Fr::from_u64(3);
        let (value, proof) = KZG::open_g1(&params, &poly, &point).expect("open");
        let ok = KZG::verify_g1(&params, &commitment, &point, &value, &proof).expect("verify");
        assert!(ok, "opening proof should verify");
    }

    #[test]
    fn kzg_batch_open_verify() {
        let mut rng = StdRng::from_entropy();
        let mut seed = [0u8; 32];
        rng.fill_bytes(&mut seed);
        let params: SRS<crate::PairingEngine> = KZG::setup(8, &seed).expect("setup");
        let coeffs: Vec<Fr> = (0..4).map(|_| Fr::random(&mut rng)).collect();
        let poly = DensePolynomial::from_coefficients_vec(coeffs);
        let commitment = KZG::commit_g1(&params, &poly).expect("commit");

        // Test batch opening at multiple points
        let points = vec![Fr::from_u64(1), Fr::from_u64(2), Fr::from_u64(3)];
        let (values, proofs) = KZG::batch_open_g1(&params, &poly, &points).expect("batch open");

        // Verify individual openings
        for i in 0..points.len() {
            let ok = KZG::verify_g1(&params, &commitment, &points[i], &values[i], &proofs[i])
                .expect("verify");
            assert!(ok, "individual proof {} should verify", i);
        }

        // Verify batch
        let ok = KZG::batch_verify_g1(&params, &commitment, &points, &values, &proofs)
            .expect("batch verify");
        assert!(ok, "batch proof should verify");
    }

    #[test]
    fn kzg_batch_verify_invalid() {
        let mut rng = StdRng::from_entropy();
        let mut seed = [0u8; 32];
        rng.fill_bytes(&mut seed);
        let params: SRS<crate::PairingEngine> = KZG::setup(8, &seed).expect("setup");
        let coeffs: Vec<Fr> = (0..4).map(|_| Fr::random(&mut rng)).collect();
        let poly = DensePolynomial::from_coefficients_vec(coeffs);
        let commitment = KZG::commit_g1(&params, &poly).expect("commit");

        let points = vec![Fr::from_u64(1), Fr::from_u64(2), Fr::from_u64(3)];
        let (mut values, proofs) = KZG::batch_open_g1(&params, &poly, &points).expect("batch open");

        // Tamper with one value
        values[1] = Fr::random(&mut rng);

        // Batch verification should fail
        let ok = KZG::batch_verify_g1(&params, &commitment, &points, &values, &proofs)
            .expect("batch verify");
        assert!(!ok, "batch proof should not verify with tampered value");
    }
}
