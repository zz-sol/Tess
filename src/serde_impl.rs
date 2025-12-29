//! Serde serialization support for TESS types.
//!
//! This module provides serde `Serialize` and `Deserialize` implementations
//! for all public TESS types, enabling easy persistence and transmission of
//! cryptographic parameters, keys, and ciphertexts.
//!
//! # Serialization Strategy
//!
//! - **Curve Points** (G1, G2): Serialized using compressed byte representations via `to_repr()`
//! - **Scalars**: Serialized as byte arrays via `to_bytes_le()`
//! - **Target Group Elements** (Gt): Serialized using byte representations via `to_repr()`
//! - **Vectors**: Serialized as arrays
//!
//! # Example
//!
//! ```rust,ignore
//! use tess::{PairingEngine, SilentThresholdScheme, ThresholdEncryption};
//! use serde_json;
//!
//! let mut rng = rand::thread_rng();
//! let scheme = SilentThresholdScheme::<PairingEngine>::new();
//! let params = scheme.param_gen(&mut rng, 5, 3).unwrap();
//!
//! // Serialize to JSON
//! let json = serde_json::to_string(&params).unwrap();
//!
//! // Deserialize from JSON
//! let params2: Params<PairingEngine> = serde_json::from_str(&json).unwrap();
//! ```

use alloc::vec::Vec;
use serde::de;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::{
    AggregateKey, Ciphertext, DecryptionResult, Fr, LagrangePowers, PairingBackend, Params,
    PartialDecryption, PublicKey, SRS, SecretKey, UnsafeKeyMaterial,
    arith::{CurvePoint, FieldElement, TargetGroup},
};

fn repr_from_bytes<R, E>(bytes: &[u8]) -> Result<R, E>
where
    R: Default + AsRef<[u8]> + AsMut<[u8]>,
    E: de::Error,
{
    let mut repr = R::default();
    let dst = repr.as_mut();
    if bytes.len() > dst.len() {
        return Err(E::custom("byte representation too long"));
    }
    dst[..bytes.len()].copy_from_slice(bytes);
    Ok(repr)
}

fn field_from_bytes<F, E>(bytes: &[u8]) -> Result<F, E>
where
    F: FieldElement,
    E: de::Error,
{
    let repr = repr_from_bytes::<F::Repr, E>(bytes)?;
    F::from_repr(&repr).map_err(E::custom)
}

fn curve_point_from_bytes<C, F, E>(bytes: &[u8]) -> Result<C, E>
where
    C: CurvePoint<F>,
    F: FieldElement,
    E: de::Error,
{
    let repr = repr_from_bytes::<C::Repr, E>(bytes)?;
    C::from_repr(&repr).map_err(E::custom)
}

fn target_group_from_bytes<T, E>(bytes: &[u8]) -> Result<T, E>
where
    T: TargetGroup,
    E: de::Error,
{
    let repr = repr_from_bytes::<T::Repr, E>(bytes)?;
    T::from_repr(&repr).map_err(E::custom)
}

// Implement Serialize and Deserialize for SecretKey
impl<B: PairingBackend> Serialize for SecretKey<B> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut state = serializer.serialize_struct("SecretKey", 2)?;
        state.serialize_field("participant_id", &self.participant_id)?;
        let scalar_bytes = self.scalar.to_repr();
        state.serialize_field("scalar", scalar_bytes.as_ref())?;
        state.end()
    }
}

impl<'de, B: PairingBackend> Deserialize<'de> for SecretKey<B> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct SecretKeyHelper {
            participant_id: usize,
            scalar: Vec<u8>,
        }

        let helper = SecretKeyHelper::deserialize(deserializer)?;
        let scalar = field_from_bytes::<B::Scalar, D::Error>(&helper.scalar)?;

        Ok(SecretKey {
            participant_id: helper.participant_id,
            scalar,
        })
    }
}

// Implement Serialize and Deserialize for PublicKey
impl<B: PairingBackend> Serialize for PublicKey<B> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut state = serializer.serialize_struct("PublicKey", 6)?;
        state.serialize_field("participant_id", &self.participant_id)?;
        state.serialize_field("bls_key", self.bls_key.to_repr().as_ref())?;
        state.serialize_field("lagrange_li", self.lagrange_li.to_repr().as_ref())?;
        state.serialize_field(
            "lagrange_li_minus0",
            self.lagrange_li_minus0.to_repr().as_ref(),
        )?;
        state.serialize_field("lagrange_li_x", self.lagrange_li_x.to_repr().as_ref())?;
        state.serialize_field(
            "lagrange_li_lj_z",
            &self
                .lagrange_li_lj_z
                .iter()
                .map(|p| p.to_repr().as_ref().to_vec())
                .collect::<Vec<Vec<u8>>>(),
        )?;
        state.end()
    }
}

impl<'de, B: PairingBackend> Deserialize<'de> for PublicKey<B> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct PublicKeyHelper {
            participant_id: usize,
            bls_key: Vec<u8>,
            lagrange_li: Vec<u8>,
            lagrange_li_minus0: Vec<u8>,
            lagrange_li_x: Vec<u8>,
            lagrange_li_lj_z: Vec<Vec<u8>>,
        }

        let helper = PublicKeyHelper::deserialize(deserializer)?;

        Ok(PublicKey {
            participant_id: helper.participant_id,
            bls_key: curve_point_from_bytes::<B::G1, B::Scalar, D::Error>(&helper.bls_key)?,
            lagrange_li: curve_point_from_bytes::<B::G1, B::Scalar, D::Error>(&helper.lagrange_li)?,
            lagrange_li_minus0: curve_point_from_bytes::<B::G1, B::Scalar, D::Error>(
                &helper.lagrange_li_minus0,
            )?,
            lagrange_li_x: curve_point_from_bytes::<B::G1, B::Scalar, D::Error>(
                &helper.lagrange_li_x,
            )?,
            lagrange_li_lj_z: helper
                .lagrange_li_lj_z
                .iter()
                .map(|bytes| curve_point_from_bytes::<B::G1, B::Scalar, D::Error>(bytes))
                .collect::<Result<Vec<_>, _>>()?,
        })
    }
}

// Implement Serialize and Deserialize for AggregateKey
impl<B: PairingBackend<Scalar = Fr>> Serialize for AggregateKey<B> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut state = serializer.serialize_struct("AggregateKey", 6)?;
        state.serialize_field("public_keys", &self.public_keys)?;
        state.serialize_field("ask", &self.ask.to_repr().as_ref())?;
        state.serialize_field("z_g2", &self.z_g2.to_repr().as_ref())?;
        state.serialize_field(
            "lagrange_row_sums",
            &self
                .lagrange_row_sums
                .iter()
                .map(|p| p.to_repr().as_ref().to_vec())
                .collect::<Vec<Vec<u8>>>(),
        )?;
        state.serialize_field(
            "precomputed_pairing",
            &self.precomputed_pairing.to_repr().as_ref(),
        )?;
        state.serialize_field("kzg_params", &self.kzg_params)?;
        state.end()
    }
}

impl<'de, B: PairingBackend<Scalar = Fr>> Deserialize<'de> for AggregateKey<B> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        #[serde(bound(deserialize = ""))]
        struct AggregateKeyHelper<B: PairingBackend<Scalar = Fr>> {
            public_keys: Vec<PublicKey<B>>,
            ask: Vec<u8>,
            z_g2: Vec<u8>,
            lagrange_row_sums: Vec<Vec<u8>>,
            precomputed_pairing: Vec<u8>,
            kzg_params: SRS<B>,
        }

        let helper = AggregateKeyHelper::deserialize(deserializer)?;

        Ok(AggregateKey {
            public_keys: helper.public_keys,
            ask: curve_point_from_bytes::<B::G1, B::Scalar, D::Error>(&helper.ask)?,
            z_g2: curve_point_from_bytes::<B::G2, B::Scalar, D::Error>(&helper.z_g2)?,
            lagrange_row_sums: helper
                .lagrange_row_sums
                .iter()
                .map(|bytes| curve_point_from_bytes::<B::G1, B::Scalar, D::Error>(bytes))
                .collect::<Result<Vec<_>, _>>()?,
            precomputed_pairing: target_group_from_bytes::<B::Target, D::Error>(
                &helper.precomputed_pairing,
            )?,
            kzg_params: helper.kzg_params,
        })
    }
}

// Implement Serialize and Deserialize for Ciphertext
impl<B: PairingBackend> Serialize for Ciphertext<B> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut state = serializer.serialize_struct("Ciphertext", 6)?;
        state.serialize_field("gamma_g2", &self.gamma_g2.to_repr().as_ref())?;
        state.serialize_field(
            "proof_g1",
            &self
                .proof_g1
                .iter()
                .map(|p| p.to_repr().as_ref().to_vec())
                .collect::<Vec<Vec<u8>>>(),
        )?;
        state.serialize_field(
            "proof_g2",
            &self
                .proof_g2
                .iter()
                .map(|p| p.to_repr().as_ref().to_vec())
                .collect::<Vec<Vec<u8>>>(),
        )?;
        state.serialize_field("shared_secret", &self.shared_secret.to_repr().as_ref())?;
        state.serialize_field("threshold", &self.threshold)?;
        state.serialize_field("payload", &self.payload)?;
        state.end()
    }
}

impl<'de, B: PairingBackend> Deserialize<'de> for Ciphertext<B> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct CiphertextHelper {
            gamma_g2: Vec<u8>,
            proof_g1: Vec<Vec<u8>>,
            proof_g2: Vec<Vec<u8>>,
            shared_secret: Vec<u8>,
            threshold: usize,
            payload: Vec<u8>,
        }

        let helper = CiphertextHelper::deserialize(deserializer)?;

        Ok(Ciphertext {
            gamma_g2: curve_point_from_bytes::<B::G2, B::Scalar, D::Error>(&helper.gamma_g2)?,
            proof_g1: helper
                .proof_g1
                .iter()
                .map(|bytes| curve_point_from_bytes::<B::G1, B::Scalar, D::Error>(bytes))
                .collect::<Result<Vec<_>, _>>()?,
            proof_g2: helper
                .proof_g2
                .iter()
                .map(|bytes| curve_point_from_bytes::<B::G2, B::Scalar, D::Error>(bytes))
                .collect::<Result<Vec<_>, _>>()?,
            shared_secret: target_group_from_bytes::<B::Target, D::Error>(&helper.shared_secret)?,
            threshold: helper.threshold,
            payload: helper.payload,
        })
    }
}

// Implement Serialize and Deserialize for PartialDecryption
impl<B: PairingBackend> Serialize for PartialDecryption<B> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut state = serializer.serialize_struct("PartialDecryption", 2)?;
        state.serialize_field("participant_id", &self.participant_id)?;
        state.serialize_field("response", &self.response.to_repr().as_ref())?;
        state.end()
    }
}

impl<'de, B: PairingBackend> Deserialize<'de> for PartialDecryption<B> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct PartialDecryptionHelper {
            participant_id: usize,
            response: Vec<u8>,
        }

        let helper = PartialDecryptionHelper::deserialize(deserializer)?;

        Ok(PartialDecryption {
            participant_id: helper.participant_id,
            response: curve_point_from_bytes::<B::G2, B::Scalar, D::Error>(&helper.response)?,
        })
    }
}

// Implement Serialize and Deserialize for DecryptionResult
impl Serialize for DecryptionResult {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut state = serializer.serialize_struct("DecryptionResult", 1)?;
        state.serialize_field("plaintext", &self.plaintext)?;
        state.end()
    }
}

impl<'de> Deserialize<'de> for DecryptionResult {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct DecryptionResultHelper {
            plaintext: Option<Vec<u8>>,
        }

        let helper = DecryptionResultHelper::deserialize(deserializer)?;
        Ok(DecryptionResult {
            plaintext: helper.plaintext,
        })
    }
}

// Implement Serialize and Deserialize for SRS
impl<B: PairingBackend<Scalar = Fr>> Serialize for SRS<B> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut state = serializer.serialize_struct("SRS", 3)?;
        state.serialize_field(
            "powers_of_g",
            &self
                .powers_of_g
                .iter()
                .map(|p| p.to_repr().as_ref().to_vec())
                .collect::<Vec<Vec<u8>>>(),
        )?;
        state.serialize_field(
            "powers_of_h",
            &self
                .powers_of_h
                .iter()
                .map(|p| p.to_repr().as_ref().to_vec())
                .collect::<Vec<Vec<u8>>>(),
        )?;
        state.serialize_field("e_gh", &self.e_gh.to_repr().as_ref())?;
        state.end()
    }
}

impl<'de, B: PairingBackend<Scalar = Fr>> Deserialize<'de> for SRS<B> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct SRSHelper {
            powers_of_g: Vec<Vec<u8>>,
            powers_of_h: Vec<Vec<u8>>,
            e_gh: Vec<u8>,
        }

        let helper = SRSHelper::deserialize(deserializer)?;

        Ok(SRS {
            powers_of_g: helper
                .powers_of_g
                .iter()
                .map(|bytes| curve_point_from_bytes::<B::G1, B::Scalar, D::Error>(bytes))
                .collect::<Result<Vec<_>, _>>()?,
            powers_of_h: helper
                .powers_of_h
                .iter()
                .map(|bytes| curve_point_from_bytes::<B::G2, B::Scalar, D::Error>(bytes))
                .collect::<Result<Vec<_>, _>>()?,
            e_gh: target_group_from_bytes::<B::Target, D::Error>(&helper.e_gh)?,
        })
    }
}

// Implement Serialize and Deserialize for LagrangePowers
impl<B: PairingBackend> Serialize for LagrangePowers<B> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut state = serializer.serialize_struct("LagrangePowers", 4)?;
        state.serialize_field(
            "li",
            &self
                .li
                .iter()
                .map(|p| p.to_repr().as_ref().to_vec())
                .collect::<Vec<Vec<u8>>>(),
        )?;
        state.serialize_field(
            "li_minus0",
            &self
                .li_minus0
                .iter()
                .map(|p| p.to_repr().as_ref().to_vec())
                .collect::<Vec<Vec<u8>>>(),
        )?;
        state.serialize_field(
            "li_x",
            &self
                .li_x
                .iter()
                .map(|p| p.to_repr().as_ref().to_vec())
                .collect::<Vec<Vec<u8>>>(),
        )?;
        state.serialize_field(
            "li_lj_z",
            &self
                .li_lj_z
                .iter()
                .map(|row| {
                    row.iter()
                        .map(|p| p.to_repr().as_ref().to_vec())
                        .collect::<Vec<Vec<u8>>>()
                })
                .collect::<Vec<_>>(),
        )?;
        state.end()
    }
}

impl<'de, B: PairingBackend> Deserialize<'de> for LagrangePowers<B> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct LagrangePowersHelper {
            li: Vec<Vec<u8>>,
            li_minus0: Vec<Vec<u8>>,
            li_x: Vec<Vec<u8>>,
            li_lj_z: Vec<Vec<Vec<u8>>>,
        }

        let helper = LagrangePowersHelper::deserialize(deserializer)?;

        Ok(LagrangePowers {
            li: helper
                .li
                .iter()
                .map(|bytes| curve_point_from_bytes::<B::G1, B::Scalar, D::Error>(bytes))
                .collect::<Result<Vec<_>, _>>()?,
            li_minus0: helper
                .li_minus0
                .iter()
                .map(|bytes| curve_point_from_bytes::<B::G1, B::Scalar, D::Error>(bytes))
                .collect::<Result<Vec<_>, _>>()?,
            li_x: helper
                .li_x
                .iter()
                .map(|bytes| curve_point_from_bytes::<B::G1, B::Scalar, D::Error>(bytes))
                .collect::<Result<Vec<_>, _>>()?,
            li_lj_z: helper
                .li_lj_z
                .iter()
                .map(|row| {
                    row.iter()
                        .map(|bytes| curve_point_from_bytes::<B::G1, B::Scalar, D::Error>(bytes))
                        .collect::<Result<Vec<_>, _>>()
                })
                .collect::<Result<Vec<_>, _>>()?,
        })
    }
}

// Implement Serialize and Deserialize for Params
impl<B: PairingBackend<Scalar = Fr>> Serialize for Params<B> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut state = serializer.serialize_struct("Params", 2)?;
        state.serialize_field("srs", &self.srs)?;
        state.serialize_field("lagrange_powers", &self.lagrange_powers)?;
        state.end()
    }
}

impl<'de, B: PairingBackend<Scalar = Fr>> Deserialize<'de> for Params<B> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        #[serde(bound(deserialize = ""))]
        struct ParamsHelper<B: PairingBackend<Scalar = Fr>> {
            srs: SRS<B>,
            lagrange_powers: LagrangePowers<B>,
        }

        let helper = ParamsHelper::deserialize(deserializer)?;
        Ok(Params {
            srs: helper.srs,
            lagrange_powers: helper.lagrange_powers,
        })
    }
}

// Implement Serialize and Deserialize for UnsafeKeyMaterial
impl<B: PairingBackend<Scalar = Fr>> Serialize for UnsafeKeyMaterial<B> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut state = serializer.serialize_struct("UnsafeKeyMaterial", 4)?;
        state.serialize_field("secret_keys", &self.secret_keys)?;
        state.serialize_field("public_keys", &self.public_keys)?;
        state.serialize_field("aggregate_key", &self.aggregate_key)?;
        state.serialize_field("kzg_params", &self.kzg_params)?;
        state.end()
    }
}

impl<'de, B: PairingBackend<Scalar = Fr>> Deserialize<'de> for UnsafeKeyMaterial<B> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        #[serde(bound(deserialize = ""))]
        struct UnsafeKeyMaterialHelper<B: PairingBackend<Scalar = Fr>> {
            secret_keys: Vec<SecretKey<B>>,
            public_keys: Vec<PublicKey<B>>,
            aggregate_key: AggregateKey<B>,
            kzg_params: SRS<B>,
        }

        let helper = UnsafeKeyMaterialHelper::deserialize(deserializer)?;
        Ok(UnsafeKeyMaterial {
            secret_keys: helper.secret_keys,
            public_keys: helper.public_keys,
            aggregate_key: helper.aggregate_key,
            kzg_params: helper.kzg_params,
        })
    }
}
