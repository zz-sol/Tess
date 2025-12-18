use ark_bls12_381::Fr as ArkFr;
use ark_ff::{FftField, Field, One as ArkOne, UniformRand, Zero};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use rand_core::RngCore;

use crate::{BackendError, FieldElement};

pub type Fr = ArkFr;

impl FieldElement for Fr {
    type Repr = Vec<u8>;

    fn zero() -> Self {
        Zero::zero()
    }

    fn one() -> Self {
        ArkOne::one()
    }

    fn random<R: RngCore + ?Sized>(rng: &mut R) -> Self {
        Fr::rand(rng)
    }

    fn invert(&self) -> Option<Self> {
        self.inverse()
    }

    fn pow(&self, exp: &[u64; 4]) -> Self {
        Field::pow(self, exp)
    }

    fn to_repr(&self) -> Self::Repr {
        let mut bytes = Vec::new();
        self.serialize_compressed(&mut bytes)
            .expect("scalar serialization");
        bytes
    }

    fn from_repr(repr: &Self::Repr) -> Result<Self, BackendError> {
        Self::deserialize_compressed(repr.as_slice())
            .map_err(|_| BackendError::Serialization("invalid scalar bytes"))
    }

    fn two_adic_root_of_unity() -> Self {
        <Fr as FftField>::get_root_of_unity(1u64 << <Fr as FftField>::TWO_ADICITY)
            .unwrap_or_else(<Fr as ArkOne>::one)
    }

    fn two_adicity_generator(n: usize) -> Self {
        if n == 1 {
            return <Fr as ArkOne>::one();
        }

        // Get the 2-adic root of unity and raise it to the appropriate power
        let root = Self::two_adic_root_of_unity();
        let k = (n - 1).next_power_of_two().trailing_zeros() as usize + 1;
        let exp_power = (1u64 << k) / n as u64;

        // Convert to [u64; 4] format for pow
        let mut exp = [0u64; 4];
        exp[0] = exp_power;
        ark_ff::Field::pow(&root, &exp)
    }

    fn batch_inversion(elements: &mut [Self]) -> Result<(), BackendError> {
        if elements.is_empty() {
            return Ok(());
        }

        let mut prod = <Fr as ArkOne>::one();
        let mut products = Vec::with_capacity(elements.len());

        for elem in elements.iter() {
            if elem.is_zero() {
                return Err(BackendError::Math("cannot invert zero element"));
            }
            products.push(prod);
            prod *= *elem;
        }

        let mut inv_prod = prod
            .inverse()
            .ok_or(BackendError::Math("batch inversion failed"))?;
        for (i, elem) in elements.iter_mut().enumerate().rev() {
            *elem = inv_prod * products[i];
            inv_prod *= *elem;
        }

        Ok(())
    }

    fn from_u64(n: u64) -> Self {
        Fr::from(n)
    }
}
