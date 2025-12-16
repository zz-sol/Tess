use rand::{SeedableRng, rngs::StdRng};
use std::time::Instant;
use tess::{TargetGroup, ThresholdScheme};
use tracing::{info, instrument};
use tracing_subscriber::fmt::init as init_tracing;

use tess::{
    config::{BackendConfig, BackendId, CurveId, ThresholdParameters},
    protocol::{ProtocolBackend, ProtocolScalar, SilentThreshold},
};

#[cfg(feature = "ark_bls12381")]
use tess::ArkworksBls12;
#[cfg(feature = "ark_bn254")]
use tess::ArkworksBn254;
#[cfg(feature = "blst")]
use tess::BlstBackend;

const PARTIES: usize = 1 << 11; // 16
const THRESHOLD: usize = 1400;

#[instrument(level = "info", skip(backend_config, backend_name), fields(backend = %backend_name))]
fn run_threshold_example<B>(
    backend_name: &str,
    backend_config: BackendConfig,
) -> Result<(), Box<dyn std::error::Error>>
where
    B: ProtocolBackend,
    B::Scalar: ProtocolScalar,
{
    let mut rng = StdRng::seed_from_u64(42);
    let scheme = SilentThreshold::<B>::default();
    let params = ThresholdParameters {
        parties: PARTIES,
        threshold: THRESHOLD,
        chunk_size: 32,
        backend: backend_config,
    };

    let srs = scheme.srs_gen(&mut rng, &params)?;

    info!("starting benchmark");
    let key_material = {
        // let _span = tracing::info_span!("keygen").entered();
        let start = Instant::now();
        let km = scheme.keygen(&mut rng, &params, &srs)?;
        info!(duration = ?start.elapsed(), "key generation finished");
        km
    };

    let message = vec![0u8; params.chunk_size];
    let ciphertext = {
        // let _span = tracing::info_span!("encrypt").entered();
        let start = Instant::now();
        let ct = scheme.encrypt(&mut rng, &key_material.aggregate_key, &params, &message)?;
        info!(duration = ?start.elapsed(), "encryption finished");
        ct
    };

    let mut selector = vec![false; PARTIES];

    // Select threshold + 1 parties for decryption
    let mut partials = Vec::with_capacity(THRESHOLD + 1);
    for idx in 0..=THRESHOLD {
        selector[idx] = true;
        // let _guard = trace_span!("partial_decrypt", idx).entered();
        let partial = scheme.partial_decrypt(&key_material.secret_keys[idx], &ciphertext)?;
        partials.push(partial);
    }

    let result = {
        // let _span = tracing::info_span!("aggregate_decrypt").entered();
        let start = Instant::now();
        let res = scheme.aggregate_decrypt(
            &ciphertext,
            &partials,
            &selector,
            &key_material.aggregate_key,
        )?;
        info!(duration = ?start.elapsed(), "aggregate decryption finished");
        res
    };

    println!(
        "shared secrets equal: {}",
        ciphertext.shared_secret.to_repr().as_ref() == result.shared_secret.to_repr().as_ref()
    );
    println!(
        "Recovered plaintext matches: {} (len = {})",
        result.plaintext.as_deref() == Some(message.as_slice()),
        result
            .plaintext
            .as_ref()
            .map(|p| p.len())
            .unwrap_or_default()
    );

    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    init_tracing();
    let mut executed = 0;

    #[cfg(feature = "blst")]
    {
        run_threshold_example::<BlstBackend>(
            "blst (BLS12-381)",
            BackendConfig::new(BackendId::Blst, CurveId::Bls12_381),
        )?;
        executed += 1;
    }

    #[cfg(feature = "ark_bls12381")]
    {
        run_threshold_example::<ArkworksBls12>(
            "arkworks (BLS12-381)",
            BackendConfig::new(BackendId::Arkworks, CurveId::Bls12_381),
        )?;
        executed += 1;
    }

    #[cfg(feature = "ark_bn254")]
    {
        run_threshold_example::<ArkworksBn254>(
            "arkworks (BN254)",
            BackendConfig::new(BackendId::Arkworks, CurveId::Bn254),
        )?;
        executed += 1;
    }

    if executed == 0 {
        eprintln!(
            "Enable at least one backend feature (e.g., `--features blst`) to run the example."
        );
    }

    Ok(())
}
