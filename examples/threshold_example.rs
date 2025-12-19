use rand::{SeedableRng, rngs::StdRng};
use tracing::info;
use tracing_subscriber::fmt;

use tess::{PairingEngine, SilentThresholdScheme, ThresholdEncryption};

const PARTIES: usize = 2048;
const THRESHOLD: usize = 1400;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    fmt()
        .with_max_level(tracing::Level::INFO)
        .with_span_events(fmt::format::FmtSpan::ENTER | fmt::format::FmtSpan::CLOSE)
        .with_target(false)
        .with_ansi(false)
        .init();

    let mut rng = StdRng::seed_from_u64(42);

    let scheme = SilentThresholdScheme::<PairingEngine>::new();

    info!(
        parties = PARTIES,
        threshold = THRESHOLD,
        "starting threshold example"
    );

    // Generate params (SRS + precomputed lagrange powers)
    let params = scheme.param_gen(&mut rng, PARTIES, THRESHOLD)?;

    // Generate key material
    let key_material = scheme.keygen(&mut rng, PARTIES, &params)?;

    // Example message
    let message = vec![0u8; 32];

    // Encrypt
    let ciphertext = scheme.encrypt(
        &mut rng,
        &key_material.aggregate_key,
        &params,
        THRESHOLD,
        &message,
    )?;

    // Prepare selector and collect partials from the first `THRESHOLD + 1` participants
    let share_count = THRESHOLD + 1;
    let mut selector = vec![false; PARTIES];
    let mut partials = Vec::with_capacity(share_count);
    for (i, selected) in selector.iter_mut().enumerate().take(share_count) {
        *selected = true;
        let p = scheme.partial_decrypt(&key_material.secret_keys[i], &ciphertext)?;
        partials.push(p);
    }

    // Aggregate decrypt
    let result = scheme.aggregate_decrypt(
        &ciphertext,
        &partials,
        &selector,
        &key_material.aggregate_key,
    )?;

    info!(
        recovered = result.plaintext.is_some(),
        "decryption finished"
    );

    if let Some(plain) = result.plaintext {
        info!(matches = (plain == message), "plaintext equals original");
    } else {
        info!("no plaintext recovered");
    }

    Ok(())
}
