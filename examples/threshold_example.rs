use rand::{SeedableRng, rngs::StdRng};

#[cfg(feature = "parallel")]
use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};
use tracing::info;
#[cfg(feature = "tracing-subscriber")]
use tracing_subscriber::fmt;

use tess::{AggregateKey, PairingEngine, SilentThresholdScheme, ThresholdEncryption};

const PARTIES: usize = 2048;
const THRESHOLD: usize = 1400;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging when tracing-subscriber is available.
    #[cfg(feature = "tracing-subscriber")]
    {
        fmt()
            .with_max_level(tracing::Level::INFO)
            .with_span_events(fmt::format::FmtSpan::ENTER | fmt::format::FmtSpan::CLOSE)
            .with_target(false)
            .with_ansi(false)
            .init();
    }

    let mut rng = StdRng::seed_from_u64(42);

    let scheme = SilentThresholdScheme::<PairingEngine>::new();

    info!(
        parties = PARTIES,
        threshold = THRESHOLD,
        "starting threshold example"
    );

    // Generate params (SRS + precomputed lagrange powers)
    let params = scheme.param_gen(&mut rng, PARTIES, THRESHOLD)?;

    // Generate key material for all participants
    #[cfg(feature = "parallel")]
    let validator_keys = {
        let rngs = (0..PARTIES)
            .map(|_| StdRng::from_rng(&mut rng).unwrap())
            .collect::<Vec<_>>();
        rngs.into_par_iter()
            .enumerate()
            .map(|(i, mut rng)| scheme.keygen_single_validator(&mut rng, i, &params))
            .collect::<Result<Vec<_>, _>>()?
    };

    #[cfg(not(feature = "parallel"))]
    let validator_keys = (0..PARTIES)
        .map(|i| scheme.keygen_single_validator(&mut rng, i, &params))
        .collect::<Result<Vec<_>, _>>()?;

    let validator_public_keys = validator_keys
        .iter()
        .map(|k| &k.1)
        .cloned()
        .collect::<Vec<_>>();
    let validator_secret_keys = validator_keys.into_iter().map(|k| k.0).collect::<Vec<_>>();

    let aggregate_key = AggregateKey::aggregate_keys(&validator_public_keys, &params, PARTIES)?;

    // Example message
    let message = vec![0u8; 32];

    // Encrypt
    let ciphertext = scheme.encrypt(&mut rng, &aggregate_key, &params, THRESHOLD, &message)?;

    // Prepare selector and collect partials from the first `THRESHOLD + 1` participants
    let share_count = THRESHOLD + 1;
    let mut selector = vec![false; PARTIES];
    let mut partials = Vec::with_capacity(share_count);
    for (i, selected) in selector.iter_mut().enumerate().take(share_count) {
        *selected = true;
        let p = scheme.partial_decrypt(&validator_secret_keys[i], &ciphertext)?;
        partials.push(p);
    }

    // Aggregate decrypt
    let result = scheme.aggregate_decrypt(&ciphertext, &partials, &selector, &aggregate_key)?;

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
