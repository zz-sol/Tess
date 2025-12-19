use criterion::{Criterion, black_box, criterion_group, criterion_main};
use rand::SeedableRng;
use rand::rngs::StdRng;

use tess::{PairingEngine, SilentThresholdScheme, ThresholdEncryption};

/// Simple benchmark that runs the full flow (setup done once):
/// - param_gen (SRS + lagrange powers) is executed once outside measured loops
/// - keygen is executed once outside measured loops
/// - encryption is measured
/// - aggregate decryption (using t partials) is measured
pub fn bench_threshold(c: &mut Criterion) {
    // Deterministic RNG for repeatable benchmarks
    let mut rng = StdRng::seed_from_u64(0xdead_beef);

    // Use the concrete pairing backend (default feature `blst` should be enabled)
    type BE = PairingEngine;

    let scheme = SilentThresholdScheme::<BE>::new();

    // Parameters
    let parties = 16usize;
    let threshold = 3usize;

    // One-time setup (not measured)
    let params = scheme
        .param_gen(&mut rng, parties, threshold)
        .expect("param_gen failed");

    let key_material = scheme
        .keygen(&mut rng, parties, &params)
        .expect("keygen failed");

    // Measure encryption
    c.bench_function("threshold_encrypt", |b| {
        b.iter(|| {
            let payload = b"The quick brown fox jumps over the lazy dog";
            let ct = scheme
                .encrypt(
                    &mut rng,
                    &key_material.aggregate_key,
                    &params,
                    threshold,
                    payload,
                )
                .expect("encrypt failed");
            black_box(ct);
        })
    });

    // Prepare a ciphertext and partial decryptions for decryption benchmark
    let payload = b"benchmark payload for threshold decryption";
    let ct = scheme
        .encrypt(
            &mut rng,
            &key_material.aggregate_key,
            &params,
            threshold,
            payload,
        )
        .expect("encrypt failed");

    // Collect partial decryptions from the first `threshold + 1` participants
    let share_count = threshold + 1;
    let mut partials = Vec::with_capacity(share_count);
    let mut selector = vec![false; parties];
    for i in 0..share_count {
        selector[i] = true;
        let p = scheme
            .partial_decrypt(&key_material.secret_keys[i], &ct)
            .expect("partial_decrypt failed");
        partials.push(p);
    }

    c.bench_function("threshold_aggregate_decrypt", |b| {
        b.iter(|| {
            let res = scheme
                .aggregate_decrypt(&ct, &partials, &selector, &key_material.aggregate_key)
                .expect("aggregate_decrypt failed");
            black_box(res);
        })
    });
}

criterion_group!(benches, bench_threshold);
criterion_main!(benches);
