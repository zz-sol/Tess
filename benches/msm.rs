use criterion::{Criterion, black_box, criterion_group, criterion_main};
use rand::{SeedableRng, rngs::StdRng};
use tess::MsmProvider;

#[cfg(feature = "blst")]
fn bench_blst(c: &mut Criterion) {
    use blstrs::{G1Projective, Scalar};
    use ff::Field;
    use group::Group;
    use tess::{BlstG1, BlstMsm};

    let mut rng = StdRng::seed_from_u64(42);
    let size = 1 << 10; // 1024 terms
    let scalars: Vec<Scalar> = (0..size).map(|_| Scalar::random(&mut rng)).collect();
    let bases: Vec<BlstG1> = (0..size)
        .map(|_| BlstG1(G1Projective::random(&mut rng)))
        .collect();

    c.bench_function("blst/msm_g1_1024", |b| {
        b.iter(|| {
            let res = BlstMsm::msm_g1(black_box(&bases), black_box(&scalars)).unwrap();
            black_box(res);
        });
    });
}

#[cfg(feature = "ark_bls12381")]
fn bench_arkworks(c: &mut Criterion) {
    use ark_bls12_381::{Fr as BlsFr, G1Projective};
    use ark_std::UniformRand;
    use tess::{ArkG1, BlsMsm, MsmProvider};

    let mut rng = StdRng::seed_from_u64(42);
    let size = 1 << 10;
    let scalars: Vec<BlsFr> = (0..size).map(|_| BlsFr::rand(&mut rng)).collect();
    let bases: Vec<ArkG1> = (0..size)
        .map(|_| ArkG1(G1Projective::rand(&mut rng)))
        .collect();

    c.bench_function("arkworks/msm_g1_1024", |b| {
        b.iter(|| {
            let res = BlsMsm::msm_g1(black_box(&bases), black_box(&scalars)).unwrap();
            black_box(res);
        });
    });
}

fn criterion_benches(c: &mut Criterion) {
    #[cfg(feature = "blst")]
    bench_blst(c);
    #[cfg(feature = "ark_bls12381")]
    bench_arkworks(c);
}

criterion_group!(benches, criterion_benches);
criterion_main!(benches);
