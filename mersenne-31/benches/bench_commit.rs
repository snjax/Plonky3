use criterion::{criterion_group, criterion_main, Criterion};
use p3_field::Field;
use p3_field::extension::BinomialExtensionField;
use p3_matrix::dense::RowMajorMatrix;
use p3_mersenne_31::{Mersenne31, Poseidon2Mersenne31};
use p3_commit::ExtensionMmcs;
use p3_circle::CirclePcs;
use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};
use rand::thread_rng;
use p3_merkle_tree::MerkleTreeMmcs;
use core::marker::PhantomData;
use p3_fri::FriConfig;
use p3_challenger::DuplexChallenger;

use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

fn commit_benchmark(c: &mut Criterion) {
    let mut rng = ChaCha8Rng::from_seed([0; 32]);

    type Val = Mersenne31;
    type Challenge = BinomialExtensionField<Mersenne31, 3>;

    type Perm = Poseidon2Mersenne31<16>;
    let perm = Perm::new_from_rng_128(&mut thread_rng());

    type MyHash = PaddingFreeSponge<Perm, 16, 8, 8>;
    let hash = MyHash::new(perm.clone());

    type MyCompress = TruncatedPermutation<Perm, 2, 8, 16>;
    let compress = MyCompress::new(perm.clone());

    type ValMmcs = MerkleTreeMmcs<
        <Val as Field>::Packing,
        <Val as Field>::Packing,
        MyHash,
        MyCompress,
        8,
    >;
    let val_mmcs = ValMmcs::new(hash, compress);

    type ChallengeMmcs = ExtensionMmcs<Val, Challenge, ValMmcs>;
    let challenge_mmcs = ChallengeMmcs::new(val_mmcs.clone());

    type Challenger = DuplexChallenger<Val, Perm, 16, 8>;

    let fri_config = FriConfig {
        log_blowup: 1,
        log_final_poly_len: 0,
        num_queries: 80,
        proof_of_work_bits: 16,
        mmcs: challenge_mmcs,
    };

    dbg!(fri_config.conjectured_soundness_bits());

    type Pcs = CirclePcs<Val, ValMmcs, ChallengeMmcs>;
    let pcs = Pcs {
        mmcs: val_mmcs,
        fri_config,
        _phantom: PhantomData,
    };

    let log_n = 14;
    let w = 64;

    let d = <Pcs as p3_commit::Pcs<Challenge, Challenger>>::natural_domain_for_degree(
        &pcs,
        1 << log_n,
    );

    let evals = RowMajorMatrix::<Val>::rand(&mut rng, 1 << log_n, w, );

    c.bench_function(&format!("commit 1<<{}x{} elements", log_n, w), |b| {
        b.iter(|| {
            let (_comm, _data) =
                <Pcs as p3_commit::Pcs<Challenge, Challenger>>::commit(&pcs, vec![(d, evals.clone())]);
        });
    });
}




criterion_group!{
    name=benches;
    config = Criterion::default();
    targets = commit_benchmark
}

criterion_main!(benches);