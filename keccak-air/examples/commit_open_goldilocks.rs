use std::time::Duration;
use p3_challenger::{DuplexChallenger, FieldChallenger};
use p3_commit::{ExtensionMmcs, Pcs};
use p3_field::extension::BinomialExtensionField;
use p3_fri::{FriConfig, TwoAdicFriPcs};
use p3_merkle_tree::FieldMerkleTreeMmcs;
use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};
use tracing_forest::util::LevelFilter;
use tracing_forest::ForestLayer;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::{EnvFilter, Registry};
use p3_dft::Radix2DitParallel;
use p3_field::Field;
use p3_goldilocks::{DiffusionMatrixGoldilocks, Goldilocks};
use p3_matrix::dense::RowMajorMatrix;
use p3_poseidon2::{Poseidon2, Poseidon2ExternalMatrixGeneral};

fn main() {
    let mut rng = rand::thread_rng();

    let env_filter = EnvFilter::builder()
        .with_default_directive(LevelFilter::INFO.into())
        .from_env_lossy();

    Registry::default()
        .with(env_filter)
        .with(ForestLayer::default())
        .init();

    type Val = Goldilocks;
    type Challenge = BinomialExtensionField<Val, 2>;


    type Perm = Poseidon2<Val, Poseidon2ExternalMatrixGeneral, DiffusionMatrixGoldilocks, 8, 7>;
    let perm = Perm::new_from_rng_128(
        Poseidon2ExternalMatrixGeneral,
        DiffusionMatrixGoldilocks,
        &mut rng,
    );

    type MyHash = PaddingFreeSponge<Perm, 8, 4, 4>;
    let hash = MyHash::new(perm.clone());

    type MyCompress = TruncatedPermutation<Perm, 2, 4, 8>;
    let compress = MyCompress::new(perm.clone());

    type ValMmcs = FieldMerkleTreeMmcs<
        <Val as Field>::Packing,
        <Val as Field>::Packing,
        MyHash,
        MyCompress,
        4,
    >;
    let val_mmcs = ValMmcs::new(hash, compress);

    type ChallengeMmcs = ExtensionMmcs<Val, Challenge, ValMmcs>;
    let challenge_mmcs = ChallengeMmcs::new(val_mmcs.clone());

    type Dft = Radix2DitParallel;
    let dft = Dft {};

    type Challenger = DuplexChallenger<Val, Perm, 8, 4>;

    let fri_config = FriConfig {
        log_blowup: 1,
        num_queries: 80,
        proof_of_work_bits: 16,
        mmcs: challenge_mmcs,
    };

    type Pcs = TwoAdicFriPcs<Val, Dft, ValMmcs, ChallengeMmcs>;
    let pcs = Pcs::new(dft, val_mmcs, fri_config);

    let mut id = 0;
    for log_n in (10..=20).step_by(2) {
        for w in (1..=64).step_by(2) {
            let d = <Pcs as p3_commit::Pcs<Challenge, Challenger>>::natural_domain_for_degree(
                &pcs,
                1 << log_n,
            );

            let evals = RowMajorMatrix::<Val>::rand(&mut rng, 1 << log_n, w);

            const NUM_TRIES: usize = 100;

            let (avg_time, (_comm, prover_data)) = run_n_times(NUM_TRIES, || {
                tracing::info_span!("commit").in_scope(|| {
                    tracing::info!("id: {}", id);
                    <Pcs as p3_commit::Pcs<Challenge, Challenger>>::commit(&pcs, vec![(d, evals.clone())])
                })
            });

            let avg_time = Duration::from_secs_f64(avg_time);
            tracing::info!("commit avg time: {:?}", avg_time);

            let mut challenger = Challenger::new(perm.clone());
            let zeta: Challenge = challenger.sample_ext_element();

            let (avg_time, _) = run_n_times(NUM_TRIES, || {
                tracing::info_span!("open").in_scope(|| {
                    tracing::info!("id: {}", id);
                    pcs.open(vec![(&prover_data, vec![vec![zeta]])], &mut challenger);
                });
            });


            let avg_time = Duration::from_secs_f64(avg_time);
            tracing::info!("open avg time: {:?}", avg_time);

            id += 1;
        }
    }
}


fn run_n_times<T>(n: usize, mut f: impl FnMut() -> T) -> (f64, T) {
    let mut times = Vec::with_capacity(n);
    let res = f();
    for _ in 0..n {
        let time = std::time::Instant::now();
        f();
        times.push(time.elapsed().as_secs_f64());
    }

    (times.iter().sum::<f64>() / n as f64, res)
}