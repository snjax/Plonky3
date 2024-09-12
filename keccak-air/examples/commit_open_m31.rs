use std::fs::{File, OpenOptions};
use std::marker::PhantomData;

use p3_challenger::{DuplexChallenger, FieldChallenger};
use p3_circle::CirclePcs;
use p3_commit::{ExtensionMmcs, Pcs};
use p3_field::extension::BinomialExtensionField;
use p3_fri::FriConfig;
use p3_merkle_tree::FieldMerkleTreeMmcs;
use p3_mersenne_31::{DiffusionMatrixMersenne31, Mersenne31};
use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};
use tracing_forest::util::LevelFilter;
use tracing_forest::ForestLayer;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::{EnvFilter, Layer, Registry};
use p3_field::Field;
use p3_matrix::dense::RowMajorMatrix;
use p3_poseidon2::{Poseidon2, Poseidon2ExternalMatrixGeneral};
use serde::Serialize;

type Val = Mersenne31;

#[derive(Serialize)]
struct StatsEntry {
    id: usize,
    log_n: usize,
    w: usize,
    log_blowup: usize,
    num_queries: usize,
    proof_of_work_bits: usize,
    avg_time: f64,
}

fn main() {
    let mut rng = rand::thread_rng();

    // write logs to file
    let env_filter = EnvFilter::builder()
        .with_default_directive(LevelFilter::INFO.into())
        .from_env_lossy();

    let log_file = File::create("commit_open_m31.log").unwrap();

    // TODO: Write either the full forest output to the file, or filter out the "grind" entries.
    Registry::default()
        .with(env_filter)
        .with(
            ForestLayer::default()
        )
        .init();

    // tracing_subscriber::fmt()
    //     .with_span_events(tracing_subscriber::fmt::format::FmtSpan::ENTER)
    //     // .json()
    //     .with_ansi(false)
    //     .with_level(false)
    //     .without_time()
    //     .with_timer(tracing_subscriber::fmt::time::ChronoUtc::rfc_3339())
    //     .pretty()
    //     .with_writer(log_file)
    //     .init();


    type Challenge = BinomialExtensionField<Mersenne31, 3>;

    type Perm = Poseidon2<Val, Poseidon2ExternalMatrixGeneral, DiffusionMatrixMersenne31, 16, 5>;
    let perm = Perm::new_from_rng_128(
        Poseidon2ExternalMatrixGeneral,
        DiffusionMatrixMersenne31,
        &mut rng,
    );
    type MyHash = PaddingFreeSponge<Perm, 16, 8, 8>;
    let hash = MyHash::new(perm.clone());

    type MyCompress = TruncatedPermutation<Perm, 2, 8, 16>;
    let compress = MyCompress::new(perm.clone());

    type ValMmcs = FieldMerkleTreeMmcs<
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

    let mut commit_stats = Vec::new();
    let mut open_stats = Vec::new();

    let mut id = 0;
    for log_n in (10..=20).step_by(2) {
        for w in (1..=64).step_by(2) {
            println!("log_n: {}, w: {}", log_n, w);

            let fri_config = FriConfig {
                log_blowup: 1,
                num_queries: 80,
                proof_of_work_bits: 16,
                mmcs: challenge_mmcs.clone(),
            };

            type Pcs = CirclePcs<Val, ValMmcs, ChallengeMmcs>;
            let pcs = Pcs {
                mmcs: val_mmcs.clone(),
                fri_config,
                _phantom: PhantomData,
            };

            let d = <Pcs as p3_commit::Pcs<Challenge, Challenger>>::natural_domain_for_degree(
                &pcs,
                1 << log_n,
            );

            let evals = RowMajorMatrix::<Val>::rand(&mut rng, 1 << log_n, w);

            let (avg_time, (_comm, prover_data)) = run_n_times(100, || {
                tracing::info_span!("commit").in_scope(|| {
                    tracing::info!("id: {}", id);
                    <Pcs as p3_commit::Pcs<Challenge, Challenger>>::commit(&pcs, vec![(d, evals.clone())])
                })
            });

            commit_stats.push(StatsEntry {
                id,
                log_n,
                w,
                log_blowup: pcs.fri_config.log_blowup,
                num_queries: pcs.fri_config.num_queries,
                proof_of_work_bits: pcs.fri_config.proof_of_work_bits,
                avg_time,
            });

            let mut challenger = Challenger::new(perm.clone());
            let zeta: Challenge = challenger.sample_ext_element();

            let (avg_time, _) = run_n_times(100, || {
                tracing::info_span!("open").in_scope(|| {
                    tracing::info!("id: {}", id);
                    pcs.open(vec![(&prover_data, vec![vec![zeta]])], &mut challenger);
                });
            });

            open_stats.push(StatsEntry {
                id,
                log_n,
                w,
                log_blowup: pcs.fri_config.log_blowup,
                num_queries: pcs.fri_config.num_queries,
                proof_of_work_bits: pcs.fri_config.proof_of_work_bits,
                avg_time,
            });

            id += 1;
        }
    }

    serde_json::to_writer_pretty(File::create("commit_m31_stats.json").unwrap(), &commit_stats).unwrap();
    serde_json::to_writer_pretty(File::create("open_m31_stats.json").unwrap(), &open_stats).unwrap();
}

fn run_n_times<T>(n: usize, mut f: impl FnMut() -> T) -> (f64, T) {
    let mut times = Vec::with_capacity(n);
    let mut res = f();
    for _ in 0..n {
        let time = std::time::Instant::now();
        res = f();
        times.push(time.elapsed().as_secs_f64());
    }

    (times.iter().sum::<f64>() / n as f64, res)
}