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
use tracing_subscriber::{EnvFilter, Registry};
use p3_field::Field;
use p3_matrix::dense::RowMajorMatrix;
use p3_poseidon2::{Poseidon2, Poseidon2ExternalMatrixGeneral};

type Val = Mersenne31;

fn main() {
    let mut rng = rand::thread_rng();

    // let env_filter = EnvFilter::builder()
    //     .with_default_directive(LevelFilter::INFO.into())
    //     .from_env_lossy();
    //
    // Registry::default()
    //     .with(env_filter)
    //     .with(ForestLayer::default())
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

            let (_comm, prover_data) = tracing::info_span!("commit").in_scope(|| {
                let time = std::time::Instant::now();
                let res = <Pcs as p3_commit::Pcs<Challenge, Challenger>>::commit(&pcs, vec![(d, evals.clone())]);
                println!("commit time: {:?}", time.elapsed());
                res
            });

            let mut challenger = Challenger::new(perm.clone());
            let zeta: Challenge = challenger.sample_ext_element();

            tracing::info_span!("open").in_scope(|| {
                let time = std::time::Instant::now();
                pcs.open(vec![(&prover_data, vec![vec![zeta]])], &mut challenger);
                println!("open time: {:?}", time.elapsed());
            });
        }
    }

}
