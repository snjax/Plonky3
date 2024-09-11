use std::fmt::Debug;

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


    type Perm = Poseidon2<Val, Poseidon2ExternalMatrixGeneral, DiffusionMatrixGoldilocks, 16, 7>;
    let perm = Perm::new_from_rng_128(
        Poseidon2ExternalMatrixGeneral,
        DiffusionMatrixGoldilocks,
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

    type Dft = Radix2DitParallel;
    let dft = Dft {};

    type Challenger = DuplexChallenger<Val, Perm, 16, 8>;

    let fri_config = FriConfig {
        log_blowup: 1,
        num_queries: 80,
        proof_of_work_bits: 16,
        mmcs: challenge_mmcs,
    };

    type Pcs = TwoAdicFriPcs<Val, Dft, ValMmcs, ChallengeMmcs>;
    let pcs = Pcs::new(dft, val_mmcs, fri_config);

    let log_n = 14;
    let w = 64;

    let d = <Pcs as p3_commit::Pcs<Challenge, Challenger>>::natural_domain_for_degree(
        &pcs,
        1 << log_n,
    );

    let evals = RowMajorMatrix::<Val>::rand(&mut rng, 1 << log_n, w);

    let (_comm, prover_data) =
        <Pcs as p3_commit::Pcs<Challenge, Challenger>>::commit(&pcs, vec![(d, evals.clone())]);

    let mut challenger = Challenger::new(perm.clone());
    let zeta: Challenge = challenger.sample_ext_element();
    pcs.open(vec![(&prover_data, vec![vec![zeta]])], &mut challenger);
}
