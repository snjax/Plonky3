use p3_challenger::{DuplexChallenger, FieldChallenger};
use p3_commit::{ExtensionMmcs, OpenedValues, Pcs, UnivariatePcs};
use p3_dft::Radix2DitParallel;
use p3_field::extension::BinomialExtensionField;
use p3_field::Field;
use p3_fri::{FriBasedPcs, FriConfigImpl, FriLdt, FriProof};
use p3_goldilocks::Goldilocks;
use p3_keccak::Keccak256Hash;
use p3_ldt::{LdtBasedPcs, QuotientMmcs};
use p3_mds::coset_mds::CosetMds;
use p3_merkle_tree::{FieldMerkleTree, FieldMerkleTreeMmcs};
use p3_poseidon2::{DiffusionMatrixGoldilocks, Poseidon2};
use p3_symmetric::{CompressionFunctionFromHasher, SerializingHasher64};
use p3_uni_stark::{prove, verify, StarkConfigImpl, VerificationError};
use rand::{random, thread_rng};
use tracing_forest::util::LevelFilter;
use tracing_forest::ForestLayer;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::{EnvFilter, Registry};
use p3_keccak_air::KeccakAir;
use p3_matrix::dense::RowMajorMatrix;


fn main() -> Result<(), VerificationError> {
    let env_filter = EnvFilter::builder()
        .with_default_directive(LevelFilter::INFO.into())
        .from_env_lossy();

    Registry::default()
        .with(env_filter)
        .with(ForestLayer::default())
        .init();

    type Val = Goldilocks;
    type Domain = Val;
    type Challenge = BinomialExtensionField<Val, 2>;
    type PackedChallenge = BinomialExtensionField<<Domain as Field>::Packing, 2>;

    type MyMds = CosetMds<Val, 8>;
    let mds = MyMds::default();

    type Perm = Poseidon2<Val, MyMds, DiffusionMatrixGoldilocks, 8, 5>;
    let perm = Perm::new_from_rng(8, 22, mds, DiffusionMatrixGoldilocks, &mut thread_rng());

    type MyHash = SerializingHasher64<Keccak256Hash>;
    let hash = MyHash::new(Keccak256Hash {});
    type MyCompress = CompressionFunctionFromHasher<Val, MyHash, 2, 4>;
    let compress = MyCompress::new(hash);

    type ValMmcs = FieldMerkleTreeMmcs<Val, MyHash, MyCompress, 4>;
    let val_mmcs = ValMmcs::new(hash, compress);

    type ChallengeMmcs = ExtensionMmcs<Val, Challenge, ValMmcs>;
    let challenge_mmcs = ChallengeMmcs::new(val_mmcs.clone());

    type Dft = Radix2DitParallel;
    let dft = Dft {};

    type Challenger = DuplexChallenger<Val, Perm, 8>;

    type Quotient = QuotientMmcs<Domain, Challenge, ValMmcs>;
    type MyFriConfig = FriConfigImpl<Val, Challenge, Quotient, ChallengeMmcs, Challenger>;
    let fri_config = MyFriConfig::new(40, challenge_mmcs);
    let ldt = FriLdt { config: fri_config };

    type Pcs = FriBasedPcs<MyFriConfig, ValMmcs, Dft, Challenger>;
    type MyConfig = StarkConfigImpl<Val, Challenge, PackedChallenge, Pcs, Challenger>;

    let pcs = Pcs::new(dft, val_mmcs, ldt);
    // let _config = StarkConfigImpl::new(pcs);

    let fsize_bytes = core::mem::size_of::<Goldilocks>();

    let kilo = 2usize.pow(10);
    let mega = 2usize.pow(20);
    let giga = 2usize.pow(30);

    for n_bytes in [kilo, 128 * kilo, 512 * kilo, mega, 128 * mega, 512 * mega, giga].iter() {
        let n_rows = n_bytes / fsize_bytes;
        tracing::info!("{n_bytes} bytes, {n_rows} rows:");

        let inputs = (0..n_rows).map(|_| random()).collect::<Vec<_>>();
        let data = RowMajorMatrix::new(inputs.clone(), 1);
        let (commitment, prover_data): ([Val; 4], FieldMerkleTree<Val, 4>) = pcs.commit_batch(data);

        let mut challenger = Challenger::new(perm.clone());
        let zeta: Challenge = challenger.sample_ext_element();
        <LdtBasedPcs<_, _, _, _, _, _> as UnivariatePcs<_, _, RowMajorMatrix<Val>, _>>::open_multi_batches(&pcs, &[(&prover_data, &[zeta])], &mut challenger);
    }

    Ok(())
}
