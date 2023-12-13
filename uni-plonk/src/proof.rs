use alloc::vec::Vec;
use p3_commit::Pcs;
use p3_matrix::dense::{RowMajorMatrix};

use crate::Config;

type Val<C> = <C as Config>::Val;
type ValMat<C> = RowMajorMatrix<Val<C>>;
type Com<C> = <<C as Config>::Pcs as Pcs<Val<C>, ValMat<C>>>::Commitment;
type PcsProof<C> = <<C as Config>::Pcs as Pcs<Val<C>, ValMat<C>>>::Proof;

pub struct Proof<C: Config> {
    pub(crate) commitments: Commitments<Com<C>>,
    pub(crate) opened_values: OpenedValues<C::Challenge>,
    pub(crate) opening_proof: PcsProof<C>,
    pub(crate) multiset_sums: Vec<C::Challenge>,
    pub(crate) log_degree: u64
}



pub struct Commitments<Com> {
    pub fixed: Com,
    pub advice: Com,
    pub multiset_f: Com,
    pub quotient: Com,
}

pub struct OpenedValues<Challenge>{
    pub fixed_local: Vec<Challenge>,
    pub fixed_next: Vec<Challenge>,
    pub advice_local: Vec<Challenge>,
    pub advice_next: Vec<Challenge>,
    pub multiset_f_local: Vec<Challenge>,
    pub multiset_f_next: Vec<Challenge>,
    pub quotient: Vec<Challenge>,

}
