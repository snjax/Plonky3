use alloc::vec;
use alloc::vec::Vec;

use p3_air::{Air, TwoRowMatrixView};
use p3_challenger::{CanObserve, FieldChallenger};
use p3_commit::UnivariatePcs;
use p3_dft::reverse_slice_index_bits;
use p3_field::{AbstractExtensionField, AbstractField, Field, TwoAdicField};
use p3_matrix::dense::RowMajorMatrix;

use crate::{Proof, Config, Engine, to_values};

use p3_uni_stark::{VerificationError};

pub fn verify<C, E>(
    config: &C,
    challenger: &mut C::Challenger,
    proof: &Proof<C>,
    instance: RowMajorMatrix<C::Val>
) -> Result<(), VerificationError>
    where
        C: Config,
        E: Engine<F=C::Val, EF=C::Challenge>,
{
    let Proof {
        commitments,
        opened_values,
        opening_proof,
        multiset_sums,
        log_degree,
    } = proof;

    let log_degree = *log_degree as usize;
    let log_quotient_degree = E::LOG_QUOTIENT_DEGREE;
    let g_subgroup = C::Val::two_adic_generator(log_degree);

    challenger.observe(commitments.fixed.clone());
    challenger.observe(commitments.advice.clone());
    challenger.observe_slice(instance.values.as_slice());

    let gamma:C::Challenge = challenger.sample_ext_element();

    challenger.observe(commitments.multiset_f.clone());
    challenger.observe_slice(to_values::<C>(multiset_sums).as_slice());

    let alpha:C::Challenge = challenger.sample_ext_element();

    challenger.observe(commitments.quotient.clone());

    let zeta:C::Challenge = challenger.sample_ext_element();

    // TODO finalize verifier




    Ok(())
}