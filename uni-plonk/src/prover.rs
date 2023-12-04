use alloc::{vec::Vec, vec};
use itertools::Itertools;

use p3_air::{Air, TwoRowMatrixView};
use p3_challenger::{CanObserve, FieldChallenger};
use p3_commit::{Pcs, UnivariatePcs, UnivariatePcsWithLde};
use p3_field::{cyclic_subgroup_coset_known_order, AbstractField, Field, PackedField, TwoAdicField, ExtensionField, AbstractExtensionField, batch_multiplicative_inverse};
use p3_matrix::dense::{RowMajorMatrix, RowMajorMatrixView};
use p3_matrix::{Matrix, MatrixGet, MatrixRowSlices};
use p3_maybe_rayon::{IndexedParallelIterator, MaybeIntoParIter, ParallelIterator, MaybeParIterMut, MaybeParChunksMut};
use p3_util::log2_strict_usize;
use tracing::{info_span, instrument};




use crate::{Config};

use crate::engine::Engine;




pub fn to_values_matrix<C:Config>(m:&RowMajorMatrix<C::Challenge>) -> RowMajorMatrix<C::Val> {
    let mut values = Vec::with_capacity(m.values.len() * C::Challenge::D);
    for c in m.values.as_slice() {
        values.extend(c.as_base_slice());
    }
    RowMajorMatrix::new(values, m.width() * C::Challenge::D)
}

#[instrument(skip_all)]
pub fn prove<C, E>(
    config: &C,
    engine: &E,
    challenger: &mut C::Challenger,
    fixed: RowMajorMatrix<C::Val>,
    advice: RowMajorMatrix<C::Val>,
    instance: RowMajorMatrix<C::Val>
)
    where
        C: Config,
        E: Engine<F=C::Val, EF=C::Challenge>,
{

    let degree = fixed.height();
    let log_degree = log2_strict_usize(degree);
    let log_quotient_degree = E::LOG_QUOTIENT_DEGREE;
    let g_subgroup = C::Val::two_adic_generator(log_degree);

    let pcs = config.pcs();

    // Compute commitment to fixed trace
    let (fixed_commit, fixed_data) =
        info_span!("commit to fixed trace data").in_scope(|| pcs.commit_batch(fixed.as_view()));

    // Observe commitment to fixed trace
    challenger.observe(fixed_commit.clone());

    // Compute commitment to advice trace
    let (advice_commit, advice_data) =
        info_span!("commit to advice trace data").in_scope(|| pcs.commit_batch(advice.as_view()));

    // Observe commitment to advice trace
    challenger.observe(advice_commit.clone());

    // Observe instance trace
    challenger.observe_slice(instance.values.as_slice());


    // Get PLONK gamma multiset challenge
    let gamma = challenger.sample_ext_element::<C::Challenge>()
        .powers().take(E::MAX_MULTISET_ELEMENT_WIDTH).collect_vec();



    // Compute partial sum multiset trace


    let mut multiset_a = RowMajorMatrix::new(vec![C::Challenge::zero(); E::MULTISET_WIDTH*degree], E::MULTISET_WIDTH);


    let id = E::id_matrix(log_degree);

    multiset_a.values.par_chunks_mut(E::MULTISET_WIDTH).enumerate().for_each(|(i, target)| {
        let next = if i+1 < degree { i } else { 0 };
        let fixed = TwoRowMatrixView::new(fixed.row_slice(i), fixed.row_slice(next));
        let advice = TwoRowMatrixView::new(advice.row_slice(i), advice.row_slice(next));
        let id = TwoRowMatrixView::new(id.row_slice(i), id.row_slice(next));
        E::eval_multiset(&gamma, id, fixed, advice, target);
    });

    let multiset_a_inverse = RowMajorMatrix::new(batch_multiplicative_inverse(multiset_a.values.as_slice()), E::MULTISET_WIDTH);

    // Compute sum of all cols of multiset_a_inverse
    let multiset_s = {
        let mut multiset_s = RowMajorMatrix::new(vec![C::Challenge::zero(); E::MULTISET_WIDTH], E::MULTISET_WIDTH);
        multiset_a_inverse.values.chunks(E::MULTISET_WIDTH).for_each(|chunk| {
            multiset_s.values.iter_mut().zip(chunk.iter()).for_each(|(s, a)| {
                *s += *a;
            });
        });
        let inv_degree = C::Val::from_canonical_usize(degree).inverse();
        multiset_s.values.iter_mut().for_each(|s| {
            *s *= inv_degree;
        });
        multiset_s
    };

    // Compute partial sum multiset trace
    // vertical additions suffix table for multiset_a_inverse, subtracting multiset_s from each row

    let multiset_f = {
        let mut multiset_f = RowMajorMatrix::new(vec![C::Challenge::zero(); E::MULTISET_WIDTH*degree], E::MULTISET_WIDTH);
        for row in 1..degree {
            for col in 0..E::MULTISET_WIDTH {
                let cell = row*E::MULTISET_WIDTH + col;
                let upper_cell = cell - E::MULTISET_WIDTH;
                let t = multiset_f.values[upper_cell] + multiset_a_inverse.values[cell] - multiset_s.values[col];
                multiset_f.values[cell] = t;
            }
        }
        multiset_f
    };

    // unroll extension field elements
    let multiset_f_values = to_values_matrix::<C>(&multiset_f);

    // Compute commitment to partial sum multiset trace
    let (multiset_commit, multiset_data) =
        info_span!("commit to multiset trace data").in_scope(|| pcs.commit_batch(multiset_f_values.as_view()));

    // Observe commitment to partial sum multiset trace

    challenger.observe(multiset_commit.clone());

    // Observe multiset sum values
    challenger.observe_slice(to_values_matrix::<C>(&multiset_s).values.as_slice());

    // Get PLONK alpha random linear combination challenge
    let alpha = challenger.sample_ext_element::<C::Challenge>();

    // Compute quotient polynomial

    // Compute commitment to quotient polynomial

    // Observe commitment to quotient polynomial

    // Get PLONK beta Schwartz-Zippel challenge

    // Compute openings at beta and beta*g for all trace polynomials

    // Compute openings at beta for quotient polynomial

    // Build proof object from all commitments and openings

    todo!();
}