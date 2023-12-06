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
use p3_uni_stark::ZerofierOnCoset;



use crate::{Config};

use crate::engine::Engine;




pub fn to_values_matrix<C:Config>(m:&RowMajorMatrix<C::Challenge>) -> RowMajorMatrix<C::Val> {
    let mut values = Vec::with_capacity(m.values.len() * C::Challenge::D);
    for c in m.values.as_slice() {
        values.extend(c.as_base_slice());
    }
    RowMajorMatrix::new(values, m.width() * C::Challenge::D)
}

pub fn as_two_row_view<'a, T>(m: &'a impl MatrixRowSlices<T>) -> TwoRowMatrixView<'a, T> {
    TwoRowMatrixView::new(m.row_slice(0), m.row_slice(1))
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
    let log_quotent_size = log_degree + log_quotient_degree;
    let g_subgroup = C::Val::two_adic_generator(log_degree);
    let g_extended = C::Val::two_adic_generator(log_quotent_size);
    let log_blowup = config.pcs().log_blowup();
    let blowup = 1 << log_blowup;

    assert!(log_blowup >= log_quotient_degree, "PCS blowup is too small for quotient degree");



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

    let gamma_packed = gamma.iter().map(|&x| C::PackedChallenge::from_f(x)).collect_vec();



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
    let (multiset_f_commit, multiset_f_data) =
        info_span!("commit to multiset trace data").in_scope(|| pcs.commit_batch(multiset_f_values.as_view()));

    // Observe commitment to partial sum multiset trace

    challenger.observe(multiset_f_commit.clone());

    // Observe multiset sum values
    challenger.observe_slice(to_values_matrix::<C>(&multiset_s).values.as_slice());

    // Get PLONK alpha random linear combination challenge
    let alpha = challenger.sample_ext_element::<C::Challenge>();

    // Compute quotient polynomial

    let _ = {
        let fixed_lde = {
            let mut t = pcs.get_ldes(&fixed_data);
            assert_eq!(t.len(), 1);
            t.pop().unwrap()
        };

        let advice_lde = {
            let mut t = pcs.get_ldes(&advice_data);
            assert_eq!(t.len(), 1);
            t.pop().unwrap()
        };

        let multiset_f_lde = {
            let mut t = pcs.get_ldes(&multiset_f_data);
            assert_eq!(t.len(), 1);
            t.pop().unwrap()
        };


        let coset_shift = config.pcs().coset_shift();

        let zerofier_on_coset = ZerofierOnCoset::new(log_degree, log_quotient_degree, coset_shift);


        let coset = cyclic_subgroup_coset_known_order(g_extended, coset_shift, log_quotent_size).collect_vec();
        let quotient_size = 1 << log_quotent_size;

        let next_step = 1 << log_quotient_degree;

        let multiset_s = TwoRowMatrixView::new(multiset_s.row_slice(0), multiset_s.row_slice(0));

        (0..quotient_size)
            .into_par_iter()
            .step_by(C::PackedVal::WIDTH)
            .flat_map_iter(|i_local_start| {
                let wrap = |i| -> usize { if i < quotient_size { i } else { i - quotient_size } };

                let i_next_start = wrap(i_local_start + next_step);
                let i_local_range = i_local_start..i_local_start + C::PackedVal::WIDTH;
                let i_next_range = i_next_start..i_next_start + C::PackedVal::WIDTH;
                let x_local = *C::PackedVal::from_slice(&coset[i_local_range.clone()]);
                let x_next = *C::PackedVal::from_slice(&coset[i_next_range.clone()]);

                // TODO: this part could be optimized at least on indexes computing and memory allocation

                let to_challenge_matrix = |m:&RowMajorMatrixView<C::Val>| -> RowMajorMatrix<C::PackedChallenge> {

                    let values = [i_local_start, i_next_start]
                        .iter().flat_map(|&i|
                        (0..m.width()).step_by(C::PackedChallenge::D).map(move |col| {
                            C::PackedChallenge::from_base_fn(|h_offset| {
                                C::PackedVal::from_fn(|v_offset| {
                                    let row = wrap(i + v_offset);
                                    m.get(row, col + h_offset)
                                })
                            })
                        })).collect_vec();

                    RowMajorMatrix::new(values, m.width() / C::PackedChallenge::D)
                };

                let to_value_matrix = |m:&RowMajorMatrixView<C::Val>| -> RowMajorMatrix<C::PackedVal> {
                    let values = [i_local_start, i_next_start]
                        .iter().flat_map(|&i|
                        (0..m.width()).map(move |col| {
                            C::PackedVal::from_fn(|v_offset| {
                                let row = wrap(i + v_offset);
                                m.get(row, col)
                            })
                        })).collect_vec();
                    RowMajorMatrix::new(values, m.width())
                };

                let fixed_data = to_value_matrix(&fixed_lde);
                let fixed = as_two_row_view(&fixed_data);
                let advice_data = to_value_matrix(&advice_lde);
                let advice = as_two_row_view(&advice_data);
                let multiset_f_data = to_challenge_matrix(&multiset_f_lde);
                let multiset_f = as_two_row_view(&multiset_f_data);
                let id_data = E::id_matrix_at(x_local, x_next);
                let id = as_two_row_view(&id_data);

                let mut multiset_a = RowMajorMatrix::new(vec![C::PackedChallenge::zero(); 2*E::MULTISET_WIDTH], E::MULTISET_WIDTH);

                let multiset_a = E::eval_multiset(&gamma_packed, id, fixed, advice, &mut multiset_a.values[0..E::MULTISET_WIDTH]);

                panic!("TODO: finish this part");
                core::iter::once(C::Challenge::zero())

            });
    };

    // Compute commitment to quotient polynomial

    // Observe commitment to quotient polynomial

    // Get PLONK beta Schwartz-Zippel challenge

    // Compute openings at beta and beta*g for all trace polynomials

    // Compute openings at beta for quotient polynomial

    // Build proof object from all commitments and openings

    todo!();
}