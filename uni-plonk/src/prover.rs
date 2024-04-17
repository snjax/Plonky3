use alloc::{vec::Vec, vec};
use itertools::Itertools;

use p3_air::{TwoRowMatrixView};
use p3_challenger::{CanObserve, FieldChallenger};
use p3_commit::{Pcs, UnivariatePcs, UnivariatePcsWithLde};
use p3_field::{cyclic_subgroup_coset_known_order, AbstractField, Field, PackedField, TwoAdicField, AbstractExtensionField, batch_multiplicative_inverse};
use p3_matrix::dense::{RowMajorMatrix, RowMajorMatrixView};
use p3_matrix::{Matrix, MatrixGet, MatrixRowSlices};
use p3_maybe_rayon::{IndexedParallelIterator, MaybeIntoParIter, ParallelIterator, MaybeParChunksMut};
use p3_util::log2_strict_usize;
use tracing::{info_span, instrument};
use p3_uni_stark::ZerofierOnCoset;

use crate::decompose::decompose_and_flatten;


use crate::{Config};
use crate::proof::{Proof, Commitments, OpenedValues};

use crate::engine::Engine;


pub fn to_values<C:Config>(m:&[C::Challenge]) -> Vec<C::Val> {
    let mut values = Vec::with_capacity(m.len() * C::Challenge::D);
    for c in m {
        values.extend(c.as_base_slice());
    }
    values
}

pub fn to_values_matrix<C:Config>(m:&RowMajorMatrix<C::Challenge>) -> RowMajorMatrix<C::Val> {
    RowMajorMatrix::new(to_values::<C>(&m.values), m.width() * C::Challenge::D)
}




#[instrument(skip_all)]
pub fn prove<C, E>(
    config: &C,
    challenger: &mut C::Challenger,
    fixed: RowMajorMatrix<C::Val>,
    advice: RowMajorMatrix<C::Val>,
    instance: RowMajorMatrix<C::Val>
) -> Proof<C>
    where
        C: Config,
        E: Engine<F=C::Val, EF=C::Challenge>,
{

    let degree = fixed.height();
    let log_degree = log2_strict_usize(degree);
    let log_quotient_degree = E::LOG_QUOTIENT_DEGREE;
    let log_quotient_size = log_degree + log_quotient_degree;
    let g_subgroup = C::Val::two_adic_generator(log_degree);
    let g_extended = C::Val::two_adic_generator(log_quotient_size);
    let log_blowup = config.pcs().log_blowup();

    let row_multiplier = 1 << (log_blowup - log_quotient_degree);

    assert!(log_blowup >= log_quotient_degree, "PCS blowup is too small for quotient degree");



    let pcs = config.pcs();

    // Compute commitment to fixed trace
    // TODO: optimize memory for matrix commitment
    let (fixed_commit, fixed_data) =
        info_span!("commit to fixed trace data").in_scope(|| pcs.commit_batch(fixed.clone()));



    // Observe commitment to fixed trace
    challenger.observe(fixed_commit.clone());

    // Compute commitment to advice trace
    let (advice_commit, advice_data) =
        info_span!("commit to advice trace data").in_scope(|| pcs.commit_batch(advice.clone()));

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
        E::eval_multiset(&gamma, &id, &fixed, &advice, target);
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
        info_span!("commit to multiset trace data").in_scope(|| {
            pcs.commit_batch(multiset_f_values)
        });

    // Observe commitment to partial sum multiset trace
    challenger.observe(multiset_f_commit.clone());

    // Observe multiset sum values
    challenger.observe_slice(to_values_matrix::<C>(&multiset_s).values.as_slice());

    // Get PLONK alpha random linear combination challenge
    // Multiply it by extension field monomials for fast evaluation
    let alpha = challenger.sample_ext_element::<C::Challenge>()
        .powers().take(E::NUM_GATES)
        .flat_map(|x| (0..C::Challenge::D).map(move |i| C::Challenge::monomial(i)*x))
        .collect_vec();

    let alpha_packed = alpha.iter().map(|&x| C::PackedChallenge::from_f(x)).collect_vec();

    // Compute quotient polynomial

    let quotient_values = {
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

        let quotient_size = 1 << log_quotient_size;
        let coset = cyclic_subgroup_coset_known_order(g_extended, coset_shift, quotient_size).collect_vec();
        
        

        let next_step = 1 << log_quotient_degree;

        let multiset_s = multiset_s.map(|e| C::PackedChallenge::from_f(e));



        (0..quotient_size)
            .into_par_iter()
            .step_by(C::PackedVal::WIDTH)
            .flat_map_iter(move |i_local_start| {
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
                                    let row = wrap(i + v_offset)*row_multiplier;
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
                                let row = wrap(i + v_offset)*row_multiplier;
                                m.get(row, col)
                            })
                        })).collect_vec();
                    RowMajorMatrix::new(values, m.width())
                };

                let fixed = to_value_matrix(&fixed_lde);

                let advice = to_value_matrix(&advice_lde);

                let multiset_f = to_challenge_matrix(&multiset_f_lde);

                let id = E::id_matrix_at(x_local, x_next);


                let mut multiset_a = RowMajorMatrix::new(vec![C::PackedChallenge::zero(); E::MULTISET_WIDTH], E::MULTISET_WIDTH);

                E::eval_multiset(&gamma_packed, &id, &fixed, &advice, &mut multiset_a.values);


                let expr = E::eval_gates(&alpha_packed, &fixed, &advice, &multiset_f, &multiset_a, &multiset_s);
                let zeroifier_inv:C::PackedVal = zerofier_on_coset.eval_inverse_packed(i_local_start);

                let quotient = expr * zeroifier_inv;


                (0..C::PackedVal::WIDTH).map(move |idx_in_packing| {
                    let quotient_value = (0..C::Challenge::D)
                        .map(|coeff_idx| quotient.as_base_slice()[coeff_idx].as_slice()[idx_in_packing])
                        .collect_vec();
                    C::Challenge::from_base_slice(&quotient_value)
                })

            }).collect::<Vec<_>>()
    };

    let quotient_chunks_flattened = info_span!("decompose quotient polynomial").in_scope(|| {
        decompose_and_flatten::<C>(
            quotient_values,
            C::Challenge::from_base(pcs.coset_shift()),
            log_quotient_degree,
        )
    });


    // Compute commitment to quotient polynomial

    let (quotient_commit, quotient_data) =
        info_span!("commit to quotient poly chunks").in_scope(|| {
            pcs.commit_shifted_batch(
                quotient_chunks_flattened,
                pcs.coset_shift().exp_power_of_2(log_quotient_degree),
            )
        });

    // Observe commitment to quotient polynomial

    challenger.observe(quotient_commit.clone());

    // Get PLONK zeta Schwartz-Zippel challenge

    let zeta: C::Challenge = challenger.sample_ext_element();

    // Compute openings at zeta and beta*g for all trace polynomials
    // Compute openings at zeta for quotient polynomial

    let (opened_values, opening_proof) = pcs.open_multi_batches(
        &[
            (&fixed_data, &[zeta, zeta * g_subgroup]),
            (&advice_data, &[zeta, zeta * g_subgroup]),
            (&multiset_f_data, &[zeta, zeta * g_subgroup]),
            (&quotient_data, &[zeta.exp_power_of_2(log_quotient_degree)]),
        ],
        challenger,
    );

    // Build proof object from all commitments and openings

    Proof {
        commitments: Commitments {
            fixed: fixed_commit,
            advice: advice_commit,
            multiset_f: multiset_f_commit,
            quotient: quotient_commit,
        },
        opened_values: OpenedValues {
            fixed_local: opened_values[0][0][0].clone(),
            fixed_next: opened_values[0][1][0].clone(),
            advice_local: opened_values[1][0][0].clone(),
            advice_next: opened_values[1][1][0].clone(),
            multiset_f_local: opened_values[2][0][0].clone(),
            multiset_f_next: opened_values[2][1][0].clone(),
            quotient: opened_values[3][0][0].clone(),
        },
        opening_proof,
        multiset_sums: multiset_s.values,
        log_degree: log_degree as u64,
    }
}