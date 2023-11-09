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



use p3_uni_stark::{
    decompose_and_flatten, Commitments, OpenedValues, Proof, ProverConstraintFolder,
    ZerofierOnCoset,
};

use crate::{Config};

use crate::engine::Engine;


// Convert matrix to packed keeping parity of rows
// even rows are packed with even, odd with odd
//
// fn to_packed_domain<C:Config>(m:RowMajorMatrix<C::Val>)->RowMajorMatrix<C::PackedDomain> {
//     let len = m.values.len();
//     let packed_width = C::PackedDomain::WIDTH;
//     let packed_len = len/packed_width;
//     let width = m.width();
//     let mut res = Vec::with_capacity(packed_len);
//
//     for offset0 in (0..len).step_by(width*packed_width*2){
//         for offset1 in 0..width*2 {
//             let offset = offset0 + offset1;
//             let t = C::PackedDomain::from_fn(|k| C::Domain::from_base(m.values[offset + 2*width*k]));
//             res.push(t);
//         }
//     }
//
//     RowMajorMatrix::new(res, width)
// }
//
// fn to_packed<C:Config>(m:RowMajorMatrix<C::Domain>)->RowMajorMatrix<C::PackedDomain> {
//     let len = m.values.len();
//     let packed_width = C::PackedDomain::WIDTH;
//     let packed_len = len/packed_width;
//     let width = m.width();
//     let mut res = Vec::with_capacity(packed_len);
//
//     for offset0 in (0..len).step_by(width*packed_width*2){
//         for offset1 in 0..width*2 {
//             let offset = offset0 + offset1;
//             let t = C::PackedDomain::from_fn(|k| m.values[offset + 2*width*k]);
//             res.push(t);
//         }
//     }
//
//     RowMajorMatrix::new(res, width)
// }
//
//
// fn get_row_packed<F:Field, P:PackedField<Scalar=F>>(m:&impl MatrixGet<F>, rows:&[usize])-> Vec<P> {
//     (0..m.width()).map(|i| P::from_fn(|k| m.get(rows[k], i))).collect_vec()
// }
//
//
//
// fn get_id_matrix<F:TwoAdicField>(log_degree:usize, width:usize) -> RowMajorMatrix<F> {
//     let g = F::two_adic_generator(log_degree);
//     let offset = F::generator();
//     let degree = 1 << log_degree;
//
//     let mut res = Vec::with_capacity(degree*width);
//
//     let mut e = F::one();
//     for _ in 0..degree {
//         let mut t = e;
//         for _ in 0..width {
//             res.push(t);
//             t *= offset;
//         }
//         e *= g;
//     }
//
//     RowMajorMatrix::new(res, width)
// }

// fn into_base_field_matrix<F:Field, EF:ExtensionField<F>>(e:RowMajorMatrix<EF>) -> RowMajorMatrix<F> {
//     let values = e.values;
//     let d = EF::D;
//     let (ptr, len, cap) = {
//         let len = values.len() * d;
//         let cap = values.capacity() * d;
//         let ptr = values.as_ptr() as *mut F;
//         core::mem::forget(values);
//         (ptr, len, cap)
//     };
//     let values = unsafe { Vec::from_raw_parts(ptr, len, cap) };
//     RowMajorMatrix::new(values, e.width()*d)
// }
//
//
// fn packed_inverse<F:PackedField<Scalar=impl Field>>(f:F) -> F {
//     let inv = batch_multiplicative_inverse(f.as_slice());
//     F::from_fn(|k| inv[k])
// }
//
// fn batch_multiplicative_inverse_packed<F:PackedField<Scalar=impl Field>>(values:impl Iterator<Item=F>) -> Vec<F> {
//     let values = values.collect_vec();
//     let mut res = values.iter().scan(F::one(), |acc, &x| {
//         *acc *= x;
//         Some(*acc)
//     }).collect_vec();
//
//     let len = res.len();
//     if len == 0 {
//         return vec![];
//     }
//
//     let mut acc = packed_inverse(*res.last().unwrap());
//
//     for i in (1..len).rev() {
//         res[i] = acc*res[i-1];
//         acc*=values[i];
//     }
//
//     res
// }

pub fn get_permutation_matrixes<C, E>(
    config:&C,
    engine:&E,
    fixed: RowMajorMatrixView<C::Val>,
    advice: RowMajorMatrixView<C::Val>,
) where
    C: Config,
    E: Engine<F=C::Domain, EF=C::Challenge>
{
    todo!()
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
        E: Engine<F=C::Domain, EF=C::Challenge>,
{

    let degree = fixed.height();
    let log_degree = log2_strict_usize(degree);
    let log_quotient_degree = E::LOG_QUOTIENT_DEGREE;
    let g_subgroup = C::Domain::two_adic_generator(log_degree);

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

    let packed_height = degree/C::PackedDomain::WIDTH;
    let mut multiset_a_packed = RowMajorMatrix::new(vec![C::PackedChallenge::zero(); E::MULTISET_WIDTH*packed_height], E::MULTISET_WIDTH);


    let id = get_id_matrix::<C::Domain>(log_degree, E::ID_WIDTH);
    let id_packed = to_packed::<C>(id);

    multiset_a_packed.values.par_chunks_mut(E::MULTISET_WIDTH).enumerate().for_each(|(i, target)| {
        let next = if i+1 < degree { i } else { 0 };
        let fixed = TwoRowMatrixView::new(fixed_packed.row_slice(i), fixed_packed.row_slice(next));
        let advice = TwoRowMatrixView::new(advice_packed.row_slice(i), advice_packed.row_slice(next));
        let id = TwoRowMatrixView::new(id_packed.row_slice(i), id_packed.row_slice(next));
        E::eval_multiset(&gamma_packed, id, fixed, advice, target);
    });






    // Compute commitment to partial sum multiset trace

    // Observe commitment to partial sum multiset trace

    // Compute multiset sum values

    // Observe multiset sum values

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