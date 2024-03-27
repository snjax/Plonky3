use alloc::vec;
use alloc::vec::Vec;
use itertools::Itertools;
use rand::prelude::SliceRandom;

use p3_field::{Field, PrimeField, PrimeField64};
use p3_matrix::dense::RowMajorMatrix;

use crate::standard_plonk::{repr_as, Advice, Fixed, LookupTable, Q, X};

#[derive(Debug, Clone, Copy)]
pub struct Var(Index);

#[derive(Debug, Clone, Copy)]
pub enum Index {
    Input(i64),
    Aux(i64),
}

impl Index {
    pub fn index(self) -> i64 {
        match self {
            Index::Input(i) => i,
            Index::Aux(i) => i,
        }
    }
}

// TODO: A more general trait for advice and input
pub trait AdviceTable<F: PrimeField>: Default {
    type Out;

    const STRIDE: usize = 4;

    fn push(&mut self, advice: Advice<F>);
    fn set_row(&mut self, index: usize, advice: Advice<F>);

    /// Returns the value of witness at index.
    fn get(&self, index: usize) -> F;

    /// Sets the value of witness at index.
    fn set(&mut self, index: usize, value: F);

    fn to_out(self) -> Self::Out;
}

impl<F: PrimeField64> AdviceTable<F> for Vec<F> {
    type Out = RowMajorMatrix<F>;

    fn push(&mut self, advice: Advice<F>) {
        self.extend_from_slice(advice.as_slice());
    }

    fn set_row(&mut self, row_index: usize, advice: Advice<F>) {
        let start = row_index * Self::STRIDE;
        let end = start + Self::STRIDE;
        self[start..end].copy_from_slice(advice.as_slice());
    }

    // TODO: Offset index by -1?
    fn get(&self, index: usize) -> F {
        self[index + (index / (Self::STRIDE - 1))]
    }

    fn set(&mut self, index: usize, value: F) {
        self[index + (index / (Self::STRIDE - 1))] = value;
    }


    fn to_out(self) -> RowMajorMatrix<F> {
        RowMajorMatrix::new(self, 4)
    }
}

impl<F: PrimeField> AdviceTable<F> for () {
    type Out = ();

    fn push(&mut self, _advice: Advice<F>) {}

    fn set_row(&mut self, _index: usize, _advice: Advice<F>) {}
    #[inline]
    fn get(&self, _index: usize) -> F {
        F::zero()
    }

    fn set(&mut self, _index: usize, _value: F) {}

    fn to_out(self) {}
}

pub struct CircuitBuilder<F: PrimeField64, A: AdviceTable<F>> {
    // TODO: Is there a better way to handle intermediate state?
    var_index: usize,
    fixed: Vec<Fixed<F>>,
    advice: A,

    /// Mapping from input index to witness index
    inputs: Vec<usize>,

    wires: Vec<Var>,
}

impl<F: PrimeField64, A: AdviceTable<F>> CircuitBuilder<F, A> {
    pub fn new() -> Self {
        // TODO: Preallocate space for fixed and advice
        CircuitBuilder {
            var_index: 0,
            fixed: Vec::new(),
            advice: A::default(),
            inputs: Vec::new(),
            wires: Vec::new(),
        }
    }

    pub fn alloc(&mut self) -> Var {
        let var = Var(Index::Aux(self.var_index as i64));
        self.var_index += 1;
        var
    }

    pub fn alloc_input(&mut self) -> Var {
        let index = self.var_index;
        let var = Var(Index::Input(index as i64));
        self.var_index += 1;
        self.inputs.push(index);
        var
    }

    pub fn add(&mut self, a: Var, b: Var) -> Var {
        let c = self.alloc();
        self.enforce(
            &[F::one(), F::one(), -F::one(), F::zero(), F::zero()],
            &[a, b, c],
        );
        c
    }

    pub fn add_constant(&mut self, a: Var, b: F) -> Var {
        let _b = self.alloc();
        let c = self.alloc();
        self.enforce(
            &[F::one(), F::zero(), -F::one(), F::zero(), b],
            &[a, Var(Index::Aux(-1)), c],
        );
        c
    }

    pub fn mul(&mut self, a: Var, b: Var) -> Var {
        let c = self.alloc();
        self.enforce(
            &[F::zero(), F::zero(), -F::one(), F::one(), F::zero()],
            &[a, b, c],
        );
        c
    }

    pub fn eq(&mut self, a: Var, b: Var) -> Var {
        let c = self.alloc();
        self.enforce(
            &[F::one(), -F::one(), F::zero(), F::zero(), F::zero()],
            &[a, b, c],
        );
        c
    }

    // TODO: Accept Q and X instead slices?
    pub fn enforce(&mut self, fixed: &[F], advice: &[Var]) {
        let zero = F::zero();
        let g = F::generator();
        let row = self.fixed.len();

        let mut sigma = [F::zero(); 3];
        for (i, advice) in advice.iter().enumerate() {
            let var_index = advice.0.index();
            let witness_index = row + i;

            if var_index >= 0 && row > 0 {
                let wire_row_index = var_index / 3;
                if wire_row_index as usize == row {
                    let prev = sigma[var_index as usize % 3];
                    sigma[i] = prev;
                    sigma[var_index as usize % 3] = g.exp_u64(witness_index as u64);
                } else {
                    let prev_sigma = self.fixed[wire_row_index as usize].sigma.as_slice_mut();
                    let prev_val = prev_sigma[i];
                    prev_sigma[i] = g.exp_u64(witness_index as u64);
                    sigma[i] = prev_val;
                }
            } else {
                sigma[i] = g.exp_u64(witness_index as u64);
            }

            self.wires.push(*advice);
        }

        let fixed = Fixed {
            q: Q {
                l: fixed[0],
                r: fixed[1],
                o: fixed[2],
                m: fixed[3],
                c: fixed[4],
            },
            sigma: repr_as::<_, X<_>>(&sigma).clone(),
            selector: zero,
            op: zero,
            table: LookupTable {
                op: zero,
                x: X {
                    a: zero,
                    b: zero,
                    c: zero,
                },
            },
        };

        let advice = Advice {
            x: X {
                a: zero,
                b: zero,
                c: zero,
            },
            lookup_right_m: zero,
        };

        self.fixed.push(fixed);
        self.advice.push(advice);
    }

    pub fn build(mut self, inputs: &[F]) -> (RowMajorMatrix<F>, A::Out) {
        for (val, input_i) in inputs.iter().zip(self.inputs.iter()) {
            self.advice.set(*input_i, *val);
        }

        for (row, Fixed { q, .. }) in self.fixed.iter().enumerate() {
            let xai = self.wires[row * 3].0.index();
            let xbi = self.wires[row * 3 + 1].0.index();
            let xa = if xai >= 0 {
                self.advice.get(xai as usize)
            } else {
                F::zero()
            };

            let xb = if xbi >= 0 {
                self.advice.get(xbi as usize)
            } else {
                F::zero()
            };

            let xc = q.l * xa + q.r * xb + q.m * xa * xb + q.c;

            let advice = Advice {
                x: X {
                    a: xa,
                    b: xb,
                    c: xc,
                },
                lookup_right_m: F::zero(),
            };

            self.advice.set_row(row, advice);
        }

        // TODO: Cast Vec<Fixed> to Vec<F> instead of allocating a new Vec.
        let mut fixed_rows = Vec::new();
        for fixed in &self.fixed {
            fixed_rows.extend_from_slice(fixed.as_slice());
        }

        // FIXME: hardcoded number of cols
        let total_num_rows = self.fixed.len().next_power_of_two();
        if fixed_rows.len() < total_num_rows * 14 {
            fixed_rows.extend_from_slice(&vec![F::zero(); total_num_rows * 14 - fixed_rows.len()]);

            for _ in 0..total_num_rows - fixed_rows.len() / 14 {
                self.advice.push(Advice {
                    x: X {
                        a: F::zero(),
                        b: F::zero(),
                        c: F::zero(),
                    },
                    lookup_right_m: F::zero(),
                });
            }
        }

        (
            RowMajorMatrix::new(fixed_rows, 14),
            self.advice.to_out(),
        )
    }
}

#[cfg(test)]
mod tests {
    use rand::{random, thread_rng};
    use p3_baby_bear::BabyBear;
    use p3_challenger::DuplexChallenger;
    use p3_commit::ExtensionMmcs;
    use p3_dft::Radix2DitParallel;
    use p3_field::AbstractField;
    use p3_field::extension::BinomialExtensionField;
    use p3_fri::{FriBasedPcs, FriConfigImpl, FriLdt};
    use p3_goldilocks::Goldilocks;
    use p3_keccak::Keccak256Hash;
    use p3_ldt::{LdtBasedPcs, QuotientMmcs};
    use p3_mds::coset_mds::CosetMds;
    use p3_merkle_tree::FieldMerkleTreeMmcs;
    use p3_symmetric::{CompressionFunctionFromHasher, SerializingHasher64};
    use p3_uni_stark::{StarkConfigImpl, verify};
    use crate::{ConfigImpl, prove};
    use super::*;

    fn build_circuit<F: PrimeField64, T: AdviceTable<F>>(inputs: &[F]) -> (RowMajorMatrix<F>, T::Out) {
        let mut builder = CircuitBuilder::<F, T>::new();
        let a = builder.alloc_input();
        let b = builder.alloc_input();
        let c = builder.add(a, b);
        builder.add_constant(c, F::from_canonical_u32(42));

        builder.eq(c, c);

        builder.build(inputs)
    }

    #[test]
    fn test_circuit_builder_build() {
        type F = BabyBear;

        let inputs = vec![
            F::from_canonical_u32(1),
            F::from_canonical_u32(2),
        ];

        // Build only fixed
        let (fixed, advice) = build_circuit::<F, ()>(&[]);

        // Build both fixed and advice
        let (fixed, advice) = build_circuit::<F, Vec<F>>(inputs.as_slice());
    }

    // #[test]
    // fn test_circuit_builder_prove() {
    //     type F = BabyBear;
    //
    //     type Val = Goldilocks;
    //     type Domain = Val;
    //     type Challenge = BinomialExtensionField<Val, 2>;
    //     type PackedChallenge = BinomialExtensionField<<Domain as Field>::Packing, 2>;
    //
    //     type MyMds = CosetMds<Val, 8>;
    //     let mds = MyMds::default();
    //
    //     type Perm = Poseidon2<Val, MyMds, DiffusionMatrixGoldilocks, 8, 5>;
    //     let perm = Perm::new_from_rng(8, 22, mds, DiffusionMatrixGoldilocks, &mut thread_rng());
    //
    //     type MyHash = SerializingHasher64<Keccak256Hash>;
    //     let hash = MyHash::new(Keccak256Hash {});
    //     type MyCompress = CompressionFunctionFromHasher<Val, MyHash, 2, 4>;
    //     let compress = MyCompress::new(hash);
    //
    //     type ValMmcs = FieldMerkleTreeMmcs<Val, MyHash, MyCompress, 4>;
    //     let val_mmcs = ValMmcs::new(hash, compress);
    //
    //     type ChallengeMmcs = ExtensionMmcs<Val, Challenge, ValMmcs>;
    //     let challenge_mmcs = ChallengeMmcs::new(val_mmcs.clone());
    //
    //     type Dft = Radix2DitParallel;
    //     let dft = Dft {};
    //
    //     type Challenger = DuplexChallenger<Val, Perm, 8>;
    //
    //     type Quotient = QuotientMmcs<Domain, Challenge, ValMmcs>;
    //     type MyFriConfig = FriConfigImpl<Val, Challenge, Quotient, ChallengeMmcs, Challenger>;
    //     let fri_config = MyFriConfig::new(40, challenge_mmcs);
    //     let ldt = FriLdt { config: fri_config };
    //
    //     type Pcs = FriBasedPcs<MyFriConfig, ValMmcs, Dft, Challenger>;
    //     type MyConfig = StarkConfigImpl<Val, Challenge, PackedChallenge, Pcs, Challenger>;
    //
    //     let pcs = Pcs::new(dft, val_mmcs, ldt);
    //     let config = ConfigImpl::new(pcs);
    //     let mut challenger = Challenger::new(perm);
    //
    //
    //
    //     let inputs = vec![
    //         F::from_canonical_u32(1),
    //         F::from_canonical_u32(2),
    //     ];
    //
    //     // Build both fixed and advice
    //     let (fixed, advice) = build_circuit::<F, Vec<F>>(inputs.as_slice());
    //
    //     prove(config, &mut challenger, fixed, advice, RowMajorMatrix::new(inputs, 2));
    // }
}
