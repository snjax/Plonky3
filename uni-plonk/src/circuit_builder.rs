use alloc::vec;
use alloc::vec::Vec;
use itertools::Itertools;
use rand::prelude::SliceRandom;

use p3_field::{Field, PrimeField, PrimeField64};
use p3_matrix::dense::RowMajorMatrix;

use crate::standard_plonk::{repr_as, Advice, Fixed, LookupTable, Q, X};

#[derive(Debug, Clone, Copy)]
pub struct Var(u64);

impl Default for Var {
    fn default() -> Self {
        Var::undefined()
    }
}

impl Var {
    #[inline]
    pub fn new(i: u64) -> Self {
        Var(i + 1)
    }

    #[inline]
    pub fn undefined() -> Self {
        Var(0)
    }

    #[inline]
    pub fn is_defined(&self) -> bool {
        self.0 != 0
    }

    #[inline]
    pub fn index(&self) -> u64 {
        self.0.saturating_sub(1)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LookupOp {
    Nop,
    Xor,
}

impl LookupOp {
    pub fn to_f<F: PrimeField64>(self) -> F {
        F::from_canonical_u64(self as u64)
    }
}

impl<F: PrimeField64> From<F> for LookupOp {
    fn from(f: F) -> Self {
        match f.as_canonical_u64() {
            0 => LookupOp::Nop,
            1 => LookupOp::Xor,
            _ => unreachable!("Invalid lookup op"),
        }
    }
}


pub trait AdviceTable<F: PrimeField> {
    type Storage: AdviceTableStorage<F>;
}

impl<F: PrimeField> AdviceTable<F> for () {
    type Storage = ();
}

impl<F: PrimeField> AdviceTable<F> for RowMajorMatrix<F> {
    type Storage = Vec<F>;
}

pub trait AdviceTableStorage<F: PrimeField>: Default {
    type Out;

    const STRIDE: usize = 4;

    fn push(&mut self, advice: Advice<F>);
    fn set_row(&mut self, index: usize, advice: Advice<F>);

    /// Returns the value of witness at index.
    fn get(&self, index: usize) -> F;

    /// Sets the value of witness at index.
    fn set(&mut self, index: usize, value: F);

    fn to_out(self) -> Self::Out;

    fn num_rows(&self) -> usize;
}

impl<F: PrimeField> AdviceTableStorage<F> for Vec<F> {
    type Out = RowMajorMatrix<F>;

    fn push(&mut self, advice: Advice<F>) {
        self.extend_from_slice(advice.as_slice());
    }

    fn set_row(&mut self, row_index: usize, advice: Advice<F>) {
        let start = row_index * Self::STRIDE;
        let end = start + Self::STRIDE;
        self[start..end].copy_from_slice(advice.as_slice());
    }

    fn get(&self, index: usize) -> F {
        self[index + (index / (Self::STRIDE - 1))]
    }

    fn set(&mut self, index: usize, value: F) {
        self[index + (index / (Self::STRIDE - 1))] = value;
    }


    fn to_out(self) -> RowMajorMatrix<F> {
        RowMajorMatrix::new(self, 4)
    }

    fn num_rows(&self) -> usize {
        self.len() / Self::STRIDE
    }
}

impl<F: PrimeField> AdviceTableStorage<F> for () {
    type Out = ();

    fn push(&mut self, _advice: Advice<F>) {}

    fn set_row(&mut self, _index: usize, _advice: Advice<F>) {}
    #[inline]
    fn get(&self, _index: usize) -> F {
        F::zero()
    }

    fn set(&mut self, _index: usize, _value: F) {}

    fn to_out(self) {}

    fn num_rows(&self) -> usize {
        0
    }
}

pub struct CircuitBuilder<F: PrimeField64, A: AdviceTable<F>> {
    // TODO: Is there a better way to handle intermediate state?
    var_index: usize,
    gate_index: usize,
    lookup_index: usize,
    fixed: Vec<Fixed<F>>,
    advice: A::Storage,

    /// Mapping input index to witness index
    inputs: Vec<usize>,

    /// Mapping witness index to Var
    wires: Vec<Var>,
}

impl<F: PrimeField64, A: AdviceTable<F>> CircuitBuilder<F, A> {
    pub fn new() -> Self {
        CircuitBuilder {
            var_index: 0,
            gate_index: 0,
            lookup_index: 0,
            fixed: Vec::new(),
            advice: A::Storage::default(),
            inputs: Vec::new(),
            wires: Vec::new(),
        }
    }

    pub fn alloc(&mut self) -> Var {
        let var = Var::new(self.var_index as u64);
        self.var_index += 1;
        var
    }

    pub fn alloc_input(&mut self) -> Var {
        let var = self.alloc();
        self.inputs.push(var.0 as usize);
        var
    }

    pub fn add(&mut self, a: Var, b: Var) -> Var {
        let c = self.alloc();
        self.enforce(
            Q {
                l: F::one(),
                r: F::one(),
                o: -F::one(),
                m: F::zero(),
                c: F::zero(),
            },
            &[a, b, c],
            F::zero(),
            F::zero(),
        );
        c
    }

    pub fn add_constant(&mut self, a: Var, b: F) -> Var {
        let _b = self.alloc();
        let c = self.alloc();

        self.enforce(
            Q {
                l: F::one(),
                r: F::zero(),
                o: -F::one(),
                m: F::zero(),
                c: b,
            },
            &[a, Var::undefined(), c],
            F::zero(),
            F::zero(),
        );

        c
    }

    pub fn add_mul(&mut self, l: F, a: Var, r: F, b: Var) -> Var {
        let c = self.alloc();

        self.enforce(
            Q { l, r, o: -F::one(), m: F::zero(), c: F::zero() },
            &[a, b, c],
            F::zero(),
            F::zero(),
        );
        c
    }

    pub fn mul(&mut self, a: Var, b: Var) -> Var {
        let c = self.alloc();

        self.enforce(
            Q {
                l: F::zero(),
                r: F::zero(),
                o: -F::one(),
                m: F::one(),
                c: F::zero(),
            },
            &[a, b, c],
            F::zero(),
            F::zero(),
        );
        c
    }

    pub fn eq(&mut self, a: Var, b: Var) -> Var {
        let c = self.alloc();
        self.enforce(
            Q {
                l: F::one(),
                r: -F::one(),
                o: F::zero(),
                m: F::zero(),
                c: F::zero(),
            },
            &[a, b, c],
            F::zero(),
            F::zero(),
        );
        c
    }

    pub fn eq_constant(&mut self, a: Var, b: F) -> Var {
        let c = self.alloc();
        self.enforce(
            Q {
                l: F::one(),
                r: -F::one(),
                o: F::zero(),
                m: F::zero(),
                c: b,
            },
            &[a, Var::undefined(), c],
            F::zero(),
            F::zero(),
        );
        c
    }

    pub fn lookup_xor(&mut self, x: Var, y: Var) -> Var {
        let z = self.alloc();

        self.enforce(
            Default::default(),
            &[x, y, z],
            F::one(),
            LookupOp::Xor.to_f(),
        );

        z
    }

    pub fn lookup_range(&mut self, x: Var) -> Var {
        let z = self.alloc();

        self.enforce(
            Default::default(),
            &[x, x, z],
            F::one(),
            LookupOp::Xor.to_f(), // Reuse the XOR lookup table
        );

        z
    }

    pub fn xor(&mut self, a: Var, b: Var) -> Var {
        // let c = self.alloc();
        //
        // // 1. Divide the element into chunks of 8 bits
        // // 2. Do a lookup for each chunk
        // // 3. Sum the results with appropriate offsets
        //
        // let mut num_chunks = core::mem::size_of::<F>() / 8;
        // let mut chunks = Vec::new(); // FIXME: Implement chunking
        // for chunk in 0..num_chunks {
        //     let z = self.lookup_xor(a, b);
        //
        //     self.advice.set_row(self.gate_index - 1, Advice {
        //         x: X {
        //             a: F::from_canonical_u64(chunk as u64), // Temporarily reuse witness for lookup metadata
        //             b: F::zero(),
        //             c: F::zero(),
        //         },
        //         lookup_right_m: F::one(), // FIXME
        //     });
        //
        //     chunks.push(z);
        // }
        //
        //
        // while num_chunks != 1 {
        //     // iterate over pairs of chunks
        //     for (i, j) in (0..num_chunks).tuples() {
        //         let shift = (num_chunks / 2) as u64; // FIXME
        //         let t = self.add_mul(
        //             F::from_canonical_u64(shift),
        //             a,
        //             F::from_canonical_u64(shift),
        //             b,
        //         );
        //
        //
        //         self.enforce(
        //             Default::default(),
        //             &[t, c, c],
        //             F::one(),
        //             F::zero(),
        //         );
        //     }
        //
        //     num_chunks /= 2;
        // }
        //
        // c

        todo!()
    }

    fn build_xor_table(&mut self, chunk_size: u32) {
        for x in 0..2u64.pow(chunk_size) {
            for y in 0..2u64.pow(chunk_size) {
                let table = LookupTable {
                    op: F::one(),
                    x: X {
                        a: F::from_canonical_u64(x),
                        b: F::from_canonical_u64(y),
                        c: F::from_canonical_u64(x ^ y),
                    },
                };

                if self.lookup_index < self.fixed.len() {
                    self.fixed[self.lookup_index].table = table;
                } else {
                    self.fixed.push(Fixed {
                        table,
                        ..Default::default()
                    });
                }
                self.lookup_index += 1;
            }
        }
    }

    pub fn enforce(&mut self, q: Q<F>, advice: &[Var], selector: F, op: F) {
        let g = F::generator();
        let row = self.fixed.len();

        let mut sigma = [F::zero(); 3];
        for (i, var) in advice.iter().enumerate() {
            let var_index = var.index();
            let witness_index = row + i;

            if row > 0 {
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

            self.wires.push(*var);
        }

        if self.fixed.len() > self.gate_index {
            let fixed = &mut self.fixed[self.gate_index];
            fixed.q = q;
            fixed.sigma = repr_as::<_, X<_>>(&sigma).clone();
            fixed.selector = selector;
            fixed.op = op;
        } else {
            let fixed = Fixed {
                q,
                sigma: repr_as::<_, X<_>>(&sigma).clone(),
                selector,
                op,
                ..Default::default()
            };

            self.fixed.push(fixed);
        }

        self.gate_index += 1;
        self.advice.push(Default::default());
    }

    pub fn build(mut self, inputs: &[F]) -> (RowMajorMatrix<F>, <<A as AdviceTable<F>>::Storage as AdviceTableStorage<F>>::Out) {
        // TODO: Make xor table optional
        self.build_xor_table(8);

        for (val, input_i) in inputs.iter().zip(self.inputs.iter()) {
            self.advice.set(*input_i, *val);
        }

        for (row, Fixed { q, selector, op, .. }) in self.fixed[..self.gate_index].iter().enumerate() {
            let xai = self.wires[row * 3];
            let xbi = self.wires[row * 3 + 1];

            let xa = if xai.is_defined() {
                self.advice.get(xai.index() as usize)
            } else {
                F::zero()
            };

            let xb = if xbi.is_defined() {
                self.advice.get(xbi.index() as usize)
            } else {
                F::zero()
            };

            let xc = if selector.is_one() {
                match LookupOp::from(*op) {
                    LookupOp::Nop => xa,
                    LookupOp::Xor => {
                        let a = xa.as_canonical_u64();
                        let b = xb.as_canonical_u64();
                        let c = a ^ b;
                        F::from_canonical_u64(c)
                    }
                }
            } else {
                q.l * xa + q.r * xb + q.m * xa * xb + q.c
            };


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


        // Pad the rows to the next power of two
        let num_fixed_rows = self.fixed.len();
        let next_power_of_two = num_fixed_rows.next_power_of_two();
        for _ in 0..next_power_of_two - num_fixed_rows {
            self.fixed.push(Fixed::default());
        }

        let num_advice_rows = self.advice.num_rows();
        for _ in 0..next_power_of_two - num_advice_rows {
            self.advice.push(Default::default());
        }

        // TODO: Cast Vec<Fixed> to Vec<F> instead of allocating a new Vec.
        let mut fixed_rows = Vec::new();
        for fixed in &self.fixed {
            fixed_rows.extend_from_slice(fixed.as_slice());
        }

        (
            RowMajorMatrix::new(fixed_rows, 14),
            self.advice.to_out(),
        )
    }
}

#[cfg(test)]
mod tests {
    use rand::{thread_rng};
    use p3_baby_bear::BabyBear;
    use p3_challenger::DuplexChallenger;
    use p3_commit::ExtensionMmcs;
    use p3_dft::Radix2DitParallel;
    use p3_field::AbstractField;
    use p3_field::extension::BinomialExtensionField;
    use p3_fri::{FriBasedPcs, FriConfigImpl, FriLdt};
    use p3_goldilocks::Goldilocks;
    use p3_keccak::Keccak256Hash;
    use p3_ldt::{QuotientMmcs};
    use p3_mds::coset_mds::CosetMds;
    use p3_merkle_tree::FieldMerkleTreeMmcs;
    use p3_symmetric::{CompressionFunctionFromHasher, SerializingHasher64};
    use p3_poseidon2::{DiffusionMatrixGoldilocks, Poseidon2};
    use crate::{ConfigImpl, prove, verify};
    use crate::standard_plonk::Plonk;
    use super::*;

    fn build_circuit<F: PrimeField64, A: AdviceTable<F>>(inputs: &[F]) -> (RowMajorMatrix<F>, <<A as AdviceTable<F>>::Storage as AdviceTableStorage<F>>::Out) {
        let mut builder = CircuitBuilder::<F, A>::new();
        let a = builder.alloc_input();
        let b = builder.alloc_input();
        let c = builder.add(a, b);
        let a_xor = builder.lookup_xor(a, a);
        builder.eq(a, a_xor);
        builder.add_constant(b, F::from_canonical_u32(42));

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
        let (fixed, _) = build_circuit::<F, ()>(&[]);

        assert!(fixed.rows().count().is_power_of_two());

        // Build both fixed and advice
        let (fixed, advice) = build_circuit::<F, RowMajorMatrix<F>>(inputs.as_slice());

        let num_fixed_rows = fixed.rows().count();
        assert!(num_fixed_rows.is_power_of_two());

        let num_advice_rows = advice.rows().count();
        assert_eq!(num_fixed_rows, num_advice_rows);
    }

    #[test]
    fn test_circuit_builder_prove() {
        type F = Goldilocks;
        type Domain = F;
        type Challenge = BinomialExtensionField<F, 2>;
        type PackedChallenge = BinomialExtensionField<<Domain as Field>::Packing, 2>;

        type MyMds = CosetMds<F, 8>;
        let mds = MyMds::default();

        type Perm = Poseidon2<F, MyMds, DiffusionMatrixGoldilocks, 8, 5>;
        let perm = Perm::new_from_rng(8, 22, mds, DiffusionMatrixGoldilocks, &mut thread_rng());

        type MyHash = SerializingHasher64<Keccak256Hash>;
        let hash = MyHash::new(Keccak256Hash {});
        type MyCompress = CompressionFunctionFromHasher<F, MyHash, 2, 4>;
        let compress = MyCompress::new(hash);

        type ValMmcs = FieldMerkleTreeMmcs<F, MyHash, MyCompress, 4>;
        let val_mmcs = ValMmcs::new(hash, compress);

        type ChallengeMmcs = ExtensionMmcs<F, Challenge, ValMmcs>;
        let challenge_mmcs = ChallengeMmcs::new(val_mmcs.clone());

        type Dft = Radix2DitParallel;
        let dft = Dft {};

        type Challenger = DuplexChallenger<F, Perm, 8>;

        type Quotient = QuotientMmcs<Domain, Challenge, ValMmcs>;
        type MyFriConfig = FriConfigImpl<F, Challenge, Quotient, ChallengeMmcs, Challenger>;
        let fri_config = MyFriConfig::new(40, challenge_mmcs);
        let ldt = FriLdt { config: fri_config };

        type Pcs = FriBasedPcs<MyFriConfig, ValMmcs, Dft, Challenger>;
        type MyConfig = ConfigImpl<F, Challenge, PackedChallenge, Pcs, Challenger>;

        let pcs = Pcs::new(dft, val_mmcs, ldt);
        let config = ConfigImpl::<F, Challenge, PackedChallenge, Pcs, Challenger>::new(pcs);
        let mut challenger = Challenger::new(perm);


        let inputs = vec![
            F::from_canonical_u32(1),
            F::from_canonical_u32(2),
        ];

        let instance = RowMajorMatrix::new(inputs.clone(), 2);

        let (fixed, advice) = build_circuit::<F, RowMajorMatrix<F>>(inputs.as_slice());

        let proof = prove::<MyConfig, Plonk<(F, Challenge)>>(&config, &mut challenger, fixed, advice, RowMajorMatrix::new(inputs, 2));
        let result = verify::<MyConfig, Plonk<(F, Challenge)>>(&config, &mut challenger, &proof, instance);

        assert!(result.is_ok());
    }
}
