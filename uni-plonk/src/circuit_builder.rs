use alloc::vec;
use alloc::vec::Vec;
use itertools::Itertools;

use p3_field::{Field, PrimeField};
use p3_matrix::dense::RowMajorMatrix;

use crate::standard_plonk::{repr_as, Advice, Fixed, LookupTable, Q, X};

#[derive(Debug, Clone, Copy)]
pub struct Var(Index);

#[derive(Debug, Clone, Copy)]
pub enum Index {
    Input(usize),
    Aux(usize),
}

impl Index {
    pub fn index(&self) -> usize {
        match self {
            Index::Input(i) => *i,
            Index::Aux(i) => *i,
        }
    }
}

// TODO: A more general trait for advice and input
pub trait AdviceTable<F: PrimeField>: Default {
    type Out;

    fn push(&mut self, advice: Advice<F>);
    fn set_row(&mut self, index: usize, advice: Advice<F>);

    fn get_row(&self, index: usize) -> Advice<F>;

    fn to_out(&self) -> Self::Out;
}

impl<F: PrimeField> AdviceTable<F> for Vec<Advice<F>> {
    type Out = RowMajorMatrix<F>;

    fn push(&mut self, advice: Advice<F>) {
        self.push(advice);
    }

    fn set_row(&mut self, index: usize, advice: Advice<F>) {
        self[index] = advice;
    }

    fn get_row(&self, index: usize) -> Advice<F> {
        self[index].clone()
    }

    fn to_out(&self) -> RowMajorMatrix<F> {
        let mut rows = Vec::new();
        for advice in self {
            rows.extend_from_slice(advice.as_slice());
        }
        RowMajorMatrix::new(rows, 4)
    }
}

impl<F: PrimeField> AdviceTable<F> for () {
    type Out = ();

    fn push(&mut self, _advice: Advice<F>) {}

    fn set_row(&mut self, _index: usize, _advice: Advice<F>) {}

    fn get_row(&self, _index: usize) -> Advice<F> {
        Advice {
            x: X {
                a: F::zero(),
                b: F::zero(),
                c: F::zero(),
            },
            lookup_right_m: F::zero(),
        }
    }

    fn to_out(&self) {}
}

pub struct CircuitBuilder<F: PrimeField, A: AdviceTable<F>> {
    var_index: usize,
    fixed: Vec<Fixed<F>>,
    advice: A,
    inputs: Vec<F>,
}

impl<F: PrimeField, A: AdviceTable<F>> CircuitBuilder<F, A> {
    pub fn new() -> Self {
        // TODO: Preallocate space for fixed and advice
        CircuitBuilder {
            var_index: 1,
            fixed: Vec::new(),
            advice: A::default(),
            inputs: Vec::new(),
        }
    }

    pub fn alloc(&mut self) -> Var {
        let var = Var(Index::Aux(self.var_index));
        self.var_index += 1;
        var
    }

    pub fn alloc_input(&mut self) -> Var {
        let index = self.var_index;
        let var = Var(Index::Input(index));
        self.var_index += 1;
        // self.inputs.push(F::generator().exp_u64(index as u64));
        self.inputs.push(F::from_canonical_usize(index));
        var
    }

    // pub fn set_value(&mut self, var: Var, val: F) {}

    pub fn add(&mut self, a: Var, b: Var) -> Var {
        let c = self.alloc();
        self.enforce(
            &[F::one(), F::one(), -F::one(), F::zero(), F::zero()],
            &[a, b, c],
        );
        c
    }

    pub fn add_constant(&mut self, a: Var, b: F) -> Var {
        let c = self.alloc();
        self.enforce(
            &[F::one(), F::zero(), -F::one(), F::zero(), b],
            &[a, Var(Index::Input(0)), c],
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

    pub fn enforce(&mut self, fixed: &[F], advice: &[Var]) {
        let zero = F::zero();
        // let g = F::generator();

        let i_a = advice[0].0.index();
        let i_b = advice[1].0.index();
        let i_c = advice[2].0.index();

        let fixed = Fixed {
            q: Q {
                l: fixed[0],
                r: fixed[1],
                o: fixed[2],
                m: fixed[3],
                c: fixed[4],
            },
            sigma: X {
                // a: g.exp_u64(i_a as u64),
                // b: g.exp_u64(i_b as u64),
                // c: g.exp_u64(i_c as u64),
                a: F::from_canonical_usize(i_a),
                b: F::from_canonical_usize(i_b),
                c: F::from_canonical_usize(i_c),
            },
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

    pub fn build(&mut self, inputs: &[F]) -> (RowMajorMatrix<F>, A::Out) {
        let total_num_rows = self.fixed.len().next_power_of_two();

        for (i, Fixed { q, sigma, .. }) in self.fixed.iter().enumerate() {
            // TODO: Optimize, DRY

            // TODO: Implement a more convenient (searchable/indexable) intermediate representation
            //       for fixed and advice.


            // Calculate x values
            let x_a = if let Some((in_i, _)) = self.inputs.iter().find_position(|&&x| x == sigma.a) {
                inputs[in_i]
            } else {
                // 1. sigma.a to index
                // 2. load value at index from advice

                // FIXME: Linear search is no go here, use an index.
                self.fixed[0..i].iter().find(|a| a.sigma.a == sigma.a).map(|a| self.advice.get_row(i).x.a).unwrap_or(F::zero())
            };

            let x_b = if let Some((in_i, _)) = self.inputs.iter().find_position(|&&x| x == sigma.b) {
                inputs[in_i]
            } else {
                // FIXME: Index
                self.fixed[0..i].iter().find(|a| a.sigma.b == sigma.b).map(|a| self.advice.get_row(i).x.b).unwrap_or(F::zero())
            };

            let x_c = q.l * x_a + q.r * x_b + q.m * x_a * x_b + q.c;

            let advice = Advice {
                x: X {
                    a: x_a,
                    b: x_b,
                    c: x_c,
                },
                lookup_right_m: F::zero(),
            };

            self.advice.set_row(i, advice);
        }

        let mut fixed_rows = Vec::new();
        for fixed in &self.fixed {
            fixed_rows.extend_from_slice(fixed.as_slice());
        }

        // FIXME: hardcoded number of cols
        if fixed_rows.len() < total_num_rows * 14 {
            fixed_rows.extend_from_slice(&vec![F::zero(); total_num_rows * 14 - fixed_rows.len()]);

            for _ in 0..total_num_rows - fixed_rows.len() {
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
    use p3_baby_bear::BabyBear;
    use p3_field::AbstractField;

    #[test]
    fn test_circuit_builder_build() {
        use super::*;

        let mut builder = CircuitBuilder::<BabyBear, ()>::new();
        let a = builder.alloc_input();
        let b = builder.alloc_input();
        let c = builder.add(a, b);
        let d = builder.add(b, c);
        builder.add_constant(d, BabyBear::from_canonical_u32(42));

        let (fixed, _) = builder.build(&[BabyBear::from_canonical_u32(1), BabyBear::from_canonical_u32(2)]);

        assert_eq!(fixed.rows().count(), 4);
        // assert_eq!(advice.rows().count(), 4);
    }
}
