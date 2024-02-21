use alloc::vec;
use alloc::vec::Vec;

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

pub struct CircuitBuilder<F: PrimeField> {
    n: usize,

    // FIXME: Calculate from rows
    var_index: usize,

    gate_index: usize,

    // FIXME: Preallocate and prefill
    fixed_rows: Vec<F>,
    advice_rows: Vec<F>,

    // FIXME: Is there a need to store values separately?
    values: Vec<F>,
}

impl<F: PrimeField> CircuitBuilder<F> {
    pub fn new(n: usize) -> Self {
        CircuitBuilder {
            n,
            var_index: 0,
            gate_index: 0,
            fixed_rows: vec![F::zero(); 2usize.pow(n as u32) * 14],
            advice_rows: vec![F::zero(); 2usize.pow(n as u32) * 3],
            values: vec![F::zero(); 2usize.pow(n as u32) * 3],
        }
    }

    pub fn alloc(&mut self) -> Var {
        let var = Var(Index::Aux(self.var_index));
        self.var_index += 1;
        var
    }

    pub fn alloc_input(&mut self) -> Var {
        let var = Var(Index::Input(self.var_index));
        self.var_index += 1;
        var
    }

    // pub fn set_value(&mut self, var: Var, val: F) {}

    pub fn add(&mut self, a: Var, b: Var) {
        let c = self.alloc();
        self.enforce(
            &[F::one(), F::one(), -F::one(), F::zero(), F::zero()],
            &[a, b, c],
        );
    }

    pub fn add_constant(&mut self, a: Var, b: F) {
        let c = self.alloc();
        self.enforce(
            &[F::one(), F::zero(), -F::one(), F::zero(), b],
            &[a, Var(Index::Input(0)), c],
        );
    }

    pub fn mul(&mut self, a: Var, b: Var) {
        let c = self.alloc();
        self.enforce(
            &[F::zero(), F::zero(), -F::one(), F::one(), F::zero()],
            &[a, b, c],
        );
    }

    pub fn enforce(&mut self, fixed: &[F], advice: &[Var]) {
        let zero = F::zero();

        // TODO: Configurable number of columns
        let fixed = Fixed {
            q: Q {
                l: fixed[0],
                r: fixed[1],
                o: fixed[2],
                m: fixed[3],
                c: fixed[4],
            },
            sigma: X {
                // FIXME: Calculate
                a: F::from_canonical_usize(advice[0].0.index()),
                b: F::from_canonical_usize(advice[1].0.index()),
                c: F::from_canonical_usize(advice[2].0.index()),
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

        let offset = self.gate_index * 14;

        self.fixed_rows[offset..offset + 14].copy_from_slice(fixed.as_slice());

        // FIXME
        let advice = Advice {
            x: X {
                a: zero,
                b: zero,
                c: zero,
            },
            lookup_right_m: zero,
        };

        let offset = self.gate_index * 4;
        self.advice_rows[offset..offset + 4].copy_from_slice(advice.as_slice());

        self.gate_index += 1;
    }

    pub fn build(self) -> (RowMajorMatrix<F>, RowMajorMatrix<F>) {
        (
            RowMajorMatrix::new(self.fixed_rows, 14),
            RowMajorMatrix::new(self.advice_rows, 4),
        )
    }
}

#[cfg(test)]
mod tests {
    use alloc::format;
    use alloc::string::String;

    use p3_baby_bear::BabyBear;
    use p3_field::AbstractField;
    use tracing::debug;

    #[test]
    fn test_circuit_builder_build() {
        use super::*;

        let mut builder = CircuitBuilder::<BabyBear>::new(2);
        let a = builder.alloc_input();
        let b = builder.alloc_input();
        let c = builder.alloc_input();
        builder.add(a, b);
        builder.add(b, c);
        builder.add_constant(c, BabyBear::from_canonical_u32(42));

        let (fixed, advice) = builder.build();

        let mut string = String::new();
        for row in fixed.rows() {
            string.push_str(&format!("{:?}\n", row));
        }
        panic!("{}", string);

        assert_eq!(fixed.rows().count(), 4);
        assert_eq!(advice.rows().count(), 4);
    }
}
