use p3_field::{PrimeField};

use crate::circuit_builder::Target;

/// Represents a base arithmetic operation in the circuit. Used to memoize results.
#[derive(Copy, Clone, Eq, PartialEq, Hash)]
pub struct BaseArithmeticOperation<F: PrimeField> {
    pub const_0: F,
    pub const_1: F,
    pub multiplicand_0: Target,
    pub multiplicand_1: Target,
    pub addend: Target,
}