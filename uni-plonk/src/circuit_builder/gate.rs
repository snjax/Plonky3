use alloc::{format, vec};
use alloc::string::String;
use alloc::sync::Arc;
use alloc::vec::Vec;
use core::any::Any;
use core::fmt::{Debug, Error, Formatter};
use core::hash::{Hash, Hasher};
use hashbrown::HashMap;
use p3_field::{ExtensionField, PrimeField};
use crate::circuit_builder::CircuitConfig;
use crate::circuit_builder::generator::{ArithmeticBaseGenerator, SimpleGenerator, WitnessGeneratorRef};

/// A gate which can perform a weighted multiply-add, i.e. `result = c0 x y + c1 z`. If the config
/// supports enough routed wires, it can support several such operations in one gate.
#[derive(Debug, Clone)]
pub struct ArithmeticGate {
    /// Number of arithmetic operations performed by an arithmetic gate.
    pub num_ops: usize,
}

impl ArithmeticGate {
    pub const fn new_from_config(config: &CircuitConfig) -> Self {
        Self {
            num_ops: Self::num_ops(config),
        }
    }

    /// Determine the maximum number of operations that can fit in one gate for the given config.
    pub(crate) const fn num_ops(config: &CircuitConfig) -> usize {
        let wires_per_op = 4;
        config.num_routed_wires / wires_per_op
    }

    pub const fn wire_ith_multiplicand_0(i: usize) -> usize {
        4 * i
    }
    pub const fn wire_ith_multiplicand_1(i: usize) -> usize {
        4 * i + 1
    }
    pub const fn wire_ith_addend(i: usize) -> usize {
        4 * i + 2
    }
    pub const fn wire_ith_output(i: usize) -> usize {
        4 * i + 3
    }
}

impl<F: PrimeField + ExtensionField<F>, const D: usize> Gate<F, D> for ArithmeticGate {
    fn id(&self) -> String {
        format!("{self:?}")
    }

    fn generators(&self, row: usize, local_constants: &[F]) -> Vec<WitnessGeneratorRef<F, D>> {
        (0..self.num_ops)
            .map(|i| {
                WitnessGeneratorRef::new(
                    ArithmeticBaseGenerator {
                        row,
                        const_0: local_constants[0],
                        const_1: local_constants[1],
                        i,
                    }
                        .adapter(),
                )
            })
            .collect()
    }

    fn num_wires(&self) -> usize {
        self.num_ops * 4
    }
    fn num_constants(&self) -> usize {
        2
    }
    fn degree(&self) -> usize {
        3
    }
    fn num_constraints(&self) -> usize {
        self.num_ops
    }
}

/// Map between gate parameters and available slots.
/// An available slot is of the form `(row, op)`, meaning the current available slot
/// is at gate index `row` in the `op`-th operation.
#[derive(Clone, Debug, Default)]
pub struct CurrentSlot<F: PrimeField + ExtensionField<F>, const D: usize> {
    pub current_slot: HashMap<Vec<F>, (usize, usize)>,
}

pub trait Gate<F: PrimeField + ExtensionField<F>, const D: usize>: 'static + Send + Sync {
    fn id(&self) -> String;
    fn num_wires(&self) -> usize;
    fn num_constants(&self) -> usize;
    fn degree(&self) -> usize;
    fn num_constraints(&self) -> usize;

    /// Number of operations performed by the gate.
    fn num_ops(&self) -> usize {
        self.generators(0, &vec![F::zero(); self.num_constants()])
            .len()
    }

    /// The generators used to populate the witness.
    /// Note: This should return exactly 1 generator per operation in the gate.
    fn generators(&self, row: usize, local_constants: &[F]) -> Vec<WitnessGeneratorRef<F, D>>;

    /// Enables gates to store some "routed constants", if they have both unused constants and
    /// unused routed wires.
    ///
    /// Each entry in the returned `Vec` has the form `(constant_index, wire_index)`. `wire_index`
    /// must correspond to a *routed* wire.
    fn extra_constant_wires(&self) -> Vec<(usize, usize)> {
        vec![]
    }
}

/// A wrapper around an `Arc<AnyGate>` which implements `PartialEq`, `Eq` and `Hash` based on gate IDs.
#[derive(Clone)]
pub struct GateRef<F: PrimeField + ExtensionField<F>, const D: usize>(pub Arc<dyn AnyGate<F, D>>);

impl<F: PrimeField + ExtensionField<F>, const D: usize> GateRef<F, D> {
    pub fn new<G: Gate<F, D>>(gate: G) -> GateRef<F, D> {
        GateRef(Arc::new(gate))
    }
}

impl<F: PrimeField + ExtensionField<F>, const D: usize> PartialEq for GateRef<F, D> {
    fn eq(&self, other: &Self) -> bool {
        self.0.id() == other.0.id()
    }
}

impl<F: PrimeField + ExtensionField<F>, const D: usize> Hash for GateRef<F, D> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.id().hash(state)
    }
}

impl<F: PrimeField + ExtensionField<F>, const D: usize> Eq for GateRef<F, D> {}

impl<F: PrimeField + ExtensionField<F>, const D: usize> Debug for GateRef<F, D> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), Error> {
        write!(f, "{}", self.0.id())
    }
}


/// A wrapper trait over a `Gate`, to allow for gate serialization.
pub trait AnyGate<F: PrimeField + ExtensionField<F>, const D: usize>: Gate<F, D> {
    fn as_any(&self) -> &dyn Any;
}

impl<T: Gate<F, D>, F: PrimeField + ExtensionField<F>, const D: usize> AnyGate<F, D> for T {
    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// A gate along with any constants used to configure it.
#[derive(Clone)]
pub struct GateInstance<F: PrimeField + ExtensionField<F>, const D: usize> {
    pub gate_ref: GateRef<F, D>,
    pub constants: Vec<F>,
}