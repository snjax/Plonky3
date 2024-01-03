pub mod gate;
mod arithmetic;
mod generator;

use alloc::vec;
use alloc::vec::Vec;
use core::fmt::Debug;
use core::ops::Range;
use hashbrown::{HashMap, HashSet};
use p3_field::{ExtensionField, Field, PrimeField};
use p3_field::extension::{BinomiallyExtendable};
use p3_matrix::dense::RowMajorMatrix;
use crate::circuit_builder::arithmetic::BaseArithmeticOperation;
use crate::circuit_builder::gate::{ArithmeticGate, CurrentSlot, Gate, GateInstance, GateRef};

/// Represents a wire in the circuit, seen as a `degree x num_wires` table.
#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub struct Wire {
    /// Row index of the wire.
    pub row: usize,
    /// Column index of the wire.
    pub column: usize,
}

impl Wire {
    pub const fn is_routable(&self, config: &CircuitConfig) -> bool {
        self.column < config.num_routed_wires
    }

    pub fn from_range(gate: usize, range: Range<usize>) -> Vec<Self> {
        range
            .map(|i| Wire {
                row: gate,
                column: i,
            })
            .collect()
    }
}

#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub enum Target {
    Wire(Wire),
    /// A target that doesn't have any inherent location in the witness (but it can be copied to
    /// another target that does). This is useful for representing intermediate values in witness
    /// generation.
    VirtualTarget {
        index: usize,
    },
}


impl Target {
    pub const fn wire(row: usize, column: usize) -> Self {
        Self::Wire(Wire { row, column })
    }

    pub const fn is_routable(&self, config: &CircuitConfig) -> bool {
        match self {
            Target::Wire(wire) => wire.is_routable(config),
            Target::VirtualTarget { .. } => true,
        }
    }

    pub fn wires_from_range(row: usize, range: Range<usize>) -> Vec<Self> {
        range.map(|i| Self::wire(row, i)).collect()
    }

    pub fn index(&self, num_wires: usize, degree: usize) -> usize {
        match self {
            Target::Wire(Wire { row, column }) => row * num_wires + column,
            Target::VirtualTarget { index } => degree * num_wires + index,
        }
    }
}

impl Default for Target {
    fn default() -> Self {
        Self::VirtualTarget { index: 0 }
    }
}

/// `Target`s representing an element of an extension field.
#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub struct ExtensionTarget<const D: usize>(pub [Target; D]);

impl<const D: usize> Default for ExtensionTarget<D> {
    fn default() -> Self {
        Self([Target::default(); D])
    }
}

impl<const D: usize> ExtensionTarget<D> {
    pub const fn to_target_array(&self) -> [Target; D] {
        self.0
    }

    pub fn from_range(row: usize, range: Range<usize>) -> Self {
        debug_assert_eq!(range.end - range.start, D);
        Target::wires_from_range(row, range).try_into().unwrap()
    }
}

impl<const D: usize> TryFrom<Vec<Target>> for ExtensionTarget<D> {
    type Error = Vec<Target>;

    fn try_from(value: Vec<Target>) -> Result<Self, Self::Error> {
        Ok(Self(value.try_into()?))
    }
}

/// A named copy constraint.
pub struct CopyConstraint {
    pub pair: (Target, Target),
}

impl From<(Target, Target)> for CopyConstraint {
    fn from(pair: (Target, Target)) -> Self {
        Self {
            pair,
        }
    }
}

impl CopyConstraint {
    pub const fn new(pair: (Target, Target)) -> Self {
        Self { pair }
    }
}

/// Generator used to fill an extra constant.
#[derive(Debug, Clone, Default)]
pub struct ConstantGenerator<F: Field> {
    pub row: usize,
    pub constant_index: usize,
    pub wire_index: usize,
    pub constant: F,
}

impl<F: Field> ConstantGenerator<F> {
    pub fn set_constant(&mut self, c: F) {
        self.constant = c;
    }
}


#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CircuitConfig {
    pub num_wires: usize,
    pub num_routed_wires: usize,
    pub num_constants: usize,
}

impl CircuitConfig {
    /// A typical recursion config, without zero-knowledge, targeting ~100 bit security.
    pub const fn standard_recursion_config() -> Self {
        Self {
            num_wires: 135,
            num_routed_wires: 80,
            num_constants: 2,
        }
    }
}

pub struct CircuitBuilder<F: PrimeField + ExtensionField<F>, const D: usize> {
    pub config: CircuitConfig,

    /// The types of gates used in this circuit.
    gates: HashSet<GateRef<F, D>>,

    /// The next available index for a `VirtualTarget`.
    virtual_target_index: usize,

    /// Targets to be made public.
    public_inputs: Vec<Target>,

    constants_to_targets: HashMap<F, Target>,
    targets_to_constants: HashMap<Target, F>,

    copy_constraints: Vec<CopyConstraint>,

    /// List of constant generators used to fill the constant wires.
    constant_generators: Vec<ConstantGenerator<F>>,

    /// The concrete placement of each gate.
    pub(crate) gate_instances: Vec<GateInstance<F, D>>,

    pub(crate) base_arithmetic_results: HashMap<BaseArithmeticOperation<F>, Target>,

    /// Map between gate type and the current gate of this type with available slots.
    current_slots: HashMap<GateRef<F, D>, CurrentSlot<F, D>>,
}

impl<F: PrimeField + ExtensionField<F>, const D: usize> CircuitBuilder<F, D> {
    pub fn new(config: CircuitConfig) -> Self {
        Self {
            config,
            gates: HashSet::new(),
            virtual_target_index: 0,
            public_inputs: Vec::new(),
            constants_to_targets: HashMap::new(),
            targets_to_constants: HashMap::new(),
            constant_generators: Vec::new(),
            copy_constraints: Vec::new(),
            gate_instances: Vec::new(),
            base_arithmetic_results: HashMap::new(),
            current_slots: HashMap::new(),
        }
    }

    pub fn add_virtual_target(&mut self) -> Target {
        let index = self.virtual_target_index;
        self.virtual_target_index += 1;
        Target::VirtualTarget { index }
    }

    /// Registers the given target as a public input.
    pub fn register_public_input(&mut self, target: Target) {
        self.public_inputs.push(target);
    }

    pub fn constant(&mut self, c: F) -> Target {
        if let Some(&target) = self.constants_to_targets.get(&c) {
            return target;
        }

        let target = self.add_virtual_target();
        self.constants_to_targets.insert(c, target);
        self.targets_to_constants.insert(target, c);

        target
    }

    /// Returns a routable target with a value of 1.
    pub fn one(&mut self) -> Target {
        self.constant(F::one())
    }

    /// Computes `x * y`.
    pub fn mul(&mut self, x: Target, y: Target) -> Target {
        // x * y = 1 * x * y + 0 * x
        self.arithmetic(F::one(), F::zero(), x, y, x)
    }

    /// Computes `C * x`.
    pub fn mul_const(&mut self, c: F, x: Target) -> Target {
        let c = self.constant(c);
        self.mul(c, x)
    }

    /// Computes `x + y`.
    pub fn add(&mut self, x: Target, y: Target) -> Target {
        let one = self.one();
        // x + y = 1 * x * 1 + 1 * y
        self.arithmetic(F::one(), F::one(), x, one, y)
    }

    /// Computes `x + C`.
    pub fn add_const(&mut self, x: Target, c: F) -> Target {
        let c = self.constant(c);
        self.add(x, c)
    }

    /// Computes `const_0 * multiplicand_0 * multiplicand_1 + const_1 * addend`.
    pub fn arithmetic(
        &mut self,
        const_0: F,
        const_1: F,
        multiplicand_0: Target,
        multiplicand_1: Target,
        addend: Target,
    ) -> Target {
        // // See if we can determine the result without adding an `ArithmeticGate`.
        // if let Some(result) =
        //     self.arithmetic_special_cases(const_0, const_1, multiplicand_0, multiplicand_1, addend)
        // {
        //     return result;
        // }

        // See if we've already computed the same operation.
        let operation = BaseArithmeticOperation {
            const_0,
            const_1,
            multiplicand_0,
            multiplicand_1,
            addend,
        };
        if let Some(&result) = self.base_arithmetic_results.get(&operation) {
            return result;
        }

        // Otherwise, we must actually perform the operation using an ArithmeticExtensionGate slot.
        let result = self.add_base_arithmetic_operation(operation);
        self.base_arithmetic_results.insert(operation, result);
        result
    }

    fn add_base_arithmetic_operation(&mut self, operation: BaseArithmeticOperation<F>) -> Target {
        let gate = ArithmeticGate::new_from_config(&self.config);
        let constants = vec![operation.const_0, operation.const_1];
        let (gate, i) = self.find_slot(gate, &constants, &constants);
        let wires_multiplicand_0 = Target::wire(gate, ArithmeticGate::wire_ith_multiplicand_0(i));
        let wires_multiplicand_1 = Target::wire(gate, ArithmeticGate::wire_ith_multiplicand_1(i));
        let wires_addend = Target::wire(gate, ArithmeticGate::wire_ith_addend(i));

        self.connect(operation.multiplicand_0, wires_multiplicand_0);
        self.connect(operation.multiplicand_1, wires_multiplicand_1);
        self.connect(operation.addend, wires_addend);

        Target::wire(gate, ArithmeticGate::wire_ith_output(i))
    }

    /// Find an available slot, of the form `(row, op)` for gate `G` using parameters `params`
    /// and constants `constants`. Parameters are any data used to differentiate which gate should be
    /// used for the given operation.
    pub fn find_slot<G: Gate<F, D> + Clone>(
        &mut self,
        gate: G,
        params: &[F],
        constants: &[F],
    ) -> (usize, usize) {
        let num_gates = self.num_gates();
        let num_ops = gate.num_ops();
        let gate_ref = GateRef::new(gate.clone());
        let gate_slot = self.current_slots.entry(gate_ref.clone()).or_default();
        let slot = gate_slot.current_slot.get(params);
        let (gate_idx, slot_idx) = if let Some(&s) = slot {
            s
        } else {
            self.add_gate(gate, constants.to_vec());
            (num_gates, 0)
        };
        let current_slot = &mut self.current_slots.get_mut(&gate_ref).unwrap().current_slot;
        if slot_idx == num_ops - 1 {
            // We've filled up the slots at this index.
            current_slot.remove(params);
        } else {
            // Increment the slot operation index.
            current_slot.insert(params.to_vec(), (gate_idx, slot_idx + 1));
        }

        (gate_idx, slot_idx)
    }

    /// Adds a gate to the circuit, and returns its index.
    pub fn add_gate<G: Gate<F, D>>(&mut self, gate_type: G, mut constants: Vec<F>) -> usize {
        self.check_gate_compatibility(&gate_type);

        assert!(
            constants.len() <= gate_type.num_constants(),
            "Too many constants."
        );
        constants.resize(gate_type.num_constants(), F::zero());

        let row = self.gate_instances.len();

        self.constant_generators
            .extend(gate_type.extra_constant_wires().into_iter().map(
                |(constant_index, wire_index)| ConstantGenerator {
                    row,
                    constant_index,
                    wire_index,
                    constant: F::zero() // Placeholder; will be replaced later.
                },
            ));

        // Note that we can't immediately add this gate's generators, because the list of constants
        // could be modified later, i.e. in the case of `ConstantGate`. We will add them later in
        // `build` instead.

        // Register this gate type if we haven't seen it before.
        let gate_ref = GateRef::new(gate_type);
        self.gates.insert(gate_ref.clone());

        self.gate_instances.push(GateInstance {
            gate_ref,
            constants,
        });

        row
    }

    fn check_gate_compatibility<G: Gate<F, D>>(&self, gate: &G) {
        assert!(
            gate.num_wires() <= self.config.num_wires,
            "{:?} requires {} wires, but our CircuitConfig has only {}",
            gate.id(),
            gate.num_wires(),
            self.config.num_wires
        );
        assert!(
            gate.num_constants() <= self.config.num_constants,
            "{:?} requires {} constants, but our CircuitConfig has only {}",
            gate.id(),
            gate.num_constants(),
            self.config.num_constants
        );
    }

    /// Uses Plonk's permutation argument to require that two elements be equal.
    /// Both elements must be routable, otherwise this method will panic.
    pub fn connect(&mut self, x: Target, y: Target) {
        assert!(
            x.is_routable(&self.config),
            "Tried to route a wire that isn't routable"
        );
        assert!(
            y.is_routable(&self.config),
            "Tried to route a wire that isn't routable"
        );
        self.copy_constraints
            .push(CopyConstraint::new((x, y)));
    }


    pub fn num_gates(&self) -> usize {
        self.gate_instances.len()
    }

    pub fn build(&self) -> CompiledCircuit<F> {
        todo!()
    }
}

struct CompiledCircuit<F: PrimeField + ExtensionField<F>> {
    pub fixed: RowMajorMatrix<F>,
    pub advice: RowMajorMatrix<F>,
}

#[cfg(test)]
mod tests {
    use p3_baby_bear::BabyBear;
    use p3_field::AbstractField;
    use crate::circuit_builder::generator::{PartialWitness, WitnessWrite};
    use super::*;

    #[test]
    fn test_build() {
        const D: usize = 2;
        type F = BabyBear;

        let config = CircuitConfig::standard_recursion_config();
        let mut builder = CircuitBuilder::<F, D>::new(config);

        // The arithmetic circuit.
        let initial = builder.add_virtual_target();
        let mut cur_target = initial;
        for i in 2..101 {
            let i_target = builder.constant(F::from_canonical_u32(i));
            cur_target = builder.mul(cur_target, i_target);
        }

        // Public inputs are the initial value (provided below) and the result (which is generated).
        builder.register_public_input(initial);
        builder.register_public_input(cur_target);

        let mut pw = PartialWitness::new();
        pw.set_target(initial, F::one());

        let data = builder.build();
    }
}

