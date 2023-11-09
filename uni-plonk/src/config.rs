use core::marker::PhantomData;

use p3_challenger::{CanObserve, FieldChallenger};
use p3_commit::{Pcs, UnivariatePcsWithLde};
use p3_field::{AbstractExtensionField, ExtensionField, Field, PackedField, TwoAdicField};
use p3_matrix::dense::{RowMajorMatrix, RowMajorMatrixView};

pub trait Config {
    /// The field over which trace data is encoded.
    type Val: Field;

    /// The domain over which trace polynomials are defined.
    type Domain: ExtensionField<Self::Val> + TwoAdicField;
    type PackedDomain: PackedField<Scalar = Self::Domain>;

    /// The field from which most random challenges are drawn.
    type Challenge: ExtensionField<Self::Val> + ExtensionField<Self::Domain> + TwoAdicField;
    type PackedChallenge: AbstractExtensionField<Self::PackedDomain, F = Self::Challenge> + 'static + Send + Sync + Copy;

    /// The PCS used to commit to trace polynomials.
    type Pcs: for<'a> UnivariatePcsWithLde<
        Self::Val,
        Self::Domain,
        Self::Challenge,
        RowMajorMatrixView<'a, Self::Val>,
        Self::Challenger,
    >;

    /// The challenger (Fiat-Shamir) implementation used.
    type Challenger: FieldChallenger<Self::Val>
    + for<'a> CanObserve<<Self::Pcs as Pcs<Self::Val, RowMajorMatrixView<'a, Self::Val>>>::Commitment>;

    fn pcs(&self) -> &Self::Pcs;
}

pub struct ConfigImpl<Val, Domain, Challenge, PackedChallenge, Pcs, Challenger> {
    pcs: Pcs,
    _phantom: PhantomData<(Val, Domain, Challenge, PackedChallenge, Challenger)>,
}

impl<Val, Domain, Challenge, PackedChallenge, Pcs, Challenger> ConfigImpl<Val, Domain, Challenge, PackedChallenge, Pcs, Challenger>
{
    pub fn new(pcs: Pcs) -> Self {
        Self {
            pcs,
            _phantom: PhantomData,
        }
    }
}

impl<Val, Domain, Challenge, PackedChallenge, Pcs, Challenger> Config
for ConfigImpl<Val, Domain, Challenge, PackedChallenge, Pcs, Challenger>
    where
        Val: Field,
        Domain: ExtensionField<Val> + TwoAdicField+Copy,
        Challenge: ExtensionField<Val> + ExtensionField<Domain> + TwoAdicField+Copy,
        PackedChallenge: AbstractExtensionField<Domain::Packing, F = Challenge> + 'static + Send + Sync+Copy,
        Pcs: for<'a> UnivariatePcsWithLde<Val, Domain, Challenge, RowMajorMatrixView<'a, Val>, Challenger>,
        Challenger: FieldChallenger<Val>
        + for <'a> CanObserve<<Pcs as p3_commit::Pcs<Val, RowMajorMatrixView<'a, Val>>>::Commitment>,
{
    type Val = Val;
    type Domain = Domain;
    type PackedDomain = Domain::Packing;
    type Challenge = Challenge;
    type PackedChallenge = PackedChallenge;
    type Pcs = Pcs;
    type Challenger = Challenger;

    fn pcs(&self) -> &Self::Pcs {
        &self.pcs
    }
}
