use p3_field::{AbstractExtensionField, AbstractField, ExtensionField, Field};
use p3_air::TwoRowMatrixView;

pub trait Engine {
    type F: Field;
    type EF: ExtensionField<Self::F>;

    const LOG_QUOTIENT_DEGREE: usize;
    const MAX_MULTISET_ELEMENT_WIDTH: usize;
    const MULTISET_WIDTH: usize;
    const ID_WIDTH: usize;
    fn eval_gates<BaseF, ExtF>(multiplier: &[ExtF],
                               id: TwoRowMatrixView<BaseF>,
                               fixed: TwoRowMatrixView<BaseF>,
                               advice: TwoRowMatrixView<BaseF>,
                               multiset_f: TwoRowMatrixView<ExtF>,
                               multiset_a: TwoRowMatrixView<ExtF>,
                               multiset_s: TwoRowMatrixView<ExtF>,
                               target:&mut ExtF)
        where BaseF: AbstractField<F=Self::F> + Copy,
              ExtF: AbstractExtensionField<BaseF, F=Self::EF> + Copy;

    fn eval_multiset<BaseF, ExtF>(multiplier: &[ExtF], id: TwoRowMatrixView<BaseF>, fixed: TwoRowMatrixView<BaseF>, advice: TwoRowMatrixView<BaseF>, target:&mut [ExtF])
        where BaseF: AbstractField<F=Self::F> + Copy,
              ExtF: AbstractExtensionField<BaseF, F=Self::EF> + Copy;
}

