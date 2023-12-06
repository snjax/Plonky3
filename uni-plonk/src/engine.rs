use p3_field::{AbstractExtensionField, AbstractField, ExtensionField, Field, TwoAdicField};
use p3_air::TwoRowMatrixView;
use p3_matrix::dense::RowMajorMatrix;

pub trait Engine {
    type F: TwoAdicField;
    type EF: ExtensionField<Self::F>;

    const LOG_QUOTIENT_DEGREE: usize;
    const MAX_MULTISET_ELEMENT_WIDTH: usize;
    const MULTISET_WIDTH: usize;
    const ID_WIDTH: usize;
    fn eval_gates<BaseF, ExtF>(multiplier: &[ExtF],
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
    
    
    fn id_matrix(log_degree:usize) -> RowMajorMatrix<Self::F>;

    fn id_matrix_at<BaseF>(x_local:BaseF, x_next:BaseF) -> RowMajorMatrix<BaseF>
        where BaseF: AbstractField<F=Self::F> + Copy;
}

