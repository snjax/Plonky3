use crate::{Engine};
use p3_field::{AbstractExtensionField, AbstractField, ExtensionField, TwoAdicField};
use p3_matrix::{MatrixRowSlices};
use core::marker::PhantomData;
use p3_air::TwoRowMatrixView;
use p3_matrix::dense::RowMajorMatrix;
use p3_util::log2_ceil_usize;
use alloc::vec;


fn repr_as<T,S>(src:&[T]) -> &S {
    let (prefix, shorts, suffix) = unsafe { src.align_to::<S>() };
    debug_assert!(prefix.is_empty(), "Data was not aligned");
    debug_assert!(suffix.is_empty(), "Data was not aligned");
    debug_assert_eq!(shorts.len(), 1);
    &shorts[0]
}



fn repr_as_mut<T,S>(src:&mut [T]) -> &mut S {
    let (prefix, shorts, suffix) = unsafe { src.align_to_mut::<S>() };
    debug_assert!(prefix.is_empty(), "Data was not aligned");
    debug_assert!(suffix.is_empty(), "Data was not aligned");
    debug_assert_eq!(shorts.len(), 1);
    &mut shorts[0]
}

//TODO: move to field implementation
fn rlc_mul_ex<E:Engine,
    BaseF:AbstractField<F=E::F>+Copy,
    ExtF: AbstractExtensionField<BaseF, F=E::EF>+Copy
>(multiplier:&[ExtF], offset:usize, expr:ExtF) -> ExtF {
    let base_expr = expr.as_base_slice();
    let m_offset = offset*ExtF::D;
    let mut res = multiplier[m_offset] * base_expr[0];
    for i in 1..ExtF::D {
        res += multiplier[m_offset+i] * base_expr[i];
    }
    res
}

fn rlc_mul<E:Engine,
    BaseF:AbstractField<F=E::F>+Copy,
    ExtF: AbstractExtensionField<BaseF, F=E::EF>+Copy
>(multiplier:&[ExtF], offset:usize, expr:BaseF) -> ExtF {
    multiplier[offset*ExtF::D] * expr
}

struct Q<T> {
    l: T,
    r: T,
    o: T,
    m: T,
    c: T
}


struct X<T> {
    a: T,
    b: T,
    c: T,
}

struct LookupTable<T> {
    op: T,
    x: X<T>
}

struct Fixed<T> {
    q: Q<T>,
    sigma: X<T>,
    selector: T,
    op: T,
    table: LookupTable<T>
}

struct Advice<T> {
    x: X<T>,
    lookup_right_m: T
}

struct Multiset<T> {
    id: X<T>,
    sigma: X<T>,
    lookup_left: T,
    lookup_right: T
}

pub struct Plonk<T>(PhantomData<T>);

impl <F, EF> Engine for Plonk<(F, EF)> where
    F:TwoAdicField,
    EF:ExtensionField<F>,
{
    type F = F;
    type EF = EF;
    const LOG_QUOTIENT_DEGREE:usize=1;
    const MAX_MULTISET_ELEMENT_WIDTH:usize=4;
    const MULTISET_WIDTH:usize=7;
    const ID_WIDTH:usize=3;

    fn eval_gates<BaseF, ExtF>(multiplier: &[ExtF],
                               fixed: TwoRowMatrixView<BaseF>,
                               advice: TwoRowMatrixView<BaseF>,
                               multiset_f: TwoRowMatrixView<ExtF>,
                               multiset_a: TwoRowMatrixView<ExtF>,
                               multiset_s: TwoRowMatrixView<ExtF>,
                               target:&mut ExtF)
    where BaseF: AbstractField<F=Self::F> + Copy,
          ExtF: AbstractExtensionField<BaseF, F=Self::EF> + Copy
    {
        let (multiset_f, multiset_next_f, multiset_a, multiset_s) = (
            repr_as::<_,Multiset<ExtF>>(multiset_f.row_slice(0)),
            repr_as::<_,Multiset<ExtF>>(multiset_f.row_slice(1)),
            repr_as::<_,Multiset<ExtF>>(multiset_a.row_slice(0)),
            repr_as::<_,Multiset<ExtF>>(multiset_s.row_slice(0))
        );


        let Fixed::<BaseF> {q, sigma: _, selector: _, op: _, table: _} = repr_as(fixed.row_slice(0));
        let Advice::<BaseF> {x, lookup_right_m} = repr_as(advice.row_slice(0));

        let one = ExtF::one();

        let mut acc = (multiset_next_f.id.a - multiset_f.id.a + multiset_s.id.a) * multiset_a.id.a - one;

        acc += {
            let expr = (multiset_next_f.id.b - multiset_f.id.b + multiset_s.id.b) * multiset_a.id.b - one;
            rlc_mul_ex::<Self,_,_>(multiplier, 0, expr)
        };

        acc += {
            let expr = (multiset_next_f.id.c - multiset_f.id.c + multiset_s.id.c) * multiset_a.id.c - one;
            rlc_mul_ex::<Self,_,_>(multiplier, 1, expr)
        };

        acc += {
            let expr = (multiset_next_f.sigma.a - multiset_f.sigma.a + multiset_s.sigma.a) * multiset_a.sigma.a - one;
            rlc_mul_ex::<Self,_,_>(multiplier, 2, expr)
        };

        acc += {
            let expr = (multiset_next_f.sigma.b - multiset_f.sigma.b + multiset_s.sigma.b) * multiset_a.sigma.b - one;
            rlc_mul_ex::<Self,_,_>(multiplier, 3, expr)
        };

        acc += {
            let expr = (multiset_next_f.sigma.c - multiset_f.sigma.c + multiset_s.sigma.c) * multiset_a.sigma.c - one;
            rlc_mul_ex::<Self,_,_>(multiplier, 4, expr)
        };

        acc += {
            let expr = (multiset_next_f.lookup_left - multiset_f.lookup_left + multiset_s.lookup_left) * multiset_a.lookup_left - one;
            rlc_mul_ex::<Self,_,_>(multiplier, 5, expr)
        };

        acc += {
            let expr = (multiset_next_f.lookup_right - multiset_f.lookup_right + multiset_s.lookup_right) * multiset_a.lookup_right - *lookup_right_m;
            rlc_mul_ex::<Self,_,_>(multiplier, 6, expr)
        };

        acc += {
            let expr = q.l * x.a + q.r * x.b + q.o * x.c + q.m * x.a * x.b + q.c;
            rlc_mul::<Self,_,ExtF>(multiplier, 7, expr)
        };

        *target = acc;
    }

    fn eval_multiset<BaseF, ExtF>(multiplier: &[ExtF], id:TwoRowMatrixView<BaseF>, fixed: TwoRowMatrixView<BaseF>, advice: TwoRowMatrixView<BaseF>, target:&mut [ExtF])
        where BaseF: AbstractField<F=Self::F> + Copy,
              ExtF: AbstractExtensionField<BaseF, F=Self::EF> + Copy
    {
        let multiset_a = repr_as_mut::<_,Multiset<ExtF>>(target);
        let id = repr_as::<_, X<BaseF>>(id.row_slice(0));

        let Fixed::<BaseF> {q: _, sigma, selector, op, table} = repr_as(fixed.row_slice(0));
        let Advice::<BaseF> {x, lookup_right_m: _} = repr_as(advice.row_slice(0));


        let gamma:&(ExtF,ExtF,ExtF,ExtF) = repr_as(&multiplier[0..4]);

        let one = ExtF::one();

        let x_a_multiplied = gamma.1 * x.a;
        let x_b_multiplied = gamma.1 * x.b;
        let x_c_multiplied = gamma.1 * x.c;

        let lookup_left = one + (gamma.0 * (*op) + gamma.1 * x.a + gamma.2 * x.b + gamma.3 * x.c) * (*selector);
        let lookup_right = one + gamma.0*table.op + gamma.1*table.x.a + gamma.2*table.x.b + gamma.3*table.x.c;

        *multiset_a = Multiset {
            id: X {
                a: one + gamma.0 * id.a + x_a_multiplied,
                b: one + gamma.0 * id.b + x_b_multiplied,
                c: one + gamma.0 * id.c + x_c_multiplied,
            },
            sigma: X {
                a: one + gamma.0 * sigma.a + x_a_multiplied,
                b: one + gamma.0 * sigma.b + x_b_multiplied,
                c: one + gamma.0 * sigma.c + x_c_multiplied,
            },
            lookup_left,
            lookup_right
        };
    }

    fn id_matrix(log_degree: usize) -> RowMajorMatrix<Self::F> {
        // Field should be big enough to represent all indices as a single field element. Otherwise we need two field elements per one index.
        let degree = 1 << log_degree;
        assert!(log2_ceil_usize(Self::ID_WIDTH) + log_degree < F::TWO_ADICITY);

        let g = F::two_adic_generator(log_degree);
        let h = F::generator();

        let h2 = h*h;

        let mut buff = vec![F::zero(); degree*Self::ID_WIDTH];

        let mut x = F::one();

        for i in 0..degree {
            *repr_as_mut(&mut buff[i*Self::ID_WIDTH..(i+1)*Self::ID_WIDTH]) = X {
                a: x,
                b: x*h,
                c: x*h2
            };
            x*=g;
        }

        RowMajorMatrix::new(buff, Self::ID_WIDTH)
    }

    fn id_matrix_at<BaseF>(x_local:BaseF, x_next:BaseF) -> RowMajorMatrix<BaseF>
        where BaseF: AbstractField<F=Self::F> + Copy
    {
        let mut buff = vec![BaseF::zero(); 2*Self::ID_WIDTH];
        let h = BaseF::generator();
        let h2 = h*h;
        *repr_as_mut(&mut buff[0..Self::ID_WIDTH]) = X {
            a: x_local,
            b: x_local*h,
            c: x_local*h2
        };

        *repr_as_mut(&mut buff[Self::ID_WIDTH..2*Self::ID_WIDTH]) = X {
            a: x_next,
            b: x_next*h,
            c: x_next*h2
        };

        RowMajorMatrix::new(buff, Self::ID_WIDTH)
    }
}