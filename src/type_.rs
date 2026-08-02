use crate::{
    ast::{self, AstKind, DeclList, DeclListExt, RangeKind, TypeEnum, UpcastToAst, type_new},
    context::{ctx, primitives},
    diagnostics::cerror,
    parser::lexer::Span,
    ptr::{OPtr, Ptr},
    sema::{accumulate_type, primitives::Primitives},
    util::{
        BigIntExt, Layout, UnwrapDebug, aligned_add, debug_only_assert, is_simple_enum,
        panic_debug, round_up_to_alignment, round_up_to_nearest_power_of_two, unreachable_debug,
        variant_count_to_tag_size_bits,
    },
};
use std::{convert::Infallible, ops::FromResidual};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CommonTypeSelection {
    Equal,
    Lhs,
    Rhs,
    Mismatch,
    NewAlloc(Ptr<ast::Type>),
}

impl FromResidual<Option<Infallible>> for CommonTypeSelection {
    fn from_residual(_: Option<Infallible>) -> Self {
        Self::Mismatch
    }
}

impl CommonTypeSelection {
    pub fn flip(self) -> CommonTypeSelection {
        match self {
            CommonTypeSelection::Lhs => CommonTypeSelection::Rhs,
            CommonTypeSelection::Rhs => CommonTypeSelection::Lhs,
            s => s,
        }
    }
}

/// symmetrical
pub fn common_type(lhs: Ptr<ast::Type>, rhs: Ptr<ast::Type>) -> OPtr<ast::Type> {
    common_type_restrict_optional_coerction(lhs, rhs, AllowOptionalCoercion::TRUE)
}

pub struct AllowOptionalCoercion {
    lhs: bool,
    rhs: bool,
}

#[allow(unused)]
impl AllowOptionalCoercion {
    pub const FALSE: AllowOptionalCoercion = AllowOptionalCoercion { lhs: false, rhs: false };
    pub const LHS: AllowOptionalCoercion = AllowOptionalCoercion { lhs: true, rhs: false };
    pub const RHS: AllowOptionalCoercion = AllowOptionalCoercion { lhs: false, rhs: true };
    pub const TRUE: AllowOptionalCoercion = AllowOptionalCoercion { lhs: true, rhs: true };
}

pub fn common_type_restrict_optional_coerction(
    lhs: Ptr<ast::Type>,
    rhs: Ptr<ast::Type>,
    allow_opt_coercion: AllowOptionalCoercion,
) -> OPtr<ast::Type> {
    match type_check(TypeCheckMode::Join, lhs, rhs, allow_opt_coercion) {
        CommonTypeSelection::Equal => Some(lhs),
        CommonTypeSelection::Lhs => Some(lhs),
        CommonTypeSelection::Rhs => Some(rhs),
        CommonTypeSelection::Mismatch => None,
        CommonTypeSelection::NewAlloc(ty) => Some(ty),
    }
}

/// might not be symmetrical
pub fn ty_match(got: Ptr<ast::Type>, expected: Ptr<ast::Type>) -> bool {
    match type_check(TypeCheckMode::Strict, got, expected, AllowOptionalCoercion::TRUE) {
        CommonTypeSelection::Equal | CommonTypeSelection::Rhs => true,
        CommonTypeSelection::Mismatch | CommonTypeSelection::Lhs => false,
        CommonTypeSelection::NewAlloc(_) => unreachable_debug(),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TypeCheckMode {
    Strict,
    Join,
}

fn type_check(
    mode: TypeCheckMode,
    got: Ptr<ast::Type>,
    expected: Ptr<ast::Type>,
    allow_opt_coercion: AllowOptionalCoercion,
    //quiet: bool,
) -> CommonTypeSelection {
    use CommonTypeSelection::*;

    let p = primitives();

    if got == expected {
        return Equal;
    }

    if expected == p.err_ty {
        return Rhs;
    } else if got == p.err_ty {
        return if mode == TypeCheckMode::Strict { Equal } else { Lhs };
    }

    if is_bottom_type(got, p) || expected == p.any {
        return Rhs;
    } else if is_bottom_type(expected, p) || got == p.any {
        return Lhs;
    }

    if let Some(expected) = expected.try_downcast::<ast::GenericDef>() {
        // resolve polymorph instantiation
        debug_assert!(mode == TypeCheckMode::Strict);
        return if accumulate_type(&mut expected.as_mut().cur_inst, got, None).is_ok() {
            Equal
        } else {
            Mismatch
        };
    } else if let Some(_got) = got.try_downcast::<ast::GenericDef>() {
        //todo!();
        //debug_assert!(got.var_ty.is_none_or(|t| t == expected));
        //got.as_mut().var_ty = Some(expected);
        return Rhs;
    }

    if let Some(expected_lvl) = number_subtyping_level(expected) {
        let got_lvl = number_subtyping_level(got)?;
        debug_assert_ne!(expected, got, "exact equality was already checked above");
        return SubtypingLevel::select(got_lvl, expected_lvl);
    }

    // must be above every non-zero `got` value.
    /*
    if let Some(expected_opt) = expected.try_downcast::<ast::OptionTy>() {
        let expected_inner = expected_opt.inner_ty.downcast_type();
        return if allow_opt_coercion
            && expected.count_optional_nesting() > got.count_optional_nesting()
            && got != p.enum_variant
            && got.is_non_zero()
        {
            ty_match_(got, expected_inner, false)
        } else if let Some(got_opt) = got.try_downcast::<ast::OptionTy>() {
            ty_match_(got_opt.inner_ty.downcast_type(), expected_inner, false)
        } else {
            false
        };
    }
    */

    fn opt_coercion(
        mode: TypeCheckMode,
        lhs: Ptr<ast::Type>,
        rhs: Ptr<ast::OptionTy>,
        allow_opt_coercion: bool,
    ) -> CommonTypeSelection {
        debug_assert!(lhs.kind != AstKind::OptionTy);
        let rhs_inner = rhs.inner_ty.downcast_type();
        return if allow_opt_coercion
            //&& lhs_inner.count_optional_nesting() >= rhs.count_optional_nesting()
            && lhs != primitives().enum_variant // why is this needed?
            && lhs.is_non_zero()
        {
            match type_check(mode, lhs, rhs_inner, AllowOptionalCoercion::FALSE) {
                Equal | Rhs => Rhs,
                Lhs => NewAlloc(type_new!(OptionTy { inner_ty: lhs.upcast() }).upcast_to_type()),
                NewAlloc(inner_common) => NewAlloc(
                    type_new!(OptionTy { inner_ty: inner_common.upcast() }).upcast_to_type(),
                ),
                Mismatch => Mismatch,
            }
        } else {
            Mismatch
        };
    }

    if let Some(expected) = expected.try_downcast::<ast::OptionTy>() {
        return if let Some(got) = got.try_downcast::<ast::OptionTy>() {
            // assumes that OptionTy is never non_zero.
            type_check(
                mode,
                got.inner_ty.downcast_type(),
                expected.inner_ty.downcast_type(),
                AllowOptionalCoercion::FALSE,
            )
        } else {
            opt_coercion(mode, got, expected, allow_opt_coercion.lhs)
        };
    } else if let Some(got) = got.try_downcast::<ast::OptionTy>()
        && mode == TypeCheckMode::Join
    {
        return opt_coercion(mode, expected, got, allow_opt_coercion.rhs).flip();
    }

    /// common_type([]T, []mut T) == []T
    /// common_type([]mut T, []T) == []T
    /// common_type([][]mut T, []mut []T) == [][]T
    ///      would require an allocation. => currently an error (TODO)
    macro_rules! mut_check {
        ($got:expr, $expected:expr, $ty:ident $child_field:ident $(,)?) => {{
            let got = $got;
            let expected = $expected;
            if ctx().do_mut_checks
                && mode == TypeCheckMode::Strict
                && expected.is_mut
                && !got.is_mut
            {
                return Mismatch;
            }
            let out_mut = expected.is_mut && got.is_mut;
            let child_sel = type_check(
                mode,
                got.$child_field.downcast_type(),
                expected.$child_field.downcast_type(),
                AllowOptionalCoercion::TRUE, // *NonNull -> *?NonNull coercion is valid
            );
            match child_sel {
                Mismatch => Mismatch,
                Lhs | Rhs => {
                    let sel = if child_sel == Lhs { got } else { expected };
                    if sel.is_mut == out_mut {
                        child_sel
                    } else {
                        NewAlloc(
                            type_new!($ty { is_mut: out_mut, $child_field: sel.$child_field })
                                .upcast_to_type(),
                        )
                    }
                },
                Equal if got.is_mut == expected.is_mut => Equal,
                Equal if got.is_mut == out_mut => Lhs,
                Equal => {
                    debug_assert!(expected.is_mut == out_mut);
                    Rhs
                },
                NewAlloc(child) => NewAlloc(
                    type_new!($ty { is_mut: out_mut, $child_field: child.upcast() })
                        .upcast_to_type(),
                ),
            }
        }};
    }

    if let Some(expected) = expected.try_downcast::<ast::PtrTy>() {
        let got = got.try_downcast::<ast::PtrTy>()?;
        return mut_check!(got, expected, PtrTy pointee);
    }

    if let Some(expected) = expected.try_downcast::<ast::SliceTy>() {
        let got = got.try_downcast::<ast::SliceTy>()?;
        return mut_check!(got, expected, SliceTy elem_ty);
    }

    if let Some(expected) = expected.try_downcast::<ast::ArrayTy>() {
        let got = got.try_downcast::<ast::ArrayTy>()?;
        if got.len.downcast::<ast::IntVal>().val != expected.len.downcast::<ast::IntVal>().val {
            return Mismatch;
        }
        return type_check(
            mode,
            got.elem_ty.downcast_type(),
            expected.elem_ty.downcast_type(),
            allow_opt_coercion,
        );
    }

    if let Some(expected_range) = expected.try_downcast::<ast::RangeTy>() {
        let got_range = got.try_downcast::<ast::RangeTy>()?;
        if got_range.rkind != expected_range.rkind {
            return Mismatch;
        }
        return type_check(mode, got_range.elem_ty, expected_range.elem_ty, allow_opt_coercion);
    }

    if let Some(got_fn) = got.try_downcast::<ast::Fn>() {
        let expected_fn = expected.try_downcast::<ast::Fn>()?;
        // Currently function types must match exactly (see <https://en.wikipedia.org/wiki/Subtyping#Function_types>)
        let eq = got_fn.params().len() == expected_fn.params().len()
            && ty_match(got_fn.ret_ty.u(), expected_fn.ret_ty.u())
            && got_fn
                .params()
                .iter()
                .map(|p| p.var_ty.u())
                .zip(expected_fn.params().iter().map(|p| p.var_ty.u()))
                .all(|(g, e)|
                    // g and e are swapped because functions are contravariant wrt. parameter types
                    // TODO: better errors messages
                    ty_match(e, g));
        return if eq { Equal } else { Mismatch };
    }

    Mismatch
}

/// The bottom type has no values and can transform into any other type.
///
/// `common({weak}, i32)` -> `i32`
/// `ty_match({weak}, i32)` -> `true`
/// `ty_match(i32, {weak})` -> `false`
#[inline]
pub fn is_bottom_type(ty: Ptr<ast::Type>, p: &Primitives) -> bool {
    ty == p.never || ty == p.rec_ret_ty
}

#[derive(Debug, Clone, Copy)]
struct SubtypingLevel {
    level: u8,
    is_leaf: bool,
}

impl SubtypingLevel {
    /// `lhs == rhs` => [`CommonTypeSelection::Mismatch`]
    fn select(lhs: Self, rhs: Self) -> CommonTypeSelection {
        match lhs.level.cmp(&rhs.level) {
            std::cmp::Ordering::Less if !lhs.is_leaf => CommonTypeSelection::Rhs,
            std::cmp::Ordering::Greater if !rhs.is_leaf => CommonTypeSelection::Lhs,
            _ => CommonTypeSelection::Mismatch,
        }
    }
}

/// # Number subtyping tree
/// ```rust,ignore
///                         top
/// 4 FloatTy
/// 3  ↖︎- float_lit  IntTy{signed=true}
/// 2        ↖︎- sint_lit -↗︎           IntTy{signed=false}
/// 1                    ↖︎- int_lit -↗︎
///                         bottom
/// ```
fn number_subtyping_level(ty: Ptr<ast::Type>) -> Option<SubtypingLevel> {
    match ty.matchable2() {
        ast::TypeMatch::IntTy(int_ty) => Some(SubtypingLevel {
            level: 1 + int_ty.is_signed as u8 + int_ty.bits.is_some() as u8,
            is_leaf: int_ty.bits.is_some(),
        }),
        ast::TypeMatch::FloatTy(float_ty) => Some(SubtypingLevel {
            level: 3 + float_ty.bits.is_some() as u8,
            is_leaf: float_ty.bits.is_some(),
        }),
        _ => None,
    }
}

/// Some expressions might cause type coercion. These expressions usually allow explicit type
/// annotations. This function has to remove any coercion to prevent later uses of `ty` from
/// failing or having to handle all coercion cases.
///
/// ```mylang
/// x: *i32 = /* ... */;
/// y: ?*i32 = x;
/// //         ^ coerced from `*i32` to `?*i32`
///
///   &x;
/// //^ Other expressions, like AddrOf, can never cause type coercion.
///
/// z: ?**i32 = &x;
/// //          ^ produces `**i32` (no coercion)
/// //          ^^ coercion to `?**i32` is caused by Decl
/// ```
pub fn finalize_ty(
    ty: &mut Ptr<ast::Type>,
    mut out_ty: Ptr<ast::Type>,
    can_have_type_coercion: bool,
) -> Ptr<ast::Type> {
    let p = primitives();
    debug_assert!(ty_match(*ty, out_ty), "{ty} matches {out_ty}");
    if *ty == p.never {
        // never is important for trimming unreachable code
        return *ty;
    }

    if can_have_type_coercion {
        remove_optional_coercion_for_finalize(*ty, &mut out_ty);
    } else {
        debug_only_assert!(
            has_no_optional_coercion(*ty, out_ty),
            "expected no type coercion (got: {ty} -> {out_ty})"
        );
    }

    // this is probably incorrect in general.
    // TODO: finalize during `ty_match`
    if !ty.is_finalized()
        || ty.iter_nested_optionals().any(|o| o.upcast_to_type().equals(p.null_ty))
    {
        *ty = out_ty;
    }
    *ty
}

fn remove_optional_coercion_for_finalize(expr_ty: Ptr<ast::Type>, out_ty: &mut Ptr<ast::Type>) {
    if let Some(out_opt) = out_ty.try_downcast::<ast::OptionTy>()
        && expr_ty.is_non_zero()
    {
        let expr_opt_depth = expr_ty.count_optional_nesting();
        let out_opt_depth = out_ty.count_optional_nesting();
        debug_assert!(expr_opt_depth <= out_opt_depth, "should have been a type mismatch");
        match out_opt_depth - expr_opt_depth {
            0 => {},
            1 => *out_ty = out_opt.inner_ty.downcast_type(),
            _ => panic_debug!("should have been a type mismatch"),
        }
    }
}

#[cfg(debug_assertions)]
fn has_no_optional_coercion(expr_ty: Ptr<ast::Type>, out_ty: Ptr<ast::Type>) -> bool {
    let mut new_out_ty = out_ty;
    remove_optional_coercion_for_finalize(expr_ty, &mut new_out_ty);
    new_out_ty == out_ty
}

const ZST_ALIGNMENT: usize = 1;

impl ast::Type {
    pub fn matches_int(self: Ptr<Self>) -> bool {
        self.kind == AstKind::IntTy || self.propagates_out()
    }

    pub fn matches_bool(self: Ptr<Self>) -> bool {
        self == primitives().bool || self.propagates_out()
    }

    pub fn matches_void(self: Ptr<Self>) -> bool {
        self == primitives().void_ty || self.propagates_out()
    }

    pub fn matches_ptr(self: Ptr<Self>) -> bool {
        self.kind == AstKind::PtrTy || self.propagates_out()
    }

    pub fn matches_str(self: Ptr<Self>) -> bool {
        let p = primitives();
        self == p.str_slice_ty
            || self
                .try_downcast::<ast::SliceTy>()
                .is_some_and(|slice| slice.elem_ty.downcast_type() == p.u8)
            || self.propagates_out()
    }

    pub fn is_finalized(&self) -> bool {
        match self.matchable().as_ref() {
            TypeEnum::SimpleTy { is_finalized, .. } => *is_finalized,
            TypeEnum::IntTy { bits, .. } | TypeEnum::FloatTy { bits, .. } => bits.is_some(),
            TypeEnum::StructDef { .. } | TypeEnum::UnionDef { .. } | TypeEnum::EnumDef { .. } => {
                true
            },
            TypeEnum::PtrTy { pointee: t, .. }
            | TypeEnum::SliceTy { elem_ty: t, .. }
            | TypeEnum::ArrayTy { elem_ty: t, .. }
            | TypeEnum::OptionTy { inner_ty: t, .. } => t.downcast_type().is_finalized(),
            TypeEnum::RangeTy { elem_ty, .. } => elem_ty.is_finalized(),
            TypeEnum::Fn { ret_ty, .. } => ret_ty.is_some_and(|t| t.is_finalized()),
            TypeEnum::ArrayLikeContainer { .. } | TypeEnum::Unset => unreachable_debug(),
            TypeEnum::GenericDef { .. } => false,
        }
    }

    /// This might mutate values behind [`Ptr`]s in `self`.
    /// Example: the value behind `elem_ty` on [`TypeInfo::Array`] might change.
    pub fn finalize(self: &mut Ptr<Self>) -> Ptr<ast::Type> {
        let p = primitives();
        debug_assert!(self.ty == p.type_ty || self.kind.is_type_kind());
        match self.matchable().as_mut() {
            TypeEnum::SimpleTy { .. } => {
                if *self == p.rec_ret_ty {
                    cerror!(Span::ZERO, "Cannot infer return type"); // TODO: correct span
                }
            },
            TypeEnum::IntTy { bits: None, .. } => *self = p.i64,
            TypeEnum::FloatTy { bits: None, .. } => *self = p.f64,
            TypeEnum::IntTy { .. }
            | TypeEnum::FloatTy { .. }
            | TypeEnum::StructDef { .. }
            | TypeEnum::UnionDef { .. }
            | TypeEnum::EnumDef { .. } => {},
            TypeEnum::PtrTy { pointee: t, .. }
            | TypeEnum::SliceTy { elem_ty: t, .. }
            | TypeEnum::ArrayTy { elem_ty: t, .. }
            | TypeEnum::OptionTy { inner_ty: t, .. } => {
                t.downcast_type_ref().finalize();
            },
            TypeEnum::RangeTy { elem_ty, .. } => {
                elem_ty.finalize();
            },
            TypeEnum::Fn { params_scope, ret_ty, .. } => {
                debug_assert!(params_scope.decls.iter().all(|p| p.var_ty.u().is_finalized()));
                debug_assert!(ret_ty.u().is_finalized());
            },
            TypeEnum::ArrayLikeContainer { .. } | TypeEnum::Unset => unreachable_debug(),
            TypeEnum::GenericDef { .. } => panic_debug!("cannot finalize GenericDef"),
        }
        debug_assert!(self.is_finalized(), "Cannot finalize `{self}`");
        *self
    }

    /// size of stack allocation in bytes
    pub fn size(self: Ptr<Self>) -> usize {
        debug_assert!(self.is_finalized(), "`{self}` is not finalized");
        const PTR_SIZE: usize = 8;
        match self.matchable().as_ref() {
            TypeEnum::SimpleTy { .. } => {
                let p = primitives();
                if self == p.void_ty || self == p.never || self == p.type_ty || self == p.any {
                    0
                } else if self == p.bool {
                    1
                } else {
                    unreachable_debug()
                }
            },
            TypeEnum::IntTy { bits, .. } | TypeEnum::FloatTy { bits, .. } => int_size(bits.u()),
            TypeEnum::PtrTy { .. } | TypeEnum::Fn { .. } => PTR_SIZE,
            TypeEnum::SliceTy { .. } => 2 * PTR_SIZE,
            TypeEnum::ArrayTy { len, elem_ty, .. } => {
                elem_ty.downcast_type().size() * len.int::<usize>()
            },
            //TypeEnum::FunctionTy { .. } => todo!(),
            TypeEnum::StructDef { fields, .. } => struct_size(fields.iter_types()),
            TypeEnum::UnionDef { fields, .. } => union_size(*fields),
            TypeEnum::EnumDef { variants, tag_ty, .. } => aligned_add(
                int_size(tag_ty.u().bits.u()),
                Layout::new(union_size(*variants), struct_alignment(variants)),
            ),
            TypeEnum::RangeTy { elem_ty, rkind, .. } => elem_ty.size() * rkind.get_field_count(),
            TypeEnum::OptionTy { inner_ty: t, .. } if t.downcast_type().is_non_zero() => {
                t.downcast_type().size()
            },
            TypeEnum::OptionTy { inner_ty: t, .. } => aligned_add(1, t.downcast_type().layout()),
            TypeEnum::ArrayLikeContainer { .. } | TypeEnum::Unset => unreachable_debug(),
            TypeEnum::GenericDef { .. } => todo!("Generic"),
        }
    }

    /// alignment of stack allocation in bytes
    pub fn alignment(self: Ptr<Self>) -> usize {
        let alignment = match self.matchable().as_ref() {
            TypeEnum::SimpleTy { .. } => {
                let p = primitives();
                if self == p.void_ty || self == p.never || self == p.type_ty || self == p.any {
                    ZST_ALIGNMENT
                } else if self == p.bool {
                    1
                } else {
                    todo!()
                }
            },
            TypeEnum::IntTy { bits, .. } | TypeEnum::FloatTy { bits, .. } => {
                int_alignment(bits.u())
            },
            TypeEnum::PtrTy { .. } | TypeEnum::SliceTy { .. } | TypeEnum::Fn { .. } => 8,
            TypeEnum::ArrayTy { elem_ty, .. } => elem_ty.downcast_type().alignment(),
            //TypeEnum::FunctionTy { .. } => todo!(),
            TypeEnum::StructDef { fields, .. } | TypeEnum::UnionDef { fields, .. } => {
                struct_alignment(fields)
            },
            TypeEnum::EnumDef { variants, .. } => enum_alignment(variants),
            TypeEnum::RangeTy { rkind: RangeKind::Full, .. } => ZST_ALIGNMENT,
            TypeEnum::RangeTy { elem_ty, .. } => elem_ty.alignment(),
            TypeEnum::OptionTy { inner_ty, .. } => inner_ty.downcast_type().alignment(),
            TypeEnum::ArrayLikeContainer { .. } | TypeEnum::Unset => unreachable_debug(),
            TypeEnum::GenericDef { .. } => todo!("Generic"),
        };
        debug_assert!(alignment.is_power_of_two());
        alignment
    }

    /// Returns `(self.size(), self.alignment())`
    pub fn layout(self: Ptr<Self>) -> Layout {
        Layout::new(self.size(), self.alignment())
    }

    /// `#sizeof(?T) == #sizeof(T)`
    pub fn is_non_zero(self: Ptr<Self>) -> bool {
        optional_repr(self).is_non_zero()
    }

    /// `not is_primitive`
    pub fn is_aggregate(self: Ptr<Self>) -> bool {
        match self.matchable().as_ref() {
            TypeEnum::SimpleTy { .. } => false,
            TypeEnum::IntTy { .. } | TypeEnum::FloatTy { .. } | TypeEnum::PtrTy { .. } => false,
            TypeEnum::OptionTy { inner_ty, .. } if inner_ty.downcast_type().is_non_zero() => {
                inner_ty.downcast_type().is_aggregate()
            },
            //TypeEnum::FunctionTy { .. } => todo!(),
            TypeEnum::SliceTy { .. }
            | TypeEnum::ArrayTy { .. }
            | TypeEnum::StructDef { .. }
            | TypeEnum::UnionDef { .. }
            | TypeEnum::RangeTy { .. }
            | TypeEnum::OptionTy { .. } => true,
            TypeEnum::EnumDef { is_simple_enum: simple, variants, .. } => {
                debug_assert_eq!(is_simple_enum(*variants), *simple);
                !*simple
            },
            TypeEnum::Fn { .. } => false,
            TypeEnum::ArrayLikeContainer { .. } | TypeEnum::Unset => unreachable_debug(),
            TypeEnum::GenericDef { .. } => todo!("Generic"),
        }
    }

    pub fn is_ffi_noundef(self: Ptr<Self>) -> bool {
        // arrays are special because they are always passed as a primitive pointer
        !self.is_aggregate() || self.kind == AstKind::ArrayTy
    }

    /// `func(arg)`
    ///  ^^^^ never => out = never
    pub fn propagates_out(self: Ptr<Self>) -> bool {
        let p = primitives();
        self == p.never || self == p.err_ty
    }

    pub fn equals(self: Ptr<Self>, other: Ptr<Self>) -> bool {
        use ast::TypeMatch as M;
        match (self.matchable2(), other.matchable2()) {
            (M::SimpleTy(l), M::SimpleTy(r)) => l == r,
            (M::IntTy(l), M::IntTy(r)) => l.is_signed == r.is_signed && l.bits == r.bits,
            (M::FloatTy(l), M::FloatTy(r)) => l.bits == r.bits,
            (M::PtrTy(l), M::PtrTy(r)) => {
                l.is_mut == r.is_mut && l.pointee.downcast_type().equals(r.pointee.downcast_type())
            },
            (M::SliceTy(l), M::SliceTy(r)) => {
                l.is_mut == r.is_mut && l.elem_ty.downcast_type().equals(r.elem_ty.downcast_type())
            },
            (M::ArrayTy(l), M::ArrayTy(r)) => {
                l.len.int::<usize>() == r.len.int()
                    && l.elem_ty.downcast_type().equals(r.elem_ty.downcast_type())
            },
            (M::StructDef(l), M::StructDef(r)) => l == r,
            (M::UnionDef(l), M::UnionDef(r)) => l == r,
            (M::EnumDef(l), M::EnumDef(r)) => l == r,
            (M::RangeTy(l), M::RangeTy(r)) => l.rkind == r.rkind && l.elem_ty.equals(r.elem_ty),
            (M::OptionTy(l), M::OptionTy(r)) => {
                l.inner_ty.downcast_type().equals(r.inner_ty.downcast_type())
            },
            (M::Fn(l), M::Fn(r)) => l == r,
            (M::ArrayLikeContainer(_), _) => unreachable_debug(),
            (_, M::ArrayLikeContainer(_)) => unreachable_debug(),
            _ => false,
        }
    }
}

#[inline]
pub fn int_size(bits: u32) -> usize {
    if bits == 0 {
        return 0;
    }
    round_up_to_nearest_power_of_two(bits as usize).div_ceil(8)
}

#[inline]
pub fn int_alignment(bits: u32) -> usize {
    int_size(bits).min(16)
}

#[inline]
pub fn struct_size(field_types: impl IntoIterator<Item = Ptr<ast::Type>>) -> usize {
    struct_layout(field_types).size
}

#[inline]
pub fn struct_alignment(fields: &[Ptr<ast::Decl>]) -> usize {
    fields.iter_types().map(ast::Type::alignment).max().unwrap_or(ZST_ALIGNMENT)
}

pub fn struct_layout(field_types: impl IntoIterator<Item = Ptr<ast::Type>>) -> Layout {
    let l = struct_layout_unaligned(field_types);
    let size = round_up_to_alignment!(l.size, l.align);
    Layout { size, ..l }
}

/// doesn't align the [`Layout::size`] to the alignment of the entire struct.
fn struct_layout_unaligned(field_types: impl IntoIterator<Item = Ptr<ast::Type>>) -> Layout {
    let mut align = ZST_ALIGNMENT;
    let size = field_types
        .into_iter()
        .map(ast::Type::layout)
        .inspect(|layout| align = align.max(layout.align))
        .fold(0, aligned_add);
    Layout { size, align }
}

pub fn struct_offset(fields: &[Ptr<ast::Decl>], f_idx: usize) -> usize {
    let f = fields.get(f_idx).u();
    let prev_offset = struct_layout_unaligned(fields[..f_idx].iter_types()).size;
    round_up_to_alignment!(prev_offset, f.var_ty.u().alignment())
}

#[inline]
pub fn union_size(fields: DeclList) -> usize {
    fields.iter_types().map(ast::Type::size).max().unwrap_or(0)
}

#[inline]
pub fn enum_alignment(variants: &[Ptr<ast::Decl>]) -> usize {
    int_alignment(variant_count_to_tag_size_bits(variants.len())).max(struct_alignment(variants))
}

#[derive(Debug)]
pub enum EnumRepr {
    /// 0 variants
    Never,
    /// 1 variant
    Transparent(Ptr<ast::Type>),
    /// 2+ variants
    Tagged,
}

pub fn enum_repr(variant_types: impl IntoIterator<Item = Ptr<ast::Type>>) -> EnumRepr {
    let p = primitives();
    // TODO: deep never check. Replace types like `struct { a: never }` with
    // `never` to make this deep check cheap.
    match variant_types.into_iter().filter(|t| *t != p.never).enumerate().last() {
        None => EnumRepr::Never,
        Some((0, only_variant)) => EnumRepr::Transparent(only_variant),
        _ => EnumRepr::Tagged,
    }
}

#[derive(Debug)]
pub enum NonZeroFieldType {
    Ptr,
    EnumTag(Ptr<ast::IntTy>),
}

#[derive(Debug)]
pub enum OptionalRepr {
    /// `null` is the only possible value of `?never`. `Some(never)` cannot be constructed.
    AlwaysNull,
    /// Iff [`OptionTy::inner_ty`] is [`Type::is_non_zero`] then `0` can be used to represent
    /// `null`. Thus a seperate tag field is not needed.
    NullOptimized { offset: usize, field_ty: NonZeroFieldType },
    /// `Optional :: struct { tag: u8, inner_ty: T }`
    Tagged,
}

pub fn optional_repr(inner_ty: Ptr<ast::Type>) -> OptionalRepr {
    use OptionalRepr::*;
    match inner_ty.matchable().as_ref() {
        TypeEnum::SimpleTy { .. } => {
            let p = primitives();
            if inner_ty == p.never {
                AlwaysNull
            } else if [p.void_ty, p.any, p.bool, p.char].contains(&inner_ty) {
                // TODO: null optimization for `?bool`
                Tagged
            } else {
                todo!("{}", inner_ty)
            }
        },
        TypeEnum::IntTy { .. } | TypeEnum::FloatTy { .. } => Tagged,
        TypeEnum::PtrTy { .. } | TypeEnum::SliceTy { .. } | TypeEnum::Fn { .. } => {
            NullOptimized { offset: 0, field_ty: NonZeroFieldType::Ptr }
        },
        TypeEnum::ArrayTy { elem_ty, .. } => optional_repr(elem_ty.downcast_type()),
        TypeEnum::StructDef { fields, .. } => fields
            .iter_types()
            .enumerate()
            .find_map(|(idx, ty)| match optional_repr(ty) {
                AlwaysNull => None, // `?never` doesn't require a tag, but can't represent `null`
                NullOptimized { offset, field_ty } => Some(NullOptimized {
                    offset: struct_size(fields.iter_types().take(idx)) + offset,
                    field_ty,
                }),
                Tagged => None,
            })
            .unwrap_or(Tagged),
        TypeEnum::UnionDef { fields, .. } => {
            let mut biggest_field_size = 0;
            let mut biggest_field = None::<OptionalRepr>;

            for field in fields.iter_types() {
                let repr = optional_repr(field);
                if !repr.is_non_zero() {
                    return Tagged;
                }

                let field_size = field.size();
                if field_size > biggest_field_size {
                    biggest_field_size = field_size;
                    debug_assert!(
                        !repr.is_always_null()
                            || biggest_field.is_none_or(|prev| prev.is_always_null())
                    );
                    biggest_field = Some(repr);
                }
            }

            match biggest_field {
                Some(OptionalRepr::AlwaysNull) | None => AlwaysNull,
                Some(r @ OptionalRepr::NullOptimized { .. }) => r,
                Some(OptionalRepr::Tagged) => unreachable_debug(),
            }
        },
        TypeEnum::EnumDef { variants, tag_ty, .. } => match enum_repr(variants.iter_types()) {
            EnumRepr::Never => AlwaysNull,
            EnumRepr::Transparent(v) => optional_repr(v),
            EnumRepr::Tagged => {
                debug_assert!(inner_ty.size() > 0);
                if variants.into_iter().any(|v| v.init.u().downcast::<ast::IntVal>().val.is_zero())
                {
                    Tagged
                } else {
                    NullOptimized { offset: 0, field_ty: NonZeroFieldType::EnumTag(tag_ty.u()) }
                }
            },
        },
        TypeEnum::RangeTy { elem_ty, .. } => optional_repr(*elem_ty),
        TypeEnum::OptionTy { .. } => Tagged,
        TypeEnum::ArrayLikeContainer { .. } | TypeEnum::Unset => unreachable_debug(),
        TypeEnum::GenericDef { .. } => todo!("Generic"),
    }
}

impl OptionalRepr {
    pub fn is_non_zero(&self) -> bool {
        match self {
            OptionalRepr::AlwaysNull => true,
            OptionalRepr::NullOptimized { .. } => true,
            OptionalRepr::Tagged => false,
        }
    }

    pub fn is_always_null(&self) -> bool {
        matches!(self, OptionalRepr::AlwaysNull)
    }
}

impl ast::OptionTy {
    pub fn repr(&self) -> OptionalRepr {
        optional_repr(self.inner_ty.downcast_type())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{cli::BuildArgs, context::CompilationContext};

    #[test]
    fn number_subtyping() {
        let _ctx = CompilationContext::empty(BuildArgs::default());
        let p = primitives();
        let int_lit = p.int_lit.upcast_to_type();
        let sint_lit = p.sint_lit.upcast_to_type();
        let float_lit = p.float_lit.upcast_to_type();

        assert_eq!(common_type(int_lit, int_lit), int_lit);
        for supertype in [p.u32, sint_lit, p.i32, float_lit, p.f32] {
            assert_eq!(common_type(int_lit, supertype), supertype);
            assert_eq!(common_type(supertype, int_lit), supertype);
            assert_eq!(common_type(supertype, supertype), supertype);
        }
    }
}
