use crate::{
    ast::{self, AstKind, DeclList, DeclListExt, RangeKind, TypeEnum, UpcastToAst, type_new},
    context::{ctx, primitives},
    diagnostics::cerror,
    parser::lexer::Span,
    ptr::{OPtr, Ptr},
    sema::primitives::Primitives,
    util::{
        Layout, UnwrapDebug, aligned_add, is_simple_enum, panic_debug, round_up_to_alignment,
        round_up_to_nearest_power_of_two, unreachable_debug, variant_count_to_tag_size_bits,
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
    match common_type_impl_(lhs, rhs, allow_opt_coercion) {
        CommonTypeSelection::Equal => Some(lhs),
        CommonTypeSelection::Lhs => Some(lhs),
        CommonTypeSelection::Rhs => Some(rhs),
        CommonTypeSelection::Mismatch => None,
        CommonTypeSelection::NewAlloc(ty) => Some(ty),
    }
}

// Problem: `common_type(*int_lit, *mut i32)` should return `*i32`
//    Can this (or something similar) even happen?
fn common_type_impl(lhs: Ptr<ast::Type>, rhs: Ptr<ast::Type>) -> CommonTypeSelection {
    common_type_impl_(lhs, rhs, AllowOptionalCoercion::TRUE)
}

fn common_type_impl_(
    lhs: Ptr<ast::Type>,
    rhs: Ptr<ast::Type>,
    allow_opt_coercion: AllowOptionalCoercion,
) -> CommonTypeSelection {
    use CommonTypeSelection::*;
    let p = primitives();

    if lhs == rhs {
        return Equal;
    }

    if lhs == p.err_ty {
        return Lhs;
    } else if rhs == p.err_ty {
        return Rhs;
    }

    if is_bottom_type(lhs, p) || rhs == p.any {
        return Rhs;
    } else if is_bottom_type(rhs, p) || lhs == p.any {
        return Lhs;
    }

    fn optional_common(
        lhs: Ptr<ast::OptionTy>,
        rhs: Ptr<ast::Type>,
        allow_opt_coercion: bool,
    ) -> CommonTypeSelection {
        let lhs_inner = lhs.inner_ty.downcast_type();
        return if allow_opt_coercion
            && lhs_inner.count_optional_nesting() >= rhs.count_optional_nesting()
            && rhs.is_non_zero()
        {
            match common_type_impl_(lhs_inner, rhs, AllowOptionalCoercion::FALSE) {
                Equal | Lhs => Lhs,
                Rhs => NewAlloc(type_new!(OptionTy { inner_ty: rhs.upcast() }).upcast_to_type()),
                NewAlloc(inner_common) => NewAlloc(
                    type_new!(OptionTy { inner_ty: inner_common.upcast() }).upcast_to_type(),
                ),
                Mismatch => Mismatch,
            }
        } else if let Some(rhs_opt) = rhs.try_downcast::<ast::OptionTy>() {
            common_type_impl_(
                lhs_inner,
                rhs_opt.inner_ty.downcast_type(),
                AllowOptionalCoercion::FALSE,
            )
        } else {
            Mismatch
        };
    }

    if let Some(lhs) = lhs.try_downcast::<ast::OptionTy>() {
        return optional_common(lhs, rhs, allow_opt_coercion.rhs);
    } else if let Some(rhs) = rhs.try_downcast::<ast::OptionTy>() {
        debug_assert_ne!(lhs.kind, AstKind::OptionTy);
        return optional_common(rhs, lhs, allow_opt_coercion.lhs).flip();
    }

    if let Some(lhs_num_lvl) = number_subtyping_level(lhs) {
        let rhs_num_lvl = number_subtyping_level(rhs)?;
        debug_assert_ne!(lhs, rhs, "exact equality was already checked above");
        return SubtypingLevel::select(lhs_num_lvl, rhs_num_lvl);
    }

    /// common_type([]T, []mut T) == []T
    /// common_type([]mut T, []T) == []T
    /// common_type([][]mut T, []mut []T) == [][]T
    ///      would require an allocation. => currently an error (TODO)
    macro_rules! common_mut {
        ($lhs:ident, $rhs:ident, $child_sel:expr, $ty:ident $child_field:ident $(,)?) => {{
            let child_sel = $child_sel;
            let out_mut = $lhs.is_mut && $rhs.is_mut;
            match child_sel {
                Mismatch => Mismatch,
                Lhs if $lhs.is_mut == out_mut => Lhs,
                Rhs if $rhs.is_mut == out_mut => Rhs,
                Lhs | Rhs => {
                    let sel = if child_sel == Lhs { $lhs } else { $rhs };
                    let mut copy = unsafe { std::ptr::read(sel.raw()) };
                    copy.is_mut = out_mut;
                    let copy = Ptr::from_ref(&copy).upcast_to_type();
                    cerror!(
                        Span::ZERO,
                        "The compiler currently cannot use `{copy}` as the combined type of `{}` \
                         and `{}`. Consider specifying `{copy}` explicitly.",
                        $lhs.upcast_to_type(),
                        $rhs.upcast_to_type(),
                    );
                    Mismatch
                },
                Equal if $lhs.is_mut == $rhs.is_mut => Equal,
                Equal if $lhs.is_mut == out_mut => Lhs,
                Equal => {
                    debug_assert!($rhs.is_mut == out_mut);
                    Rhs
                },
                // TODO: Maybe allocate in the other problematic case aswell
                NewAlloc(child) => NewAlloc(
                    type_new!($ty { is_mut: out_mut, $child_field: child.upcast() })
                        .upcast_to_type(),
                ),
            }
        }};
    }

    if let Some(lhs) = lhs.try_downcast::<ast::PtrTy>() {
        let rhs = rhs.try_downcast::<ast::PtrTy>()?;
        return common_mut!(
            lhs,
            rhs,
            common_type_impl(lhs.pointee.downcast_type(), rhs.pointee.downcast_type()),
            PtrTy pointee,
        );
    }

    if let Some(lhs) = lhs.try_downcast::<ast::SliceTy>() {
        let rhs = rhs.try_downcast::<ast::SliceTy>()?;
        return common_mut!(
            lhs,
            rhs,
            common_type_impl(lhs.elem_ty.downcast_type(), rhs.elem_ty.downcast_type()),
            SliceTy elem_ty,
        );
    }

    if let Some(lhs) = lhs.try_downcast::<ast::ArrayTy>() {
        return match rhs.try_downcast::<ast::ArrayTy>() {
            Some(rhs) if lhs.len.int::<u64>() == rhs.len.int::<u64>() => {
                common_type_impl(lhs.elem_ty.downcast_type(), rhs.elem_ty.downcast_type())
            },
            _ => Mismatch,
        };
    }

    if let Some(lhs) = lhs.try_downcast::<ast::RangeTy>() {
        return match rhs.try_downcast::<ast::RangeTy>() {
            Some(rhs) if lhs.rkind == rhs.rkind => common_type_impl(lhs.elem_ty, rhs.elem_ty),
            _ => Mismatch,
        };
    }

    if let Some(lhs) = lhs.try_downcast::<ast::Fn>() {
        if let Some(rhs) = rhs.try_downcast::<ast::Fn>()
            // Currently function types must match exactly (see <https://en.wikipedia.org/wiki/Subtyping#Function_types>)
            && ty_match(lhs.upcast_to_type(), rhs.upcast_to_type())
        {
            return Equal;
        }
        return Mismatch;
    }

    Mismatch
}

/// might not be symmetrical
pub fn ty_match(got: Ptr<ast::Type>, expected: Ptr<ast::Type>) -> bool {
    ty_match_(got, expected, true)
}

fn ty_match_(got: Ptr<ast::Type>, expected: Ptr<ast::Type>, allow_opt_coercion: bool) -> bool {
    let p = primitives();

    if got == expected {
        return true;
    }

    if got == p.err_ty || expected == p.err_ty {
        return true;
    }

    if is_bottom_type(got, p) || expected == p.any {
        return true;
    } else if is_bottom_type(expected, p) || got == p.any {
        return false;
    }

    if let Some(expected_lvl) = number_subtyping_level(expected) {
        let Some(got_lvl) = number_subtyping_level(got) else { return false };
        debug_assert_ne!(expected, got, "exact equality was already checked above");
        return SubtypingLevel::select(expected_lvl, got_lvl) != CommonTypeSelection::Mismatch;
    }

    // must be above every non-zero `got` value.
    if let Some(expected_opt) = expected.try_downcast::<ast::OptionTy>() {
        let expected_inner = expected_opt.inner_ty.downcast_type();
        return if allow_opt_coercion
            && expected.count_optional_nesting() > got.count_optional_nesting()
            && got.is_non_zero()
        {
            ty_match_(got, expected_inner, false)
        } else if let Some(got_opt) = got.try_downcast::<ast::OptionTy>() {
            ty_match_(got_opt.inner_ty.downcast_type(), expected_inner, false)
        } else {
            false
        };
    }

    if let Some(expected_ptr) = expected.try_downcast::<ast::PtrTy>() {
        let Some(got_ptr) = got.try_downcast::<ast::PtrTy>() else { return false };
        if ctx().do_mut_checks && expected_ptr.is_mut && !got_ptr.is_mut {
            return false;
        }
        return ty_match(got_ptr.pointee.downcast_type(), expected_ptr.pointee.downcast_type());
    }

    if let Some(expected_slice) = expected.try_downcast::<ast::SliceTy>() {
        let Some(got_slice) = got.try_downcast::<ast::SliceTy>() else { return false };
        if ctx().do_mut_checks && expected_slice.is_mut && !got_slice.is_mut {
            return false;
        }
        return ty_match(got_slice.elem_ty.downcast_type(), expected_slice.elem_ty.downcast_type());
    }

    if let Some(expected_arr) = expected.try_downcast::<ast::ArrayTy>() {
        let Some(got_arr) = got.try_downcast::<ast::ArrayTy>() else { return false };
        return got_arr.len.int::<u64>() == expected_arr.len.int::<u64>()
            && ty_match(got_arr.elem_ty.downcast_type(), expected_arr.elem_ty.downcast_type());
    }

    if let Some(expected_range) = expected.try_downcast::<ast::RangeTy>() {
        let Some(got_range) = got.try_downcast::<ast::RangeTy>() else { return false };
        return got_range.rkind == expected_range.rkind
            && ty_match(got_range.elem_ty, expected_range.elem_ty);
    }

    if let Some(got_fn) = got.try_downcast::<ast::Fn>() {
        let Some(expected_fn) = expected.try_downcast::<ast::Fn>() else { return false };
        return got_fn.params().len() == expected_fn.params().len()
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
    }

    false
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

pub fn remove_type_coercion_for_finalize(expr_ty: Ptr<ast::Type>, out_ty: &mut Ptr<ast::Type>) {
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
pub fn has_no_type_coercion(expr_ty: Ptr<ast::Type>, out_ty: Ptr<ast::Type>) -> bool {
    let mut new_out_ty = out_ty;
    remove_type_coercion_for_finalize(expr_ty, &mut new_out_ty);
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
        }
        debug_assert!(self.is_finalized(), "Cannot finalize `{self}`");
        *self
    }

    /// size of stack allocation in bytes
    pub fn size(self: Ptr<Self>) -> usize {
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
        TypeEnum::EnumDef { variants, variant_tags, tag_ty, .. } => {
            // TODO: precompute is_non_zero for enums?
            let variant_tags = variant_tags.u();
            debug_assert_eq!(variants.len(), variant_tags.len());
            match enum_repr(variants.iter_types()) {
                EnumRepr::Never => AlwaysNull,
                EnumRepr::Transparent(v) => optional_repr(v),
                EnumRepr::Tagged => {
                    debug_assert!(inner_ty.size() > 0);
                    if variant_tags.contains(&0) {
                        Tagged
                    } else {
                        NullOptimized { offset: 0, field_ty: NonZeroFieldType::EnumTag(tag_ty.u()) }
                    }
                },
            }
        },
        TypeEnum::RangeTy { elem_ty, .. } => optional_repr(*elem_ty),
        TypeEnum::OptionTy { .. } => Tagged,
        TypeEnum::ArrayLikeContainer { .. } | TypeEnum::Unset => unreachable_debug(),
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
