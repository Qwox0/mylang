use crate::{
    ast::{self, AstKind, DeclList, DeclListExt, TypeEnum},
    context::{ctx, primitives},
    diagnostics::cerror,
    parser::lexer::Span,
    ptr::{OPtr, Ptr},
    sema::primitives::Primitives,
    util::{
        Layout, UnwrapDebug, aligned_add, is_simple_enum, round_up_to_alignment,
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
}

impl FromResidual<Option<Infallible>> for CommonTypeSelection {
    fn from_residual(_: Option<Infallible>) -> Self {
        Self::Mismatch
    }
}

// Problem: `common_type(*int_lit, *mut i32)` should return `*i32`
//    Can this (or something similar) even happen?
fn common_type_impl(mut lhs: Ptr<ast::Type>, mut rhs: Ptr<ast::Type>) -> CommonTypeSelection {
    use CommonTypeSelection::*;
    let p = primitives();

    if lhs == rhs {
        return Equal;
    }

    if is_bottom_type(lhs, p) || rhs == p.any {
        return Rhs;
    } else if is_bottom_type(rhs, p) || lhs == p.any {
        return Lhs;
    }

    let swap_inputs = rhs.kind == AstKind::OptionTy;
    if swap_inputs {
        let tmp = rhs;
        rhs = lhs;
        lhs = tmp;
    }

    if let Some(lhs) = lhs.try_downcast::<ast::OptionTy>() {
        let lhs_inner = lhs.inner_ty.downcast_type();
        return match rhs.try_downcast::<ast::OptionTy>() {
            Some(rhs) => common_type_impl(lhs_inner, rhs.inner_ty.downcast_type()),
            // ?ilit + i32
            None if rhs.is_non_null() => common_type_impl(lhs_inner, rhs),
            None => Mismatch,
        };
    }

    debug_assert!(!swap_inputs);

    if let Some(lhs_num_lvl) = number_subtyping_level(lhs) {
        let rhs_num_lvl = number_subtyping_level(rhs)?;
        debug_assert_ne!(lhs, rhs, "was already checked above");
        return SubtypingLevel::select(lhs_num_lvl, rhs_num_lvl);
    }

    /// common_type([]T, []mut T) == []T
    /// common_type([]mut T, []T) == []T
    /// common_type([][]mut T, []mut []T) == [][]T
    ///      would require an allocation. => currently an error (TODO)
    macro_rules! common_mut {
        ($lhs:ident, $rhs:ident, $child_sel:expr $(,)?) => {{
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
            }
        }};
    }

    if let Some(lhs) = lhs.try_downcast::<ast::PtrTy>() {
        let rhs = rhs.try_downcast::<ast::PtrTy>()?;
        return common_mut!(
            lhs,
            rhs,
            common_type_impl(lhs.pointee.downcast_type(), rhs.pointee.downcast_type()),
        );
    }

    if let Some(lhs) = lhs.try_downcast::<ast::SliceTy>() {
        let rhs = rhs.try_downcast::<ast::SliceTy>()?;
        return common_mut!(
            lhs,
            rhs,
            common_type_impl(lhs.elem_ty.downcast_type(), rhs.elem_ty.downcast_type()),
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

/// symmetrical
pub fn common_type(lhs: Ptr<ast::Type>, rhs: Ptr<ast::Type>) -> OPtr<ast::Type> {
    match common_type_impl(lhs, rhs) {
        CommonTypeSelection::Equal => Some(lhs),
        CommonTypeSelection::Lhs => Some(lhs),
        CommonTypeSelection::Rhs => Some(rhs),
        CommonTypeSelection::Mismatch => None,
    }
}

/// might not be symmetrical
pub fn ty_match(got: Ptr<ast::Type>, expected: Ptr<ast::Type>) -> bool {
    let p = primitives();

    if got == expected {
        return true;
    }

    if is_bottom_type(got, p) || expected == p.any {
        return true;
    } else if is_bottom_type(expected, p) || got == p.any {
        return false;
    }

    if got == p.int_lit {
        return matches!(expected.kind, AstKind::IntTy | AstKind::FloatTy)
            || expected == p.float_lit;
    } else if got == p.sint_lit {
        return expected.is_sint() || expected.kind == AstKind::FloatTy || expected == p.float_lit;
    }

    if got == p.float_lit {
        return expected.kind == AstKind::FloatTy || expected.is_int_lit();
    }

    // needs to be above every non_null `got` value.
    if let Some(expected) = expected.try_downcast::<ast::OptionTy>() {
        let expected_inner = expected.inner_ty.downcast_type();
        return match got.try_downcast::<ast::OptionTy>() {
            Some(got) => ty_match(got.inner_ty.downcast_type(), expected_inner),
            None if got.is_non_null() => ty_match(got, expected_inner),
            None => false,
        };
    }

    // TODO: remove this rule when implementing option lifting
    if let Some(got_opt) = got.try_downcast::<ast::OptionTy>()
        && expected.kind == AstKind::PtrTy
        && got_opt.inner_ty.rep().kind == AstKind::PtrTy
    {
        return true;
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
                .all(|(g, e)| ty_match(g, e));
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
    ty == p.never || ty == p.unknown_ty
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
    let p = primitives();
    Some(if ty == p.int_lit {
        SubtypingLevel { level: 1, is_leaf: false }
    } else if ty == p.sint_lit {
        SubtypingLevel { level: 2, is_leaf: false }
    } else if ty == p.float_lit {
        SubtypingLevel { level: 3, is_leaf: false }
    } else if let Some(int_ty) = ty.try_downcast::<ast::IntTy>() {
        SubtypingLevel { level: 2 + int_ty.is_signed as u8, is_leaf: true }
    } else if ty.kind == AstKind::FloatTy {
        SubtypingLevel { level: 4, is_leaf: true }
    } else {
        return None;
    })
}

const ZST_ALIGNMENT: usize = 1;

impl ast::Type {
    pub fn matches_int(self: Ptr<Self>) -> bool {
        self.kind == AstKind::IntTy || self == primitives().never
    }

    pub fn matches_bool(self: Ptr<Self>) -> bool {
        let p = primitives();
        self == p.bool || self == p.never
    }

    pub fn matches_void(self: Ptr<Self>) -> bool {
        let p = primitives();
        self == p.void_ty || self == p.never
    }

    pub fn matches_ptr(self: Ptr<Self>) -> bool {
        self.kind == AstKind::PtrTy || self == primitives().never
    }

    pub fn matches_str(self: Ptr<Self>) -> bool {
        let p = primitives();
        self == p.str_slice_ty
            || self
                .try_downcast::<ast::SliceTy>()
                .is_some_and(|slice| slice.elem_ty.downcast_type() == p.u8)
    }

    pub fn is_finalized(&self) -> bool {
        match self.matchable().as_ref() {
            TypeEnum::SimpleTy { is_finalized, .. } => *is_finalized,
            TypeEnum::IntTy { .. }
            | TypeEnum::FloatTy { .. }
            | TypeEnum::StructDef { .. }
            | TypeEnum::UnionDef { .. }
            | TypeEnum::EnumDef { .. } => true,
            TypeEnum::PtrTy { pointee: t, .. }
            | TypeEnum::SliceTy { elem_ty: t, .. }
            | TypeEnum::ArrayTy { elem_ty: t, .. }
            | TypeEnum::OptionTy { inner_ty: t, .. } => t.downcast_type().is_finalized(),
            TypeEnum::RangeTy { elem_ty, .. } => elem_ty.is_finalized(),
            TypeEnum::Fn { ret_ty, .. } => ret_ty.is_some_and(|t| t.is_finalized()),
            TypeEnum::Unset => unreachable_debug(),
        }
    }

    /// This might mutate values behind [`Ptr`]s in `self`.
    /// Example: the value behind `elem_ty` on [`TypeInfo::Array`] might change.
    pub fn finalize(self: &mut Ptr<Self>) -> Ptr<ast::Type> {
        let p = primitives();
        debug_assert!(self.ty == p.type_ty || self.kind.is_type_kind());
        match self.matchable().as_mut() {
            TypeEnum::SimpleTy { .. } => {
                if self.is_int_lit() {
                    *self = p.i64;
                } else if *self == p.float_lit {
                    *self = p.f64;
                }
            },
            TypeEnum::IntTy { .. }
            | TypeEnum::FloatTy { .. }
            | TypeEnum::StructDef { .. }
            | TypeEnum::UnionDef { .. }
            | TypeEnum::EnumDef { .. } => {},
            TypeEnum::PtrTy { pointee: t, .. }
            | TypeEnum::SliceTy { elem_ty: t, .. }
            | TypeEnum::ArrayTy { elem_ty: t, .. }
            | TypeEnum::OptionTy { inner_ty: t, .. } => {
                let ty = t.rep_mut().downcast_type_ref();
                ty.finalize();
            },
            TypeEnum::RangeTy { elem_ty, .. } => {
                elem_ty.finalize();
            },
            TypeEnum::Fn { params_scope, ret_ty, .. } => {
                debug_assert!(params_scope.decls.iter().all(|p| p.var_ty.u().is_finalized()));
                debug_assert!(ret_ty.u().is_finalized());
            },
            TypeEnum::Unset => unreachable_debug(),
        }
        debug_assert!(self.is_finalized());
        *self
    }

    /// size of stack allocation in bytes
    pub fn size(self: Ptr<Self>) -> usize {
        const PTR_SIZE: usize = 8;
        match self.matchable().as_ref() {
            TypeEnum::SimpleTy { .. } => {
                let p = primitives();
                if self == p.void_ty || self == p.never || self == p.type_ty {
                    0
                } else if self == p.bool {
                    1
                } else {
                    unreachable_debug()
                }
            },
            TypeEnum::IntTy { bits, .. } | TypeEnum::FloatTy { bits, .. } => int_size(*bits),
            TypeEnum::PtrTy { .. } | TypeEnum::Fn { .. } => PTR_SIZE,
            TypeEnum::SliceTy { .. } => 2 * PTR_SIZE,
            TypeEnum::ArrayTy { len, elem_ty, .. } => {
                elem_ty.downcast_type().size() * len.int::<usize>()
            },
            //TypeEnum::FunctionTy { .. } => todo!(),
            TypeEnum::StructDef { fields, .. } => struct_size(fields),
            TypeEnum::UnionDef { fields, .. } => union_size(*fields),
            TypeEnum::EnumDef { variants, tag_ty, .. } => aligned_add(
                int_size(tag_ty.u().bits),
                Layout::new(union_size(*variants), struct_alignment(variants)),
            ),
            TypeEnum::RangeTy { elem_ty, rkind, .. } => elem_ty.size() * rkind.get_field_count(),
            TypeEnum::OptionTy { inner_ty: t, .. } if t.downcast_type().is_non_null() => {
                t.downcast_type().size()
            },
            TypeEnum::OptionTy { inner_ty: t, .. } => aligned_add(1, t.downcast_type().layout()),
            TypeEnum::Unset => unreachable_debug(),
        }
    }

    /// alignment of stack allocation in bytes
    pub fn alignment(self: Ptr<Self>) -> usize {
        let alignment = match self.matchable().as_ref() {
            TypeEnum::SimpleTy { .. } => {
                let p = primitives();
                if self == p.void_ty || self == p.never || self == p.type_ty {
                    ZST_ALIGNMENT
                } else if self == p.bool {
                    1
                } else {
                    todo!()
                }
            },
            TypeEnum::IntTy { bits, .. } | TypeEnum::FloatTy { bits, .. } => int_alignment(*bits),
            TypeEnum::PtrTy { .. } | TypeEnum::SliceTy { .. } | TypeEnum::Fn { .. } => 8,
            TypeEnum::ArrayTy { elem_ty, .. } => elem_ty.downcast_type().alignment(),
            //TypeEnum::FunctionTy { .. } => todo!(),
            TypeEnum::StructDef { fields, .. } | TypeEnum::UnionDef { fields, .. } => {
                struct_alignment(fields)
            },
            TypeEnum::EnumDef { variants, .. } => enum_alignment(variants),
            TypeEnum::RangeTy { rkind: RangeKind::Full, .. } => ZST_ALIGNMENT,
            TypeEnum::RangeTy { elem_ty, .. } => elem_ty.alignment(),
            TypeEnum::OptionTy { ty, .. } => ty.u().alignment(),
            TypeEnum::Unset => unreachable_debug(),
        };
        debug_assert!(alignment.is_power_of_two());
        alignment
    }

    /// Returns `(self.size(), self.alignment())`
    pub fn layout(self: Ptr<Self>) -> Layout {
        Layout::new(self.size(), self.alignment())
    }

    pub fn is_non_null(self: Ptr<Self>) -> bool {
        match self.matchable().as_ref() {
            TypeEnum::SimpleTy { .. } => {
                let p = primitives();
                if self == p.void_ty { false } else { todo!("{:?}", self) }
            },
            TypeEnum::IntTy { .. } | TypeEnum::FloatTy { .. } => false,
            TypeEnum::PtrTy { .. } | TypeEnum::SliceTy { .. } | TypeEnum::Fn { .. } => true,
            TypeEnum::ArrayTy { elem_ty, .. } => elem_ty.downcast_type().is_non_null(),
            //TypeEnum::FunctionTy { .. } => todo!(),
            TypeEnum::StructDef { fields, .. } => fields.iter_types().any(ast::Type::is_non_null),
            TypeEnum::UnionDef { .. } => todo!(),
            TypeEnum::EnumDef { .. } => todo!(),
            TypeEnum::RangeTy { .. } => todo!(),
            TypeEnum::OptionTy { .. } => false,
            TypeEnum::Unset => unreachable_debug(),
        }
    }

    /// `not is_primitive`
    pub fn is_aggregate(self: Ptr<Self>) -> bool {
        match self.matchable().as_ref() {
            TypeEnum::SimpleTy { .. } => false,
            TypeEnum::IntTy { .. } | TypeEnum::FloatTy { .. } | TypeEnum::PtrTy { .. } => false,
            TypeEnum::OptionTy { inner_ty, .. } if inner_ty.downcast_type().is_non_null() => {
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
            TypeEnum::Unset => unreachable_debug(),
        }
    }

    pub fn is_ffi_noundef(self: Ptr<Self>) -> bool {
        // arrays are special because they are always passed as a primitive pointer
        !self.is_aggregate() || self.kind == AstKind::ArrayTy
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
pub fn struct_size(fields: &[Ptr<ast::Decl>]) -> usize {
    struct_layout(fields).size
}

#[inline]
pub fn struct_alignment(fields: &[Ptr<ast::Decl>]) -> usize {
    fields.iter_types().map(ast::Type::alignment).max().unwrap_or(ZST_ALIGNMENT)
}

pub fn struct_layout(fields: &[Ptr<ast::Decl>]) -> Layout {
    let l = struct_layout_unaligned(fields);
    let size = round_up_to_alignment!(l.size, l.align);
    Layout { size, ..l }
}

/// doesn't align the [`Layout::size`] to the alignment of the entire struct.
fn struct_layout_unaligned(fields: &[Ptr<ast::Decl>]) -> Layout {
    let mut align = ZST_ALIGNMENT;
    let size = fields
        .iter_types()
        .map(ast::Type::layout)
        .inspect(|layout| align = align.max(layout.align))
        .fold(0, aligned_add);
    Layout { size, align }
}

pub fn struct_offset(fields: &[Ptr<ast::Decl>], f_idx: usize) -> usize {
    let f = fields.get(f_idx).u();
    let prev_offset = struct_layout_unaligned(&fields[..f_idx]).size;
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RangeKind {
    /// `..`
    Full,
    /// `start..`
    From,
    /// `..end`
    To,
    /// `..=end`
    ToInclusive,
    /// `start..end`
    Both,
    /// `start..=end`
    BothInclusive,
}

impl RangeKind {
    pub fn get_field_count(self) -> usize {
        match self {
            RangeKind::Full => 0,
            RangeKind::From | RangeKind::To | RangeKind::ToInclusive => 1,
            RangeKind::Both | RangeKind::BothInclusive => 2,
        }
    }

    pub fn has_start(self) -> bool {
        match self {
            RangeKind::Full | RangeKind::To | RangeKind::ToInclusive => false,
            RangeKind::From | RangeKind::Both | RangeKind::BothInclusive => true,
        }
    }

    pub fn has_end(self) -> bool {
        match self {
            RangeKind::Full | RangeKind::From => false,
            RangeKind::To | RangeKind::ToInclusive | RangeKind::Both | RangeKind::BothInclusive => {
                true
            },
        }
    }

    pub fn is_inclusive(self) -> bool {
        match self {
            RangeKind::Full | RangeKind::From | RangeKind::To | RangeKind::Both => false,
            RangeKind::ToInclusive | RangeKind::BothInclusive => true,
        }
    }

    pub fn type_name(self) -> &'static str {
        match self {
            RangeKind::Full => "RangeFull",
            RangeKind::From => "RangeFrom",
            RangeKind::To => "RangeTo",
            RangeKind::ToInclusive => "RangeToInclusive",
            RangeKind::Both => "Range",
            RangeKind::BothInclusive => "RangeInclusive",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{cli::BuildArgs, context::CompilationContext};

    #[test]
    fn number_subtyping() {
        let _ctx = CompilationContext::new(BuildArgs::default());
        let p = primitives();
        assert_eq!(common_type(p.int_lit, p.int_lit), p.int_lit);
        for supertype in [p.u32, p.sint_lit, p.i32, p.float_lit, p.f32] {
            assert_eq!(common_type(p.int_lit, supertype), supertype);
            assert_eq!(common_type(supertype, p.int_lit), supertype);
            assert_eq!(common_type(supertype, supertype), supertype);
        }
    }
}
