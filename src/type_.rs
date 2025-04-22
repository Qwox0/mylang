use crate::{
    ast::{self, AstKind, DeclList, DeclListExt, TypeEnum},
    context::{ctx, primitives},
    ptr::{OPtr, Ptr},
    util::{
        Layout, UnwrapDebug, aligned_add, round_up_to_nearest_power_of_two, unreachable_debug,
        variant_count_to_tag_size_bits, variant_count_to_tag_size_bytes,
    },
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CommonTypeSelection {
    Lhs,
    Rhs,
    Mismatch,
}

impl CommonTypeSelection {
    #[inline]
    pub fn flip(self) -> CommonTypeSelection {
        match self {
            CommonTypeSelection::Lhs => CommonTypeSelection::Rhs,
            CommonTypeSelection::Rhs => CommonTypeSelection::Lhs,
            CommonTypeSelection::Mismatch => CommonTypeSelection::Mismatch,
        }
    }

    #[inline]
    pub fn flip_if(self, cond: bool) -> CommonTypeSelection {
        if cond { self.flip() } else { self }
    }
}

// Problem: `common_type(*int_lit, *mut i32)` should return `*i32`
//    Can this (or something similar) even happen?
fn common_type_impl(mut lhs: Ptr<ast::Type>, mut rhs: Ptr<ast::Type>) -> CommonTypeSelection {
    use CommonTypeSelection::*;
    let p = primitives();

    #[cfg(debug_assertions)]
    if lhs == p.type_ty || rhs == p.type_ty {
        println!("WARN: checking common_type_impl for primitive `type`");
    }

    if lhs == rhs {
        return Lhs;
    }

    if lhs == p.never {
        return Rhs;
    } else if rhs == p.never {
        return Lhs;
    }

    let swap_inputs = rhs == p.int_lit || rhs == p.float_lit || rhs.kind == AstKind::OptionTy;
    if swap_inputs {
        let tmp = rhs;
        rhs = lhs;
        lhs = tmp;
    }

    if lhs == p.int_lit {
        debug_assert_ne!(rhs, p.int_lit, "was already checked above");
        let matches = matches!(rhs.kind, AstKind::IntTy | AstKind::FloatTy) || rhs == p.float_lit;
        return if matches { Rhs.flip_if(swap_inputs) } else { Mismatch };
    }

    if lhs == p.float_lit {
        // Note: float_lit and int_lit match but float_lit and IntTy don't.
        debug_assert_ne!(rhs, p.float_lit, "was already checked above");
        return if rhs.kind == AstKind::FloatTy {
            Rhs.flip_if(swap_inputs)
        } else if rhs == p.int_lit {
            Lhs.flip_if(swap_inputs)
        } else {
            Mismatch
        };
    }

    if let Some(lhs) = lhs.try_downcast::<ast::OptionTy>() {
        let lhs_inner = lhs.inner_ty.downcast_type();
        return match rhs.try_downcast::<ast::OptionTy>() {
            Some(rhs) => common_type_impl(lhs_inner, rhs.inner_ty.downcast_type()),
            // i32 + ?ilit
            None if rhs.is_non_null() => common_type_impl(lhs_inner, rhs),
            None => Mismatch,
        };
    }

    debug_assert!(!swap_inputs);

    if let Some(lhs) = lhs.try_downcast::<ast::PtrTy>() {
        let lhs_pointee = lhs.pointee.downcast_type();
        if let Some(rhs) = rhs.try_downcast::<ast::PtrTy>() {
            return common_type_impl(lhs_pointee, rhs.pointee.downcast_type());
        }
        return Mismatch;
    }

    if let Some(lhs) = lhs.try_downcast::<ast::SliceTy>() {
        return match rhs.try_downcast::<ast::SliceTy>() {
            Some(rhs) => common_type_impl(lhs.elem_ty.downcast_type(), rhs.elem_ty.downcast_type()),
            None => Mismatch,
        };
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
            && lhs.params.len() == rhs.params.len()
            && ty_match(lhs.ret_ty.u(), rhs.ret_ty.u())
            && lhs
                .params
                .iter()
                .map(|p| p.var_ty.u())
                .zip(rhs.params.iter().map(|p| p.var_ty.u()))
                .all(|(g, e)| ty_match(g, e))
        {
            return Lhs; // or Rhs? idk
        }
        return Mismatch;
    }

    Mismatch
}

/// symmetrical
pub fn common_type(lhs: Ptr<ast::Type>, rhs: Ptr<ast::Type>) -> OPtr<ast::Type> {
    match common_type_impl(lhs, rhs) {
        CommonTypeSelection::Lhs => Some(lhs),
        CommonTypeSelection::Rhs => Some(rhs),
        CommonTypeSelection::Mismatch => None,
    }
}

/// might not be symmetrical
pub fn ty_match(got: Ptr<ast::Type>, expected: Ptr<ast::Type>) -> bool {
    let p = primitives();

    #[cfg(debug_assertions)]
    if got == p.type_ty || expected == p.type_ty {
        println!("WARN: checking ty_match for primitive `type`");
    }

    if got == expected {
        return true;
    }

    if got == p.never || expected == p.any {
        return true;
    } else if expected == p.never || got == p.any {
        return false;
    }

    debug_assert!(expected != p.int_lit && expected != p.float_lit);

    if got == p.int_lit {
        return matches!(expected.kind, AstKind::IntTy | AstKind::FloatTy)
            || expected == p.float_lit;
    }

    if got == p.float_lit {
        return expected.kind == AstKind::FloatTy || expected == p.int_lit;
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
        && got_opt.inner_ty.kind == AstKind::PtrTy
    {
        return true;
    }

    #[allow(unused)]
    if let Some(got_ptr) = got.try_downcast::<ast::PtrTy>() {
        let got_pointee = got_ptr.pointee.downcast_type();
        if let Some(expected_ptr) = expected.try_downcast::<ast::PtrTy>() {
            if ctx().do_mut_checks && expected_ptr.is_mut && !got_ptr.is_mut {
                return false;
            }
            return ty_match(got_pointee, expected_ptr.pointee.downcast_type());
        }
        return false;
    }

    if let Some(got_slice) = got.try_downcast::<ast::SliceTy>() {
        return match expected.try_downcast::<ast::SliceTy>() {
            Some(expected_slice) => {
                ty_match(got_slice.elem_ty.downcast_type(), expected_slice.elem_ty.downcast_type())
            },
            None => false,
        };
    }

    if let Some(got_arr) = got.try_downcast::<ast::ArrayTy>() {
        return match expected.try_downcast::<ast::ArrayTy>() {
            Some(expected_arr) if got_arr.len.int::<u64>() == expected_arr.len.int::<u64>() => {
                ty_match(got_arr.elem_ty.downcast_type(), expected_arr.elem_ty.downcast_type())
            },
            _ => false,
        };
    }

    if let Some(got_range) = got.try_downcast::<ast::RangeTy>() {
        return match expected.try_downcast::<ast::RangeTy>() {
            Some(expected_range) if got_range.rkind == expected_range.rkind => {
                ty_match(got_range.elem_ty, expected_range.elem_ty)
            },
            _ => false,
        };
    }

    if let Some(expected_fn) = expected.try_downcast::<ast::Fn>() {
        if let Some(got_fn) = got.try_downcast::<ast::Fn>()
            && got_fn.params.len() == expected_fn.params.len()
            && ty_match(got_fn.ret_ty.u(), expected_fn.ret_ty.u())
            && got_fn
                .params
                .iter()
                .map(|p| p.var_ty.u())
                .zip(expected_fn.params.iter().map(|p| p.var_ty.u()))
                .all(|(g, e)| ty_match(g, e))
        {
            return true;
        } else if got == p.fn_val {
            return true; // TODO: better checks here
        }
        return false;
    }

    false
}

const ZST_ALIGNMENT: usize = 1;

impl Ptr<ast::Type> {
    pub fn matches_int(self) -> bool {
        self.kind == AstKind::IntTy || self == primitives().never
    }

    pub fn matches_bool(self) -> bool {
        let p = primitives();
        self == p.bool || self == p.never
    }

    pub fn matches_void(self) -> bool {
        let p = primitives();
        self == p.void_ty || self == p.never
    }

    pub fn matches_ptr(self) -> bool {
        self.kind == AstKind::PtrTy || self == primitives().never
    }

    pub fn matches_str(self) -> bool {
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
            TypeEnum::Fn { .. } => true,
            TypeEnum::Unset => unreachable_debug(),
        }
    }

    /// This might mutate values behind [`Ptr`]s in `self`.
    /// Example: the value behind `elem_ty` on [`TypeInfo::Array`] might change.
    pub fn finalize(&mut self) -> Ptr<ast::Type> {
        let p = primitives();
        debug_assert!(self.ty == p.type_ty || self.kind == AstKind::Fn);
        match self.matchable().as_mut() {
            TypeEnum::SimpleTy { .. } => {
                if *self == p.int_lit {
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
            TypeEnum::Fn { .. } => {
                //todo!()
            },
            TypeEnum::Unset => unreachable_debug(),
        }
        debug_assert!(self.is_finalized());
        *self
    }

    /// size of stack allocation in bytes
    pub fn size(self) -> usize {
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
            TypeEnum::PtrTy { .. } => 8,
            TypeEnum::SliceTy { .. } => 16,
            TypeEnum::ArrayTy { len, elem_ty, .. } => {
                elem_ty.downcast_type().size() * len.int::<usize>()
            },
            //TypeEnum::FunctionTy { .. } => todo!(),
            TypeEnum::StructDef { fields, .. } => struct_size(*fields),
            TypeEnum::UnionDef { fields, .. } => union_size(*fields),
            TypeEnum::EnumDef { variants, .. } => aligned_add(
                variant_count_to_tag_size_bytes(variants.len()) as usize,
                Layout::new(union_size(*variants), struct_alignment(*variants)),
            ),
            TypeEnum::RangeTy { elem_ty, rkind, .. } => elem_ty.size() * rkind.get_field_count(),
            TypeEnum::OptionTy { inner_ty: t, .. } if t.downcast_type().is_non_null() => {
                t.downcast_type().size()
            },
            TypeEnum::OptionTy { inner_ty: t, .. } => aligned_add(1, t.downcast_type().layout()),
            TypeEnum::Fn { .. } => todo!(),
            TypeEnum::Unset => unreachable_debug(),
        }
    }

    /// alignment of stack allocation in bytes
    pub fn alignment(self) -> usize {
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
            TypeEnum::PtrTy { .. } | TypeEnum::SliceTy { .. } => 8,
            TypeEnum::ArrayTy { elem_ty, .. } => elem_ty.downcast_type().alignment(),
            //TypeEnum::FunctionTy { .. } => todo!(),
            TypeEnum::StructDef { fields, .. } | TypeEnum::UnionDef { fields, .. } => {
                struct_alignment(*fields)
            },
            TypeEnum::EnumDef { variants, .. } => enum_alignment(*variants),
            TypeEnum::RangeTy { rkind: RangeKind::Full, .. } => ZST_ALIGNMENT,
            TypeEnum::RangeTy { elem_ty, .. } => elem_ty.alignment(),
            TypeEnum::OptionTy { ty, .. } => ty.u().alignment(),
            TypeEnum::Fn { .. } => todo!(),
            TypeEnum::Unset => unreachable_debug(),
        };
        debug_assert!(alignment.is_power_of_two());
        alignment
    }

    /// Returns `(self.size(), self.alignment())`
    pub fn layout(self) -> Layout {
        Layout::new(self.size(), self.alignment())
    }

    pub fn is_non_null(self) -> bool {
        match self.matchable().as_ref() {
            TypeEnum::SimpleTy { .. } => todo!("{:?}", self),
            TypeEnum::IntTy { .. } | TypeEnum::FloatTy { .. } => false,
            TypeEnum::PtrTy { .. } | TypeEnum::SliceTy { .. } => true,
            TypeEnum::ArrayTy { elem_ty, .. } => elem_ty.downcast_type().is_non_null(),
            //TypeEnum::FunctionTy { .. } => todo!(),
            TypeEnum::StructDef { fields, .. } => fields.iter_types().any(Ptr::is_non_null),
            TypeEnum::UnionDef { .. } => todo!(),
            TypeEnum::EnumDef { .. } => todo!(),
            TypeEnum::RangeTy { .. } => todo!(),
            TypeEnum::OptionTy { .. } => false,
            TypeEnum::Fn { .. } => todo!(),
            TypeEnum::Unset => unreachable_debug(),
        }
    }

    pub fn is_aggregate(self) -> bool {
        match self.matchable().as_ref() {
            TypeEnum::SimpleTy { .. } => false,
            TypeEnum::IntTy { .. } | TypeEnum::FloatTy { .. } | TypeEnum::PtrTy { .. } => false,
            TypeEnum::OptionTy { ty, .. } if ty.u().is_non_null() => ty.u().is_aggregate(),
            //TypeEnum::FunctionTy { .. } => todo!(),
            TypeEnum::SliceTy { .. }
            | TypeEnum::ArrayTy { .. }
            | TypeEnum::StructDef { .. }
            | TypeEnum::UnionDef { .. }
            | TypeEnum::EnumDef { .. }
            | TypeEnum::RangeTy { .. }
            | TypeEnum::OptionTy { .. } => true,
            TypeEnum::Fn { .. } => todo!(),
            TypeEnum::Unset => unreachable_debug(),
        }
    }
}

#[inline]
pub fn int_size(bits: u32) -> usize {
    round_up_to_nearest_power_of_two(bits as usize).div_ceil(8)
}

#[inline]
pub fn int_alignment(bits: u32) -> usize {
    int_size(bits).min(16)
}

#[inline]
pub fn struct_size(fields: DeclList) -> usize {
    fields.iter_types().map(Ptr::layout).fold(0, aligned_add)
}

#[inline]
pub fn struct_alignment(fields: DeclList) -> usize {
    fields.iter_types().map(Ptr::alignment).max().unwrap_or(ZST_ALIGNMENT)
}

#[inline]
pub fn union_size(fields: DeclList) -> usize {
    fields.iter_types().map(Ptr::size).max().unwrap_or(0)
}

#[inline]
pub fn enum_alignment(variants: DeclList) -> usize {
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
