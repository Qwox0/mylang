use super::err::{SemaErrorKind, SemaResult};
use crate::{parser::lexer::Span, ptr::Ptr, type_::Type};
use std::ptr::NonNull;

#[derive(Debug, Clone, Copy)]
pub struct SemaValue {
    pub ty: Type,
    /// [`None`] if this is a runtime value
    /// [`Some`] if this is a compiletime value
    /// cast correctly based on `self.ty`.
    pub const_val: Option<Ptr<()>>,
}

impl SemaValue {
    pub fn new(ty: Type) -> SemaValue {
        SemaValue { ty, const_val: None }
    }

    pub fn new_const(ty: Type, val: Ptr<()>) -> SemaValue {
        SemaValue { ty, const_val: Some(val) }
    }

    pub fn never() -> SemaValue {
        SemaValue::new_const(Type::Never, EMPTY_PTR)
    }

    pub fn void() -> SemaValue {
        SemaValue::new_const(Type::Void, EMPTY_PTR)
    }

    pub fn type_(ty: Ptr<Type>) -> SemaValue {
        SemaValue::new_const(Type::Type(ty), EMPTY_PTR)
    }

    pub fn const_bool(p: Ptr<bool>) -> SemaValue {
        SemaValue::new_const(Type::Bool, p.cast())
    }

    pub fn is_const(&self) -> bool {
        self.const_val.is_some()
    }

    pub fn check_constness(&self, has_to_be_const: bool) -> bool {
        !has_to_be_const || self.is_const()
    }

    pub fn into_const_checked(&self, has_to_be_const: bool, span: Span) -> SemaResult<SemaValue> {
        if self.check_constness(has_to_be_const) {
            SemaResult::Ok(*self)
        } else {
            super::err(SemaErrorKind::NotAConstExpr, span)
        }
    }

    pub fn is_int(self) -> bool {
        matches!(self.ty, Type::Int { .. } | Type::IntLiteral)
    }

    pub fn int(self) -> Result<i128, SemaErrorKind> {
        if !self.is_int() {
            return Err(SemaErrorKind::ExpectedNumber { got: self.ty });
        }
        let Some(count_val) = self.const_val else {
            return Err(SemaErrorKind::NotAConstExpr);
        };
        Ok(*count_val.cast::<i128>())
    }

    pub fn finalize_ty(mut self) -> SemaValue {
        self.ty = self.ty.finalize();
        self
    }
}

const EMPTY: () = ();
pub const EMPTY_PTR: Ptr<()> =
    Ptr::new(unsafe { NonNull::new_unchecked(&EMPTY as *const () as *mut ()) });
