use crate::{
    ast::{self, Ast},
    diagnostics::HandledErr,
    parser::lexer::Span,
    ptr::Ptr,
};
use SemaResult::*;
use std::{
    convert::Infallible,
    ops::{FromResidual, Try},
};

#[derive(Debug, Clone, thiserror::Error)]
#[error("{self:?}")]
pub enum SemaErrorKind {
    ConstDeclWithoutInit,

    #[error("MismatchedTypesBinOp (lhs: {lhs_ty}, rhs: {rhs_ty})")]
    MismatchedTypesBinOp {
        lhs_ty: Ptr<ast::Type>,
        rhs_ty: Ptr<ast::Type>,
    },
    CannotInferAutocastTy,
    ReturnNotInAFunction,
    NotAType,

    MissingArg,

    CannotInfer,
    UnionFieldWithDefaultValue,

    UnexpectedTopLevelExpr(Ptr<Ast>),

    HandledErr,
}

#[derive(Debug, Clone)]
pub struct SemaError {
    pub kind: SemaErrorKind,
    pub span: Span,
}

impl SemaError {
    #[allow(non_upper_case_globals)]
    pub const HandledErr: SemaError =
        SemaError { kind: SemaErrorKind::HandledErr, span: Span::ZERO };
}

impl From<crate::diagnostics::HandledErr> for SemaError {
    fn from(_: crate::diagnostics::HandledErr) -> Self {
        SemaError::HandledErr
    }
}

impl From<!> for SemaError {
    fn from(never: !) -> Self {
        never
    }
}

#[derive(Debug, PartialEq, Eq)]
#[must_use]
pub enum SemaResult<T, E = SemaError> {
    Ok(T),
    //NotFinished(Option<Type>),
    NotFinished,
    Err(E),
}

impl<T> SemaResult<T> {
    #[allow(non_upper_case_globals)]
    pub const HandledErr: SemaResult<T> = Err(SemaError::HandledErr);
}

impl<T, E> SemaResult<T, E> {
    pub fn map_ok<U>(self, f: impl FnOnce(T) -> U) -> SemaResult<U, E> {
        match self {
            Ok(t) => Ok(f(t)),
            NotFinished => NotFinished,
            Err(err) => Err(err),
        }
    }

    pub fn map_err<E2>(self, f: impl FnOnce(E) -> E2) -> SemaResult<T, E2> {
        match self {
            Ok(t) => Ok(t),
            NotFinished => NotFinished,
            Err(err) => Err(f(err)),
        }
    }

    pub fn and_then<U>(self, f: impl FnOnce(T) -> SemaResult<U, E>) -> SemaResult<U, E> {
        match self {
            Ok(t) => f(t),
            NotFinished => NotFinished,
            Err(err) => Err(err),
        }
    }

    pub fn inspect_err(self, f: impl FnOnce(&E)) -> Self {
        if let Err(e) = &self {
            f(e);
        }
        self
    }

    pub fn is_ok(&self) -> bool {
        matches!(self, Ok(_))
    }

    pub fn is_err(&self) -> bool {
        matches!(self, Err(_))
    }

    pub fn is_ok_and(&self, cond: impl FnOnce(&T) -> bool) -> bool {
        match self {
            Ok(t) => cond(t),
            _ => false,
        }
    }

    pub fn ok(self) -> Option<T> {
        match self {
            Ok(t) => Some(t),
            _ => None,
        }
    }
}

impl<T, E> Try for SemaResult<T, E> {
    type Output = T;
    type Residual = SemaResult<!, E>;

    fn from_output(output: Self::Output) -> Self {
        Ok(output)
    }

    fn branch(self) -> std::ops::ControlFlow<Self::Residual, Self::Output> {
        match self {
            Ok(ty) => std::ops::ControlFlow::Continue(ty),
            NotFinished => std::ops::ControlFlow::Break(SemaResult::NotFinished),
            Err(err) => std::ops::ControlFlow::Break(SemaResult::Err(err)),
        }
    }
}

impl<T, E> FromResidual<SemaResult<!, E>> for SemaResult<T, E> {
    fn from_residual(residual: SemaResult<!, E>) -> Self {
        match residual {
            NotFinished => SemaResult::NotFinished,
            Err(err) => SemaResult::Err(err),
        }
    }
}

impl<T> FromResidual<SemaResult<!, HandledErr>> for SemaResult<T> {
    fn from_residual(residual: SemaResult<!, HandledErr>) -> Self {
        match residual {
            NotFinished => SemaResult::NotFinished,
            Err(err) => SemaResult::Err(err.into()),
        }
    }
}

impl<T> FromResidual<SemaResult<!, !>> for SemaResult<T> {
    fn from_residual(residual: SemaResult<!, !>) -> Self {
        match residual {
            NotFinished => SemaResult::NotFinished,
        }
    }
}

impl<T, E, E2: From<E>> FromResidual<Result<Infallible, E>> for SemaResult<T, E2> {
    fn from_residual(residual: Result<Infallible, E>) -> Self {
        match residual {
            Result::Err(err) => Err(err.into()),
        }
    }
}

impl<T, E> FromResidual<Option<Infallible>> for SemaResult<Option<T>, E> {
    fn from_residual(residual: Option<Infallible>) -> Self {
        match residual {
            None => Ok(None),
        }
    }
}

impl<T> From<HandledErr> for SemaResult<T> {
    fn from(_: HandledErr) -> Self {
        Err(SemaError::HandledErr)
    }
}
