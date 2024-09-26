use crate::{
    ast::{Expr, PreOpKind},
    parser::lexer::Span,
    ptr::Ptr,
    type_::Type,
};
use std::ops::{FromResidual, Try};
use SemaResult::*;

#[derive(Debug, Clone)]
pub enum SemaErrorKind {
    ConstDeclWithoutInit,
    /// TODO: maybe infer the type from the usage
    VarDeclNoType,

    MismatchedTypes {
        expected: Type,
        got: Type,
    },
    MismatchedTypesBinOp {
        lhs_ty: Type,
        rhs_ty: Type,
    },
    /// rust error:
    /// ```notest
    /// error[E0600]: cannot apply unary operator `!` to type `&'static str`
    ///   --> src/sema/mod.rs:70:17
    ///    |
    /// 70 |         let a = !"";
    ///    |                 ^^^ cannot apply unary operator `!`
    /// ```
    InvalidPreOp {
        ty: Type,
        kind: PreOpKind,
    },
    CannotApplyInitializer {
        ty: Type,
    },
    DuplicateInInitializer,
    MissingFieldInInitializer {
        field: Ptr<str>,
    },
    CallOfANotFunction,
    ReturnNotInAFunction,
    NotAType,

    MissingArg,
    MissingElseBranch,
    IncompatibleBranches {
        expected: Type,
        got: Type,
    },

    UnknownIdent(Ptr<str>),
    UnknownField {
        ty: Type,
        field: Ptr<str>,
    },
    CannotInfer,

    TopLevelDuplicate,
    UnexpectedTopLevelExpr(Ptr<Expr>),

    NotAConstExpr,

    AllocErr(bumpalo::AllocErr),
}

#[derive(Debug, Clone)]
pub struct SemaError {
    pub kind: SemaErrorKind,
    pub span: Span,
}

#[derive(Debug)]
#[must_use]
pub enum SemaResult<T> {
    Ok(T),
    //NotFinished(Option<Type>),
    NotFinished,
    Err(SemaError),
}

impl<T> SemaResult<T> {
    pub fn map_ok<U>(self, f: impl FnOnce(T) -> U) -> SemaResult<U> {
        match self {
            Ok(t) => Ok(f(t)),
            NotFinished => NotFinished,
            Err(err) => Err(err),
        }
    }

    pub fn inspect_err(self, f: impl FnOnce(&SemaError)) -> Self {
        if let Err(e) = &self {
            f(e);
        }
        self
    }
}

impl<T> Try for SemaResult<T> {
    type Output = T;
    type Residual = SemaResult<!>;

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

impl<T> FromResidual<SemaResult<!>> for SemaResult<T> {
    fn from_residual(residual: SemaResult<!>) -> Self {
        match residual {
            Ok(never) => never,
            NotFinished => SemaResult::NotFinished,
            Err(err) => SemaResult::Err(err),
        }
    }
}

impl<T> FromResidual<Result<!, SemaError>> for SemaResult<T> {
    fn from_residual(residual: Result<!, SemaError>) -> Self {
        match residual {
            Result::Ok(never) => never,
            Result::Err(err) => Err(err),
        }
    }
}