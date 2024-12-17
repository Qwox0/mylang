use crate::{
    ast::{Expr, UnaryOpKind},
    parser::lexer::Span,
    ptr::Ptr,
    type_::Type,
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
    /// TODO: maybe infer the type from the usage
    VarDeclNoType,

    #[error("MismatchedTypes (expected: {expected}, got: {got})")]
    MismatchedTypes {
        expected: Type,
        got: Type,
    },
    #[error("MismatchedTypesBinOp (lhs: {lhs_ty}, rhs: {rhs_ty})")]
    MismatchedTypesBinOp {
        lhs_ty: Type,
        rhs_ty: Type,
    },
    #[error("ExpectedNumber (got: {got})")]
    ExpectedNumber {
        got: Type,
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
        kind: UnaryOpKind,
    },
    DuplicateEnumVariant,
    DuplicateField,
    #[error("CannotApplyInitializer to {ty}")]
    CannotApplyInitializer {
        ty: Type,
    },
    CannotInferPositionalInitializerTy,
    CannotInferNamedInitializerTy,
    MultiplePossibleInitializerTy,
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
    /// unknown struct or union field or enum variant
    UnknownField {
        ty: Type,
        field: Ptr<str>,
    },
    CannotInfer,
    UnionFieldWithDefaultValue,

    TopLevelDuplicate,
    UnexpectedTopLevelExpr(Ptr<Expr>),

    NotAConstExpr,
    AssignToConst,
    AssignToNotMut,

    NegativeArrayLen,
    CanOnlyIndexArrays,
    CannotReturnFromLoop,

    AllocErr(bumpalo::AllocErr),
}

#[derive(Debug, Clone)]
pub struct SemaError {
    pub kind: SemaErrorKind,
    pub span: Span,
}

#[derive(Debug, PartialEq, Eq)]
#[must_use]
pub enum SemaResult<T, E = SemaError> {
    Ok(T),
    //NotFinished(Option<Type>),
    NotFinished,
    Err(E),
}

impl<T, E> SemaResult<T, E> {
    pub fn map_ok<U>(self, f: impl FnOnce(T) -> U) -> SemaResult<U, E> {
        match self {
            Ok(t) => Ok(f(t)),
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
        match self {
            Ok(_) => true,
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

impl<T> FromResidual<Result<Infallible, SemaError>> for SemaResult<T> {
    fn from_residual(residual: Result<Infallible, SemaError>) -> Self {
        match residual {
            Result::Err(err) => Err(err),
        }
    }
}
