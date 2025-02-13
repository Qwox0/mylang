use crate::{
    ast::{self, Ast, UnaryOpKind},
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
    /// TODO: maybe infer the type from the usage
    VarDeclNoType,

    #[error("MismatchedTypes (expected: {expected}, got: {got})")]
    MismatchedTypes {
        expected: Ptr<ast::Type>,
        got: Ptr<ast::Type>,
    },
    #[error("MismatchedTypesBinOp (lhs: {lhs_ty}, rhs: {rhs_ty})")]
    MismatchedTypesBinOp {
        lhs_ty: Ptr<ast::Type>,
        rhs_ty: Ptr<ast::Type>,
    },
    #[error("ExpectedNumber (got: {got})")]
    ExpectedNumber {
        got: Ptr<ast::Type>,
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
        ty: Ptr<ast::Type>,
        op: UnaryOpKind,
    },
    DuplicateEnumVariant,
    DuplicateField,
    #[error("CannotApplyInitializer to {ty}")]
    CannotApplyInitializer {
        ty: Ptr<ast::Type>,
    },
    CannotInferInitializerTy,
    CannotInferAutocastTy,
    MultiplePossibleInitializerTy,
    DuplicateInInitializer,
    MissingFieldInInitializer {
        field: Ptr<str>,
    },
    CallOfANonFunction,
    ReturnNotInAFunction,
    NotAType,

    MissingArg,
    MissingElseBranch,
    #[error("IncompatibleBranches (expected: {expected}, got: {got})")]
    IncompatibleBranches {
        expected: Ptr<ast::Type>,
        got: Ptr<ast::Type>,
    },

    #[error("unknown ident `{}`", &**_0)]
    UnknownIdent(Ptr<str>),
    /// unknown struct or union field or enum variant
    #[error("no field `{}` on type `{ty}`", &**field)]
    UnknownField {
        ty: Ptr<ast::Type>,
        field: Ptr<str>,
    },
    CannotInfer,
    UnionFieldWithDefaultValue,

    TopLevelDuplicate,
    UnexpectedTopLevelExpr(Ptr<Ast>),

    NotAConstExpr,
    AssignToConst,
    AssignToNotMut,

    NegativeArrayLen,
    MismatchedArrayLen {
        expected: usize,
        got: usize,
    },
    CanOnlyIndexArrays,
    #[error("cannot index into array with {}", ty)]
    InvalidArrayIndex {
        ty: Ptr<ast::Type>,
    },
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
