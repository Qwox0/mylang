use crate::{diagnostics::HandledErr, sema::UnitDependency};
use SemaResult::*;
use std::{
    convert::Infallible,
    ops::{ControlFlow, FromResidual, Residual, Try},
};

pub type SemaError = HandledErr;

#[derive(Debug, PartialEq, Eq)]
#[must_use]
pub enum SemaResult<T> {
    Ok(T),
    NotFinished(UnitDependency),
    Err(SemaError),
}

impl<T> SemaResult<T> {
    #[allow(non_upper_case_globals)]
    pub const HandledErr: SemaResult<T> = Err(HandledErr);
}

impl<T> SemaResult<T> {
    pub fn map<U>(self, f: impl FnOnce(T) -> U) -> SemaResult<U> {
        match self {
            Ok(t) => Ok(f(t)),
            NotFinished(dep) => NotFinished(dep),
            Err(HandledErr) => Err(HandledErr),
        }
    }

    pub fn is_ok(self) -> SemaResult<bool> {
        match self {
            Ok(_) => Ok(true),
            NotFinished(dep) => NotFinished(dep),
            Err(HandledErr) => Ok(false),
        }
    }
}
impl SemaResult<()> {
    pub fn ignore_error(self) -> Self {
        match self {
            Err(HandledErr) => Ok(()),
            res => res,
        }
    }
}

impl<T> Try for SemaResult<T> {
    type Output = T;
    type Residual = SemaResult<!>;

    fn from_output(output: Self::Output) -> Self {
        Ok(output)
    }

    fn branch(self) -> ControlFlow<Self::Residual, Self::Output> {
        match self {
            Ok(ty) => ControlFlow::Continue(ty),
            NotFinished(dep) => ControlFlow::Break(SemaResult::NotFinished(dep)),
            Err(err) => ControlFlow::Break(SemaResult::Err(err)),
        }
    }
}

impl<T> FromResidual<SemaResult<!>> for SemaResult<T> {
    fn from_residual(residual: SemaResult<!>) -> Self {
        match residual {
            NotFinished(dep) => SemaResult::NotFinished(dep),
            Err(err) => SemaResult::Err(err),
        }
    }
}

impl<T> Residual<T> for SemaResult<!> {
    type TryType = SemaResult<T>;
}

impl<T, E: Into<SemaError>> FromResidual<Result<Infallible, E>> for SemaResult<T> {
    fn from_residual(residual: Result<Infallible, E>) -> Self {
        match residual {
            Result::Err(err) => SemaResult::Err(err.into()),
        }
    }
}

impl<T> FromResidual<Option<Infallible>> for SemaResult<Option<T>> {
    fn from_residual(residual: Option<Infallible>) -> Self {
        match residual {
            None => Ok(None),
        }
    }
}

impl<T> From<HandledErr> for SemaResult<T> {
    fn from(_: HandledErr) -> Self {
        Err(HandledErr)
    }
}
