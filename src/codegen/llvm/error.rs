use CodegenResult::*;
use inkwell::{builder::BuilderError, execution_engine::FunctionLookupError, support::LLVMString};
use std::{
    convert::Infallible,
    ops::{ControlFlow, FromResidual, Try},
};

#[derive(Debug, thiserror::Error)]
#[error("{:?}", self)]
pub enum CodegenError {
    BuilderError(BuilderError),
    InvalidGeneratedFunction,
    FunctionLookupError(FunctionLookupError),
    CannotOptimizeModule(LLVMString),
    CannotCompileObjFile(LLVMString),
    CannotCreateJit(LLVMString),
}

unsafe impl Send for CodegenError {}
unsafe impl Sync for CodegenError {}

#[derive(Debug)]
#[must_use]
pub enum CodegenResult<T, U = !> {
    Ok(T),
    /// Represents the [`unreachable`](https://llvm.org/docs/LangRef.html#i-unreachable) Terminator
    /// Instruction.
    Unreachable(U),
    #[cfg(not(debug_assertions))]
    Err(CodegenError),
    #[cfg(debug_assertions)]
    Err(anyhow::Error),
}

pub type CodegenResultAndControlFlow<T> = CodegenResult<T, ()>;

impl<T, U> CodegenResult<T, U> {
    pub fn map<T2>(self, mapper: impl FnOnce(T) -> T2) -> CodegenResult<T2, U> {
        match self {
            Ok(t) => Ok(mapper(t)),
            Unreachable(u) => Unreachable(u),
            Err(error) => Err(error),
        }
    }
}

impl<T> CodegenResultAndControlFlow<T> {
    pub fn do_continue(&self) -> bool {
        matches!(self, Ok(_))
    }

    pub fn handle_unreachable(self) -> CodegenResult<Option<T>> {
        match self {
            Ok(t) => Ok(Some(t)),
            Unreachable(_) => Ok(None),
            Err(error) => Err(error),
        }
    }

    pub fn as_do_continue(self) -> CodegenResult<bool> {
        self.handle_unreachable().map(|t| t.is_some())
    }
}

impl<T> CodegenResult<T, !> {
    pub fn unwrap(self) -> T {
        match self {
            Ok(t) => t,
            Err(error) => panic!("Called `unwrap` on CodegenResult::Err: {error:?}"),
        }
    }

    #[inline]
    pub fn coerce(self) -> CodegenResultAndControlFlow<T> {
        match self {
            Ok(t) => Ok(t),
            Err(error) => Err(error),
        }
    }
}

impl<T, U> Try for CodegenResult<T, U> {
    type Output = T;
    type Residual = CodegenResult<!, U>;

    fn from_output(output: Self::Output) -> Self {
        Ok(output)
    }

    fn branch(self) -> ControlFlow<Self::Residual, Self::Output> {
        match self {
            Ok(t) => ControlFlow::Continue(t),
            Unreachable(u) => ControlFlow::Break(Unreachable(u)),
            Err(error) => ControlFlow::Break(Err(error)),
        }
    }
}

impl<T, U> FromResidual<CodegenResult<!, U>> for CodegenResult<T, U> {
    fn from_residual(residual: CodegenResult<!, U>) -> Self {
        match residual {
            Unreachable(u) => Unreachable(u),
            Err(error) => Err(error),
        }
    }
}

impl<T> FromResidual<CodegenResult<!>> for CodegenResultAndControlFlow<T> {
    fn from_residual(residual: CodegenResult<!>) -> Self {
        match residual {
            Err(err) => CodegenResult::Err(err.into()),
        }
    }
}

impl<T, U, E> FromResidual<Result<Infallible, E>> for CodegenResult<T, U>
where E: Into<CodegenError>
{
    fn from_residual(residual: Result<Infallible, E>) -> Self {
        match residual {
            Result::Err(error) => Err(error.into().into()),
        }
    }
}

impl From<BuilderError> for CodegenError {
    fn from(e: BuilderError) -> Self {
        CodegenError::BuilderError(e)
    }
}

impl From<FunctionLookupError> for CodegenError {
    fn from(e: FunctionLookupError) -> Self {
        CodegenError::FunctionLookupError(e)
    }
}
