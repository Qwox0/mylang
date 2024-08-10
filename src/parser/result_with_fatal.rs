use super::util::OptionExt;
use core::fmt;
use std::ops::{ControlFlow, FromResidual, Try};
use ResultWithFatal::*;

#[derive(Debug)]
pub enum ResultWithFatal<T, E> {
    Ok(T),
    /// error. Following `alternative` parsers can still match and recover from
    /// this error.
    Err(E),
    /// fatal error. This asserts that no following parser can recover from the
    /// error.
    Fatal(E),
}

impl<T, E> FromResidual<ResultWithFatal<!, E>> for ResultWithFatal<T, E> {
    fn from_residual(residual: ResultWithFatal<!, E>) -> Self {
        match residual {
            Ok(never) => never,
            Err(err) => Err(err),
            Fatal(err) => Fatal(err),
        }
    }
}

impl<T, E> FromResidual<ResultWithFatal<!, E>> for Result<T, E> {
    fn from_residual(residual: ResultWithFatal<!, E>) -> Self {
        match residual {
            Ok(never) => never,
            Err(err) | Fatal(err) => Result::Err(err),
        }
    }
}

impl<T, E> Try for ResultWithFatal<T, E> {
    type Output = T;
    type Residual = ResultWithFatal<!, E>;

    fn from_output(output: Self::Output) -> Self {
        Ok(output)
    }

    fn branch(self) -> ControlFlow<Self::Residual, Self::Output> {
        match self {
            Ok(ok) => ControlFlow::Continue(ok),
            Err(err) => ControlFlow::Break(Err(err)),
            Fatal(err) => ControlFlow::Break(Fatal(err)),
        }
    }
}

impl<T, E> ResultWithFatal<T, E> {
    /// ```text
    /// | res           | output        |
    /// |:-------------:|:-------------:|
    /// | Ok            | Ok            |
    /// | Err           | Err           |
    /// ```
    pub fn from_res(res: Result<T, E>) -> Self {
        match res {
            Result::Ok(ok) => Ok(ok),
            Result::Err(err) => Err(err),
        }
    }

    /// ```text
    /// | res           | output        |
    /// |:-------------:|:-------------:|
    /// | Ok            | Ok            |
    /// | Err           | Fatal         |
    /// ```
    pub fn from_res_fatal(res: Result<T, E>) -> Self {
        match res {
            Result::Ok(ok) => Ok(ok),
            Result::Err(err) => Fatal(err),
        }
    }

    pub fn is_ok(&self) -> bool {
        matches!(self, Ok(_))
    }

    pub fn is_any_err(&self) -> bool {
        matches!(self, Err(_) | Fatal(_))
    }

    pub fn is_nonfatal_err(&self) -> bool {
        matches!(self, Err(_))
    }

    pub fn is_fatal_err(&self) -> bool {
        matches!(self, Fatal(_))
    }

    pub fn into_fatal(self) -> Self {
        match self {
            Err(err) => Fatal(err),
            res => res,
        }
    }

    /// ```text
    /// | self          | output        |
    /// |:-------------:|:-------------:|
    /// | Ok t          | Ok f(t)       |
    /// | Err           | Err           |
    /// | Fatal         | Fatal         |
    /// ```
    pub fn map<U>(self, f: impl FnOnce(T) -> U) -> ResultWithFatal<U, E> {
        //Ok(f(self?))
        match self {
            Ok(t) => Ok(f(t)),
            Err(e) => Err(e),
            Fatal(e) => Fatal(e),
        }
    }

    pub fn map_nonfatal_err(self, f: impl FnOnce(E) -> E) -> ResultWithFatal<T, E> {
        match self {
            Ok(t) => Ok(t),
            Err(e) => Err(f(e)),
            Fatal(e) => Fatal(e),
        }
    }

    /*
    pub fn map_res<U>(
        self,
        f: impl FnOnce(Result<T, E>) -> ResultWithFatal<U, E>,
    ) -> ResultWithFatal<U, E> {
        match self {
            Fatal(err) => Fatal(err),
            res => f(Result::from(res)),
        }
    }
    */

    /// ```text
    /// | self          | output        |
    /// |:-------------:|:-------------:|
    /// | Ok t          | f(t)          |
    /// | Err           | Err           |
    /// | Fatal         | Fatal         |
    /// ```
    pub fn and_then<U>(self, f: impl FnOnce(T) -> ResultWithFatal<U, E>) -> ResultWithFatal<U, E> {
        //f(self?)
        match self {
            Ok(t) => f(t),
            Err(e) => Err(e),
            Fatal(e) => Fatal(e),
        }
    }

    /// ```text
    /// | self          | output        |
    /// |:-------------:|:-------------:|
    /// | Ok            | Ok            |
    /// | Err e         | f(e)          |
    /// | Fatal         | Fatal         |
    /// ```
    pub fn or_else(self, f: impl FnOnce(E) -> ResultWithFatal<T, E>) -> ResultWithFatal<T, E> {
        match self {
            Err(err) => f(err),
            res => res,
        }
    }

    pub fn inspect(self, f: impl FnOnce(&T)) -> Self {
        if let Ok(t) = &self {
            f(t)
        }
        self
    }

    pub fn inspect_err(self, f: impl FnOnce(&E)) -> Self {
        if let Err(e) = &self {
            f(e)
        }
        self
    }

    pub fn inspect_fatal(self, f: impl FnOnce(&E)) -> Self {
        if let Fatal(e) = &self {
            f(e)
        }
        self
    }

    pub fn unwrap(self) -> T
    where E: fmt::Debug {
        Result::from(self).unwrap()
    }

    pub fn unwrap_or(self, default: T) -> T {
        Result::from(self).unwrap_or(default)
    }

    pub fn unwrap_or_else(self, f: impl FnOnce(E) -> T) -> T {
        Result::from(self).unwrap_or_else(f)
    }
}

impl<T, E> From<ResultWithFatal<T, E>> for Result<T, E> {
    fn from(value: ResultWithFatal<T, E>) -> Self {
        match value {
            Ok(ok) => Result::Ok(ok),
            Err(err) | Fatal(err) => Result::Err(err),
        }
    }
}
