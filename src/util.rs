use crate::{
    error::SpannedError,
    parser::lexer::{Code, Span},
};
use core::fmt;
use std::{hint::unreachable_unchecked, ops::Try};

pub fn display_spanned_error(err: &impl SpannedError, code: &Code) {
    eprintln!("ERROR: {}", err.get_text());
    display_span_in_code(err.span(), code)
}

pub fn display_span_in_code(span: Span, code: &Code) {
    display_span_in_code_with_label(span, code, "")
}

pub fn display_span_in_code_with_label(span: Span, code: &Code, label: impl fmt::Display) {
    let start_offset = if code.0[..span.start].ends_with("\n") {
        0
    } else {
        code.0[..span.start].lines().last().map(str::len).unwrap_or(span.start)
    };
    let end_offset = code.0[span.end..].lines().next().map(str::len).unwrap_or(span.end);
    let line = &code.0[span.start - start_offset..span.end + end_offset];

    let linecount_in_span = code[span].lines().count();
    eprintln!("|");
    eprintln!("| {}", line.lines().intersperse("\\n").collect::<String>());
    let offset = " ".repeat(start_offset);
    eprintln!("| {offset}{} {label}", "^".repeat(span.len() + linecount_in_span - 1));
    eprintln!("|");
}

pub trait UnwrapDebug {
    type Inner;

    /// like [`Option::unwrap`] but UB in release mode.
    #[track_caller]
    fn unwrap_debug(self) -> Self::Inner;
}

impl<T> UnwrapDebug for Option<T> {
    type Inner = T;

    fn unwrap_debug(self) -> Self::Inner {
        if cfg!(debug_assertions) {
            self.unwrap()
        } else {
            unsafe { self.unwrap_unchecked() }
        }
    }
}

impl<T, E: fmt::Debug> UnwrapDebug for Result<T, E> {
    type Inner = T;

    fn unwrap_debug(self) -> Self::Inner {
        if cfg!(debug_assertions) {
            self.unwrap()
        } else {
            unsafe { self.unwrap_unchecked() }
        }
    }
}

/// like [`unreachable`] but UB in release mode.
#[track_caller]
#[inline]
pub fn unreachable_debug() -> ! {
    if cfg!(debug_assertions) {
        unreachable!()
    } else {
        unsafe { unreachable_unchecked() }
    }
}

/// like [`panic`] but UB in release mode.
#[track_caller]
#[inline]
pub fn panic_debug(msg: &str) -> ! {
    if cfg!(debug_assertions) {
        panic!("{}", msg)
    } else {
        unsafe { unreachable_unchecked() }
    }
}

pub fn collect_all_result_errors<T, E>(
    i: impl IntoIterator<Item = Result<T, E>>,
) -> Result<Vec<T>, Vec<E>> {
    let iter = i.into_iter();
    let mut res = Ok(Vec::with_capacity(iter.size_hint().0));
    for x in iter {
        match (x, &mut res) {
            (Ok(t), Ok(ok_list)) => ok_list.push(t),
            (Ok(_), Err(_)) => continue,
            (Err(err), Ok(_)) => res = Err(vec![err]),
            (Err(e), Err(err_list)) => err_list.push(e),
        }
    }
    res
}

#[inline]
pub unsafe fn forget_lifetime<'a, T: ?Sized>(r: &T) -> &'a T {
    unsafe { &*(r as *const T) }
}

pub trait OkOrWithTry<T> {
    fn ok_or2<Result>(self, err: Result::Residual) -> Result
    where Result: Try<Output = T>;

    fn ok_or_else2<Result>(self, err: impl FnOnce() -> Result::Residual) -> Result
    where Result: Try<Output = T>;
}

impl<T> OkOrWithTry<T> for Option<T> {
    #[inline]
    fn ok_or2<Result>(self, err: Result::Residual) -> Result
    where Result: Try<Output = T> {
        match self {
            Some(t) => Result::from_output(t),
            None => Result::from_residual(err),
        }
    }

    #[inline]
    fn ok_or_else2<Result>(self, err: impl FnOnce() -> Result::Residual) -> Result
    where Result: Try<Output = T> {
        match self {
            Some(t) => Result::from_output(t),
            None => Result::from_residual(err()),
        }
    }
}
