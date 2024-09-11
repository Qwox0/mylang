use crate::parser::{
    lexer::{Code, Span},
    ParseError,
};
use std::fmt::Debug;

pub trait SpannedError: Debug {
    fn span(&self) -> Span;
}

impl SpannedError for ParseError {
    fn span(&self) -> Span {
        self.span
    }
}

pub fn display_spanned_error(err: impl SpannedError, code: &Code) {
    eprintln!("ERROR: {:?}", err);
    display_span_in_code(err.span(), code)
}

pub fn display_span_in_code(span: Span, code: &Code) {
    let start_offset = code.0[..span.start].lines().last().map(str::len).unwrap_or(span.start);
    let end_offset = code.0[span.end..].lines().next().map(str::len).unwrap_or(span.end);
    let line = &code.0[span.start - start_offset..span.end + end_offset];

    let linecount_in_span = code[span].lines().count();
    eprintln!(" {}", line.lines().intersperse("\\n").collect::<String>());
    eprintln!(" {}{}", " ".repeat(start_offset), "^".repeat(span.len() + linecount_in_span - 1));
}

pub trait UnwrapDebug {
    type Inner;

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
