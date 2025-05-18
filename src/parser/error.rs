use super::lexer::{Token, TokenKind};
use crate::{diagnostics::cerror2, util::IteratorExt};
use core::fmt;

#[derive(Debug, PartialEq, Eq)]
pub struct ParseError;

pub type ParseResult<T> = Result<T, ParseError>;

#[track_caller]
pub fn unexpected_token<T>(t: Token, expected: &[TokenKind]) -> ParseResult<T> {
    match expected {
        [] => cerror2!(t.span, "unexpected token: {}", t.kind),
        [e] => unexpected_token_expect1(t, e),
        many => cerror2!(t.span, "expected one {}, got of {}", many.iter().join(", "), t.kind),
    }
}

#[track_caller]
pub fn unexpected_token_expect1<T>(t: Token, expected: impl fmt::Display) -> ParseResult<T> {
    cerror2!(t.span, "expected {expected}, got {}", t.kind)
}

impl From<crate::diagnostics::HandledErr> for ParseError {
    fn from(_: crate::diagnostics::HandledErr) -> Self {
        ParseError
    }
}
