use super::lexer::{Token, TokenKind};
use crate::{
    diagnostics::{cerror, cerror2, HandledErr},
    parser::lexer::Span,
    util::{IteratorExt, UnwrapDebug},
};
use core::fmt;

pub type ParseError = HandledErr;

pub type ParseResult<T> = Result<T, ParseError>;

#[track_caller]
pub fn unexpected_token<T>(t: Token, expected: &[TokenKind]) -> ParseResult<T> {
    match format_expected_tokens(expected) {
        Some(expected) => cerror2!(t.span, "expected {expected}, got {}", t.kind),
        None => cerror2!(t.span, "unexpected token: {}", t.kind),
    }
}

#[track_caller]
pub fn unexpected_token_expect1<T>(t: Token, expected: impl fmt::Display) -> ParseResult<T> {
    cerror2!(t.span, "expected {expected}, got {}", t.kind)
}

pub fn expected_token(after_expr_span: Span, expected: &[TokenKind]) -> ParseError {
    cerror!(after_expr_span, "expected {}", format_expected_tokens(expected).u())
}

fn format_expected_tokens(expected: &[TokenKind]) -> Option<String> {
    Some(match expected {
        [] => return None,
        [e] => format!("{e}"),
        [a, b] => format!("{a} or {b}"),
        _ => {
            let (&last, many) = expected.split_last().u();
            many.iter().join(", ") + ", or " + &last.to_string()
        },
    })
}
