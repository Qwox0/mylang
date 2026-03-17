use super::lexer::{Token, TokenKind};
use crate::{
    ast,
    diagnostics::{HandledErr, cerror2},
    parser::lexer::Span,
    ptr::Ptr,
    util::{IteratorExt, UnwrapDebug},
};
use core::fmt;

#[cfg(not(debug_assertions))]
pub type ParseError = HandledErr;
#[cfg(debug_assertions)]
pub type ParseError = anyhow::Error;

pub type ParseResult<T> = Result<T, ParseError>;

pub fn parse_err() -> ParseError {
    HandledErr.into()
}

#[track_caller]
pub fn unexpected_token(t: Token, expected: &[TokenKind]) -> ParseError {
    debug_assert!(!expected.contains(&t.kind));
    if t.kind == TokenKind::HandledErr {
        return parse_err();
    }
    match format_expected_tokens(expected) {
        Some(expected) => cerror2!(t.span, "expected {expected}, got {}", t.kind),
        None => cerror2!(t.span, "unexpected token: {}", t.kind),
    }
}

#[track_caller]
pub fn unexpected_token_expect1(t: Token, expected: impl fmt::Display) -> ParseError {
    cerror2!(t.span, "expected {expected}, got {}", t.kind)
}

#[track_caller]
pub fn unexpected_expr(expr: Ptr<ast::Ast>, expected: impl fmt::Display) -> ParseError {
    cerror2!(expr.full_span(), "expected {expected}, got an expression")
}

#[track_caller]
pub fn expected_token(after_expr_span: Span, expected: &[TokenKind]) -> ParseError {
    cerror2!(after_expr_span, "expected {}", format_expected_tokens(expected).u())
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
