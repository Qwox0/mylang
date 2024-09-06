use super::{
    lexer::{Token, TokenKind},
    DeclMarkerKind,
};
use crate::parser::lexer::Span;
use core::fmt;

#[derive(Debug, PartialEq, Eq)]
pub enum ParseErrorKind {
    Unimplemented,
    NoInput,
    UnexpectedToken(TokenKind),
    MissingToken(TokenKind),
    NotAKeyword,
    NotAnIdent,
    DuplicateDeclMarker(DeclMarkerKind),
    AllocErr(bumpalo::AllocErr),

    Finished,
}

pub struct ParseError {
    pub kind: ParseErrorKind,
    pub span: Span,

    #[cfg(debug_assertions)]
    pub context: anyhow::Error,
}

macro_rules! err_impl {
    ($kind:ident, $span:expr) => {
        ParseError::new(ParseErrorKind::$kind , $span)
    };
    ($kind:ident ( $( $field:expr ),* $(,)? ), $span:expr) => {
        ParseError::new(ParseErrorKind::$kind ( $($field),* ) , $span)
    };
    ($kind:ident { $( $field:ident $( : $val:expr )? ),* $(,)? } , $span:expr) => {
        ParseError::new(ParseErrorKind::$kind { $($field $(: $val)?),* }, $span)
    };
}
pub(crate) use err_impl;

macro_rules! err {
    (x $($t:tt)*) => {
        err_impl!($($t)*)
    };
    ($($t:tt)*) => {
        Err(err_impl!($($t)*))
    };
}
/*
macro_rules! err {
    (x $kind:ident, $span:expr) => {
        ParseError::new(ParseErrorKind::$kind , $span)
    };
    (x $kind:ident ( $( $field:expr ),* $(,)? ), $span:expr) => {
        ParseError::new(ParseErrorKind::$kind ( $($field),* ) , $span)
    };
    (x $kind:ident { $( $field:ident $( : $val:expr )? ),* $(,)? } , $span:expr) => {
        ParseError::new(ParseErrorKind::$kind { $($field $(: $val)?),* }, $span)
    };
    ($($t:tt)*) => {
        Err(err!(x $($t)*))
    };
}
*/
pub(crate) use err;

impl std::fmt::Debug for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut f = f.debug_struct("ParseError");
        let f = f.field("kind", &self.kind).field("span", &self.span);
        #[cfg(debug_assertions)]
        let f = f.field("context", &format!("{:#}", &self.context));
        f.finish()
    }
}

impl ParseError {
    pub fn new(kind: ParseErrorKind, span: Span) -> Self {
        Self {
            kind,
            span,
            #[cfg(debug_assertions)]
            context: anyhow::Error::msg("ERROR"),
        }
    }

    pub fn unexpected_token(t: Token) -> Result<!, ParseError>{
        Err(ParseError::new(ParseErrorKind::UnexpectedToken(t.kind), t.span))
    }

    #[cfg(debug_assertions)]
    pub fn add_context(self, context: impl fmt::Display + Send + Sync + 'static) -> Self {
        let context = self.context.context(context);
        Self { context, ..self }
    }

    #[cfg(not(debug_assertions))]
    pub fn add_context(self, _context: impl fmt::Display + Send + Sync + 'static) -> Self {
        self
    }
}

pub type ParseResult<T> = Result<T, ParseError>;

pub trait MyContext {
    fn context(self, context: impl std::fmt::Display + Send + Sync + 'static) -> Self;
    fn with_context<C>(self, f: impl FnOnce() -> C) -> Self
    where C: std::fmt::Display + Send + Sync + 'static;
}

impl<T> MyContext for ParseResult<T> {
    fn context(self, context: impl std::fmt::Display + Send + Sync + 'static) -> Self {
        self.map_err(|err| err.add_context(context))
    }

    fn with_context<C>(self, f: impl FnOnce() -> C) -> Self
    where C: std::fmt::Display + Send + Sync + 'static {
        self.map_err(|err| err.add_context(f()))
    }
}
