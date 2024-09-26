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
    /// Like [`ParseErrorKind::UnexpectedToken`] but this might not be a real
    /// error is some cases.
    UnexpectedStart(TokenKind),
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

pub fn err<T>(kind: ParseErrorKind, span: Span) -> ParseResult<T> {
    Err(ParseError::new(kind, span))
}

pub fn err_val(kind: ParseErrorKind, span: Span) -> ParseError {
    ParseError::new(kind, span)
}

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

    pub fn unexpected_token<T>(t: Token) -> Result<T, ParseError> {
        Err(ParseError::new(ParseErrorKind::UnexpectedToken(t.kind), t.span))
    }

    pub fn unexpected_start_token<T>(t: Token) -> Result<T, ParseError> {
        Err(ParseError::new(ParseErrorKind::UnexpectedStart(t.kind), t.span))
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

pub trait ParseResultExt<T> {
    fn opt(self) -> ParseResult<Option<T>>;
}

impl<T> ParseResultExt<T> for ParseResult<T> {
    /// [`ParseErrorKind::UnexpectedStart`] -> [`None`]
    /// [`ParseErrorKind::NoInput`] -> [`None`]
    fn opt(self) -> ParseResult<Option<T>> {
        match self {
            Ok(t) => Ok(Some(t)),
            Err(ParseError {
                kind: ParseErrorKind::UnexpectedStart(_) | ParseErrorKind::NoInput,
                ..
            }) => Ok(None),
            Err(e) => Err(e),
        }
    }
}

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
