use super::lexer::TokenKind;
use crate::parser::{lexer::Span, result_with_fatal::ResultWithFatal};
use core::fmt;

#[derive(Debug, PartialEq, Eq)]
pub enum ParseErrorKind {
    Unimplemented,
    NoInput,
    UnexpectedToken { expected: TokenKind, got: TokenKind },
    UnexpectedToken2(TokenKind),
    MissingToken(TokenKind),
    NotAKeyword,
    NotAnIdent,
    AllocErr(bumpalo::AllocErr),

    Finished,
}

pub struct ParseError {
    pub kind: ParseErrorKind,
    pub span: Span,

    #[cfg(debug_assertions)]
    pub context: anyhow::Error,
}

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
            context: anyhow::Error::msg("root"),
        }
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

pub type ParseResult<T> = ResultWithFatal<T, ParseError>;

impl<T> ParseResult<T> {
    pub fn context(self, context: impl std::fmt::Display + Send + Sync + 'static) -> Self {
        match self {
            ResultWithFatal::Err(err) => ResultWithFatal::Err(err.add_context(context)),
            ResultWithFatal::Fatal(err) => ResultWithFatal::Fatal(err.add_context(context)),
            ok => ok,
        }
    }

    pub fn with_context<C>(self, f: impl FnOnce() -> C) -> Self
    where C: std::fmt::Display + Send + Sync + 'static {
        match self {
            ResultWithFatal::Err(err) => ResultWithFatal::Err(err.add_context(f())),
            ResultWithFatal::Fatal(err) => ResultWithFatal::Fatal(err.add_context(f())),
            ok => ok,
        }
    }
}
