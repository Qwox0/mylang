use super::lexer::{Lexer, Span};
use std::num::NonZeroU32;

pub struct Symbol {
    id: NonZeroU32,
}

pub enum Delimiter {
    /// `( ... )`
    Parenthesis,
    /// `{ ... }`
    Brace,
    /// `[ ... ]`
    Bracket,
    None,
}

pub struct Group {
    span: Span,
    delimiter: Delimiter,
}

pub struct Ident {
    span: Span,
    symbol: Symbol,
}

#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub enum LitKind {
    Byte,
    Char,
    Integer,
    Float,
    Str,
    /// # Example
    ///
    /// ```rust
    /// r#"The " character is valid in a raw string literal"#; // -> StrRaw(1)
    /// ```
    StrRaw(u8),
    Err,
}

#[derive(Debug)]
pub struct Literal {}

pub struct Parser<'c> {
    code: &'c str,
    lexer: Lexer<'c>,
}

impl<'c> Parser<'c> {
    pub fn new(code: &'c str) -> Parser<'c> {
        let lexer = Lexer::new(code);
        Parser { code, lexer }
    }
}
