use crate::parser::LitKind;
use core::fmt;
use std::ops::{Index, Range};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CommentType {
    /// `// ...` and `/* ... */`
    Comment,
    /// `//!` and `/*! ... */`
    DocInner,
    /// `///` and `/** ... */`
    DocOuter,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenKind {
    /// see [`is_whitespace`]
    Whitespace,

    /// `// ...`
    LineComment(CommentType),
    /// `/* ... */`
    BlockComment(CommentType),

    /// `let`, `my_var`, `MyStruct`, `_`
    Ident,

    /// Note: Bool literals are Idents
    Literal(LitKind),

    /// `(`
    OpenParenthesis,
    /// `)`
    CloseParenthesis,
    /// `[`
    OpenBracket,
    /// `]`
    CloseBracket,
    /// `{`
    OpenBrace,
    /// `}`
    CloseBrace,

    /// `=`
    Equal,
    /// `==`
    EqualEqual,
    /// `=>`
    FatArrow,
    /// `!`
    Bang,
    /// `!=`
    BangEqual,
    /// `<`
    Lt,
    /// `<=`
    LtEqual,
    /// `<<`
    LtLt,
    /// `<<=`
    LtLtEqual,
    /// `>`
    Gt,
    /// `>=`
    GtEqual,
    /// `>>`
    GtGt,
    /// `>>=`
    GtGtEqual,

    /// `+`
    Plus,
    /// `+=`
    PlusEqual,
    /// `-`
    Minus,
    /// `-=`
    MinusEqual,
    /// `->`
    Arrow,
    /// `*`
    Asterisk,
    /// `*=`
    AsteriskEqual,
    /// `/`
    Slash,
    /// `/=`
    SlashEqual,
    /// `%`
    Percent,
    /// `%=`
    PercentEqual,

    /// `&`
    Ampersand,
    /// `&&`
    AmpersandAmpersand,
    /// `&=`
    AmpersandEqual,
    /// `|`
    Pipe,
    /// `||`
    PipePipe,
    /// `|=`
    PipeEqual,
    /// `^`
    Caret,
    /// `^=`
    CaretEqual,

    /// `.`
    Dot,
    /// `..`
    DotDot,
    /// `.*`
    DotAsterisk,
    /// `.&`
    DotAmpersand,
    /// `,`
    Comma,
    /// `:`
    Colon,
    /// `;`
    Semicolon,
    /// `?`
    Question,
    /// `#`
    Pound,
    /// `$`
    Dollar,
    /// `@`
    At,
    /// `~`
    Tilde,
    /// `\`
    /// maybe: <https://ziglang.org/documentation/0.11.0/#Multiline-String-Literals>
    BackSlash,
    /// `
    BackTick,

    Unknown,
}

impl TokenKind {
    pub fn is_ignored(&self) -> bool {
        match self {
            TokenKind::Whitespace | TokenKind::LineComment(_) | TokenKind::BlockComment(_) => true,
            _ => false,
        }
    }
}

/// Stores the type of the token and it's position in the original code `&str`
/// but not the token content.
#[derive(Debug, Clone, Copy)]
pub struct Token {
    pub kind: TokenKind,
    pub span: Span,
}

impl Token {
    pub fn new(kind: TokenKind, span: Range<usize>) -> Self {
        Self { kind, span: Span::from(span) }
    }
}

/// byte range offset for a [`Code`].
#[derive(Clone, Copy)]
pub struct Span {
    pub start: usize,
    pub end: usize,
}

impl fmt::Debug for Span {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Span{}..{}", &self.start, &self.end)
    }
}

impl Span {
    pub fn new(start: usize, end: usize) -> Span {
        Span { start, end }
    }

    pub fn pos(pos: usize) -> Span {
        Self::new(pos, pos + 1)
    }

    pub fn len(&self) -> usize {
        (self.start..self.end).len()
    }

    pub fn bytes(self) -> Range<usize> {
        self.into()
    }

    /// ```text
    /// join(1..3, 3..10) = 1..10
    /// join(1..3, 9..10) = 1..10
    /// join(1..9, 3..10) = 1..10
    /// join(1..10, 3..9) = 1..10
    /// ```
    /// this function is commutative
    pub fn join(self, other: Span) -> Span {
        Span::new(self.start.min(other.start), self.end.max(other.end))
    }

    pub fn multi_join(spans: impl IntoIterator<Item = Span>) -> Option<Span> {
        let mut iter = spans.into_iter();
        let first = iter.next();
        let last = iter.last();
        first.map(|first| last.map(|last| first.join(last)).unwrap_or(first))
    }
}

impl From<Range<usize>> for Span {
    fn from(bytes: Range<usize>) -> Self {
        let Range { start, end } = bytes;
        Span::new(start, end)
    }
}

impl From<Span> for Range<usize> {
    fn from(span: Span) -> Self {
        span.start..span.end
    }
}

impl fmt::Display for Span {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}..{}", self.start, self.end)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Code<'a>(pub &'a str);

impl<'a> Index<Span> for Code<'a> {
    type Output = str;

    fn index(&self, span: Span) -> &Self::Output {
        &self.0[span.bytes()]
    }
}

pub fn is_whitespace(c: char) -> bool {
    matches!(
        c,
        // Usual ASCII suspects
        '\u{0009}'   // \t
        | '\u{000A}' // \n
        | '\u{000B}' // vertical tab
        | '\u{000C}' // form feed
        | '\u{000D}' // \r
        | '\u{0020}' // space

        // NEXT LINE from latin1
        | '\u{0085}'

        // Bidi markers
        | '\u{200E}' // LEFT-TO-RIGHT MARK
        | '\u{200F}' // RIGHT-TO-LEFT MARK

        // Dedicated whitespace characters from Unicode
        | '\u{2028}' // LINE SEPARATOR
        | '\u{2029}' // PARAGRAPH SEPARATOR
    )
}

pub fn is_id_start(c: char) -> bool {
    c == '_' || unicode_xid::UnicodeXID::is_xid_start(c)
}

pub fn is_id_continue(c: char) -> bool {
    unicode_xid::UnicodeXID::is_xid_continue(c)
}

#[derive(Debug, Clone, Copy)]
pub struct Lexer<'c> {
    pub code: Cursor<'c>,
}

impl<'c> Lexer<'c> {
    pub fn new(code: &'c str) -> Lexer<'c> {
        let mut lex = Self { code: Cursor::new(code) };
        if lex.peek().is_some_and(|t| t.kind == TokenKind::Whitespace) {
            lex.advance();
        }
        lex
    }

    pub fn get_code(&self) -> Code<'c> {
        self.code.code
    }

    pub fn get_pos(&self) -> usize {
        self.code.pos
    }

    pub fn next(&mut self) -> Option<Token> {
        /// peeks at the next char, matches the patterns and advances `self` if
        /// needed. the `default` value is returns if no pattern matches
        /// the peeked char
        macro_rules! maybe_followed_by {
            (
                default: $default:expr,
                $( $peek_char_option:pat => $option_res:expr ),* $(,)?
            ) => {
                match self.code.peek() {
                    $(Some($peek_char_option) => {
                        self.code.advance();
                        $option_res
                    },)*
                    _ => $default,
                }
            };
        }

        let start = self.code.pos;
        let kind = match self.code.next()? {
            w if is_whitespace(w) => {
                self.code.advance_while(is_whitespace);
                TokenKind::Whitespace
            },
            '"' => self.string_literal(),
            '\'' => self.char_literal(),
            '0'..='9' => self.num_literal(),

            '(' => TokenKind::OpenParenthesis,
            ')' => TokenKind::CloseParenthesis,
            '[' => TokenKind::OpenBracket,
            ']' => TokenKind::CloseBracket,
            '{' => TokenKind::OpenBrace,
            '}' => TokenKind::CloseBrace,

            '=' => maybe_followed_by! {
                default: TokenKind::Equal,
                '=' => TokenKind::EqualEqual,
                '>' => TokenKind::FatArrow,
            },
            '!' => maybe_followed_by! {
                default: TokenKind::Bang,
                '=' => TokenKind::BangEqual,
            },
            '<' => maybe_followed_by! {
                default: TokenKind::Lt,
                '=' => TokenKind::LtEqual,
                '<' => maybe_followed_by! {
                    default: TokenKind::LtLt,
                    '=' => TokenKind::LtLtEqual,
                },
            },
            '>' => maybe_followed_by! {
                default: TokenKind::Gt,
                '=' => TokenKind::GtEqual,
                '>' => maybe_followed_by! {
                    default: TokenKind::GtGt,
                    '=' => TokenKind::GtGtEqual,
                },
            },

            '+' => maybe_followed_by! {
                default: TokenKind::Plus,
                '=' => TokenKind::PlusEqual,
            },
            '-' => maybe_followed_by! {
                default: TokenKind::Minus,
                '=' => TokenKind::MinusEqual,
                '>' => TokenKind::Arrow,
            },
            '*' => maybe_followed_by! {
                default: TokenKind::Asterisk,
                '=' => TokenKind::AsteriskEqual,
            },
            '/' => maybe_followed_by! {
                default: TokenKind::Slash,
                '/' => self.line_comment(),
                '*' => self.block_comment(),
                '=' => TokenKind::SlashEqual,
            },
            '%' => maybe_followed_by! {
                default: TokenKind::Percent,
                '=' => TokenKind::PercentEqual,
            },

            '&' => maybe_followed_by! {
                default: TokenKind::Ampersand,
                '&' => TokenKind::AmpersandAmpersand,
                '=' => TokenKind::AmpersandEqual,
            },
            '|' => maybe_followed_by! {
                default: TokenKind::Pipe,
                '|' => TokenKind::PipePipe,
                '=' => TokenKind::PipeEqual,
            },
            '^' => maybe_followed_by! {
                default: TokenKind::Caret,
                '=' => TokenKind::CaretEqual,
            },

            '.' => maybe_followed_by! {
                default: TokenKind::Dot,
                '.' => TokenKind::DotDot,
                '*' => TokenKind::DotAsterisk,
                '&' => TokenKind::DotAmpersand,
            },
            ',' => TokenKind::Comma,
            ':' => TokenKind::Colon,
            ';' => TokenKind::Semicolon,
            '?' => TokenKind::Question,
            '#' => TokenKind::Pound,
            '$' => TokenKind::Dollar,
            '@' => TokenKind::At,
            '~' => TokenKind::Tilde,
            '\\' => todo!("BackSlash"),

            'b' => maybe_followed_by! {
                default: self.ident(),
                '\'' => self.bchar_literal()
            },
            c if is_id_start(c) => self.ident(),

            _ => TokenKind::Unknown,
        };
        Some(Token::new(kind, start..self.code.pos))
    }

    pub fn peek(&self) -> Option<Token> {
        self.clone().next()
    }

    pub fn advance(&mut self) {
        self.next();
    }

    pub fn advanced(mut self) -> Self {
        self.advance();
        self
    }

    fn string_literal(&mut self) -> TokenKind {
        while !matches!(self.code.next(), Some('"' | '\n')) {}
        // TODO: check for invalid literal
        TokenKind::Literal(LitKind::Str)
    }

    fn bchar_literal(&mut self) -> TokenKind {
        while !matches!(self.code.next(), Some('\'' | '\n')) {}
        // TODO: check for invalid literal
        TokenKind::Literal(LitKind::BChar)
    }

    fn char_literal(&mut self) -> TokenKind {
        while !matches!(self.code.next(), Some('\'' | '\n')) {}
        // TODO: check for invalid literal
        TokenKind::Literal(LitKind::Char)
    }

    fn num_literal(&mut self) -> TokenKind {
        self.code.advance_while(|c| matches!(c, '0'..='9' | '_'));
        match self.code.peek().zip(self.code.peek2()) {
            // an integer might be followed by a range operator `..` or a method `1.foo()`
            Some(('.', c)) if c != '.' && !is_id_start(c) => {
                self.code.advance();
                self.code.advance_while(|c| matches!(c, '0'..='9' | '_'));
                TokenKind::Literal(LitKind::Float)
            },
            _ => TokenKind::Literal(LitKind::Int),
        }
    }

    fn line_comment(&mut self) -> TokenKind {
        let comment_type = match self.code.peek() {
            Some('!') => CommentType::DocInner,
            Some('/') => CommentType::DocOuter,
            _ => CommentType::Comment,
        };
        while !matches!(self.code.next(), None | Some('\n')) {}
        TokenKind::LineComment(comment_type)
    }

    fn block_comment(&mut self) -> TokenKind {
        let comment_type = match self.code.peek() {
            Some('!') => CommentType::DocInner,
            // `/**/` => CommentType::Comment
            Some('*') if self.code.peek2() != Some('/') => CommentType::DocOuter,
            _ => CommentType::Comment,
        };

        let mut depth: usize = 1;
        while let Some(c) = self.code.next() {
            match c {
                '/' if self.code.advance_if(|c| c == '*') => depth += 1,
                '*' if self.code.advance_if(|c| c == '/') => {
                    depth -= 1;
                    if depth == 0 {
                        break;
                    }
                },
                _ => (),
            }
        }

        TokenKind::BlockComment(comment_type)
    }

    fn ident(&mut self) -> TokenKind {
        self.code.advance_while(is_id_continue);
        TokenKind::Ident
    }
}

impl<'c> Iterator for Lexer<'c> {
    type Item = Token;

    fn next(&mut self) -> Option<Self::Item> {
        self.next()
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Cursor<'c> {
    code: Code<'c>,
    pos: usize,
}

impl<'c> Cursor<'c> {
    pub fn new(code: &'c str) -> Cursor<'c> {
        Cursor { code: Code(code), pos: 0 }
    }

    pub fn next(&mut self) -> Option<char> {
        let c = self.peek();
        self.advance();
        c
    }

    pub fn get_rem(self) -> &'c str {
        &self.code.0[self.pos..]
    }

    pub fn peek(self) -> Option<char> {
        self.get_rem().chars().next()
    }

    fn peek2(&self) -> Option<char> {
        let mut c = self.get_rem().chars();
        c.next();
        c.next()
    }

    pub fn advance(&mut self) {
        let offset = self.get_rem().char_indices().next().map(|(idx, _)| idx).unwrap_or_default();
        self.pos += offset + 1;
    }

    /// Advances the inner [`Chars`] [`Iterator`] while a condition is true.
    fn advance_while(&mut self, mut f: impl FnMut(char) -> bool) {
        while self.peek().is_some_and(&mut f) {
            self.advance();
        }
    }

    fn advance_if(&mut self, mut f: impl FnMut(char) -> bool) -> bool {
        let do_advance = self.peek().is_some_and(&mut f);
        if do_advance {
            self.advance();
        }
        do_advance
    }
}
