use super::ParseErrorKind;
use crate::{
    parser::parser_helper::ParserInterface,
    ptr::{OPtr, Ptr},
    source_file::SourceFile,
    util::UnwrapDebug,
};
use core::{fmt, ops::Range};
use std::{
    assert_matches::debug_assert_matches,
    mem,
    ops::{Deref, Index},
    str::FromStr,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenKind {
    /// see [`is_whitespace`]
    Whitespace,

    /// `// ...`
    LineComment,
    /// `//!`
    LineDocInner,
    /// `///`
    LineDocOuter,

    /// `/* ... */`
    BlockComment,
    /// `/*! ... */`
    BlockDocInner,
    /// `/** ... */`
    BlockDocOuter,

    /// `let`, `my_var`, `MyStruct`, `_`
    Ident,

    Keyword(Keyword),

    IntLit,
    FloatLit,
    BoolLitTrue,
    BoolLitFalse,
    CharLit,
    BCharLit,
    StrLit,
    MultilineStrLitLine,

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
    Eq,
    /// `==`
    EqEq,
    /// `=>`
    FatArrow,
    /// `!`
    Bang,
    /// `!=`
    BangEq,
    /// `<`
    Lt,
    /// `<=`
    LtEq,
    /// `<<`
    LtLt,
    /// `<<=`
    LtLtEq,
    /// `>`
    Gt,
    /// `>=`
    GtEq,
    /// `>>`
    GtGt,
    /// `>>=`
    GtGtEq,

    /// `+`
    Plus,
    /// `+=`
    PlusEq,
    /// `-`
    Minus,
    /// `-=`
    MinusEq,
    /// `->`
    Arrow,
    /// `*`
    Asterisk,
    /// `*=`
    AsteriskEq,
    /// `/`
    Slash,
    /// `/=`
    SlashEq,
    /// `%`
    Percent,
    /// `%=`
    PercentEq,

    /// `&`
    Ampersand,
    /// `&&`
    AmpersandAmpersand,
    /// `&&=`
    AmpersandAmpersandEq,
    /// `&=`
    AmpersandEq,
    /// `|`
    Pipe,
    /// `||`
    PipePipe,
    /// `||=`
    PipePipeEq,
    /// `|=`
    PipeEq,
    /// `|>`
    PipeGt,
    /// `^`
    Caret,
    /// `^=`
    CaretEq,

    /// `.`
    Dot,
    /// `..`
    DotDot,
    /// `..=`
    DotDotEq,
    /// `.*`
    DotAsterisk,
    /// `.&`
    DotAmpersand,
    /// `.(`
    DotOpenParenthesis,
    /// `.[`
    DotOpenBracket,
    /// `.{`
    DotOpenBrace,
    /// `,`
    Comma,
    /// `:`
    Colon,
    /// `::`
    ColonColon,
    /// `:=`
    ColonEq,
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
    Backslash,
    /// `
    Backtick,

    Unknown,
}

impl TokenKind {
    pub fn is_whitespace(&self) -> bool {
        matches!(
            self,
            TokenKind::Whitespace
                | TokenKind::LineComment
                | TokenKind::LineDocInner
                | TokenKind::LineDocOuter
                | TokenKind::BlockComment
                | TokenKind::BlockDocInner
                | TokenKind::BlockDocOuter
        )
    }

    /// `true` also means the token ends parsing of the previous expression
    /// independent of precedence.
    #[inline]
    pub fn is_invalid_start(self) -> bool {
        use TokenKind as K;
        match self {
            //K::Whitespace(whitespace) => todo!(),
            K::Keyword(Keyword::Else) | K::CloseParenthesis | K::CloseBracket | K::CloseBrace |
            //K::FatArrow => todo!(),
            //K::Arrow => todo!(),
            K::PipeGt |
            K::Comma |
            //K::Colon => todo!(),
            //K::ColonColon => todo!(),
            //K::ColonEq => todo!(),
            K::Semicolon => true,
            //K::Pound => todo!(),
            //K::Dollar => todo!(),
            //K::At => todo!(),
            //K::Tilde => todo!(),
            //K::Backtick => todo!(),
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
    // TODO: reduce size (see <https://youtu.be/IroPQ150F6c?t=2100>)
    //pub span_start: usize,
}

impl Token {
    /*
    fn span(&self, code: &Code) -> Span {
        let mut lexer = Lexer::new(code.offset(self.span_start));
        panic!("{}", lexer.pos_span());
        todo!()
    }
    */

    /// `true` also means the token ends parsing of the previous expression
    /// independent of precedence.
    #[inline]
    pub fn is_invalid_start(&self) -> bool {
        self.kind.is_invalid_start()
    }
}

macro_rules! keywords {
    ( $($enum_variant:ident = $text:literal),* $(,)? ) => {
        #[derive(Debug, Clone, Copy, PartialEq, Eq)]
        pub enum Keyword {
            $($enum_variant),*
        }

        impl FromStr for Keyword {
            type Err = ParseErrorKind;

            fn from_str(s: &str) -> Result<Self, Self::Err> {
                match s {
                    $($text => Result::Ok(Keyword::$enum_variant),)*
                    _ => Result::Err(ParseErrorKind::NotAKeyword)
                }
            }
        }

        impl Keyword {
            pub fn as_str(self) -> &'static str {
                match self {
                    $(Keyword::$enum_variant => $text,)*
                }
            }
        }
    };
}

keywords! {
    Mut = "mut",
    Rec = "rec",
    Pub = "pub",
    Struct = "struct",
    Union = "union",
    Enum = "enum",
    Unsafe = "unsafe",
    Extern = "extern",
    And = "and",
    AndEq = "and=",
    Or = "or",
    OrEq = "or=",
    If = "if",
    Then = "then",
    Else = "else",
    Match = "match",
    For = "for",
    While = "while",
    Do = "do",
    Return = "return",
    Break = "break",
    Continue = "continue",
    Autocast = "xx",
    Defer = "defer",
}

/// byte range offset for a [`Code`].
#[derive(Clone, Copy, PartialEq)]
pub struct Span {
    pub start: usize,
    pub end: usize,

    /// Every Span containing this pointer seems overkill, but it has to be fine for now.
    pub file: OPtr<SourceFile>,
}

impl fmt::Debug for Span {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Span{}..{}", &self.start, &self.end)
    }
}

impl Span {
    pub const ZERO: Span = Span::new(0..0, None);

    pub const fn new(range: Range<usize>, file: OPtr<SourceFile>) -> Span {
        let Range { start, end } = range;
        Span { start, end, file }
    }

    pub fn pos(pos: usize, file: OPtr<SourceFile>) -> Span {
        Self::new(pos..pos + 1, file)
    }

    pub fn start(&self) -> Span {
        Span::pos(self.start, self.file)
    }

    pub fn end(&self) -> Span {
        Span::pos(self.end - 1, self.file)
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
        debug_assert_eq!(self.file, other.file);
        Span::new(self.start.min(other.start)..self.end.max(other.end), self.file)
    }

    pub fn multi_join(spans: impl IntoIterator<Item = Span>) -> Option<Span> {
        let mut iter = spans.into_iter();
        let first = iter.next();
        let last = iter.last();
        first.map(|first| last.map(|last| first.join(last)).unwrap_or(first))
    }

    pub fn start_pos(self) -> Span {
        Span::pos(self.start, self.file)
    }

    pub fn after(self) -> Span {
        Span::pos(self.end, self.file)
    }

    pub fn range(&self) -> Range<usize> {
        (self.start..self.end).into()
    }

    pub fn get_text(self) -> Ptr<str> {
        Ptr::from_ref(&self.file.u().code.as_ref()[self])
    }
}

impl From<Span> for Range<usize> {
    fn from(span: Span) -> Self {
        span.range()
    }
}

impl fmt::Display for Span {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}..{}", self.start, self.end)
    }
}

#[derive(Debug)]
#[repr(transparent)]
pub struct Code(pub str);

impl Code {
    pub fn new<'c>(code: &'c str) -> &'c Code {
        // SAFETY: `Code` is identical to `str`
        unsafe { mem::transmute(code) }
    }

    /*
    pub fn offset(&self, offset: usize) -> &Self {
        Code::new(&self.0[offset..])
    }
    */
}

impl AsRef<Code> for String {
    fn as_ref(&self) -> &Code {
        Code::new(self)
    }
}

impl AsRef<Code> for str {
    fn as_ref(&self) -> &Code {
        Code::new(self)
    }
}

impl fmt::Display for &Code {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", &self.0)
    }
}

impl Deref for Code {
    type Target = str;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl Index<Span> for &Code {
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
//#[derive(Debug, Clone)]
pub struct Lexer {
    pub code: Cursor,
    next_tok: Option<Token>,

    pub file: Ptr<SourceFile>,
}

impl Lexer {
    pub fn new(file: Ptr<SourceFile>) -> Lexer {
        let code = Cursor::new(file.code);
        let mut lex = Self { code, file, next_tok: None };
        let first_tok = lex.load_next();
        lex.next_tok = first_tok;
        lex
    }

    #[inline]
    pub fn get_code(&self) -> &Code {
        &self.code.code
    }

    pub fn pos_span(&self) -> Span {
        self.next_tok
            .map(|t| t.span)
            .unwrap_or_else(|| Span::pos(self.get_code().len(), Some(self.file)))
    }

    /*
    pub fn span_to(&self, other: Lexer<'_>) -> Span {
        self.pos_span().join(other.pos_span())
    }
    */

    pub fn advanced(mut self) -> Self {
        self.advance();
        self
    }

    fn load_next(&mut self) -> Option<Token> {
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
            '0' => self.num_literal_with_prefix(),
            '0'..='9' => self.num_literal(),

            '(' => TokenKind::OpenParenthesis,
            ')' => TokenKind::CloseParenthesis,
            '[' => TokenKind::OpenBracket,
            ']' => TokenKind::CloseBracket,
            '{' => TokenKind::OpenBrace,
            '}' => TokenKind::CloseBrace,

            '=' => maybe_followed_by! {
                default: TokenKind::Eq,
                '=' => TokenKind::EqEq,
                '>' => TokenKind::FatArrow,
            },
            '!' => maybe_followed_by! {
                default: TokenKind::Bang,
                '=' => TokenKind::BangEq,
            },
            '<' => maybe_followed_by! {
                default: TokenKind::Lt,
                '=' => TokenKind::LtEq,
                '<' => maybe_followed_by! {
                    default: TokenKind::LtLt,
                    '=' => TokenKind::LtLtEq,
                },
            },
            '>' => maybe_followed_by! {
                default: TokenKind::Gt,
                '=' => TokenKind::GtEq,
                '>' => maybe_followed_by! {
                    default: TokenKind::GtGt,
                    '=' => TokenKind::GtGtEq,
                },
            },

            '+' => maybe_followed_by! {
                default: TokenKind::Plus,
                '=' => TokenKind::PlusEq,
            },
            '-' => maybe_followed_by! {
                default: TokenKind::Minus,
                '=' => TokenKind::MinusEq,
                '>' => TokenKind::Arrow,
            },
            '*' => maybe_followed_by! {
                default: TokenKind::Asterisk,
                '=' => TokenKind::AsteriskEq,
            },
            '/' => maybe_followed_by! {
                default: TokenKind::Slash,
                '/' => self.line_comment(),
                '*' => self.block_comment(),
                '=' => TokenKind::SlashEq,
            },
            '%' => maybe_followed_by! {
                default: TokenKind::Percent,
                '=' => TokenKind::PercentEq,
            },

            '&' => maybe_followed_by! {
                default: TokenKind::Ampersand,
                '&' => maybe_followed_by! {
                    default: TokenKind::AmpersandAmpersand,
                    '=' => TokenKind::AmpersandAmpersandEq,
                },
                '=' => TokenKind::AmpersandEq,
            },
            '|' => maybe_followed_by! {
                default: TokenKind::Pipe,
                '|' => maybe_followed_by! {
                    default: TokenKind::PipePipe,
                    '=' => TokenKind::PipePipeEq,
                },
                '=' => TokenKind::PipeEq,
                '>' => TokenKind::PipeGt,
            },
            '^' => maybe_followed_by! {
                default: TokenKind::Caret,
                '=' => TokenKind::CaretEq,
            },

            '.' => maybe_followed_by! {
                default: TokenKind::Dot,
                '.' => maybe_followed_by! {
                    default: TokenKind::DotDot,
                    '=' => TokenKind::DotDotEq,
                },
                '*' => TokenKind::DotAsterisk,
                '&' => TokenKind::DotAmpersand,
                '(' => TokenKind::DotOpenParenthesis,
                '[' => TokenKind::DotOpenBracket,
                '{' => TokenKind::DotOpenBrace,
            },
            ',' => TokenKind::Comma,
            ':' => maybe_followed_by! {
                default: TokenKind::Colon,
                ':' => TokenKind::ColonColon,
                '=' => TokenKind::ColonEq,
            },
            ';' => TokenKind::Semicolon,
            '?' => TokenKind::Question,
            '#' => TokenKind::Pound,
            '$' => TokenKind::Dollar,
            '@' => TokenKind::At,
            '~' => TokenKind::Tilde,
            '\\' => maybe_followed_by! {
                default: TokenKind::Backslash,
                '\\' => self.multiline_string_literal_line(),
            },

            'b' => maybe_followed_by! {
                default: self.ident_like(start),
                '\'' => self.bchar_literal()
            },
            c if is_id_start(c) => self.ident_like(start),

            _ => TokenKind::Unknown,
        };
        Some(Token { kind, span: Span::new(start..self.code.pos, Some(self.file)) })
    }

    fn string_literal(&mut self) -> TokenKind {
        while let Some(c) = self.code.next() {
            match c {
                '\"' => break,
                '\\' => {
                    let escape_char = self.code.next();
                    debug_assert_matches!(
                        escape_char,
                        Some('n' | 'r' | 't' | '\\' | '0' | '\'' | '\"')
                    )
                },
                _ => {},
            }
        }
        TokenKind::StrLit
    }

    fn multiline_string_literal_line(&mut self) -> TokenKind {
        while self.code.next().is_some_and(|c| c != '\n') {}
        TokenKind::MultilineStrLitLine
    }

    fn bchar_literal(&mut self) -> TokenKind {
        self.char_literal();
        TokenKind::BCharLit
    }

    fn char_literal(&mut self) -> TokenKind {
        while let Some(c) = self.code.next() {
            match c {
                '\'' | '\n' => break,
                '\\' => {
                    let escape_char = self.code.next();
                    debug_assert_matches!(
                        escape_char,
                        Some('n' | 'r' | 't' | '\\' | '0' | '\'' | '\"')
                    )
                },
                _ => {},
            }
        }
        TokenKind::CharLit
    }

    fn num_literal(&mut self) -> TokenKind {
        self.code.advance_while(|c| matches!(c, '0'..='9' | '_'));
        match self.code.peek().zip(self.code.peek2()) {
            // an integer might be followed by a range operator `..` or a method `1.foo()`
            Some(('.', c)) if c != '.' && !is_id_start(c) => {
                self.code.advance();
                self.code.advance_while(|c| matches!(c, '0'..='9' | '_'));
                TokenKind::FloatLit
            },
            _ => TokenKind::IntLit,
        }
    }

    fn num_literal_with_prefix(&mut self) -> TokenKind {
        if self.code.advance_if(|c| matches!(c, 'b' | 'o' | 'x')) {
            self.code.advance_while(|c| matches!(c, '0'..='9' | '_'));
            TokenKind::IntLit
        } else {
            self.num_literal()
        }
    }

    fn line_comment(&mut self) -> TokenKind {
        let t = match self.code.peek() {
            Some('!') => TokenKind::LineDocInner,
            Some('/') => TokenKind::LineDocOuter,
            _ => TokenKind::LineComment,
        };
        while !matches!(self.code.next(), None | Some('\n')) {}
        t
    }

    fn block_comment(&mut self) -> TokenKind {
        let t = match self.code.peek() {
            Some('!') => TokenKind::BlockDocInner,
            // `/**/` => CommentType::Comment
            Some('*') if self.code.peek2() != Some('/') => TokenKind::BlockDocOuter,
            _ => TokenKind::BlockComment,
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

        t
    }

    fn ident_like(&mut self, start: usize) -> TokenKind {
        self.code.advance_while(is_id_continue);
        match &self.get_code().0[start..self.code.pos] {
            "true" => TokenKind::BoolLitTrue,
            "false" => TokenKind::BoolLitFalse,
            s => s.parse::<Keyword>().map(TokenKind::Keyword).unwrap_or(TokenKind::Ident),
        }
    }

    #[inline]
    pub fn next_if_kind(&mut self, kind: TokenKind) -> Option<Token> {
        self.next_if(|t| t.kind == kind)
    }

    #[inline]
    pub fn advance_if_kind(&mut self, kind: TokenKind) -> bool {
        self.advance_if(|t| t.kind == kind)
    }
}

impl ParserInterface for Lexer {
    type Item = Token;
    type PeekedItem = Token;

    fn next(&mut self) -> Option<Self::Item> {
        let t = self.next_tok;
        self.next_tok = self.load_next();
        t
    }

    fn peek(&self) -> Option<Self::PeekedItem> {
        self.next_tok
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Cursor {
    code: Ptr<Code>,
    pos: usize,
}

impl Cursor {
    pub fn new(code: Ptr<Code>) -> Cursor {
        Cursor { code, pos: 0 }
    }

    pub fn get_rem(&self) -> &str {
        &self.code.0[self.pos..]
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

    pub fn get_pos(&self) -> usize {
        self.pos
    }

    pub unsafe fn set_pos(&mut self, pos: usize) {
        self.pos = pos;
    }

    pub fn set_pos_checked(&mut self, pos: usize) -> bool {
        let is_valid = pos < self.code.len() && self.code.0.is_char_boundary(pos);
        if is_valid {
            self.pos = pos;
        }
        is_valid
    }
}

impl ParserInterface for Cursor {
    type Item = char;
    type PeekedItem = char;

    fn next(&mut self) -> Option<Self::Item> {
        let mut i = self.get_rem().char_indices();
        let (_, char) = i.next()?;
        self.pos = match i.next() {
            Some((offset, _)) => self.pos + offset,
            None => self.code.0.len(),
        };
        Some(char)
    }

    fn peek(&self) -> Option<Self::PeekedItem> {
        self.get_rem().chars().next()
    }
}

/*
#[test]
fn parse_token_span() {
    let code = "
";
    let lex = Lexer::new(code.as_ref());
    panic!()
}
*/
