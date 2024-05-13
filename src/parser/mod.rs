use self::lexer::TokenKind;
use crate::parser::lexer::{Lexer, Token};
use std::{
    iter::Peekable,
    ops::{Index, Range},
};

mod lexer;
mod util;

#[derive(Debug, Clone)]
pub enum ParseError {
    NoInput,
    MissingToken(TokenKind),
    MissingLetIdent,
    /// `let mut mut a = ...`
    /// `        ^^^` span
    MultipleMut(Span),
    MultipleRec(Span),
    UnexpectedToken(Token),
}

pub type ParseResult<T> = Result<T, ParseError>;

/*
impl<T> FromResidual<ParseResult<!>> for ParseResult<T> {
    fn from_residual(residual: ParseResult<!>) -> Self {
        match residual {
            ParseResult::Ok(never) => never,
            ParseResult::Err(e) => ParseResult::Err(e),
            ParseResult::None => ParseResult::None,
        }
    }
}

impl<T> Try for ParseResult<T> {
    type Output = T;
    type Residual = ParseResult<!>;

    fn from_output(output: Self::Output) -> Self {
        ParseResult::Ok(output)
    }

    fn branch(self) -> std::ops::ControlFlow<Self::Residual, Self::Output> {
        match self {
            ParseResult::Ok(t) => std::ops::ControlFlow::Continue(t),
            ParseResult::Err(e) => std::ops::ControlFlow::Break(ParseResult::Err(e)),
            ParseResult::None => std::ops::ControlFlow::Break(ParseResult::None),
        }
    }
}
*/

#[derive(Debug, Clone, Copy)]
pub struct Code<'a>(pub &'a str);

impl<'a> Index<Span> for Code<'a> {
    type Output = str;

    fn index(&self, span: Span) -> &Self::Output {
        &self.0[span.bytes()]
    }
}

/// byte range offset for a [`Code`].
#[derive(Debug, Clone, Copy)]
pub struct Span {
    pub start: usize,
    pub end: usize,
}

impl Span {
    pub fn new(start: usize, end: usize) -> Span {
        Span { start, end }
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
        Span::new(self.start.min(other.start), self.end.min(other.end))
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

#[derive(Debug, Clone)]
pub enum Stmt {
    /// `let mut rec <name>: <type> = <rhs>`
    Let {
        let_: Token,
        is_mut: bool,
        is_rec: bool,
        name: Ident,
        type_: Option<Box<Expr>>,
        kind: LetKind,
    },

    /// `<expr>;`
    Semicolon {
        expr: Box<Expr>,
        span: Span,
    },

    Expr(Box<Expr>),
}

impl Stmt {
    pub fn parse<'p>(parser: &mut Parser<'p, impl Iterator<Item = Token>>) -> ParseResult<Self> {
        let token = parser.lexer.next().ok_or(ParseError::NoInput)?;
        match token.kind {
            TokenKind::Ident if &parser.code[token.span] == "let" => Stmt::parse_let(token, parser),
            _ => Err(ParseError::UnexpectedToken(token)),
        }
    }

    pub fn parse_let<'c>(
        let_: Token,
        parser: &mut Parser<'c, impl Iterator<Item = Token>>,
    ) -> ParseResult<Stmt> {
        let mut is_mut = false;
        let mut is_rec = false;

        let name = loop {
            let span = match parser.lexer.next() {
                Some(Token { kind: TokenKind::Ident, span }) => span,
                Some(t) => return Err(ParseError::UnexpectedToken(t)),
                None => return Err(ParseError::MissingLetIdent),
            };
            match &parser.code[span] {
                "mut" => {
                    if is_mut {
                        return Err(ParseError::MultipleMut(span));
                    } else {
                        is_mut = true;
                    }
                },
                "rec" => {
                    if is_rec {
                        return Err(ParseError::MultipleRec(span));
                    } else {
                        is_rec = true;
                    }
                },
                _ => break Ident { span },
            }
        };

        let type_ = if let Some(Token { kind: TokenKind::Colon, .. }) = parser.lexer.peek() {
            parser.lexer.next();
            Some(Expr::parse(parser).map(Box::new)?)
        } else {
            None
        };

        let kind = if let Some(Token { kind: TokenKind::Equal, .. }) = parser.lexer.peek() {
            parser.lexer.next();
            LetKind::Init(Box::new(Expr::parse(parser)?))
        } else {
            LetKind::Decl
        };

        Ok(Stmt::Let { let_, is_mut, is_rec, name, type_, kind })
    }

    /// returns the [`Span`] for the entire expression.
    pub fn span(&self) -> Span {
        match self {
            Stmt::Let { let_, name, type_, kind, .. } => let_.span.join(
                kind.span().or_else(|| type_.as_ref().map(|e| e.span())).unwrap_or(name.span),
            ),
            Stmt::Semicolon { span, .. } => *span,
            Stmt::Expr(expr) => expr.span(),
        }
    }
}

/// grouped and ordered by precedence
#[derive(Debug, Clone)]
pub enum Expr {
    Ident(Ident),
    /// `[<val>; <count>]`
    /// both for types and literals
    ArrayShort {
        val: Box<Expr>,
        count: Box<Expr>,
        span: Span,
    },
    /// `[<expr>, <expr>, ..., <expr>,]`
    ArrayInit {
        elements: Vec<Expr>,
        span: Span,
    },
    /// `(<expr>, <expr>, ..., <expr>,)`
    /// both for types and literals
    Tuple {
        elements: Vec<Expr>,
        span: Span,
    },
    Literal {
        kind: LitKind,
        span: Span,
    },
    //Type(Type),
    Fn {
        signature: (),
        body: Box<Expr>,
        span: Span,
    },
    /// `struct { ... }`
    StructDef {
        fields: Vec<StructFields>,
        span: Span,
    },
    /// `MyStruct { a: <expr>, b, }`
    StructInit {
        fields: Vec<StructFields>,
        span: Span,
    },
    /// `struct(...)`
    TupleStructDef {
        fields: Vec<Expr>,
        span: Span,
    },
    /// `union { ... }`
    Union {
        span: Span,
    },
    /// `enum { ... }`
    Enum {
        span: Span,
    },

    /// `{ ... }`
    Block {
        nodes: Vec<Stmt>,
        span: Span,
    },
    /// `( <expr> )`
    Parenthesis {
        expr: Box<Expr>,
        span: Span,
    },

    /// `<lhs> . <rhs>`
    Dot {
        lhs: Box<Expr>,
        rhs: Ident,
    },
    /// examples: `<expr>?`, `<expr>.*`
    PostOp {
        kind: PostOpKind,
        expr: Box<Expr>,
        span: Span,
    },
    /// `<lhs> [ <idx> ]`
    Index {
        lhs: Box<Expr>,
        idx: Box<Expr>,
        span: Span,
    },

    /// `<lhs> : <rhs>`
    Colon {
        lhs: Box<Expr>,
        rhs: Box<Expr>,
    },

    /// `<func> < <params> >`
    CompCall {
        func: Box<Expr>,
        params: Vec<Expr>,
        span: Span,
    },
    /// `<func> ( <params> )`
    Call {
        func: Box<Expr>,
        params: Vec<Expr>,
        span: Span,
    },

    /// examples: `&<expr>`, `- <expr>`
    PreOp {
        kind: PreOpKind,
        expr: Box<Expr>,
        span: Span,
    },
    BinOp {
        lhs: Box<Expr>,
        op: BinOpKind,
        rhs: Box<Expr>,
    },
    BinOpAssign {
        lhs: Box<Expr>,
        op: BinOpKind,
        rhs: Box<Expr>,
    },
}

impl Expr {
    pub fn parse<'p>(parser: &mut Parser<'p, impl Iterator<Item = Token>>) -> ParseResult<Expr> {
        while let Some(t) = parser.lexer.next() {
            let span = t.span;
            match t.kind {
                TokenKind::Whitespace | TokenKind::LineComment(_) | TokenKind::BlockComment(_) => {
                    continue;
                },
                TokenKind::Ident => {
                    let text = &parser.code[t.span];
                    return ParseResult::Ok(match text {
                        //"let" => Expr::parse_let(t, parser)?,
                        "true" | "false" => Expr::Literal { kind: LitKind::Bool, span: t.span },
                        _ => Expr::Ident(Ident { span }),
                    });
                },
                TokenKind::Literal(l) => todo!("Literal({:?})", l),
                TokenKind::OpenParenthesis => todo!("OpenParenthesis"),
                TokenKind::CloseParenthesis => todo!("CloseParenthesis"),
                TokenKind::OpenBracket => todo!("OpenBracket"),
                TokenKind::CloseBracket => todo!("CloseBracket"),
                TokenKind::OpenBrace => todo!("OpenBrace"),
                TokenKind::CloseBrace => todo!("CloseBrace"),
                TokenKind::Equal => todo!("Equal"),
                TokenKind::EqualEqual => todo!("EqualEqual"),
                TokenKind::FatArrow => todo!("FatArrow"),
                TokenKind::Bang => todo!("Bang"),
                TokenKind::BangEqual => todo!("BangEqual"),
                TokenKind::Lt => todo!("Lt"),
                TokenKind::LtEqual => todo!("LtEqual"),
                TokenKind::LtLt => todo!("LtLt"),
                TokenKind::LtLtEqual => todo!("LtLtEqual"),
                TokenKind::Gt => todo!("Gt"),
                TokenKind::GtEqual => todo!("GtEqual"),
                TokenKind::GtGt => todo!("GtGt"),
                TokenKind::GtGtEqual => todo!("GtGtEqual"),
                TokenKind::Plus => todo!("Plus"),
                TokenKind::PlusEqual => todo!("PlusEqual"),
                TokenKind::Minus => todo!("Minus"),
                TokenKind::MinusEqual => todo!("MinusEqual"),
                TokenKind::Arrow => todo!("Arrow"),
                TokenKind::Asterisk => todo!("Asterisk"),
                TokenKind::AsteriskEqual => todo!("AsteriskEqual"),
                TokenKind::Slash => todo!("Slash"),
                TokenKind::SlashEqual => todo!("SlashEqual"),
                TokenKind::Percent => todo!("Percent"),
                TokenKind::PercentEqual => todo!("PercentEqual"),
                TokenKind::Ampersand => todo!("Ampersand"),
                TokenKind::AmpersandAmpersand => todo!("AmpersandAmpersand"),
                TokenKind::AmpersandEqual => todo!("AmpersandEqual"),
                TokenKind::Pipe => todo!("Pipe"),
                TokenKind::PipePipe => todo!("PipePipe"),
                TokenKind::PipeEqual => todo!("PipeEqual"),
                TokenKind::Caret => todo!("Caret"),
                TokenKind::CaretEqual => todo!("CaretEqual"),
                TokenKind::Dot => todo!("Dot"),
                TokenKind::DotDot => todo!("DotDot"),
                TokenKind::DotAsterisk => todo!("DotAsterisk"),
                TokenKind::DotAmpersand => todo!("DotAmpersand"),
                TokenKind::Comma => todo!("Comma"),
                TokenKind::Colon => todo!("Colon"),
                TokenKind::Semicolon => todo!("Semicolon"),
                TokenKind::Question => todo!("Question"),
                TokenKind::Pound => todo!("Pound"),
                TokenKind::BackSlash => todo!("BackSlash"),
                TokenKind::Dollar
                | TokenKind::At
                | TokenKind::Tilde
                | TokenKind::BackTick
                | TokenKind::Unknown => return Err(ParseError::UnexpectedToken(t)),
            }
        }
        Err(ParseError::NoInput)
    }

    /// returns the [`Span`] for the entire expression.
    pub fn span(&self) -> Span {
        match self {
            Expr::Ident(i) => i.span,
            Expr::Dot { lhs, rhs } => lhs.span().join(rhs.span),
            Expr::Colon { lhs, rhs }
            | Expr::BinOp { lhs, rhs, .. }
            | Expr::BinOpAssign { lhs, rhs, .. } => lhs.span().join(rhs.span()),
            Expr::ArrayShort { span, .. }
            | Expr::StructDef { span, .. }
            | Expr::StructInit { span, .. }
            | Expr::Fn { span, .. }
            | Expr::TupleStructDef { span, .. }
            | Expr::Union { span, .. }
            | Expr::Enum { span, .. }
            | Expr::ArrayInit { span, .. }
            | Expr::Tuple { span, .. }
            | Expr::Literal { span, .. }
            | Expr::Block { span, .. }
            | Expr::Parenthesis { span, .. }
            | Expr::PostOp { span, .. }
            | Expr::PreOp { span, .. }
            | Expr::Index { span, .. }
            | Expr::CompCall { span, .. }
            | Expr::Call { span, .. } => *span,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Ident {
    span: Span,
}

#[derive(Debug, Clone)]
pub struct StructFields {
    name_ident: Expr,
    type_: Expr,
}

#[derive(Debug, Clone)]
pub enum LetKind {
    Decl,
    Init(Box<Expr>),
}

impl LetKind {
    pub fn span(&self) -> Option<Span> {
        match self {
            LetKind::Decl => None,
            LetKind::Init(expr) => Some(expr.span()),
        }
    }
}

#[derive(Debug, Clone)]
pub enum PreOpKind {
    /// `& <expr>`
    AddrOf,
    /// `&mut <expr>`
    AddrMutOf,
    /// `* <expr>`
    Deref,
    /// `! <expr>`
    Not,
    /// `- <expr>`
    Neg,
}

#[derive(Debug, Clone)]
pub enum PostOpKind {
    /// `<expr>.&`
    AddrOf,
    /// `<expr>.&mut`
    AddrMutOf,
    /// `<expr>.*`
    Deref,
    /// `<expr>?`
    Try,
    /// `<expr>!`
    Force,
    /// `<expr>!unsafe`
    ForceUnsafe,
    /// `<expr>.type`
    TypeOf,
}

#[derive(Debug, Clone, Copy)]
pub enum BinOpKind {
    /// `*`
    Mul,
    /// `/`
    Div,
    /// `%`
    Mod,

    /// `+`
    Plus,
    /// `-`
    Minus,

    /// `<<`
    ShiftL,
    /// `>>`
    ShiftR,

    /// `&`
    BitAnd,

    /// `^`
    BitXor,

    /// `|`
    BitOr,

    /// `==`
    Eq,
    /// `!=`
    Ne,
    /// `<`
    Lt,
    /// `<=`
    Le,
    /// `>`
    Gt,
    /// `>=`
    Ge,

    /// `&&`
    And,

    /// `||`
    Or,

    /// `..`
    Range,
    /// `..=`
    RangeInclusive,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LitKind {
    /// `true`, `false`
    Bool,
    /// `'a'`
    Char,
    /// `b'a'`
    BChar,
    /// `1`, `-10`, `10_000`
    /// allowed but not recommended: `1_0_0_0`
    Int,
    /// `1.5`
    Float,
    /// `"literal"`
    Str,
}

#[derive(Debug, Clone)]
pub struct Parser<'c, Lex: Iterator<Item = Token>> {
    code: Code<'c>,
    lexer: Peekable<Lex>,
}

pub fn parser_new<'c>(code: &'c str) -> Parser<'c, impl Iterator<Item = Token> + 'c> {
    let lexer = Lexer::new(code).filter(|t| !t.kind.is_ignored()).peekable();
    let code = Code(code);
    Parser { code, lexer }
}

impl<'c, Lex: Iterator<Item = Token>> Parser<'c, Lex> {
    pub fn peek_kind_is(&mut self, rhs: TokenKind) -> bool {
        self.lexer.peek().is_some_and(|t| t.kind == rhs)
    }
}

impl<'c, Lex: Iterator<Item = Token>> Iterator for Parser<'c, Lex> {
    type Item = Result<Stmt, ParseError>;

    fn next(&mut self) -> Option<Self::Item> {
        match Stmt::parse(self) {
            ParseResult::Ok(stmt) => Some(Ok(stmt)),
            ParseResult::Err(err) => Some(Err(err)),
            //ParseResult::None => None,
        }
    }
}

fn debug_lexer(code: &str, lexer: impl Iterator<Item = Token>) {
    let mut full = String::new();
    let mut prev_was_ident = false;

    for Token { kind, span } in lexer {
        let text = code.get(span.bytes()).expect("correctly parsed span");
        let is_ident = kind == TokenKind::Ident;
        if prev_was_ident && is_ident {
            full.push(' ');
        }
        full.push_str(&text);
        prev_was_ident = is_ident;
        let text = format!("{:?}", text);
        println!("{:<20} -> {:?}", text, kind);
    }

    println!("+++ full code:\n{}", full);
}

pub fn debug_tokens(code: &str) {
    let lexer = Lexer::new(code).filter(|t| {
        !matches!(
            t.kind,
            TokenKind::Whitespace | TokenKind::LineComment(_) | TokenKind::BlockComment(_)
        )
    });

    debug_lexer(code, lexer);
}

pub fn parse(code: &str) -> Result<Vec<Stmt>, ParseError> {
    debug_tokens(code);
    //parser_new(code).collect()
    parser_new(code).next().unwrap().map(|a| vec![a])
}
