use lexer::{Lexer, Span, Token, TokenKind};

pub mod lexer;

#[allow(unused)]
macro_rules! todo {
    ($msg:literal, $span:expr) => {
        return Err(ParseError::Tmp(String::leak(format!("TODO: {}", $msg)), $span))
    };
}

macro_rules! always {
    ($val:expr) => {
        |lex| Ok(($val, lex))
    };
}

#[derive(Debug)]
pub enum ParseError {
    NoInput,
    UnexpectedToken(Token),
    NotAnIdent,
    NotAnKeyword,
    /// `let mut = ...`
    MissingLetIdent,
    /// `let mut mut x;`
    /// `        ^^^`
    DoubleLetMarker(Token),
    /// `let x x;`
    /// `      ^`
    TooManyLetIdents(Span),

    NotWasFound,

    Tmp(&'static str, Span),
}

type ParseResult<'l, T> = Result<(T, Lexer<'l>), ParseError>;

pub trait Parser<T>: Fn(Lexer<'_>) -> ParseResult<'_, T> + Copy + Sized {
    fn new(f: Self) -> impl Parser<T> {
        f
    }

    /// ```text
    /// | self output   | other output  | output        |
    /// |:-------------:|:-------------:|:-------------:|
    /// | Err           | Err           | Err           |
    /// | Err           | Ok t2         | Ok t2         |
    /// | Ok t1         | ?             | Ok t1         |
    /// ```
    fn or(self, other: impl Parser<T>) -> impl Parser<T> {
        move |lex| self(lex.clone()).or_else(|_| other(lex))
    }

    /// ```text
    /// | self output   | other output  | output        |
    /// |:-------------:|:-------------:|:-------------:|
    /// | Err           | Err           | Err           |
    /// | Ok t1         | Err           | Err           |
    /// | Ok t1         | Ok t2         | Ok (t1, t2)   |
    /// ```
    fn and<T2>(self, other: impl Parser<T2>) -> impl Parser<(T, T2)> {
        move |lex| self(lex).and_then(|(t1, lex)| other(lex).map(|(t2, lex)| ((t1, t2), lex)))
    }

    /// like [`Parser::and`] but only keeps the lhs
    fn has_suffix<T2>(self, other: impl Parser<T2>) -> impl Parser<T> {
        self.and(other).map(|(lhs, _)| lhs)
    }

    /// like [`Parser::and`] but only keeps the rhs
    fn prefix_of<T2>(self, other: impl Parser<T2>) -> impl Parser<T2> {
        self.and(other).map(|(_, rhs)| rhs)
    }

    /// ```text
    /// | self output   | output        |
    /// |:-------------:|:-------------:|
    /// | Err           | Err           |
    /// | Ok t1         | Ok f(t1)      |
    /// ```
    fn map<U>(self, f: impl Fn(T) -> U + Copy) -> impl Parser<U> {
        move |lex| self(lex).map(|(t, lex)| (f(t), lex))
    }

    fn map_and<U, P2: Parser<U>>(self, other: impl Fn(T) -> P2 + Copy) -> impl Parser<U> {
        move |lex| self(lex).and_then(|(t1, lex)| other(t1)(lex))
    }

    fn map_with_data<U>(self, f: impl Fn(T, &Lexer<'_>) -> U + Copy) -> impl Parser<U> {
        move |lex| self(lex).map(|(t, lex)| (f(t, &lex), lex))
    }

    /// ```text
    /// | self output   | output        |
    /// |:-------------:|:-------------:|
    /// | Err           | Err           |
    /// | Ok t1         | f(t1)         |
    /// ```
    fn flat_map<U>(self, f: impl Fn(T) -> Result<U, ParseError> + Copy) -> impl Parser<U> {
        move |lex| self(lex).and_then(|(t, lex)| Ok((f(t)?, lex)))
    }

    fn flat_map_with_data<U>(
        self,
        f: impl Fn(T, &Lexer<'_>) -> Result<U, ParseError> + Copy,
    ) -> impl Parser<U> {
        move |lex| self(lex).and_then(|(t, lex)| Ok((f(t, &lex)?, lex)))
    }

    fn not(self) -> impl Parser<()> {
        move |lex| match self(lex) {
            Ok(_) => Err(ParseError::NotWasFound),
            Err(_) => Ok(((), lex)),
        }
    }

    /// ```text
    /// | self output   | output        |
    /// |:-------------:|:-------------:|
    /// | Err           | Ok None       |
    /// | Ok t1         | Ok Some(t1)   |
    /// ```
    fn opt(self) -> impl Parser<Option<T>> {
        self.map(Some).or(always!(None))
    }

    /// `self` 0 or more times
    /// ```text
    /// | self output   | output        |
    /// |:-------------:|:-------------:|
    /// | Err           | Ok []         |
    /// | Ok t1         | Ok [...]      |
    /// ```
    fn many0(self) -> impl Parser<Vec<T>> {
        self.many0_default(Vec::new)
    }

    /// `self` 1 or more times
    /// ```text
    /// | self output   | output        |
    /// |:-------------:|:-------------:|
    /// | Err           | Err           |
    /// | Ok t1         | Ok [...]      |
    /// ```
    fn many1(self) -> impl Parser<Vec<T>> {
        move |lex| {
            let (first, mut lex) = self(lex)?;
            let mut vec = vec![first];
            while let Ok((t, new_lex)) = self(lex) {
                vec.push(t);
                lex = new_lex;
            }
            Ok((vec, lex))
        }
    }

    /// ```text
    /// | self output   | output        |
    /// |:-------------:|:-------------:|
    /// | Err           | Ok [d]        |
    /// | Ok t1         | Ok [d, ...]   |
    /// ```
    fn many0_default(self, def: impl Fn() -> Vec<T> + Copy) -> impl Parser<Vec<T>> {
        move |mut lex| {
            let mut vec = def();
            while let Ok((t, new)) = self(lex) {
                vec.push(t);
                lex = new;
            }
            Ok((vec, lex))
        }
    }

    fn many_until<U>(self, until: impl Parser<U>) -> impl Parser<(Vec<T>, U)> {
        move |mut lex| {
            let mut vec = vec![];
            let u = loop {
                match until(lex) {
                    Ok((u, new_lex)) => {
                        lex = new_lex;
                        break u;
                    },
                    Err(_) => {
                        let (t, new_lex) = self(lex)?;
                        lex = new_lex;
                        vec.push(t)
                    },
                }
            };
            Ok(((vec, u), lex))
        }
    }

    fn sep_by1<Sep>(self, sep: impl Parser<Sep>) -> impl Parser<Vec<T>>
    where Self: Copy {
        self.and(sep.prefix_of(self).many0()).map(|(first, mut rest)| {
            rest.insert(0, first);
            rest
        })
    }

    fn sep_by0<Sep>(self, sep: impl Parser<Sep>) -> impl Parser<Vec<T>>
    where Self: Copy {
        self.sep_by1(sep).or(|lex| Ok((Vec::new(), lex)))
    }

    /// creates a [`Span`] for the entire match
    /// ```text
    /// | self output   | output        |
    /// |:-------------:|:-------------:|
    /// | Err           | Err           |
    /// | Ok t1         | Ok (t1, span) |
    /// ```
    fn spaned(self) -> impl Parser<(T, Span)> {
        Parser::new(|lex: Lexer<'_>| Ok((lex.get_pos(), lex)))
            .and(self)
            .map_with_data(|(start, t), lex| (t, Span::new(start, lex.get_pos())))
    }

    fn to_spaned(self) -> impl Parser<Span> {
        self.spaned().map(|a| a.1)
    }

    fn to_expr(self) -> impl Parser<Expr>
    where Expr: From<(T, Span)> {
        self.spaned().map(Expr::from)
    }

    fn to_stmt(self) -> impl Parser<Stmt>
    where Stmt: From<(T, Span)> {
        self.spaned().map(Stmt::from)
    }
}

impl<T, F: Fn(Lexer<'_>) -> Result<(T, Lexer<'_>), ParseError> + Copy> Parser<T> for F {}

/// no whitespace skipping after the token
pub fn tok_matches_(f: impl Fn(&TokenKind) -> bool + Copy) -> impl Parser<Token> {
    move |lex| match lex.peek() {
        Some(t) if f(&t.kind) => Ok((t, lex.advanced())),
        Some(t) => Err(ParseError::UnexpectedToken(t)),
        None => Err(ParseError::NoInput),
    }
}

/// parses any whitespace or comment
pub fn ws(lex: Lexer<'_>) -> ParseResult<'_, Span> {
    tok_matches_(|t| t.is_ignored()).many0().to_spaned()(lex)
}

pub fn tok_matches(f: impl Fn(&TokenKind) -> bool + Copy) -> impl Parser<Token> {
    tok_matches_(f).has_suffix(ws)
}

pub fn any_tok() -> impl Parser<Token> {
    tok_matches(|_| true)
}

pub fn tok(kind: TokenKind) -> impl Parser<Token> {
    tok_matches(move |lex| *lex == kind)
}

macro_rules! tok {
    ('(') => { tok(TokenKind::OpenParenthesis) };
    (')') => { tok(TokenKind::CloseParenthesis) };
    ('[') => { tok(TokenKind::OpenBracket) };
    (']') => { tok(TokenKind::CloseBracket) };
    ('{') => { tok(TokenKind::OpenBrace) };
    ('}') => { tok(TokenKind::CloseBrace) };
    ('=') => { tok(TokenKind::Equal) };
    ("==") => { tok(TokenKind::EqualEqual) };
    ("=>") => { tok(TokenKind::FatArrow) };
    ('!') => { tok(TokenKind::Bang) };
    ("!=") => { tok(TokenKind::BangEqual) };
    ('<') => { tok(TokenKind::Lt) };
    ("<=") => { tok(TokenKind::LtEqual) };
    ("<<") => { tok(TokenKind::LtLt) };
    ("<<=") => { tok(TokenKind::LtLtEqual) };
    ('>') => { tok(TokenKind::Gt) };
    (">=") => { tok(TokenKind::GtEqual) };
    (">>") => { tok(TokenKind::GtGt) };
    (">>=") => { tok(TokenKind::GtGtEqual) };
    ('+') => { tok(TokenKind::Plus) };
    ("+=") => { tok(TokenKind::PlusEqual) };
    ('-') => { tok(TokenKind::Minus) };
    ("-=") => { tok(TokenKind::MinusEqual) };
    ("->") => { tok(TokenKind::Arrow) };
    ('*') => { tok(TokenKind::Asterisk) };
    ("*=") => { tok(TokenKind::AsteriskEqual) };
    ('/') => { tok(TokenKind::Slash) };
    ("/=") => { tok(TokenKind::SlashEqual) };
    ('%') => { tok(TokenKind::Percent) };
    ("%=") => { tok(TokenKind::PercentEqual) };

    ('&') => { tok(TokenKind::Ampersand) };
    ("&&") => { tok(TokenKind::AmpersandAmpersand) };
    ("&=") => { tok(TokenKind::AmpersandEqual) };
    ('|') => { tok(TokenKind::Pipe) };
    ("||") => { tok(TokenKind::PipePipe) };
    ("|=") => { tok(TokenKind::PipeEqual) };
    ('^') => { tok(TokenKind::Caret) };
    ("^=") => { tok(TokenKind::CaretEqual) };
    ('.') => { tok(TokenKind::Dot) };
    ("..") => { tok(TokenKind::DotDot) };
    (".*") => { tok(TokenKind::DotAsterisk) };
    (".&") => { tok(TokenKind::DotAmpersand) };
    (',') => { tok(TokenKind::Comma) };
    (':') => { tok(TokenKind::Colon) };
    (';') => { tok(TokenKind::Semicolon) };
    ('?') => { tok(TokenKind::Question) };
    ('#') => { tok(TokenKind::Pound) };
    ('$') => { tok(TokenKind::Dollar) };
    ('@') => { tok(TokenKind::At) };
    ('~') => { tok(TokenKind::Tilde) };
    ('\\') => { compile_error!("TODO: BackSlash") };
    ($($t:tt)*) => {
        compile_error!(concat!("cannot convert \"", $($t)*, "\" to TokenKind"))
    };
}

pub fn keyword(keyword: &'static str) -> impl Parser<&'static str> {
    move |lex| {
        ident_token.flat_map(|t| match &lex.get_code()[t.span] {
            t if t == keyword => Ok(keyword),
            _ => Err(ParseError::NotAnKeyword),
        })(lex)
    }
}

const KEYWORDS: &[&'static str] = &["let", "mut", "rec", "true", "false"];

/// (`p` (,`p`)* `,`? )?
/// this function parses expression until a seperator token of `close` is
/// reached. the `close` token is not consumed!
pub fn comma_chain<T>(p: impl Parser<T> + Copy) -> impl Parser<Vec<T>> {
    p.sep_by0(tok!(',')).has_suffix(tok!(',').opt())
}

// -----------------------

/// [`Expr`] but always returning `()`
#[derive(Debug, Clone)]
pub enum StmtKind {
    /// `let mut rec <name> `(`: <ty>`)?` `(`= <rhs>`)`;`
    Let { is_mut: bool, is_rec: bool, name: Ident, ty: Option<Box<Expr>>, kind: LetKind },

    /// `<expr>;`
    Semicolon(Box<Expr>),
}

#[derive(Debug, Clone)]
pub struct Stmt {
    kind: StmtKind,
    #[allow(unused)]
    span: Span,
}

impl Stmt {
    /// This generates new Code and doesn't read the orignal code!
    pub fn to_text(&self) -> String {
        match &self.kind {
            StmtKind::Let { is_mut, is_rec, name, ty, kind } => {
                format!(
                    "let{}{} {}{}{};",
                    if *is_mut { " mut" } else { "" },
                    if *is_rec { " rec" } else { "" },
                    name.to_text(),
                    ty.as_ref().map(|ty| format!(":{}", ty.to_text())).unwrap_or_default(),
                    if let LetKind::Init(rhs) = kind {
                        format!("={}", rhs.to_text())
                    } else {
                        "".to_string()
                    }
                )
            },
            StmtKind::Semicolon(expr) => format!("{};", expr.to_text()),
        }
    }

    pub fn print_tree(&self) {
        let mut lines = TreeLines::default();
        let len = self.to_text().len();
        match &self.kind {
            StmtKind::Let { is_mut, is_rec, name, ty, kind } => {
                lines.write(&format!(
                    "let{}{} ",
                    if *is_mut { " mut" } else { "" },
                    if *is_rec { " rec" } else { "" }
                ));

                lines.scope_next_line(|l| name.write_tree(l));
                lines.write(&format!("{}", "-".repeat(name.span.len())));

                if let Some(ty) = ty {
                    lines.write(":");
                    lines.scope_next_line(|l| ty.write_tree(l));
                    lines.write(&"-".repeat(ty.to_text().len()));
                }

                if let LetKind::Init(rhs) = kind {
                    lines.write("=");
                    lines.scope_next_line(|l| rhs.write_tree(l));
                    lines.write(&"-".repeat(rhs.to_text().len()));
                    lines.write(";");
                }
            },
            StmtKind::Semicolon(expr) => {
                lines.write(&"-".repeat(len - 1));
                lines.write(";");
                lines.next_line();
                expr.write_tree(&mut lines)
            },
        }
        for l in lines.lines {
            println!("| {}", l.0);
        }
    }
}

#[derive(Default)]
pub struct TreeLine(pub String);

impl TreeLine {
    pub fn ensure_len(&mut self, len: usize) {
        let pad = " ".repeat(len.saturating_sub(self.0.len()));
        self.0.push_str(&pad);
    }

    pub fn overwrite(&mut self, offset: usize, text: &str) {
        self.ensure_len(offset + text.len());
        self.0.replace_range(offset..offset + text.len(), text);
    }
}

#[derive(Default)]
pub struct TreeLines {
    lines: Vec<TreeLine>,
    cur_line: usize,
    cur_offset: usize,
}

impl TreeLines {
    pub fn ensure_lines(&mut self, idx: usize) {
        while self.lines.get(idx).is_none() {
            self.lines.push(TreeLine(String::new()))
        }
    }

    pub fn get_cur(&mut self) -> &mut TreeLine {
        self.ensure_lines(self.cur_line);
        self.lines.get_mut(self.cur_line).unwrap()
    }

    pub fn write(&mut self, text: &str) {
        let offset = self.cur_offset;
        self.get_cur().overwrite(offset, text);
        self.cur_offset += text.len();
    }

    pub fn write_minus(&mut self, len: usize) {
        self.write(&"-".repeat(len))
    }

    pub fn scope_next_line(&mut self, f: impl FnOnce(&mut Self)) {
        let state = (self.cur_line, self.cur_offset);
        self.next_line();
        f(self);
        self.cur_line = state.0;
        self.cur_offset = state.1;
    }

    pub fn next_line(&mut self) {
        self.cur_line += 1;
    }

    pub fn prev_line(&mut self) {
        self.cur_line -= 1;
    }

    pub fn set_offset(&mut self, offset: usize) {
        self.cur_offset = offset;
    }
}

impl From<(StmtKind, Span)> for Stmt {
    fn from((kind, span): (StmtKind, Span)) -> Self {
        Stmt { kind, span }
    }
}

pub fn stmt(lex: Lexer<'_>) -> ParseResult<'_, Stmt> {
    let let_ = keyword("let")
        .prefix_of(ident_token.many1())
        .and(tok!(':').prefix_of(expr).opt())
        .and(let_kind)
        .has_suffix(tok!(';')) // ?
        .flat_map_with_data(|((idents, ty), kind), lex| {
            let mut is_mut = false;
            let mut is_rec = false;
            let mut idents = idents.into_iter();
            let name = loop {
                let Some(t) = idents.next() else {
                    return Err(ParseError::MissingLetIdent);
                };
                let text = &lex.get_code()[t.span];
                match text {
                    "rec" if !is_rec => is_rec = true,
                    "mut" if !is_mut => is_mut = true,
                    "let" | "rec" | "mut" => return Err(ParseError::DoubleLetMarker(t)),
                    _ => break Ident::try_from_tok(t, &lex)?,
                }
            };
            if let Some(rem) = Span::multi_join(idents.map(|t| t.span)) {
                return Err(ParseError::TooManyLetIdents(rem));
            }
            let ty = ty.map(Box::new);
            Ok(StmtKind::Let { is_mut, is_rec, name, ty, kind })
        });
    let semicolon = expr.has_suffix(tok!(';')).map(|expr| StmtKind::Semicolon(Box::new(expr)));
    let_.or(semicolon).to_stmt()(lex)
}

#[derive(Debug, Clone)]
pub enum LetKind {
    Decl,
    Init(Box<Expr>),
}

pub fn let_kind(lex: Lexer<'_>) -> ParseResult<'_, LetKind> {
    tok!('=').prefix_of(expr).opt().map(|expr| match expr {
        Some(expr) => LetKind::Init(Box::new(expr)),
        None => LetKind::Decl,
    })(lex)
}

/// grouped and ordered by precedence
///
/// storing all the tokens is inefficient and not needed, but helps with the
/// implementation
#[derive(Debug, Clone)]
pub enum ExprKind {
    Ident,
    /// `[<val>; <count>]`
    /// both for types and literals
    ArrayShort {
        val: Box<Expr>,
        count: Box<Expr>,
    },
    /// `[<expr>, <expr>, ..., <expr>,]`
    ArrayInit {
        elements: Vec<Expr>,
    },
    /// `(<expr>, <expr>, ..., <expr>,)`
    /// both for types and literals
    Tuple {
        elements: Vec<Expr>,
    },
    /// `( <expr> )`
    Parenthesis {
        expr: Box<Expr>,
    },
    Literal(LitKind),
    /// `(<expr>, <expr>: <ty>, ..., <expr>,) -> <body>`
    /// `-> <body>`
    Fn {
        params: Vec<(Expr, Option<Expr>)>,
        body: Box<Expr>,
    },
    /// `struct { ... }`
    StructDef {
        fields: Vec<StructFields>,
    },
    /// `MyStruct { a: <expr>, b, }`
    StructInit {
        fields: Vec<StructFields>,
        span: Span,
    },
    /// `struct(...)`
    TupleStructDef {
        fields: Vec<ExprKind>,
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
    /// `?<ty>`
    OptionShort {
        ty: Box<ExprKind>,
        span: Span,
    },

    /// `{ <stmt>`*` }`
    Block {
        stmts: Vec<Stmt>,
    },

    /// [`expr`] . [`expr`]
    Dot {
        lhs: Box<Expr>,
        rhs: Ident,
    },
    /// [`dot`] : [`dot`]
    Colon {
        lhs: Box<Expr>,
        rhs: Box<Expr>,
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

    /// `<func> < <params> >`
    CompCall {
        func: Box<Expr>,
        args: Vec<Expr>,
    },
    /// [`colon`] `(` [`comma_chain`] ([`expr`]) `)`
    Call {
        func: Box<Expr>,
        args: Vec<Expr>,
    },

    /// examples: `&<expr>`, `- <expr>`
    PreOp {
        kind: PreOpKind,
        expr: Box<Expr>,
        span: Span,
    },
    /// `<lhs> op <lhs>`
    BinOp {
        lhs: Box<Expr>,
        op: BinOpKind,
        rhs: Box<Expr>,
    },
    /// `<lhs> op= <lhs>`
    BinOpAssign {
        lhs: Box<Expr>,
        op: BinOpKind,
        rhs: Box<Expr>,
    },
}

#[derive(Debug, Clone)]
pub struct Expr {
    kind: ExprKind,
    span: Span,
}

impl Expr {
    /// This generates new Code and doesn't read the orignal code!
    pub fn to_text(&self) -> String {
        #[allow(unused)]
        match &self.kind {
            ExprKind::Ident => "X".repeat(self.span.len()),
            ExprKind::ArrayShort { val, count } => {
                format!("[{};{}]", val.to_text(), count.to_text())
            },
            ExprKind::ArrayInit { elements } => format!(
                "[{}]",
                elements
                    .iter()
                    .map(|e| e.to_text())
                    .intersperse(",".to_string())
                    .collect::<String>()
            ),
            ExprKind::Tuple { elements } => format!(
                "({})",
                elements
                    .iter()
                    .map(|e| e.to_text())
                    .intersperse(",".to_string())
                    .collect::<String>()
            ),
            ExprKind::Parenthesis { expr } => format!("({})", expr.to_text()),
            ExprKind::Literal(LitKind::Bool) => "0".repeat(self.span.len()),
            ExprKind::Literal(LitKind::Char) => "0".repeat(self.span.len()),
            ExprKind::Literal(LitKind::BChar) => "0".repeat(self.span.len()),
            ExprKind::Literal(LitKind::Int) => "0".repeat(self.span.len()),
            ExprKind::Literal(LitKind::Float) => "0".repeat(self.span.len()),
            ExprKind::Literal(LitKind::Str) => "0".repeat(self.span.len()),
            ExprKind::Fn { params, body } => panic!(),
            ExprKind::StructDef { fields } => panic!(),
            ExprKind::StructInit { fields, span } => panic!(),
            ExprKind::TupleStructDef { fields, span } => panic!(),
            ExprKind::Union { span } => panic!(),
            ExprKind::Enum { span } => panic!(),
            ExprKind::OptionShort { ty, span } => panic!(),
            ExprKind::Block { stmts } => panic!(),
            ExprKind::Dot { lhs, rhs } => {
                format!("{}.{}", lhs.to_text(), rhs.to_text())
            },
            ExprKind::Colon { lhs, rhs } => {
                format!("{}:{}", lhs.to_text(), rhs.to_text())
            },
            ExprKind::PostOp { kind, expr, span } => panic!(),
            ExprKind::Index { lhs, idx, span } => panic!(),
            ExprKind::CompCall { func, args } => panic!(),
            ExprKind::Call { func, args } => format!(
                "{}({})",
                func.to_text(),
                args.iter()
                    .map(|e| e.to_text())
                    .intersperse(",".to_string())
                    .collect::<String>()
            ),
            ExprKind::PreOp { kind, expr, span } => panic!(),
            ExprKind::BinOp { lhs, op, rhs } => panic!(),
            ExprKind::BinOpAssign { lhs, op, rhs } => panic!(),
        }
    }

    pub fn write_tree(&self, lines: &mut TreeLines) {
        match &self.kind {
            ExprKind::Ident
            | ExprKind::Literal(LitKind::Bool)
            | ExprKind::Literal(LitKind::Char)
            | ExprKind::Literal(LitKind::BChar)
            | ExprKind::Literal(LitKind::Int)
            | ExprKind::Literal(LitKind::Float)
            | ExprKind::Literal(LitKind::Str) => lines.write(&self.to_text()),
            /*
            ExprKind::ArrayShort { val, count } => {
                format!("[{};{}]", val.to_text(), count.to_text())
            },
            ExprKind::ArrayInit { elements } => format!(
                "[{}]",
                elements
                    .iter()
                    .map(|e| e.to_text())
                    .intersperse(",".to_string())
                    .collect::<String>()
            ),
            ExprKind::Tuple { elements } => format!(
                "({})",
                elements
                    .iter()
                    .map(|e| e.to_text())
                    .intersperse(",".to_string())
                    .collect::<String>()
            ),
            ExprKind::Parenthesis { expr } => format!("({})", expr.to_text()),
            ExprKind::Fn { params, body } => panic!(),
            ExprKind::StructDef { fields } => panic!(),
            ExprKind::StructInit { fields, span } => panic!(),
            ExprKind::TupleStructDef { fields, span } => panic!(),
            ExprKind::Union { span } => panic!(),
            ExprKind::Enum { span } => panic!(),
            ExprKind::OptionShort { ty, span } => panic!(),
            ExprKind::Block { stmts } => panic!(),
            */
            ExprKind::Dot { lhs, rhs } => {
                lines.scope_next_line(|l| lhs.write_tree(l));
                lines.write_minus(lhs.to_text().len());
                lines.write(".");
                lines.scope_next_line(|l| rhs.write_tree(l));
                lines.write_minus(rhs.span.len());
            },
            ExprKind::Colon { lhs, rhs } => {
                lines.scope_next_line(|l| lhs.write_tree(l));
                lines.write_minus(lhs.to_text().len());
                lines.write(":");
                lines.scope_next_line(|l| rhs.write_tree(l));
                lines.write_minus(rhs.span.len());
            },
            /*
            ExprKind::PostOp { kind, expr, span } => panic!(),
            ExprKind::Index { lhs, idx, span } => panic!(),
            ExprKind::CompCall { func, args } => panic!(),
            */
            ExprKind::Call { func, args } => {
                lines.scope_next_line(|l| func.write_tree(l));
                lines.write_minus(func.to_text().len());
                lines.write("(");
                lines.write(&format!(
                    "{})",
                    args.iter()
                        .map(|e| e.to_text())
                        .intersperse(",".to_string())
                        .collect::<String>()
                ));
            },
            /*
            ExprKind::PreOp { kind, expr, span } => panic!(),
            ExprKind::BinOp { lhs, op, rhs } => panic!(),
            ExprKind::BinOpAssign { lhs, op, rhs } => panic!(),
            */
            k => panic!("{:?}", k),
        }
    }
}

impl From<(ExprKind, Span)> for Expr {
    fn from((kind, span): (ExprKind, Span)) -> Self {
        Expr { kind, span }
    }
}

pub fn expr(lex: Lexer<'_>) -> ParseResult<'_, Expr> {
    let (mut expr, mut lex) = expr_value(lex)?;
    while let Ok((tail, l)) = expr_tail(lex) {
        lex = l;
        expr = tail.into_expr(expr, &lex);
    }
    Ok((expr, lex))
}

pub fn expr_value(lex: Lexer<'_>) -> ParseResult<'_, Expr> {
    ident.map(Ident::into_expr).or(array).or(tuple_or_paren).or(literal).or(fn_lit)(lex)

    // /// `struct { ... }`
    // StructDef {
    //     fields: Vec<StructFields>,
    // },
    // /// `MyStruct { a: <expr>, b, }`
    // StructInit {
    //     fields: Vec<StructFields>,
    //     span: Span,
    // },
    // /// `struct(...)`
    // TupleStructDef {
    //     fields: Vec<ExprKind>,
    //     span: Span,
    // },
    // /// `union { ... }`
    // Union {
    //     span: Span,
    // },
    // /// `enum { ... }`
    // Enum {
    //     span: Span,
    // },
    // /// `?<ty>`
    // OptionShort {
    //     ty: Box<ExprKind>,
    //     span: Span,
    // },
}

/// [`expr_value`] | `{` ( [`stmt`] )* `}`
pub fn block(lex: Lexer<'_>) -> ParseResult<'_, Expr> {
    let block = tok!('{')
        .prefix_of(stmt.many0())
        .has_suffix(tok!('}'))
        .map(|stmts| ExprKind::Block { stmts })
        .to_expr();
    expr_value.or(block)(lex)
}

#[derive(Debug, Clone)]
pub enum ExprTail {
    Dot(Ident),
    Colon(Box<Expr>),
    Call(Vec<Expr>),
}

pub fn expr_tail(lex: Lexer<'_>) -> ParseResult<'_, ExprTail> {
    dot.or(colon).or(call)(lex)
}

impl ExprTail {
    pub fn into_expr(self, lhs: Expr, lex: &Lexer<'_>) -> Expr {
        match self {
            ExprTail::Dot(i) => {
                let span = lhs.span.join(i.span);
                let kind = ExprKind::Dot { lhs: Box::new(lhs), rhs: i };
                Expr { kind, span }
            },
            ExprTail::Colon(rhs) => {
                let span = lhs.span.join(rhs.span);
                let kind = ExprKind::Colon { lhs: Box::new(lhs), rhs };
                Expr { kind, span }
            },
            ExprTail::Call(args) => {
                let span = lhs.span.join(Span::pos(lex.get_pos()));
                let kind = ExprKind::Call { func: Box::new(lhs), args };
                Expr { kind, span }
            },
        }
    }
}

/// `.` [`ident`]
pub fn dot(lex: Lexer<'_>) -> ParseResult<'_, ExprTail> {
    tok!('.').prefix_of(ident).map(ExprTail::Dot)(lex)
}

/// `:` [`dot`]
///
/// Note: `1:x.y` = `x.y(1)`
/// Otherwise: `1:x.y`
// `            ^^^^^` ERR: "func `1:x` has no prop `y`"
pub fn colon(lex: Lexer<'_>) -> ParseResult<'_, ExprTail> {
    let rhs = ident.map(Ident::into_expr).and(dot.opt()).map_with_data(|(i, t), lex| match t {
        Some(t) => t.into_expr(i, lex),
        None => i,
    });
    tok!(':').prefix_of(rhs).map(|rhs| ExprTail::Colon(Box::new(rhs)))(lex)
}

// /// examples: `<expr>?`, `<expr>.*`
// PostOp {
//     kind: PostOpKind,
//     expr: Box<ExprKind>,
//     span: Span,
// },
// /// `<lhs> [ <idx> ]`
// Index {
//     lhs: Box<ExprKind>,
//     idx: Box<ExprKind>,
//     span: Span,
// },

// /// `<func> < <params> >`
// CompCall {
//     func: Box<ExprKind>,
//     params: Vec<ExprKind>,
//     span: Span,
// },

/// `(` [`comma_chain`] ([`expr`]) `)`
pub fn call(lex: Lexer<'_>) -> ParseResult<'_, ExprTail> {
    tok!('(').prefix_of(comma_chain(expr)).has_suffix(tok!(')')).map(ExprTail::Call)(lex)
}

#[derive(Debug, Clone, Copy)]
pub struct Ident {
    pub span: Span,
}

impl Ident {
    pub fn try_from_tok(t: Token, lex: &Lexer<'_>) -> Result<Ident, ParseError> {
        if KEYWORDS.contains(&&lex.get_code()[t.span]) {
            return Err(ParseError::NotAnIdent);
        }
        Ok(Ident { span: t.span })
    }

    pub fn into_expr(self) -> Expr {
        Expr { kind: ExprKind::Ident, span: self.span }
    }

    pub fn to_text(&self) -> String {
        "X".repeat(self.span.len())
    }

    pub fn write_tree(&self, lines: &mut TreeLines) {
        lines.write(&self.to_text());
    }
}

/// returns an [`Ident`] or a keyword
pub fn ident_token(lex: Lexer<'_>) -> ParseResult<'_, Token> {
    tok(TokenKind::Ident)(lex)
}

/// returns an [`Ident`] but not keywords
pub fn ident(lex: Lexer<'_>) -> ParseResult<'_, Ident> {
    ident_token.flat_map_with_data(Ident::try_from_tok)(lex)
}

/// `[<val>; <count>]`
/// `[<expr>, <expr>, ..., <expr>,]`
pub fn array(lex: Lexer<'_>) -> ParseResult<'_, Expr> {
    let array_semicolon = expr
        .has_suffix(tok!(';'))
        .and(expr)
        .map(|(val, count)| ExprKind::ArrayShort { val: Box::new(val), count: Box::new(count) });
    let array_comma = comma_chain(expr).map(|elements| ExprKind::ArrayInit { elements });
    let content = array_semicolon.or(array_comma);
    tok!('[').prefix_of(content).has_suffix(tok!(']')).to_expr()(lex)
}

/// `(<expr>, <expr>, ..., <expr>,)` -> Tuple
/// `()` -> Tuple
/// `(<expr>)` -> Parenthesis
/// `(<expr>,)` -> Tuple
pub fn tuple_or_paren(lex: Lexer<'_>) -> ParseResult<'_, Expr> {
    let comma_chain = expr.sep_by0(tok!(',')).and(tok!(',').opt());
    tok!('(')
        .prefix_of(comma_chain)
        .has_suffix(tok!(')'))
        .map(|(elements, suffix_comma)| match suffix_comma {
            Some(_) if elements.len() == 1 => {
                let expr = elements.into_iter().next().map(Box::new).expect("len == 1");
                ExprKind::Parenthesis { expr }
            },
            _ => ExprKind::Tuple { elements },
        })
        .to_expr()(lex)
}

pub fn literal(lex: Lexer<'_>) -> ParseResult<'_, Expr> {
    any_tok().flat_map(|t| match t.kind {
        TokenKind::Literal(l) => {
            let kind = ExprKind::Literal(l);
            Ok(Expr { kind, span: t.span })
        },
        _ => Err(ParseError::UnexpectedToken(t)),
    })(lex)
}

/// `(<expr>, <expr>, ..., <expr>,) -> <body>`
/// `-> <body>`
pub fn fn_lit(lex: Lexer<'_>) -> ParseResult<'_, Expr> {
    let arg = expr.and(tok!(':').prefix_of(expr).opt());
    let input = tok!('(').prefix_of(comma_chain(arg)).has_suffix(tok!(')')).opt();
    input
        .has_suffix(tok!("->"))
        .and(expr)
        .map(|(input, body)| ExprKind::Fn {
            params: input.unwrap_or_default(),
            body: Box::new(body),
        })
        .to_expr()(lex)
}

///// `struct { ... }`
//StructDef {
//    fields: Vec<StructFields>,
//},
///// `MyStruct { a: <expr>, b, }`
//StructInit {
//    fields: Vec<StructFields>,
//    span: Span,
//},
///// `struct(...)`
//TupleStructDef {
//    fields: Vec<ExprKind>,
//    span: Span,
//},
///// `union { ... }`
//Union {
//    span: Span,
//},
///// `enum { ... }`
//Enum {
//    span: Span,
//},
///// `?<ty>`
//OptionShort {
//    ty: Box<ExprKind>,
//    span: Span,
//},

// -------------------------

/*
 */

///// `<lhs> . <rhs>`
//Dot {
//    lhs: Box<ExprKind>,
//    rhs: Ident,
//},
///// `<lhs> : <rhs>`
//Colon {
//    lhs: Box<ExprKind>,
//    rhs: Box<ExprKind>,
//},
///// examples: `<expr>?`, `<expr>.*`
//PostOp {
//    kind: PostOpKind,
//    expr: Box<ExprKind>,
//    span: Span,
//},
///// `<lhs> [ <idx> ]`
//Index {
//    lhs: Box<ExprKind>,
//    idx: Box<ExprKind>,
//    span: Span,
//},

///// `<func> < <params> >`
//CompCall {
//    func: Box<ExprKind>,
//    params: Vec<ExprKind>,
//    span: Span,
//},
///// `<func> ( <params> )`
//Call {
//    func: Box<ExprKind>,
//    params: Vec<ExprKind>,
//    span: Span,
//},

///// examples: `&<expr>`, `- <expr>`
//PreOp {
//    kind: PreOpKind,
//    expr: Box<ExprKind>,
//    span: Span,
//},
///// `<lhs> op <lhs>`
//BinOp {
//    lhs: Box<ExprKind>,
//    op: BinOpKind,
//    rhs: Box<ExprKind>,
//},
///// `<lhs> op= <lhs>`
//BinOpAssign {
//    lhs: Box<ExprKind>,
//    op: BinOpKind,
//    rhs: Box<ExprKind>,
//},

// -----------------------

// /// `struct { ... }`
// /// `MyStruct { a: <expr>, b, }`
// /// `struct(...)`
// /// `union { ... }`
// /// `enum { ... }`
// /// `?<ty>`

#[allow(unused)]
pub struct Pattern {
    kind: ExprKind, // TODO: own kind enum
    span: Span,
}

#[allow(unused)]
#[derive(Debug, Clone)]
pub struct StructFields {
    name_ident: ExprKind,
    type_: ExprKind,
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
