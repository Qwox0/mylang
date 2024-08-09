//use self::debug::TreeLines;
//use self::util::TupleMap0;
use self::{
    debug::TreeLines,
    lexer::Code,
    parser::{always, err, f, peek, PErrKind, PResult, ParseError, Parser},
    result_with_fatal::ResultWithFatal,
};
use crate::parser::parser::choice;
use lexer::{Lexer, Span, Token, TokenKind};
use parser::opt;
use result_with_fatal::ResultWithFatal::*;
use std::str::FromStr;

pub mod lexer;
pub mod parser;
pub mod result_with_fatal;
mod util;

#[derive(Clone)]
pub struct StmtIter<'c> {
    lex: Lexer<'c>,
}

impl<'c> Iterator for StmtIter<'c> {
    type Item = ResultWithFatal<Expr, ParseError>;

    fn next(&mut self) -> Option<Self::Item> {
        use ResultWithFatal::*;
        Some(match stmt().run(self.lex) {
            Ok((stmt, l)) => {
                self.lex = l;
                Ok(stmt)
            },
            Err(ParseError { kind: PErrKind::NoInput, .. }) => return None,
            Err(e) => Err(e),
            Fatal(e) => Fatal(e),
        })
    }
}

impl<'c> StmtIter<'c> {
    /// Parses top-level items until the end of the [`Code`] or until an
    /// [`PError`] occurs.
    pub fn parse(code: &'c Code) -> Self {
        let mut lex = Lexer::new(code);
        if let ResultWithFatal::Ok((_, new_lex)) = ws0().run(lex) {
            lex = new_lex;
        }
        Self { lex }
    }
}

/// A top-level item
pub struct Item<'c> {
    pub markers: DeclMarkers,
    pub ident: Ident,
    pub ty: Option<Box<Expr>>,
    pub value: Box<Expr>,
    pub code: &'c Code,
}

/// no whitespace skipping after the token
pub fn tok_matches(f: impl Fn(&TokenKind) -> bool + 'static) -> Parser<Token> {
    Parser::new(move |lex| {
        match lex.peek() {
            Some(t) if f(&t.kind) => Ok((t, lex.advanced())),
            //Some(t) => Err(PError::UnexpectedToken(t)),
            Some(t) => err!(UnexpectedToken(t), lex.pos_span()),
            None => err!(NoInput, lex.pos_span()),
        }
    })
}

pub fn ws0() -> Parser<Vec<Token>> {
    tok_matches(|t| t.is_ignored()).many0()
}

pub fn ws1() -> Parser<Vec<Token>> {
    tok_matches(|t| t.is_ignored()).many1().context("ws1")
}

pub fn any_tok() -> Parser<Token> {
    tok_matches(|_| true)
}

pub fn tok(kind: TokenKind) -> Parser<Token> {
    tok_matches(move |lex| *lex == kind)
}

macro_rules! tok {
    ('(') => { tok(TokenKind::OpenParenthesis).context("Token: `(`") };
    (')') => { tok(TokenKind::CloseParenthesis).context("Token: `)`") };
    ('[') => { tok(TokenKind::OpenBracket).context("Token: `[`") };
    (']') => { tok(TokenKind::CloseBracket).context("Token: `]`") };
    ('{') => { tok(TokenKind::OpenBrace).context("Token: `{`") };
    ('}') => { tok(TokenKind::CloseBrace).context("Token: `}`") };
    (=) => { tok(TokenKind::Eq).context("Token: `=`") };
    (==) => { tok(TokenKind::EqEq).context("Token: `==`") };
    (=>) => { tok(TokenKind::FatArrow).context("Token: `=>`") };
    (!) => { tok(TokenKind::Bang).context("Token: `!`") };
    (!=) => { tok(TokenKind::BangEq).context("Token: `!=`") };
    (<) => { tok(TokenKind::Lt).context("Token: `<`") };
    (<=) => { tok(TokenKind::LtEq).context("Token: `<=`") };
    (<<) => { tok(TokenKind::LtLt).context("Token: `<<`") };
    (<<=) => { tok(TokenKind::LtLtEq).context("Token: `<<=`") };
    (>) => { tok(TokenKind::Gt).context("Token: `>`") };
    (>=) => { tok(TokenKind::GtEq).context("Token: `>=`") };
    (>>) => { tok(TokenKind::GtGt).context("Token: `>>`") };
    (>>=) => { tok(TokenKind::GtGtEq).context("Token: `>>=`") };
    (+) => { tok(TokenKind::Plus).context("Token: `+`") };
    (+=) => { tok(TokenKind::PlusEq).context("Token: `+=`") };
    (-) => { tok(TokenKind::Minus).context("Token: `-`") };
    (-=) => { tok(TokenKind::MinusEq).context("Token: `-=`") };
    (->) => { tok(TokenKind::Arrow).context("Token: `->`") };
    (*) => { tok(TokenKind::Asterisk).context("Token: `*`") };
    (*=) => { tok(TokenKind::AsteriskEq).context("Token: `*=`") };
    (/) => { tok(TokenKind::Slash).context("Token: `/`") };
    (/=) => { tok(TokenKind::SlashEq).context("Token: `/=`") };
    (%) => { tok(TokenKind::Percent).context("Token: `%`") };
    (%=) => { tok(TokenKind::PercentEq).context("Token: `%=`") };
    (&) => { tok(TokenKind::Ampersand).context("Token: `&`") };
    (&&) => { tok(TokenKind::AmpersandAmpersand).context("Token: `&&`") };
    (&=) => { tok(TokenKind::AmpersandEq).context("Token: `&=`") };
    (|) => { tok(TokenKind::Pipe).context("Token: `|`") };
    (||) => { tok(TokenKind::PipePipe).context("Token: `||`") };
    (|=) => { tok(TokenKind::PipeEq).context("Token: `|=`") };
    (^) => { tok(TokenKind::Caret).context("Token: `^`") };
    (^=) => { tok(TokenKind::CaretEq).context("Token: `^=`") };
    (.) => { tok(TokenKind::Dot).context("Token: `.`") };
    (..) => { tok(TokenKind::DotDot).context("Token: `..`") };
    (..=) => { tok(TokenKind::DotDotEq).context("Token: `..=`") };
    (.*) => { tok(TokenKind::DotAsterisk).context("Token: `.*`") };
    (.&) => { tok(TokenKind::DotAmpersand).context("Token: `.&`") };
    (,) => { tok(TokenKind::Comma).context("Token: `,`") };
    (:) => { tok(TokenKind::Colon).context("Token: `:`") };
    (::) => { tok(TokenKind::ColonColon).context("Token: `::`") };
    (:=) => { tok(TokenKind::ColonEq).context("Token: `:=`") };
    (;) => { tok(TokenKind::Semicolon).context("Token: `;`") };
    (?) => { tok(TokenKind::Question).context("Token: `?`") };
    (#) => { tok(TokenKind::Pound).context("Token: `#`") };
    ($) => { tok(TokenKind::Dollar).context("Token: `$`") };
    (@) => { tok(TokenKind::At).context("Token: `@`") };
    (~) => { tok(TokenKind::Tilde).context("Token: `~`") };
    ('\\') => { compile_error!(TODO: BackSlash) };
    ($($t:tt)*) => {
        compile_error!(concat!("cannot convert \"", stringify!($($t)*), "\" to TokenKind"))
    };
}

pub fn any_keyword() -> Parser<Keyword> {
    ident_token()
        .flat_map(move |t, lex| {
            ResultWithFatal::from_res(
                lex.get_code()[t.span].parse().map_err(|e| ParseError::new(e, lex.pos_span())),
            )
        })
        .context("any_keyword")
}

pub fn keyword(keyword: Keyword) -> Parser<Keyword> {
    let text = keyword.as_str();
    ident_token()
        .flat_map(move |t, lex| match &lex.get_code()[t.span] {
            t if t == text => Ok(keyword),
            _ => err!(NotAKeyword, lex.pos_span()),
        })
        .context(format!("keyword: {:?}", keyword))
}

macro_rules! keywords {
    ( $($enum_variant:ident = $text:literal),* $(,)? ) => {
        #[derive(Debug, Clone, Copy)]
        pub enum Keyword {
            $($enum_variant),*
        }

        impl FromStr for Keyword {
            type Err = PErrKind;

            fn from_str(s: &str) -> Result<Self, Self::Err> {
                match s {
                    $($text => Result::Ok(Keyword::$enum_variant),)*
                    _ => Result::Err(PErrKind::NotAKeyword)
                }
            }
        }

        impl Keyword {
            fn as_str(self) -> &'static str {
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
    Struct = "struct",
    Union = "union",
    Enum = "enum",
    Unsafe = "unsafe",
    True = "true",
    False = "false",
    If = "if",
    Else = "else",
    Match = "match",
    For = "for",
    While = "while",
}

/// (`p` (,`p`)* )?
pub fn comma_chain_no_trailing_comma<T: 'static>(p: Parser<T>) -> Parser<Vec<T>> {
    p.ws0().sep_by0(tok!(,).ws0())
}

/// (`p` (,`p`)* `,`? )?
/// this function parses expression until a seperator token of `close` is
/// reached. the `close` token is not consumed!
pub fn comma_chain<T: 'static>(p: Parser<T>) -> Parser<Vec<T>> {
    comma_chain_no_trailing_comma(p).and_l(opt(tok!(,)))
}

// -----------------------

/*
/// [`Expr`] but always returning `()`
#[derive(Debug, Clone)]
pub enum StmtKind {

    /// `<expr>;`
    Semicolon(Box<Expr>),

    Expr(Box<Expr>),
}

#[derive(Debug, Clone)]
pub struct Stmt {
    pub kind: StmtKind,
    #[allow(unused)]
    pub span: Span,
}

impl From<(StmtKind, Span)> for Stmt {
    fn from((kind, span): (StmtKind, Span)) -> Self {
        Stmt { kind, span }
    }
}

pub fn stmt() -> Parser<Stmt> {
    let semicolon = expr().map(Box::new).and(opt(tok!(;))).map(|(expr, semi)| {
        if semi.is_some() { StmtKind::Semicolon(expr) } else { StmtKind::Expr(expr) }
    });
    decl().or(semicolon).to_stmt().ws0()
}
*/

pub type Stmt = Expr;

fn wrap_semicolons((mut expr, semicolons): (Expr, Vec<Token>)) -> Expr {
    for semicolon in semicolons {
        let span = expr.span.join(semicolon.span);
        expr = Expr { kind: ExprKind::Semicolon(Box::new(expr)), span };
    }
    expr
}

pub fn stmt() -> Parser<Expr> {
    decl()
        .to_expr()
        .or(expr())
        .ws0()
        .and(tok!(;).ws0().many1())
        .map(wrap_semicolons)
        .context("stmt")
}

/// `mut rec <name>: <ty> = <init>`
/// `mut rec <name>: <ty>`
/// `mut rec <name> := <init>`
///
/// `mut rec <name>: <ty> : <init>`
/// `mut rec <name> :: <init>`
pub fn decl() -> Parser<ExprKind> {
    enum Init {
        Init(Box<Expr>),
        Const(Box<Expr>),
    }

    let eq_init = tok!(=).ws0().and_r_fatal(expr()).map(Box::new).map(Init::Init);
    let colon_init = tok!(:).ws0().and_r_fatal(expr()).map(Box::new).map(Init::Const);
    let colon = tok!(:)
        .ws0()
        .and_r_fatal(ty().ws0().map(Box::new))
        .and(opt(eq_init.or(colon_init)))
        .map(|(ty, init)| match init {
            Some(Init::Init(e)) => DeclKind::WithTy { ty, init: Some(e) },
            Some(Init::Const(init)) => DeclKind::Const { ty: Some(ty), init },
            None => DeclKind::WithTy { ty, init: None },
        })
        .context("decl with type");
    let colon_eq = tok!(:=)
        .ws0()
        .and_r_fatal(expr())
        .map(|init| DeclKind::InferTy { init: Box::new(init) })
        .context("decl infer type");
    let colon_colon = tok!(::)
        .ws0()
        .and_r_fatal(expr())
        .map(|init| DeclKind::Const { ty: None, init: Box::new(init) })
        .context("decl infer type");

    let after_markers = ident().ws0().and(colon.or(colon_eq).or(colon_colon));

    let with_markers = var_decl_markers1().and_fatal(after_markers.clone());
    let without_markers = always!(DeclMarkers::default()).and(after_markers);
    with_markers
        .or(without_markers)
        .map(|(markers, (ident, kind))| ExprKind::Decl { markers, ident, kind })
        .context("variable declaration")
}

#[derive(Debug, Clone, Copy, Default)]
pub struct DeclMarkers {
    is_pub: bool,
    is_mut: bool,
    is_rec: bool,
}

impl DeclMarkers {
    pub fn is_empty(&self) -> bool {
        !(self.is_pub || self.is_mut || self.is_rec)
    }
}

#[derive(Debug, Clone, Copy)]
pub enum VarDeclMarkerKind {
    Pub,
    Mut,
    Rec,
}

pub fn var_decl_markers1() -> Parser<DeclMarkers> {
    choice([
        keyword(Keyword::Mut).map(|_| VarDeclMarkerKind::Mut),
        keyword(Keyword::Rec).map(|_| VarDeclMarkerKind::Rec),
    ])
    .ws1()
    .many1()
    .flat_map(|vec, lex| {
        let mut is_pub = false;
        let mut is_mut = false;
        let mut is_rec = false;
        for i in vec {
            match i {
                VarDeclMarkerKind::Pub if !is_pub => is_pub = true,
                VarDeclMarkerKind::Mut if !is_mut => is_mut = true,
                VarDeclMarkerKind::Rec if !is_rec => is_rec = true,
                m => return err!(DoubleLetMarker(m), lex.pos_span()),
            }
        }
        Ok(DeclMarkers { is_pub, is_mut, is_rec })
    })
    .context("variable declaration markers")
}

#[derive(Debug, Clone)]
pub enum DeclKind {
    /// `<name>: <ty>;`
    /// `<name>: <ty> = <init>;`
    WithTy { ty: Box<Expr>, init: Option<Box<Expr>> },
    /// `<name> := <init>;`
    InferTy { init: Box<Expr> },

    /// `<name>: <ty> : <init>;`
    /// `<name> :: <init>;`
    Const { ty: Option<Box<Expr>>, init: Box<Expr> },
}

impl DeclKind {
    pub fn into_ty_val(self) -> (Option<Box<Expr>>, Option<Box<Expr>>) {
        match self {
            DeclKind::WithTy { ty, init } => (Some(ty), init),
            DeclKind::InferTy { init } => (None, Some(init)),
            DeclKind::Const { ty, init } => (ty, Some(init)),
        }
    }
}

#[derive(Debug, Clone)]
pub enum ExprKind {
    Ident,
    Literal(LitKind),

    /// `[<val>; <count>]`
    /// both for types and literals
    ArraySemi {
        val: Box<Expr>,
        count: Box<Expr>,
    },
    /// `[<expr>, <expr>, ..., <expr>,]`
    ArrayComma {
        elements: Vec<Expr>,
    },
    /// `(<expr>, <expr>, ..., <expr>,)`
    /// both for types and literals
    Tuple {
        elements: Vec<Expr>,
    },
    /// `(<ident>, <ident>: <ty>, ..., <ident>,) -> <type> { <body> }`
    /// `(<ident>, <ident>: <ty>, ..., <ident>,) -> <body>`
    /// `-> <type> { <body> }`
    /// `-> <body>`
    Fn {
        params: Vec<(Ident, Option<Type>)>,
        ret_type: Option<Box<Type>>,
        body: Box<Expr>,
    },
    /// `( <expr> )`
    Parenthesis {
        expr: Box<Expr>,
    },
    /// `{ <stmt>`*` }`
    Block {
        stmts: Vec<Expr>,
    },

    /// `struct { a: int, b: String, c: (u8, u32) }`
    StructDef(Vec<(Ident, Type)>),
    /// `MyStruct { a: <expr>, b, }` or
    /// `MyStruct { a = <expr>, b, }` ?
    StructInit {
        name: Ident,
        fields: Vec<(Ident, Option<Expr>)>,
    },
    /// `struct(...)`
    TupleStructDef(Vec<Type>),
    /// `union { ... }`
    Union {},
    /// `enum { ... }`
    Enum {},
    /// `?<ty>`
    OptionShort(Box<Expr>),

    /// [`expr`] . [`expr`]
    Dot {
        lhs: Box<Expr>,
        rhs: Ident,
    },
    /// examples: `<expr>?`, `<expr>.*`
    PostOp {
        kind: PostOpKind,
        expr: Box<Expr>,
    },
    /// `<lhs> [ <idx> ]`
    Index {
        lhs: Box<Expr>,
        idx: Box<Expr>,
    },

    /*
    /// `<func> < <params> >`
    CompCall {
        func: Box<Expr>,
        args: Vec<Expr>,
    },
    */
    /// [`colon`] `(` [`comma_chain`] ([`expr`]) `)`
    Call {
        func: Box<Expr>,
        args: Vec<Expr>,
    },

    /// examples: `&<expr>`, `- <expr>`
    PreOp {
        kind: PreOpKind,
        expr: Box<Expr>,
    },
    /// `<lhs> op <lhs>`
    BinOp {
        lhs: Box<Expr>,
        op: BinOpKind,
        rhs: Box<Expr>,
    },
    /// `<lhs> = <lhs>`
    Assign {
        lhs: Box<LValue>,
        rhs: Box<Expr>,
    },
    /// `<lhs> op= <lhs>`
    BinOpAssign {
        lhs: Box<LValue>,
        op: BinOpKind,
        rhs: Box<Expr>,
    },

    /// variable declaration (and optional initialization)
    /// `mut rec <name>: <ty>`
    /// `mut rec <name>: <ty> = <init>`
    /// `mut rec <name> := <init>`
    /// `mut rec <name>: <ty> : <init>`
    /// `mut rec <name> :: <init>`
    Decl {
        markers: DeclMarkers,
        ident: Ident,
        kind: DeclKind,
    },

    If {
        condition: Box<Expr>,
        then_body: Box<Expr>,
        else_body: Option<Box<Expr>>,
    },

    Semicolon(Box<Expr>),
}

#[derive(Debug, Clone)]
pub struct Expr {
    pub kind: ExprKind,
    pub span: Span,
}

impl From<(ExprKind, Span)> for Expr {
    fn from((kind, span): (ExprKind, Span)) -> Self {
        Expr { kind, span }
    }
}

/// doesn't parse `;`!
pub fn expr() -> Parser<Expr> {
    /*if_().or(assign())*/ assign().or(rvalue()).context("expression")
}

pub type Type = Expr;

pub fn ty() -> Parser<Type> {
    rvalue().context("type")
}

pub fn if_() -> Parser<Expr> {
    let expr_parser = ws1().and_r(expr().map(Box::new));
    keyword(Keyword::If)
        .and_r_fatal(expr_parser.clone())
        .and_fatal(expr_parser.clone())
        .and(opt(ws1().and(keyword(Keyword::Else)).and_r_fatal(expr_parser)))
        .map(|((condition, then_body), else_body)| ExprKind::If { condition, then_body, else_body })
        .to_expr()
}

#[derive(Debug, Clone, Copy)]
pub enum BinOpKind {
    /// `*`, `*=`
    Mul,
    /// `/`, `/=`
    Div,
    /// `%`, `%=`
    Mod,

    /// `+`, `+=`
    Add,
    /// `-`, `-=`
    Sub,

    /// `<<`, `<<=`
    ShiftL,
    /// `>>`, `>>=`
    ShiftR,

    /// `&`, `&=`
    BitAnd,

    /// `^`, `^=`
    BitXor,

    /// `|`, `|=`
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

impl BinOpKind {
    pub fn to_binop_text(self) -> &'static str {
        match self {
            BinOpKind::Mul => "*",
            BinOpKind::Div => "/",
            BinOpKind::Mod => "%",
            BinOpKind::Add => "+",
            BinOpKind::Sub => "-",
            BinOpKind::ShiftL => "<<",
            BinOpKind::ShiftR => ">>",
            BinOpKind::BitAnd => "&",
            BinOpKind::BitXor => "^",
            BinOpKind::BitOr => "|",
            BinOpKind::Eq => "==",
            BinOpKind::Ne => "!=",
            BinOpKind::Lt => "<",
            BinOpKind::Le => "<=",
            BinOpKind::Gt => ">",
            BinOpKind::Ge => ">=",
            BinOpKind::And => "&&",
            BinOpKind::Or => "||",
            BinOpKind::Range => "..",
            BinOpKind::RangeInclusive => "..=",
        }
    }

    pub fn to_binop_assign_text(self) -> &'static str {
        match self {
            BinOpKind::Mul => "*=",
            BinOpKind::Div => "/=",
            BinOpKind::Mod => "%=",
            BinOpKind::Add => "+=",
            BinOpKind::Sub => "-=",
            BinOpKind::ShiftL => "<<=",
            BinOpKind::ShiftR => ">>=",
            BinOpKind::BitAnd => "&=",
            BinOpKind::BitXor => "^=",
            BinOpKind::BitOr => "|=",
            k => panic!("Unexpected binop kind: {:?}", k),
        }
    }
}

macro_rules! choose_bin_op {
    ( $first_op:tt : $first_kind:expr $( , $op:tt : $kind:expr )* $(,)? ) => {
        tok!($first_op).map(|_| $first_kind)
        $(
            .or(tok!($op).map(|_| $kind))
        )*
        .ws0()
    };
}

/// for [`ExprKind::Assign`] and [`ExprKind::BinOpAssign`]
pub fn assign() -> Parser<Expr> {
    let assign = tok!(=)
        .context("`=`")
        .ws0()
        .and_r_fatal(rvalue())
        .context("assign")
        .map(|rhs| (None, rhs));
    let bin_op_assign = choose_bin_op!(
        += : BinOpKind::Add,
        -= : BinOpKind::Sub,
        *= : BinOpKind::Mul,
        /= : BinOpKind::Div,
        %= : BinOpKind::Mod,
        <<= : BinOpKind::ShiftL,
        >>= : BinOpKind::ShiftR,
        &= : BinOpKind::BitAnd,
        ^= : BinOpKind::BitXor,
        |= : BinOpKind::BitOr,
    )
    .ws0()
    .and_fatal(rvalue())
    .context("binop assign")
    .map(|(op, rhs)| (Some(op), rhs));

    lvalue()
        .ws0()
        .and(assign.or(bin_op_assign))
        .map(|(lhs, (op, rhs))| {
            let lhs = Box::new(lhs);
            let rhs = Box::new(rhs);
            match op {
                None => ExprKind::Assign { lhs, rhs },
                Some(op) => ExprKind::BinOpAssign { lhs, op, rhs },
            }
        })
        .to_expr()
}

/// TODO
pub type LValue = Ident;

/// TODO
pub fn lvalue() -> Parser<Ident> {
    ident().context("lvalue")
}

pub fn rvalue() -> Parser<Expr> {
    bin_op().context("rvalue")
}

pub fn bin_op() -> Parser<Expr> {
    fn reduce_bin_op(lhs: Expr, op: BinOpKind, rhs: Expr) -> Expr {
        let span = lhs.span.join(rhs.span);
        let lhs = Box::new(lhs);
        let rhs = Box::new(rhs);
        let kind = ExprKind::BinOp { lhs, op, rhs };
        Expr { kind, span }
    }

    macro_rules! op_chain {
        ( $sub_term:expr => $( $op:tt : $kind:expr ),+ $(,)? ) => {
            $sub_term.ws0().sep_reduce1(choose_bin_op!($( $op : $kind ),+), reduce_bin_op)
        };
    }

    let mul_chain = op_chain!(f(factor) => *: BinOpKind::Mul, /: BinOpKind::Div, %: BinOpKind::Mod);
    let add_chain = op_chain!(mul_chain => +: BinOpKind::Add, -: BinOpKind::Sub);
    let shift_chain = op_chain!(add_chain => <<: BinOpKind::ShiftL, >>: BinOpKind::ShiftR);
    let bit_and_chain = op_chain!(shift_chain => &: BinOpKind::BitAnd);
    let bit_xor_chain = op_chain!(bit_and_chain => ^: BinOpKind::BitXor);
    let bit_or_chain = op_chain!(bit_xor_chain => ^: BinOpKind::BitOr);
    let cmp_chain = op_chain!(bit_or_chain =>
        == : BinOpKind::Eq,
        != : BinOpKind::Ne,
        < : BinOpKind::Lt,
        <= : BinOpKind::Le,
        > : BinOpKind::Gt,
        >= : BinOpKind::Ge,
    );
    let and_chain = op_chain!(cmp_chain => &&: BinOpKind::And);
    let or_chain = op_chain!(and_chain => &&: BinOpKind::Or);

    let range_tail = choose_bin_op!(..: BinOpKind::Range, ..= : BinOpKind::RangeInclusive)
        .and_fatal(or_chain.clone());

    or_chain.and(opt(range_tail)).map(|(lhs, range_tail)| match range_tail {
        Some((range, rhs)) => {
            let span = lhs.span.join(rhs.span);
            let kind = ExprKind::BinOp { lhs: Box::new(lhs), op: range, rhs: Box::new(rhs) };
            Expr { kind, span }
        },
        None => lhs,
    })
}

pub type Factor = Expr;

pub fn factor(lex: Lexer<'_>) -> PResult<'_, Factor> {
    let (mut factor, mut lex) = factor_start().ws0().context("factor").run(lex)?;
    loop {
        match factor_ext().ws0().run(lex) {
            Err(ParseError { kind: PErrKind::NoInput, .. }) | Ok((FactorExt::None, _)) => {
                return Ok((factor, lex));
            },
            Ok((ext, l)) => {
                lex = l;
                factor = ext.extend(factor, &lex);
            },
            Err(e) => return Err(e),
            Fatal(e) => return Fatal(e),
        }
    }
}

pub fn factor_start() -> Parser<Factor> {
    // `[...]`, `(...)`, `(...) -> ...` or `{...}`
    let group = expr_bracket().or(expr_paren()).or(block()).context("group");
    // `-> ...`
    let special_fn = tail_function();
    let custom_type_def = custom_struct_def()
        .or(custom_union_def())
        .or(custom_enum_def())
        .or(option_short())
        .context("custom_type_def");

    let other = group.or(special_fn).or(custom_type_def).or(struct_init());
    factor_start_ident().or(literal()).or(other.to_expr())
}

/// returns an [`Ident`] or a keyword
pub fn ident_token() -> Parser<Token> {
    tok(TokenKind::Ident)
}

/// returns an [`Ident`] but not keywords
pub fn ident() -> Parser<Ident> {
    ident_token().flat_map(Ident::try_from_tok).context("ident")
}

/// parses an expression starting with an [`ident`]
pub fn factor_start_ident() -> Parser<Expr> {
    ident()
        .ws0()
        .and_fatal(opt(fn_tail().context("one param fn")))
        .map(|(ident, fn_)| match fn_ {
            Some((ret_type, body)) => {
                let span = ident.span.join(body.span);
                let kind = ExprKind::Fn {
                    params: vec![(ident, None)],
                    ret_type: ret_type.map(Box::new),
                    body: Box::new(body),
                };
                Expr { kind, span }
            },
            None => ident.into_expr(),
        })
        .context("ident")
}

pub fn literal() -> Parser<Expr> {
    any_tok()
        .flat_map(|t, lex| match t.kind {
            TokenKind::Literal(l) => {
                let kind = ExprKind::Literal(l);
                Ok(Expr { kind, span: t.span })
            },
            _ => err!(UnexpectedToken(t), lex.pos_span()),
        })
        .context("literal")
}

/// parses an expression starting with `(`
pub fn expr_paren() -> Parser<ExprKind> {
    let param = ident().ws0().and(opt(tok!(:).ws0().and_r_fatal(ty()))).context("fn param");
    let params = comma_chain(param).ws0().and_l(tok!(')'));
    let fn_ = params.ws0().and(fn_tail()).map(|(params, (ret_type, body))| {
        let ret_type = ret_type.map(Box::new);
        ExprKind::Fn { params, ret_type, body: Box::new(body) }
    });

    // `(<expr>, <expr>, ..., <expr>,)` = Tuple
    // `()` = Tuple
    // `(<expr>)` = Parenthesis
    // `(<expr>,)` = Tuple
    let tuple_or_paren = comma_chain_no_trailing_comma(expr())
        .and(opt(tok!(,).ws0()))
        .map(|(elements, suffix_comma)| match suffix_comma {
            None if elements.len() == 1 => {
                let expr = elements.into_iter().next().map(Box::new).expect("len == 1");
                ExprKind::Parenthesis { expr }
            },
            _ => ExprKind::Tuple { elements },
        })
        .and_l_fatal(tok!(')'))
        .context("( ... )");

    tok!('(').ws0().and_r_fatal(fn_.or(tuple_or_paren)).ws0()
}

/// `-> expr`
/// `-> type { expr }`
/// Parser returns the type and body as expressions
pub fn fn_tail() -> Parser<(Option<Expr>, Expr)> {
    let rhs = ty()
        .ws0()
        .and(opt(block().to_expr()))
        .map(|(ty, block)| {
            match block {
                Some(block) => (Some(ty), block),
                // types can be return values:
                None => (None, ty),
                //None => (Some(ty.clone()), ty),
            }
        })
        .or(expr().map(|body| (None, body)))
        .context("function body");
    tok!(->).ws0().and_r_fatal(rhs).context("function")
}

/// `-> { ... }`
pub fn tail_function() -> Parser<ExprKind> {
    fn_tail()
        .map(|(ret_type, body)| {
            let ret_type = ret_type.map(Box::new);
            ExprKind::Fn { params: vec![], ret_type, body: Box::new(body) }
        })
        .context("tail function")
}

/// parses an expression starting with `{`
pub fn block() -> Parser<ExprKind> {
    let item = decl().to_expr().or(expr()).context("block item");
    let stmts = item
        .clone()
        .ws0()
        .and(tok!(;).ws0().many1())
        .map(wrap_semicolons)
        .many0()
        .and(opt(item.and(tok!(;).ws0().many0()).map(wrap_semicolons)))
        .ws0()
        .map(|(mut stmts, ret_expr)| {
            if let Some(ret_expr) = ret_expr {
                stmts.push(ret_expr);
            }
            ExprKind::Block { stmts }
        });
    tok!('{').ws0().and_r_fatal(stmts).and_l_fatal(tok!('}')).context("{ ... }")
}

/// parses an expression starting with `[`
pub fn expr_bracket() -> Parser<ExprKind> {
    let array_semi = expr()
        .and_l(tok!(;).ws0())
        .and_fatal(expr())
        .map(|(val, count)| ExprKind::ArraySemi { val: Box::new(val), count: Box::new(count) });
    let array_comma = comma_chain(expr()).map(|elements| ExprKind::ArrayComma { elements });
    let kind = array_semi.or(array_comma);
    tok!('[')
        .ws0()
        .and_r_fatal(kind.ws0())
        .and_l_fatal(tok!(']'))
        .context("[ ... ]")
}

/// `struct { a: int, b: String, c: (u8, u32) }`
/// `struct(int, String, (u8, u32))`
pub fn custom_struct_def() -> Parser<ExprKind> {
    let struct_field = ident().ws0().and_l(tok!(:)).ws0().and_fatal(expr());
    let struct_fields = comma_chain(struct_field);
    let struct_ = tok!('{')
        .and_r_fatal(struct_fields)
        .and_l_fatal(tok!('}'))
        .map(ExprKind::StructDef);
    let tuple_struct = tok!('(')
        .and_r_fatal(comma_chain(expr()))
        .and_l_fatal(tok!(')'))
        .map(ExprKind::TupleStructDef);
    keyword(Keyword::Struct).ws0().and_r_fatal(struct_.or(tuple_struct))
}

/// TODO
pub fn custom_union_def() -> Parser<ExprKind> {
    let union_ = f(|lex| Ok((ExprKind::Union {}, lex)));
    keyword(Keyword::Union).ws0().and_r_fatal(union_)
}

/// TODO
pub fn custom_enum_def() -> Parser<ExprKind> {
    let enum_ = f(|lex| Ok((ExprKind::Enum {}, lex)));
    keyword(Keyword::Enum).ws0().and_r_fatal(enum_)
}

/// `MyStruct { a: <expr>, b, }` or
/// `MyStruct { a = <expr>, b, }` ?
pub fn struct_init() -> Parser<ExprKind> {
    // TODO: `:` or `=`
    let init_field_value = tok!(:).or(tok!(=)).ws0().and_r_fatal(expr());
    let init_field = ident().ws0().and(opt(init_field_value));
    ident()
        .ws0()
        .and_l(tok!('{'))
        .ws0()
        .and_fatal(init_field.many0())
        .ws0()
        .and_l_fatal(tok!('}'))
        .map(|(name, fields)| ExprKind::StructInit { name, fields })
        .context("struct init")
}

/// `?<ty>`
pub fn option_short() -> Parser<ExprKind> {
    tok!(?)
        .ws0()
        .and_r_fatal(ty())
        .map(Box::new)
        .map(ExprKind::OptionShort)
        .context("option short")
}

/// Anything that count trail a [`factor`] and extend it.
///
/// # Example
///
/// * small factor:  `1`
/// * extension 1:   ` .add`    -> [`FactorTrail::Dot`]
/// * extension 2:   `     (1)` -> [`FactorTrail::Call`]
/// * bigger factor: `1.add(1)`
#[derive(Debug, Clone)]
pub enum FactorExt {
    None,

    Dot(Ident),
    //Colon(Box<Expr>),
    //CompCall(Vec<Expr>),
    Call(Vec<Expr>),
    Index(Box<Expr>),

    PostOp(PostOpKind),
}

impl FactorExt {
    pub fn extend(self, lhs: Factor, lex: &Lexer<'_>) -> Expr {
        let span_start = lhs.span.start;
        let new_expr = |span_end, kind| {
            let span = Span::new(span_start, span_end);
            Expr { kind, span }
        };

        match self {
            FactorExt::None => lhs,

            FactorExt::Dot(i) => new_expr(i.span.end, ExprKind::Dot { lhs: Box::new(lhs), rhs: i }),
            /*
            FactorExt::Colon(rhs) => {
                new_expr(rhs.span.end, ExprKind::Colon { lhs: Box::new(lhs), rhs })
            },
            FactorExt::CompCall(args) => {
                new_expr(lex.get_pos(), ExprKind::CompCall { func: Box::new(lhs), args })
            },
            */
            FactorExt::Call(args) => {
                new_expr(lex.get_pos(), ExprKind::Call { func: Box::new(lhs), args })
            },

            FactorExt::PostOp(kind) => {
                new_expr(lex.get_pos(), ExprKind::PostOp { kind, expr: Box::new(lhs) })
            },
            FactorExt::Index(idx) => {
                new_expr(lex.get_pos(), ExprKind::Index { lhs: Box::new(lhs), idx })
            },
        }
    }
}

/// All Tokens which could end the factor extension chain.
pub const FACTOR_END: [char; 2] = ['a', 'b'];

pub fn factor_ext() -> Parser<FactorExt> {
    let factor_end = choice([
        tok!(')'),
        tok!(']'),
        tok!('{'),
        tok!('}'),
        tok!(=),
        tok!(==),
        tok!(=>),
        // !
        tok!(!=),
        tok!(<),
        tok!(<=),
        tok!(<<),
        tok!(<<=),
        tok!(>),
        tok!(>=),
        tok!(>>),
        tok!(>>=),
        tok!(+),
        tok!(+=),
        tok!(-),
        tok!(-=),
        // ->
        tok!(*),
        tok!(*=),
        tok!(/),
        tok!(/=),
        tok!(%),
        tok!(%=),
        tok!(&),
        tok!(&&),
        tok!(&=),
        tok!(|),
        tok!(||),
        tok!(|=),
        tok!(^),
        tok!(^=),
        tok!(..),
        tok!(..=),
        tok!(,),
        tok!(;),
        tok!(#),
        tok!($),
        tok!(@),
        tok!(~),
        // \
        // `
    ])
    .map(|_| FactorExt::None);

    let more_post_op = try_().or(force()).map(FactorExt::PostOp);

    dot()
        .or(more_post_op)
        .or(colon())
        .or(call())
        .or(index())
        .ws0()
        .or(peek(factor_end))
        .context("factor extension")
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
    // /// `<expr>.type`
    // TypeOf,
}

/// parses an expression extension starting with `.`
/// ([`FactorExt::Dot`], [`FactorExt::PostOp`])
pub fn dot() -> Parser<FactorExt> {
    let addr_of = tok!(&)
        .and_r_fatal(opt(keyword(Keyword::Mut)))
        .map(|t_mut| if t_mut.is_some() { PostOpKind::AddrMutOf } else { PostOpKind::AddrOf })
        .context("factor.&");
    let deref = tok!(*).map(|_| PostOpKind::Deref).context("factor.&");
    let dot_post_op = addr_of.or(deref).map(FactorExt::PostOp);
    let member_access = ident().map(FactorExt::Dot).context("member");
    tok!(.)
        .and_r_fatal(dot_post_op.or(ws0().and_r(member_access)))
        .context("factor.member")
}

pub fn try_() -> Parser<PostOpKind> {
    tok!(?).map(|_| PostOpKind::Try).context("try")
}

pub fn force() -> Parser<PostOpKind> {
    tok!(!)
        .and_r_fatal(opt(keyword(Keyword::Unsafe)))
        .map(
            |t_unsafe| {
                if t_unsafe.is_some() { PostOpKind::ForceUnsafe } else { PostOpKind::Force }
            },
        )
        .context("force")
}

/// `:` [`dot`]
///
/// Note: `1:x.y` = `x.y(1)`
/// Otherwise: `1:x.y`
// `            ^^^^^` ERR: "func `1:x` has no prop `y`"
pub fn colon() -> Parser<FactorExt> {
    Parser::new(|l| err!(TODO, l.pos_span()))
    /*
    let member_access = ident().ws0().sep_by1(tok!(.)).map_with_lex(|d, lex| {
        let mut idents = d.into_iter();
        let first = idents.next().expect("sep_by1").into_expr();
        idents.map(FactorExt::Dot).fold(first, |lhs, dot_rhs| dot_rhs.extend(lhs, &lex)) // TODO: reference should be to an older lexer
    });
    tok!(:).ws0().and_r_fatal(member_access).map(Box::new).map(FactorExt::Colon)
    */
}

/// `<` [`expr`], [`expr`], ..., [`expr`], `>`
pub fn compcall() -> Parser<FactorExt> {
    Parser::new(|l| err!(TODO, l.pos_span()))
    /*
    tok!(<)
        .and_r_fatal(comma_chain(expr()))
        .and_l_fatal(tok!(>))
        .map(FactorExt::CompCall)
        */
}

/// `(` [`expr`], [`expr`], ..., [`expr`], `)`
pub fn call() -> Parser<FactorExt> {
    tok!('(')
        .and_r_fatal(comma_chain(expr()))
        .and_l_fatal(tok!(')'))
        .map(FactorExt::Call)
}

/// `[` [`expr`] `]`
pub fn index() -> Parser<FactorExt> {
    tok!('[')
        .and_r_fatal(expr())
        .and_l_fatal(tok!(']'))
        .map(Box::new)
        .map(FactorExt::Index)
}

// PostOp { kind: PostOpKind, expr: Box<Expr>, span: Span, },
// /// `<lhs> [ <idx> ]`
// Index { lhs: Box<Expr>, idx: Box<Expr>, span: Span, },
// ----------------------
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

// ----------------------------

#[derive(Debug, Clone, Copy)]
pub struct Ident {
    pub span: Span,
}

impl Ident {
    pub fn try_from_tok(t: Token, lex: &Lexer<'_>) -> ResultWithFatal<Ident, ParseError> {
        if Keyword::from_str(&&lex.get_code()[t.span]).is_ok() {
            return err!(NotAnIdent, lex.pos_span());
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

#[allow(unused)]
pub struct Pattern {
    kind: ExprKind, // TODO: own kind enum
    span: Span,
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

pub mod debug {
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
        pub lines: Vec<TreeLine>,
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

        pub fn get_cur_offset(&self) -> usize {
            self.cur_offset
        }

        pub fn write(&mut self, text: &str) {
            let offset = self.cur_offset;
            self.get_cur().overwrite(offset, text);
            self.cur_offset += text.len();
        }

        pub fn write_minus(&mut self, len: usize) {
            self.write(&"-".repeat(len))
        }

        pub fn scope_next_line<O>(&mut self, f: impl FnOnce(&mut Self) -> O) -> O {
            let state = (self.cur_line, self.cur_offset);
            self.next_line();
            let out = f(self);
            self.cur_line = state.0;
            self.cur_offset = state.1;
            out
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
}

/*
impl Stmt {
    /// This generates new Code and doesn't read the orignal code!
    pub fn to_text(&self) -> String {
        match &self.kind {
            StmtKind::Decl { markers, ident, kind } => {
                let DeclMarkers { is_pub, is_mut, is_rec } = markers;
                format!(
                    "{}{}{}{}{};",
                    if *is_pub { "pub " } else { "" },
                    if *is_mut { "mut " } else { "" },
                    if *is_rec { "rec " } else { "" },
                    ident.to_text(),
                    match kind {
                        DeclKind::WithTy { ty, init: Some(init) } =>
                            format!(": {} = {}", ty.to_text(), init.to_text()),
                        DeclKind::WithTy { ty, init: None } => format!(": {}", ty.to_text()),
                        DeclKind::InferTy { init } => format!(" := {}", init.to_text()),
                        DeclKind::Const { ty: Some(ty), init } =>
                            format!(": {} : {}", ty.to_text(), init.to_text()),
                        DeclKind::Const { ty: None, init } => format!(" :: {}", init.to_text()),
                    },
                )
            },
            StmtKind::Semicolon(expr) => format!("{};", expr.to_text()),
            StmtKind::Expr(expr) => expr.to_text(),
        }
    }

    pub fn write_tree(&self, lines: &mut TreeLines) {
        let len = self.to_text().len();
        match &self.kind {
            StmtKind::Decl { markers, ident, kind } => {
                let DeclMarkers { is_pub, is_mut, is_rec } = markers;
                lines.write(&format!(
                    "{}{}{}",
                    if *is_pub { "pub " } else { "" },
                    if *is_mut { "mut " } else { "" },
                    if *is_rec { "rec " } else { "" }
                ));

                lines.scope_next_line(|l| ident.write_tree(l));
                lines.write_minus(ident.span.len());

                match kind {
                    DeclKind::WithTy { ty, init } => {
                        lines.write(": ");
                        lines.scope_next_line(|l| ty.write_tree(l));
                        lines.write_minus(ty.span.len());
                        if let Some(init) = init {
                            lines.write(" = ");
                            lines.scope_next_line(|l| init.write_tree(l));
                            lines.write_minus(init.span.len());
                        }
                    },
                    DeclKind::InferTy { init } => {
                        lines.write(" := ");
                        lines.scope_next_line(|l| init.write_tree(l));
                        lines.write_minus(init.span.len());
                    },
                    DeclKind::Const { ty: Some(ty), init } => {
                        lines.write(": ");
                        lines.scope_next_line(|l| ty.write_tree(l));
                        lines.write_minus(ty.span.len());
                        lines.write(" : ");
                        lines.scope_next_line(|l| init.write_tree(l));
                        lines.write_minus(init.span.len());
                    },
                    DeclKind::Const { ty: None, init } => {
                        lines.write(" :: ");
                        lines.scope_next_line(|l| init.write_tree(l));
                        lines.write_minus(init.span.len());
                    },
                }
                lines.write(";");
            },
            StmtKind::Semicolon(expr) => {
                lines.scope_next_line(|l| expr.write_tree(l));
                lines.write(&"-".repeat(len - 1));
                lines.write(";");
            },
            StmtKind::Expr(expr) => expr.write_tree(lines),
        }
    }

    pub fn print_tree(&self) {
        let mut lines = TreeLines::default();
        self.write_tree(&mut lines);
        for l in lines.lines {
            println!("| {}", l.0);
        }
    }
}
*/

impl Expr {
    /// This generates new Code and doesn't read the orignal code!
    pub fn to_text(&self) -> String {
        #[allow(unused)]
        match &self.kind {
            ExprKind::Ident => "X".repeat(self.span.len()),
            ExprKind::ArraySemi { val, count } => {
                format!("[{};{}]", val.to_text(), count.to_text())
            },
            ExprKind::ArrayComma { elements } => format!(
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
            ExprKind::Fn { params, ret_type, body } => format!(
                "({})->{}",
                params
                    .iter()
                    .map(|(ident, ty)| {
                        let ty =
                            ty.as_ref().map(|e| format!(":{}", e.to_text())).unwrap_or_default();
                        format!("{}{}", ident.to_text(), ty)
                    })
                    .intersperse(",".to_string())
                    .collect::<String>(),
                {
                    let body = body.to_text();
                    match ret_type {
                        Some(ret_type) => format!("{} {{{}}}", ret_type.to_text(), body),
                        None => body,
                    }
                }
            ),
            ExprKind::StructDef(..) => panic!(),
            ExprKind::StructInit { name, fields } => panic!(),
            ExprKind::TupleStructDef(..) => panic!(),
            ExprKind::Union {} => panic!(),
            ExprKind::Enum {} => panic!(),
            ExprKind::OptionShort(ty) => panic!(),
            ExprKind::Block { stmts } => {
                format!("{{{}}}", stmts.iter().map(|a| a.to_text()).collect::<String>())
            },
            ExprKind::Dot { lhs, rhs } => {
                format!("{}.{}", lhs.to_text(), rhs.to_text())
            },
            //ExprKind::Colon { lhs, rhs } => { format!("{}:{}", lhs.to_text(), rhs.to_text()) },
            ExprKind::PostOp { kind, expr } => panic!(),
            ExprKind::Index { lhs, idx } => panic!(),
            //ExprKind::CompCall { func, args } => panic!(),
            ExprKind::Call { func, args } => format!(
                "{}({})",
                func.to_text(),
                args.iter()
                    .map(|e| e.to_text())
                    .intersperse(",".to_string())
                    .collect::<String>()
            ),
            ExprKind::PreOp { kind, expr } => panic!(),
            ExprKind::BinOp { lhs, op, rhs } => {
                format!("{}{}{}", lhs.to_text(), op.to_binop_text(), rhs.to_text())
            },
            ExprKind::Assign { lhs, rhs } => todo!(),
            ExprKind::BinOpAssign { lhs, op, rhs } => {
                format!("{}{}{}", lhs.to_text(), op.to_binop_assign_text(), rhs.to_text())
            },
            ExprKind::If { condition, then_body, else_body } => {
                format!(
                    "if {} {{ {} }}{}",
                    condition.to_text(),
                    then_body.to_text(),
                    else_body
                        .as_ref()
                        .map(|e| format!(" else {}", e.to_text()))
                        .unwrap_or_default()
                )
            },
            ExprKind::Decl { markers, ident, kind } => {
                let DeclMarkers { is_pub, is_mut, is_rec } = markers;
                format!(
                    "{}{}{}{}{};",
                    if *is_pub { "pub " } else { "" },
                    if *is_mut { "mut " } else { "" },
                    if *is_rec { "rec " } else { "" },
                    ident.to_text(),
                    match kind {
                        DeclKind::WithTy { ty, init: Some(init) } =>
                            format!(": {} = {}", ty.to_text(), init.to_text()),
                        DeclKind::WithTy { ty, init: None } => format!(": {}", ty.to_text()),
                        DeclKind::InferTy { init } => format!(" := {}", init.to_text()),
                        DeclKind::Const { ty: Some(ty), init } =>
                            format!(": {} : {}", ty.to_text(), init.to_text()),
                        DeclKind::Const { ty: None, init } => format!(" :: {}", init.to_text()),
                    },
                )
            },
            ExprKind::Semicolon(expr) => format!("{}", expr.to_text()),
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
            */
            ExprKind::Tuple { elements } => {
                lines.write("(");
                for (idx, e) in elements.into_iter().enumerate() {
                    if idx != 0 {
                        lines.write(",");
                    }

                    lines.scope_next_line(|l| e.write_tree(l));
                    lines.write_minus(e.to_text().len());
                }
                lines.write(")");
            },
            ExprKind::Parenthesis { expr } => {
                lines.write("(");
                lines.scope_next_line(|l| expr.write_tree(l));
                lines.write_minus(expr.to_text().len());
                lines.write(")");
            },
            ExprKind::Fn { params, ret_type, body } => {
                lines.write("(");

                for (idx, (ident, ty)) in params.into_iter().enumerate() {
                    if idx != 0 {
                        lines.write(",");
                    }

                    lines.scope_next_line(|l| ident.write_tree(l));
                    lines.write_minus(ident.to_text().len());
                    if let Some(ty) = ty {
                        lines.write(":");
                        lines.scope_next_line(|l| ty.write_tree(l));
                        lines.write_minus(ty.to_text().len());
                    }
                }
                lines.write(")->");
                match ret_type {
                    Some(ret_type) => {
                        lines.scope_next_line(|l| ret_type.write_tree(l));
                        lines.write_minus(ret_type.to_text().len());
                        lines.write("{");
                        lines.scope_next_line(|l| body.write_tree(l));
                        lines.write_minus(body.to_text().len());
                        lines.write("}");
                    },
                    None => {
                        lines.scope_next_line(|l| body.write_tree(l));
                        lines.write_minus(body.to_text().len());
                    },
                }
            },
            /*
            ExprKind::StructDef { fields } => panic!(),
            ExprKind::StructInit { fields, span } => panic!(),
            ExprKind::TupleStructDef { fields, span } => panic!(),
            ExprKind::Union { span } => panic!(),
            ExprKind::Enum { span } => panic!(),
            ExprKind::OptionShort { ty, span } => panic!(),
            */
            ExprKind::Block { stmts } => {
                lines.write("{");
                let start = lines.get_cur_offset();
                let end = lines.scope_next_line(|l| {
                    for s in stmts {
                        s.write_tree(l)
                    }
                    l.get_cur_offset()
                });
                lines.write_minus(end - start);
                lines.write("}");
            },
            ExprKind::Dot { lhs, rhs } => {
                lines.scope_next_line(|l| lhs.write_tree(l));
                lines.write_minus(lhs.to_text().len());
                lines.write(".");
                lines.scope_next_line(|l| rhs.write_tree(l));
                lines.write_minus(rhs.span.len());
            },
            /*
            ExprKind::Colon { lhs, rhs } => {
                lines.scope_next_line(|l| lhs.write_tree(l));
                lines.write_minus(lhs.to_text().len());
                lines.write(":");
                lines.scope_next_line(|l| rhs.write_tree(l));
                lines.write_minus(rhs.span.len());
            },
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
            ExprKind::PreOp { kind, expr } => panic!(),
            ExprKind::BinOp { lhs, op, rhs } => {
                lines.scope_next_line(|l| lhs.write_tree(l));
                lines.write_minus(lhs.to_text().len());
                lines.write(op.to_binop_text());
                lines.scope_next_line(|l| rhs.write_tree(l));
                lines.write_minus(rhs.to_text().len());
            },
            ExprKind::Assign { lhs, rhs } => {
                lines.scope_next_line(|l| lhs.write_tree(l));
                lines.write_minus(lhs.to_text().len());
                lines.write("=");
                lines.scope_next_line(|l| rhs.write_tree(l));
                lines.write_minus(rhs.to_text().len());
            },
            ExprKind::BinOpAssign { lhs, op, rhs } => {
                lines.scope_next_line(|l| lhs.write_tree(l));
                lines.write_minus(lhs.to_text().len());
                lines.write(op.to_binop_assign_text());
                lines.scope_next_line(|l| rhs.write_tree(l));
                lines.write_minus(rhs.to_text().len());
            },
            ExprKind::If { condition, then_body, else_body } => {
                lines.write("if ");
                lines.scope_next_line(|l| condition.write_tree(l));
                lines.write_minus(condition.to_text().len());
                lines.write("{");
                lines.scope_next_line(|l| then_body.write_tree(l));
                lines.write_minus(then_body.to_text().len());
                if let Some(else_body) = else_body {
                    lines.write("} else ");
                    lines.scope_next_line(|l| else_body.write_tree(l));
                    lines.write_minus(else_body.to_text().len());
                }
            },
            ExprKind::Decl { markers, ident, kind } => {
                let DeclMarkers { is_pub, is_mut, is_rec } = markers;
                lines.write(&format!(
                    "{}{}{}",
                    if *is_pub { "pub " } else { "" },
                    if *is_mut { "mut " } else { "" },
                    if *is_rec { "rec " } else { "" }
                ));

                lines.scope_next_line(|l| ident.write_tree(l));
                lines.write_minus(ident.span.len());

                match kind {
                    DeclKind::WithTy { ty, init } => {
                        lines.write(": ");
                        lines.scope_next_line(|l| ty.write_tree(l));
                        lines.write_minus(ty.span.len());
                        if let Some(init) = init {
                            lines.write(" = ");
                            lines.scope_next_line(|l| init.write_tree(l));
                            lines.write_minus(init.span.len());
                        }
                    },
                    DeclKind::InferTy { init } => {
                        lines.write(" := ");
                        lines.scope_next_line(|l| init.write_tree(l));
                        lines.write_minus(init.span.len());
                    },
                    DeclKind::Const { ty: Some(ty), init } => {
                        lines.write(": ");
                        lines.scope_next_line(|l| ty.write_tree(l));
                        lines.write_minus(ty.span.len());
                        lines.write(" : ");
                        lines.scope_next_line(|l| init.write_tree(l));
                        lines.write_minus(init.span.len());
                    },
                    DeclKind::Const { ty: None, init } => {
                        lines.write(" :: ");
                        lines.scope_next_line(|l| init.write_tree(l));
                        lines.write_minus(init.span.len());
                    },
                }
                lines.write(";");
            },
            ExprKind::Semicolon(expr) => {
                let len = self.to_text().len();
                lines.scope_next_line(|l| expr.write_tree(l));
                lines.write_minus(len - 1);
                lines.write(";");
            },

            k => panic!("{:?}", k),
        }
    }

    pub fn print_tree(&self) {
        let mut lines = TreeLines::default();
        self.write_tree(&mut lines);
        for l in lines.lines {
            println!("| {}", l.0);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lexer::Code;

    #[test]
    fn test_let_() {
        let code = Code::new(" add := ( a , b : int ) -> a + b; ");
        println!("{:?}", code);
        println!(
            "{:?}",
            (0..code.len()).map(|x| if x % 5 == 0 { '.' } else { ' ' }).collect::<String>()
        );
        let res = decl().run(Lexer::new(code));
        match res {
            Ok(ok) => println!("OK: {:#?}", ok),
            Err(e) | Fatal(e) => panic!("{}", e.display(code)),
        }
    }
}
