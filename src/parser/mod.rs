//use self::debug::TreeLines;
//use self::util::TupleMap0;
use self::{
    debug::TreeLines,
    parser::{f, peek, PErrKind, PError, PResult, Parser},
    result_with_fatal::ResultWithFatal,
};
use crate::parser::parser::choice;
use lexer::{Lexer, Span, Token, TokenKind};
use parser::opt;
use result_with_fatal::ResultWithFatal::*;

pub mod lexer;
pub mod parser;
pub mod result_with_fatal;
mod util;

macro_rules! err {
    ($kind:ident $( ( $( $field:expr ),* $(,)? ) )? , $span:expr) => {
        Err(PError { kind: PErrKind::$kind $( ( $($field),* ) )?, span: $span })
    };
}

/*
#[allow(unused)]
macro_rules! todo {
    ($msg:literal, $span:expr) => {
        return Err(PError::Tmp(String::leak(format!("TODO: {}", $msg)), $span))
    };
}
*/

//pub type PResult<'l, T> = ResultWithFatal<(T, Lexer<'l>), PError>;
/*
pub trait ParserOption<T>: Parser<Option<T>> {
    fn replace_opt_val<U>(self, other: Parser<U>) -> Parser<Option<U>> {
        move |lex| {
            let (t, lex) = self(lex)?;
            Ok(match t {
                Some(_) => other(lex)?.map0(Some),
                None => (None, lex),
            })
        }
    }
}

impl<T, P: Parser<Option<T>>> ParserOption<T> for P {}
*/

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
    tok_matches(|t| t.is_ignored()).many1()
}

pub fn any_tok() -> Parser<Token> {
    tok_matches(|_| true)
}

pub fn tok(kind: TokenKind) -> Parser<Token> {
    tok_matches(move |lex| *lex == kind)
}

macro_rules! tok {
    ('(') => { tok(TokenKind::OpenParenthesis) };
    (')') => { tok(TokenKind::CloseParenthesis) };
    ('[') => { tok(TokenKind::OpenBracket) };
    (']') => { tok(TokenKind::CloseBracket) };
    ('{') => { tok(TokenKind::OpenBrace) };
    ('}') => { tok(TokenKind::CloseBrace) };
    (=) => { tok(TokenKind::Eq) };
    (==) => { tok(TokenKind::EqEq) };
    (=>) => { tok(TokenKind::FatArrow) };
    (!) => { tok(TokenKind::Bang) };
    (!=) => { tok(TokenKind::BangEq) };
    (<) => { tok(TokenKind::Lt) };
    (<=) => { tok(TokenKind::LtEq) };
    (<<) => { tok(TokenKind::LtLt) };
    (<<=) => { tok(TokenKind::LtLtEq) };
    (>) => { tok(TokenKind::Gt) };
    (>=) => { tok(TokenKind::GtEq) };
    (>>) => { tok(TokenKind::GtGt) };
    (>>=) => { tok(TokenKind::GtGtEq) };
    (+) => { tok(TokenKind::Plus) };
    (+=) => { tok(TokenKind::PlusEq) };
    (-) => { tok(TokenKind::Minus) };
    (-=) => { tok(TokenKind::MinusEq) };
    (->) => { tok(TokenKind::Arrow) };
    (*) => { tok(TokenKind::Asterisk) };
    (*=) => { tok(TokenKind::AsteriskEq) };
    (/) => { tok(TokenKind::Slash) };
    (/=) => { tok(TokenKind::SlashEq) };
    (%) => { tok(TokenKind::Percent) };
    (%=) => { tok(TokenKind::PercentEq) };

    (&) => { tok(TokenKind::Ampersand) };
    (&&) => { tok(TokenKind::AmpersandAmpersand) };
    (&=) => { tok(TokenKind::AmpersandEq) };
    (|) => { tok(TokenKind::Pipe) };
    (||) => { tok(TokenKind::PipePipe) };
    (|=) => { tok(TokenKind::PipeEq) };
    (^) => { tok(TokenKind::Caret) };
    (^=) => { tok(TokenKind::CaretEq) };
    (.) => { tok(TokenKind::Dot) };
    (..) => { tok(TokenKind::DotDot) };
    (..=) => { tok(TokenKind::DotDotEq) };
    (.*) => { tok(TokenKind::DotAsterisk) };
    (.&) => { tok(TokenKind::DotAmpersand) };
    (,) => { tok(TokenKind::Comma) };
    (:) => { tok(TokenKind::Colon) };
    (;) => { tok(TokenKind::Semicolon) };
    (?) => { tok(TokenKind::Question) };
    (#) => { tok(TokenKind::Pound) };
    ($) => { tok(TokenKind::Dollar) };
    (@) => { tok(TokenKind::At) };
    (~) => { tok(TokenKind::Tilde) };
    ('\\') => { compile_error!(TODO: BackSlash) };
    ($($t:tt)*) => {
        compile_error!(concat!("cannot convert \"", stringify!($($t)*), "\" to TokenKind"))
    };
}

pub fn keyword(keyword: &'static str) -> Parser<&'static str> {
    ident_token().flat_map(move |t, lex| match &lex.get_code()[t.span] {
        t if t == keyword => Ok(keyword),
        _ => err!(NotAnKeyword, lex.pos_span()),
    })
}

const KEYWORDS: &[&'static str] = &["let", "mut", "rec", "true", "false"];

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

impl From<(StmtKind, Span)> for Stmt {
    fn from((kind, span): (StmtKind, Span)) -> Self {
        Stmt { kind, span }
    }
}

pub fn stmt() -> Parser<Stmt> {
    let semicolon = f(expr).and_l(tok!(;)).map(|expr| StmtKind::Semicolon(Box::new(expr)));
    f(let_).or(semicolon).to_stmt().ws0()
}

pub fn let_(lex: Lexer<'_>) -> PResult<'_, StmtKind> {
    let ty = tok!(:).ws0().and_r_fatal(f(expr));
    let tail = ws1()
        .and_r(ident_token())
        .many1()
        .ws0()
        .and(opt(ty.ws0()))
        .and(let_kind().ws0())
        .and_l(tok!(;))
        .flat_map(|((idents, ty), kind), lex| {
            let mut is_mut = false;
            let mut is_rec = false;
            let mut idents = idents.into_iter();
            let name = loop {
                let Some(t) = idents.next() else {
                    return err!(MissingLetIdent, lex.pos_span());
                };
                let text = &lex.get_code()[t.span];
                match text {
                    "rec" if !is_rec => is_rec = true,
                    "mut" if !is_mut => is_mut = true,
                    "let" | "rec" | "mut" => return err!(DoubleLetMarker(t), lex.pos_span()),
                    _ => break Ident::try_from_tok(t, &lex)?,
                }
            };
            if let Some(rem) = Span::multi_join(idents.map(|t| t.span)) {
                return err!(TooManyLetIdents(rem), lex.pos_span());
            }
            let ty = ty.map(Box::new);
            Ok(StmtKind::Let { is_mut, is_rec, name, ty, kind })
        });
    keyword("let").and_r_fatal(tail).run(lex)
}

#[derive(Debug, Clone)]
pub enum LetKind {
    Decl,
    Init(Box<Expr>),
}

pub fn let_kind() -> Parser<LetKind> {
    let init = tok!(=).ws0().and_r_fatal(f(expr));
    opt(init).map(|a| match a {
        Some(expr) => LetKind::Init(Box::new(expr)),
        None => LetKind::Decl,
    })
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
    /// `(<ident>, <ident>: <ty>, ..., <ident>,) -> <body>`
    /// `-> <body>`
    Fn {
        params: Vec<(Ident, Option<Type>)>,
        body: Box<Expr>,
    },
    /// `( <expr> )`
    Parenthesis {
        expr: Box<Expr>,
    },
    /// `{ <stmt>`*` }`
    Block {
        stmts: Vec<Stmt>,
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
    /// [`dot`] : [`dot`]
    Colon {
        lhs: Box<Expr>,
        rhs: Box<Expr>,
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
}

#[derive(Debug, Clone)]
pub struct Expr {
    kind: ExprKind,
    span: Span,
}

impl From<(ExprKind, Span)> for Expr {
    fn from((kind, span): (ExprKind, Span)) -> Self {
        Expr { kind, span }
    }
}

pub fn expr(lex: Lexer<'_>) -> PResult<'_, Expr> {
    assign().run(lex)
    /*
    let (mut expr, mut lex) =
        factor().ws0().run(lex).inspect(|e| println!("EXPR VALUE: {:?}", e))?;
    loop {
        match expr_extension().ws0().run(lex) {
            Err(PError { kind: PErrKind::NoInput, .. }) | Ok((ExprTail::None, _)) => {
                return Ok((expr, lex));
            },
            Ok((tail, l)) => {
                lex = l;
                expr = tail.into_expr(expr, &lex);
            },
            Err(e) => return Err(e),
            Fatal(e) => return Fatal(e),
        }
    }
    */
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
    let assign = tok!(=).ws0().and_r_fatal(rvalue()).map(|rhs| (None, rhs));
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
    ident()
}

pub fn rvalue() -> Parser<Expr> {
    bin_op()
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

    let mul_chain = op_chain!(factor() => *: BinOpKind::Mul, /: BinOpKind::Div, %: BinOpKind::Mod);
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

pub fn factor() -> Parser<Expr> {
    let ident = ident().map(Ident::into_expr);
    // `[...]`, `(...)` or `{...}`
    let group = expr_bracket().or(expr_paren()).or(expr_brace());
    let custom_type_def = custom_struct_def()
        .or(custom_union_def())
        .or(custom_enum_def())
        .or(option_short());

    let other = group.or(custom_type_def).or(struct_init());
    ident.or(literal()).or(other.to_expr())
}

/// returns an [`Ident`] or a keyword
pub fn ident_token() -> Parser<Token> {
    tok(TokenKind::Ident)
}

/// returns an [`Ident`] but not keywords
pub fn ident() -> Parser<Ident> {
    ident_token().flat_map(Ident::try_from_tok)
}

pub fn literal() -> Parser<Expr> {
    any_tok().flat_map(|t, lex| match t.kind {
        TokenKind::Literal(l) => {
            let kind = ExprKind::Literal(l);
            Ok(Expr { kind, span: t.span })
        },
        _ => err!(UnexpectedToken(t), lex.pos_span()),
    })
}

/// parses an expression starting with `(`
pub fn expr_paren() -> Parser<ExprKind> {
    let param = ident().ws0().and(opt(tok!(:).ws0().and_r_fatal(f(expr))));
    let params = comma_chain(param).ws0().and_l(tok!(')'));
    let fn_ = params
        .ws0()
        .and_l(tok!(->).ws0())
        .and_fatal(f(expr))
        .map(|(params, body)| ExprKind::Fn { params, body: Box::new(body) });

    // `(<expr>, <expr>, ..., <expr>,)` = Tuple
    // `()` = Tuple
    // `(<expr>)` = Parenthesis
    // `(<expr>,)` = Tuple
    let tuple_or_paren = comma_chain_no_trailing_comma(f(expr)).and(opt(tok!(,).ws0())).map(
        |(elements, suffix_comma)| match suffix_comma {
            Some(_) if elements.len() == 1 => {
                let expr = elements.into_iter().next().map(Box::new).expect("len == 1");
                ExprKind::Parenthesis { expr }
            },
            _ => ExprKind::Tuple { elements },
        },
    );

    let options = fn_.or(tuple_or_paren);
    tok!('(').ws0().and_r_fatal(options.ws0()).and_l_fatal(tok!(')'))
}

/// parses an expression starting with `{`
pub fn expr_brace() -> Parser<ExprKind> {
    let stmts = stmt().ws0().many0().map(|stmts| ExprKind::Block { stmts });
    tok!('{').ws0().and_r_fatal(stmts).and_l_fatal(tok!('}'))
}

/// parses an expression starting with `[`
pub fn expr_bracket() -> Parser<ExprKind> {
    let array_semi = f(expr)
        .and_l(tok!(;).ws0())
        .and_fatal(f(expr))
        .map(|(val, count)| ExprKind::ArraySemi { val: Box::new(val), count: Box::new(count) });
    let array_comma = comma_chain(f(expr)).map(|elements| ExprKind::ArrayComma { elements });
    let kind = array_semi.or(array_comma);
    tok!('[').ws0().and_r_fatal(kind.ws0()).and_l_fatal(tok!(']'))
}

/// `struct { a: int, b: String, c: (u8, u32) }`
/// `struct(int, String, (u8, u32))`
pub fn custom_struct_def() -> Parser<ExprKind> {
    let struct_field = ident().ws0().and_l(tok!(:)).ws0().and_fatal(f(expr));
    let struct_fields = comma_chain(struct_field);
    let struct_ = tok!('{')
        .and_r_fatal(struct_fields)
        .and_l_fatal(tok!('}'))
        .map(ExprKind::StructDef);
    let tuple_struct = tok!('(')
        .and_r_fatal(comma_chain(f(expr)))
        .and_l_fatal(tok!(')'))
        .map(ExprKind::TupleStructDef);
    keyword("struct").ws0().and_r_fatal(struct_.or(tuple_struct))
}

/// TODO
pub fn custom_union_def() -> Parser<ExprKind> {
    let union_ = f(|lex| Ok((ExprKind::Union {}, lex)));
    keyword("union").ws0().and_r_fatal(union_)
}

/// TODO
pub fn custom_enum_def() -> Parser<ExprKind> {
    let enum_ = f(|lex| Ok((ExprKind::Enum {}, lex)));
    keyword("enum").ws0().and_r_fatal(enum_)
}

/// `MyStruct { a: <expr>, b, }` or
/// `MyStruct { a = <expr>, b, }` ?
pub fn struct_init() -> Parser<ExprKind> {
    // TODO: `:` or `=`
    let init_field_value = tok!(:).or(tok!(=)).ws0().and_r_fatal(f(expr));
    let init_field = ident().ws0().and(opt(init_field_value));
    ident()
        .ws0()
        .and_l(tok!('{'))
        .ws0()
        .and_fatal(init_field.many0())
        .ws0()
        .and_l_fatal(tok!('}'))
        .map(|(name, fields)| ExprKind::StructInit { name, fields })
}

/// `?<ty>`
pub fn option_short() -> Parser<ExprKind> {
    tok!(?).ws0().and_r_fatal(ty()).map(Box::new).map(ExprKind::OptionShort)
}

// ----------------------

#[derive(Debug, Clone)]
pub enum ExprTail {
    None,

    Dot(Ident),
    Colon(Box<Expr>),
    CompCall(Vec<Expr>),
    Call(Vec<Expr>),

    PostOp(PostOpKind),
}

impl ExprTail {
    pub fn into_expr(self, lhs: Expr, lex: &Lexer<'_>) -> Expr {
        let span_start = lhs.span.start;
        let new_expr = |span_end, kind| {
            let span = Span::new(span_start, span_end);
            Expr { kind, span }
        };

        match self {
            ExprTail::None => lhs,

            ExprTail::Dot(i) => new_expr(i.span.end, ExprKind::Dot { lhs: Box::new(lhs), rhs: i }),
            ExprTail::Colon(rhs) => {
                new_expr(rhs.span.end, ExprKind::Colon { lhs: Box::new(lhs), rhs })
            },
            ExprTail::CompCall(args) => {
                new_expr(lex.get_pos(), ExprKind::CompCall { func: Box::new(lhs), args })
            },
            ExprTail::Call(args) => {
                new_expr(lex.get_pos(), ExprKind::Call { func: Box::new(lhs), args })
            },

            ExprTail::PostOp(kind) => {
                new_expr(lex.get_pos(), ExprKind::PostOp { kind, expr: Box::new(lhs) })
            },
        }
    }
}

pub fn expr_extension() -> Parser<ExprTail> {
    let expr_end = choice([
        tok!(')'),
        tok!(']'),
        tok!('}'),
        tok!(=>),
        tok!(!),
        tok!(,),
        tok!(;),
        tok!(#),
        tok!($),
        tok!(@),
        tok!(~),
    ])
    .map(|_| ExprTail::None);

    let more_post_op = try_().or(force()).map(ExprTail::PostOp);

    dot().or(more_post_op).or(colon()).or(call()).ws0().or(peek(expr_end))
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
/// ([`ExprTail::Dot`], [`ExprTail::PostOp`])
pub fn dot() -> Parser<ExprTail> {
    let addr_of = tok!(&)
        .and_r_fatal(opt(keyword("mut")))
        .map(|t_mut| if t_mut.is_some() { PostOpKind::AddrMutOf } else { PostOpKind::AddrOf });
    let deref = tok!(*).map(|_| PostOpKind::Deref);
    let dot_post_op = addr_of.or(deref).map(ExprTail::PostOp);
    let member_access = ident().map(ExprTail::Dot);
    tok!(.).and_r_fatal(dot_post_op.or(ws0().and_r(member_access)))
}

pub fn try_() -> Parser<PostOpKind> {
    tok!(?).map(|_| PostOpKind::Try)
}

pub fn force() -> Parser<PostOpKind> {
    tok!(!).and_r_fatal(opt(keyword("unsafe"))).map(|t_unsafe| {
        if t_unsafe.is_some() { PostOpKind::ForceUnsafe } else { PostOpKind::Force }
    })
}

/// `:` [`dot`]
///
/// Note: `1:x.y` = `x.y(1)`
/// Otherwise: `1:x.y`
// `            ^^^^^` ERR: "func `1:x` has no prop `y`"
pub fn colon() -> Parser<ExprTail> {
    let member_access = ident().ws0().sep_by1(tok!(.)).map_with_lex(|d, lex| {
        let mut idents = d.into_iter();
        let first = idents.next().expect("sep_by1").into_expr();
        idents
            .map(ExprTail::Dot)
            .fold(first, |lhs, dot_rhs| dot_rhs.into_expr(lhs, &lex)) // TODO: reference should be to an older lexer
    });
    tok!(:).ws0().and_r_fatal(member_access).map(Box::new).map(ExprTail::Colon)
}

/// `<` [`expr`], [`expr`], ..., [`expr`], `>`
pub fn compcall() -> Parser<ExprTail> {
    tok!(<)
        .and_r_fatal(comma_chain(f(expr)))
        .and_l_fatal(tok!(>))
        .map(ExprTail::CompCall)
}

/// `(` [`expr`], [`expr`], ..., [`expr`], `)`
pub fn call() -> Parser<ExprTail> {
    tok!('(')
        .and_r_fatal(comma_chain(f(expr)))
        .and_l_fatal(tok!(')'))
        .map(ExprTail::Call)
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

pub type Type = Expr;

pub fn ty() -> Parser<Type> {
    f(expr)
}

#[derive(Debug, Clone, Copy)]
pub struct Ident {
    pub span: Span,
}

impl Ident {
    pub fn try_from_tok(t: Token, lex: &Lexer<'_>) -> ResultWithFatal<Ident, PError> {
        if KEYWORDS.contains(&&lex.get_code()[t.span]) {
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
            ExprKind::Fn { params, body } => panic!(),
            ExprKind::StructDef(..) => panic!(),
            ExprKind::StructInit { name, fields } => panic!(),
            ExprKind::TupleStructDef(..) => panic!(),
            ExprKind::Union {} => panic!(),
            ExprKind::Enum {} => panic!(),
            ExprKind::OptionShort(ty) => panic!(),
            ExprKind::Block { stmts } => panic!(),
            ExprKind::Dot { lhs, rhs } => {
                format!("{}.{}", lhs.to_text(), rhs.to_text())
            },
            ExprKind::Colon { lhs, rhs } => {
                format!("{}:{}", lhs.to_text(), rhs.to_text())
            },
            ExprKind::PostOp { kind, expr } => panic!(),
            ExprKind::Index { lhs, idx } => panic!(),
            ExprKind::CompCall { func, args } => panic!(),
            ExprKind::Call { func, args } => format!(
                "{}({})",
                func.to_text(),
                args.iter()
                    .map(|e| e.to_text())
                    .intersperse(",".to_string())
                    .collect::<String>()
            ),
            ExprKind::PreOp { kind, expr } => panic!(),
            ExprKind::BinOp { lhs, op, rhs } => panic!(),
            ExprKind::Assign { lhs, rhs } => todo!(),
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_let_() {
        let code = " let add = ( a , b : int ) -> a + b; ";
        println!("{:?}", code);
        println!(
            "{:?}",
            (0..code.len()).map(|x| if x % 5 == 0 { '^' } else { ' ' }).collect::<String>()
        );
        let res = let_(Lexer::new(code));
        match res {
            Ok(ok) => panic!("OK: {:#?}", ok),
            Err(e) | Fatal(e) => panic!("{}", e.display(code)),
        }
    }

    #[test]
    fn test_let_2() {
        let code = " let add = (a, b) -> a + b; ";
        println!("{:?}", code);
        println!(
            "{:?}",
            (0..code.len()).map(|x| if x % 5 == 0 { '^' } else { ' ' }).collect::<String>()
        );
        let res = let_(Lexer::new(code));
        match res {
            Ok(ok) => panic!("OK: {:#?}", ok),
            Err(e) | Fatal(e) => panic!("{}", e.display(code)),
        }
    }
}
