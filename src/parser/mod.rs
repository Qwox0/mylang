use core::range::Range;
use debug::TreeLines;
pub use error::*;
use lexer::{Code, Keyword, Lexer, Span, Token, TokenKind};
use parser_helper::Parser;
use std::{ptr::NonNull, str::FromStr};
use util::Join;

pub mod error;
pub mod lexer;
pub mod parser_helper;
mod util;

#[derive(Debug, Clone)]
pub enum ExprKind {
    Ident,
    Literal(LitKind),
    /// `true`, `false`
    BoolLit(bool),

    /// `[<val>; <count>]`
    /// both for types and literals
    ArraySemi {
        val: NonNull<Expr>,
        count: NonNull<Expr>,
    },
    /// `[<expr>, <expr>, ..., <expr>,]`
    ArrayComma {
        elements: Box<[Expr]>,
    },
    /// `(<expr>, <expr>, ..., <expr>,)`
    /// both for types and literals
    Tuple {
        elements: Box<[Expr]>,
    },
    /// `(<ident>, <ident>: <ty>, ..., <ident>,) -> <type> { <body> }`
    /// `(<ident>, <ident>: <ty>, ..., <ident>,) -> <body>`
    /// `-> <type> { <body> }`
    /// `-> <body>`
    Fn {
        params: Box<[(Ident, Option<NonNull<Expr>>)]>,
        ret_type: Option<NonNull<Expr>>,
        body: NonNull<Expr>,
    },
    /// `( <expr> )`
    Parenthesis {
        expr: NonNull<Expr>,
    },
    /// `{ <stmt>`*` }`
    Block {
        stmts: Box<[NonNull<Expr>]>,
        has_trailing_semicolon: bool,
    },

    /// `struct { a: int, b: String, c: (u8, u32) }`
    StructDef(Box<[(Ident, Expr)]>),
    /// `struct(...)`
    TupleStructDef(Box<[Expr]>),
    /// `union { ... }`
    Union {},
    /// `enum { ... }`
    Enum {},
    /// `?<ty>`
    OptionShort(NonNull<Expr>),
    /// `*<ty>`
    Ptr {
        is_mut: bool,
        ty: NonNull<Expr>,
    },

    /// `alloc(MyStruct).{ a = <expr>, b, }`
    Initializer {
        lhs: Option<NonNull<Expr>>,
        fields: Box<[(Ident, Option<NonNull<Expr>>)]>,
    },

    /// [`expr`] . [`expr`]
    Dot {
        lhs: NonNull<Expr>,
        rhs: Ident,
    },
    /// examples: `<expr>?`, `<expr>.*`
    PostOp {
        expr: NonNull<Expr>,
        kind: PostOpKind,
    },
    /// `<lhs> [ <idx> ]`
    Index {
        lhs: NonNull<Expr>,
        idx: NonNull<Expr>,
    },

    /*
    /// `<func> < <params> >`
    CompCall {
        func: NonNull<Expr>,
        args: Vec<Expr>,
    },
    */
    /// [`colon`] `(` [`comma_chain`] ([`expr`]) `)`
    Call {
        func: NonNull<Expr>,
        args: Box<[NonNull<Expr>]>,
    },

    /// examples: `&<expr>`, `- <expr>`
    PreOp {
        kind: PreOpKind,
        expr: NonNull<Expr>,
    },
    /// `<lhs> op <lhs>`
    BinOp {
        lhs: NonNull<Expr>,
        op: BinOpKind,
        rhs: NonNull<Expr>,
    },
    /// `<lhs> = <lhs>`
    Assign {
        //lhs: NonNull<LValue>,
        lhs: NonNull<Expr>,
        rhs: NonNull<Expr>,
    },
    /// `<lhs> op= <lhs>`
    BinOpAssign {
        //lhs: NonNull<LValue>,
        lhs: NonNull<Expr>,
        op: BinOpKind,
        rhs: NonNull<Expr>,
    },

    /// variable declaration (and optional initialization)
    /// `mut rec <name>: <ty>`
    /// `mut rec <name>: <ty> = <init>`
    /// `mut rec <name> := <init>`
    VarDecl {
        markers: DeclMarkers,
        //ident: Ident,
        ident: NonNull<Expr>,
        kind: VarDeclKind,
    },

    /// `mut rec <name>: <ty> : <init>`
    /// `mut rec <name> :: <init>`
    ConstDecl {
        markers: DeclMarkers,
        //ident: Ident,
        ident: NonNull<Expr>,
        ty: Option<NonNull<Expr>>,
        init: NonNull<Expr>,
    },

    // /// `pub extern my_fn: (a: i32, b: f64) -> bool`
    // ExternDecl {
    //     is_pub: bool,
    //     ident: NonNull<Expr>,
    //     ty: NonNull<Expr>,
    // },
    /// `if <cond> <then>` (`else <else>`)
    If {
        condition: NonNull<Expr>,
        then_body: NonNull<Expr>,
        else_body: Option<NonNull<Expr>>,
    },
    /// `match <val> <body>` (`else <else>`)
    Match {
        val: NonNull<Expr>,
        // TODO
        else_body: Option<NonNull<Expr>>,
    },

    /// TODO: normal syntax
    /// `<source> | for <iter_var> <body>`
    For {
        source: NonNull<Expr>,
        iter_var: Ident,
        body: NonNull<Expr>,
    },
    /// `while <cond> <body>`
    While {
        condition: NonNull<Expr>,
        body: NonNull<Expr>,
    },

    /// `lhs catch ...`
    Catch {
        lhs: NonNull<Expr>,
        // TODO
    },

    /// `lhs | rhs`
    /// Note: `lhs | if ...`, `lhs | match ...`, `lhs | for ...` and
    /// `lhs | while ...` are inlined during parsing
    Pipe {
        lhs: NonNull<Expr>,
        // TODO
    },

    Return {
        expr: Option<NonNull<Expr>>,
    },

    Semicolon(NonNull<Expr>),
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

impl Expr {
    pub fn new(kind: ExprKind, span: Span) -> Self {
        Self { kind, span }
    }
}

#[derive(Debug, Clone)]
pub struct ExprParser<'code, 'alloc> {
    lex: Lexer<'code>,
    alloc: &'alloc bumpalo::Bump,
}

// methods may advance the parser even on error
// resetting is done by the callee
impl<'code, 'alloc> ExprParser<'code, 'alloc> {
    pub fn new(lex: Lexer<'code>, alloc: &'alloc bumpalo::Bump) -> ExprParser<'code, 'alloc> {
        Self { lex, alloc }
    }

    pub fn top_level_item(&mut self) -> ParseResult<NonNull<Expr>> {
        let res = self.expr().map_err(|err| {
            if err.kind == ParseErrorKind::NoInput {
                ParseError { kind: ParseErrorKind::Finished, ..err }
            } else {
                err
            }
        });
        self.lex.advance_if(|t| t.kind == TokenKind::Semicolon);
        res
    }

    pub fn expr(&mut self) -> ParseResult<NonNull<Expr>> {
        self.expr_(MIN_PRECEDENCE)
    }

    pub fn expr_(&mut self, min_precedence: usize) -> ParseResult<NonNull<Expr>> {
        self.ws0();
        let mut lhs = self.value(min_precedence).context("expr lhs")?;
        loop {
            match self.op_chain(lhs, min_precedence) {
                Ok(node) => lhs = node,
                Err(ParseError { kind: ParseErrorKind::Finished, .. }) => return Ok(lhs),
                err => return err.context("expr op chain element"),
            };
        }
    }

    pub fn op_chain(
        &mut self,
        lhs: NonNull<Expr>,
        min_precedence: usize,
    ) -> ParseResult<NonNull<Expr>> {
        self.ws0();
        let Some(Token { kind, span }) = self.lex.peek() else {
            return err!(Finished, self.lex.pos_span());
        };

        let op = match FollowingOperator::new(kind) {
            None => return err!(Finished, span),
            Some(FollowingOperator::BinOp(binop)) if binop.precedence() <= min_precedence => {
                return err!(Finished, span);
            },
            Some(FollowingOperator::Pipe) if PIPE_PRECEDENCE <= min_precedence => {
                return err!(Finished, span);
            },
            Some(op) => op,
        };
        self.lex.advance();

        macro_rules! expr {
            ($kind:ident, $span:expr) => {
                Expr::new(ExprKind::$kind, $span)
            };
            ($kind:ident { $( $field:ident $( : $val:expr )? ),* $(,)? }, $span:expr ) => {
                Expr::new(ExprKind::$kind{$($field $(:$val)?),*}, $span)
            };
        }

        let expr = match op {
            FollowingOperator::Dot => {
                self.ws0();
                let rhs = self.ident().context("dot rhs")?;
                expr!(Dot { lhs, rhs }, span.join(rhs.span))
            },
            FollowingOperator::Call => return self.call(lhs, span, Vec::new()),
            FollowingOperator::Index => {
                let idx = self.expr()?;
                let close = self.tok(TokenKind::CloseBracket)?;
                expr!(Index { lhs, idx }, span.join(close.span))
            },
            FollowingOperator::Initializer => {
                let mut fields = Vec::new();
                loop {
                    self.ws0();
                    if self.lex.advance_if(|t| t.kind == TokenKind::CloseBrace) {
                        break;
                    }
                    let ident = self.ident().context("dot rhs")?;
                    let mut t = self.next_tok().context("expected '=', ',' or '}'")?;

                    let init = if t.kind == TokenKind::Eq {
                        let init = self.expr().context("init expr")?;
                        t = self.ws0_and_next_tok()?;
                        Some(init)
                    } else {
                        None
                    };
                    fields.push((ident, init));

                    match t.kind {
                        TokenKind::Comma => continue,
                        TokenKind::CloseBrace => break,
                        k => {
                            return err!(UnexpectedToken2(k), t.span)
                                .context("expected '=', ',' or '}'");
                        },
                    }
                }
                let close = self.tok(TokenKind::CloseBrace)?;
                let fields = fields.into_boxed_slice();
                expr!(Initializer { lhs: Some(lhs), fields }, span.join(close.span))
            },
            FollowingOperator::SingleArgNoParenFn => {
                let lhs = unsafe { lhs.as_ref() };
                let Expr { kind: ExprKind::Ident, span: param_span } = lhs else {
                    panic!("SingleArgFn: unknown rhs")
                };
                let param = (Ident { span: *param_span }, None);
                return self.function_tail(Box::new([param]), span);
            },
            FollowingOperator::PostOp(kind) => {
                expr!(PostOp { expr: lhs, kind }, span)
            },
            FollowingOperator::BinOp(op) => {
                let rhs = self.expr_(op.precedence())?;
                let span = unsafe { lhs.as_ref().span.join(rhs.as_ref().span) };
                expr!(BinOp { lhs, op, rhs }, span)
            },
            FollowingOperator::Assign => {
                let rhs = self.expr()?;
                expr!(Assign { lhs, rhs }, span)
            },
            FollowingOperator::BinOpAssign(op) => {
                let rhs = self.expr()?;
                expr!(BinOpAssign { lhs, op, rhs }, span)
            },
            FollowingOperator::VarDecl => {
                let markers = DeclMarkers::default();
                let init = self.expr().context(":= init")?;
                expr!(VarDecl { markers, ident: lhs, kind: VarDeclKind::InferTy { init } }, span)
            },
            FollowingOperator::ConstDecl => {
                let markers = DeclMarkers::default();
                let init = self.expr().context(":: init")?;
                expr!(ConstDecl { markers, ident: lhs, ty: None, init }, span)
            },
            FollowingOperator::TypedDecl => return self.typed_decl(DeclMarkers::default(), lhs),
            FollowingOperator::Pipe => {
                let t = self.ws0_and_next_tok()?;
                return match t.kind {
                    TokenKind::Keyword(Keyword::If) => self.if_after_cond(lhs, span).context("if"),
                    TokenKind::Keyword(Keyword::Match) => {
                        todo!("match")
                    },
                    TokenKind::Keyword(Keyword::For) => {
                        let iter_var = self.ident().context("for iter var")?;
                        let body = self.expr().context("for body")?;
                        self.alloc(expr!(For { source: lhs, iter_var, body }, t.span))
                    },
                    TokenKind::Keyword(Keyword::While) => {
                        let body = self.expr().context("while body")?;
                        self.alloc(expr!(While { condition: lhs, body }, t.span))
                    },
                    TokenKind::Ident => {
                        let func = self.alloc(Expr::new(ExprKind::Ident, t.span))?;
                        let open_paren = self
                            .tok(TokenKind::OpenParenthesis)
                            .context("pipe call: expect '('")?;
                        self.call(func, open_paren.span, vec![lhs])
                    },
                    k => {
                        return err!(UnexpectedToken2(k), t.span)
                            .context("expected fn call, 'if', 'match', 'for' or 'while'");
                    },
                };
            },
        };
        self.alloc(expr)
    }

    /// anything which has higher precedence than any operator
    pub fn value(&mut self, min_precedence: usize) -> ParseResult<NonNull<Expr>> {
        let Token { kind, span } = self.next_tok()?;

        macro_rules! expr {
            ($kind:ident) => {
                Expr::new(ExprKind::$kind, span)
            };
            ($kind:ident ( $( $val:expr ),* $(,)? ) ) => {
                Expr::new(ExprKind::$kind($($val),*), span)
            };
            ($kind:ident ( $( $val:expr ),* $(,)? ), $span:expr ) => {
                Expr::new(ExprKind::$kind($($val),*), $span)
            };
            ($kind:ident { $( $field:ident $( : $val:expr )? ),* $(,)? }, $span:expr ) => {
                Expr::new(ExprKind::$kind{$($field $(:$val)?),*}, $span)
            };
        }

        let expr = match kind {
            TokenKind::Ident => expr!(Ident),
            TokenKind::Keyword(Keyword::Mut)
            | TokenKind::Keyword(Keyword::Rec)
            | TokenKind::Keyword(Keyword::Pub) => {
                let mut markers = DeclMarkers::default();
                let mut t = Token { kind, span };

                macro_rules! set_marker {
                    ($variant:ident $field:ident) => {
                        if markers.$field {
                            return err!(DuplicateDeclMarker(DeclMarkerKind::$variant), t.span);
                        } else {
                            markers.$field = true
                        }
                    };
                }

                let ident = loop {
                    match t.kind {
                        TokenKind::Ident => break Expr::new(ExprKind::Ident, t.span), /* break Ident { span: t.span }, */
                        TokenKind::Keyword(Keyword::Mut) => set_marker!(Mut is_mut),
                        TokenKind::Keyword(Keyword::Rec) => set_marker!(Rec is_rec),
                        TokenKind::Keyword(Keyword::Pub) => set_marker!(Pub is_pub),
                        t => {
                            return err!(UnexpectedToken2(t), span)
                                .context("expected decl marker or ident");
                        },
                    }
                    self.ws1()?;
                    t = self.next_tok().context("missing decl ident")?;
                };
                let ident = self.alloc(ident)?;

                let decl = self.ws0_and_next_tok().context("expected ':', ':=' or '::'")?;
                match decl.kind {
                    TokenKind::Colon => return self.typed_decl(markers, ident),
                    TokenKind::ColonEq => {
                        let init = self.expr().context(":= init")?;
                        expr!(VarDecl { markers, ident, kind: VarDeclKind::InferTy { init } }, span)
                    },
                    TokenKind::ColonColon => {
                        let init = self.expr().context(":: init")?;
                        expr!(ConstDecl { markers, ident, ty: None, init }, span)
                    },
                    t => {
                        return err!(UnexpectedToken2(t), decl.span)
                            .context("expected decl marker or ident");
                    },
                }
            },
            TokenKind::Keyword(Keyword::Struct) => todo!("struct"),
            TokenKind::Keyword(Keyword::Union) => todo!("union"),
            TokenKind::Keyword(Keyword::Enum) => todo!("enum"),
            TokenKind::Keyword(Keyword::Unsafe) => todo!("unsafe"),
            TokenKind::Keyword(Keyword::If) => {
                let condition = self.expr().context("if condition")?;
                return self.if_after_cond(condition, span).context("if");
            },
            //TokenKind::Keyword(Keyword::Else) => todo!("else"),
            TokenKind::Keyword(Keyword::Match) => {
                let val = self.expr().context("match value")?;
                todo!("match body");
                self.ws0();
                let else_body = self
                    .lex
                    .next_if(|t| t.kind == TokenKind::Keyword(Keyword::Else))
                    .map(|_else| {
                        self.ws1()?;
                        self.expr().context("match else body")
                    })
                    .transpose()?;
                expr!(Match { val, else_body }, span)
            },
            TokenKind::Keyword(Keyword::For) => {
                let (source, iter_var, body) = todo!("for");
                expr!(For { source, iter_var, body }, span)
            },
            TokenKind::Keyword(Keyword::While) => {
                let condition = self.expr().context("while condition")?;
                let body = self.expr().context("while body")?;
                expr!(While { condition, body }, span)
            },
            TokenKind::Keyword(Keyword::Return) => {
                let expr = match self.expr_(min_precedence) {
                    Ok(expr) => Some(expr),
                    Err(ParseError {
                        kind: ParseErrorKind::NoInput | ParseErrorKind::Finished,
                        ..
                    }) => None,
                    Err(err) => return Err(err).context("return expr"),
                };
                expr!(Return { expr }, span)
            },
            TokenKind::Keyword(Keyword::Break) => todo!(),
            TokenKind::Keyword(Keyword::Continue) => todo!(),
            //TokenKind::Keyword(_) => todo!("//TokenKind::Keyword(_)"),
            TokenKind::Literal(l) => expr!(Literal(l)),
            TokenKind::BoolLit(b) => expr!(BoolLit(b)),
            TokenKind::OpenParenthesis => {
                self.ws0();
                // TODO: currently no tuples allowed! only:
                // () -> ...
                // (expr)
                // (ident) -> ...
                // (ident)
                // ( (ident (:ty)? ),* ) -> ...
                if self.lex.advance_if(|t| t.kind == TokenKind::CloseParenthesis) {
                    return self.function_tail(Box::new([]), span);
                }
                let first_expr = self.expr().context("first expr in (...)")?;
                self.ws0();
                let first_ident = if let Expr { kind: ExprKind::Ident, span } =
                    unsafe { first_expr.as_ref() }
                {
                    Ident { span: *span }
                } else {
                    let close_p = self.tok(TokenKind::CloseParenthesis).context("(expr)")?;
                    return self
                        .alloc(expr!(Parenthesis { expr: first_expr }, span.join(close_p.span)));
                };

                if let Some(close_p) = self.lex.next_if(|t| t.kind == TokenKind::CloseParenthesis) {
                    self.ws0();
                    return if self.lex.advance_if(|t| t.kind == TokenKind::Arrow) {
                        let param = (first_ident, None);
                        self.function_tail(Box::new([param]), span)
                    } else {
                        self.alloc(expr!(Parenthesis { expr: first_expr }, span.join(close_p.span)))
                    };
                }

                let mut ident = first_ident;
                let mut params = Vec::new();
                loop {
                    let mut t = self.ws0_and_next_tok()?;
                    let ty = if t.kind == TokenKind::Colon {
                        let ty = self.expr().context("param type")?;
                        t = self.ws0_and_next_tok()?;
                        Some(ty)
                    } else {
                        None
                    };
                    params.push((ident, ty));

                    if t.kind == TokenKind::Comma {
                        t = self.ws0_and_next_tok()?;
                        if t.kind == TokenKind::Ident {
                            ident = Ident { span: t.span };
                            continue;
                        }
                    }

                    if t.kind == TokenKind::CloseParenthesis {
                        break;
                    } else {
                        return err!(UnexpectedToken2(t.kind), t.span)
                            .context("expected ident or ')'");
                    }

                    /*
                    match t.kind {
                        TokenKind::CloseParenthesis => break,
                        TokenKind::Comma => {
                            let t = self.ws0_and_next_tok()?;
                            match t.kind {
                                TokenKind::Ident => ident = Ident { span: t.span },
                                TokenKind::CloseParenthesis => break,
                                k => {
                                    return err!(UnexpectedToken2(k), t.span)
                                        .context("expected ident or ')'");
                                },
                            }
                        },
                        k => {
                            return err!(UnexpectedToken2(k), t.span)
                                .context("expected ',' or ')'");
                        },
                    }
                    */
                }
                self.ws0();
                self.tok(TokenKind::Arrow).context("'->'")?;
                return self.function_tail(params.into_boxed_slice(), span);
            },
            TokenKind::OpenBracket => {
                self.expr().context("[...]")?;
                let Ok(Token { kind, span }) = self.ws0_and_next_tok() else {
                    return err!(MissingToken(TokenKind::CloseBracket), self.lex.pos_span())
                        .context("[...]");
                };
                todo!();
            },
            TokenKind::OpenBrace => return self.block(span),
            TokenKind::Bang => {
                let expr = self.expr_(usize::MAX).context("! expr")?;
                expr!(PreOp { kind: PreOpKind::Not, expr }, span)
            },
            TokenKind::Plus => todo!("TokenKind::Plus"),
            TokenKind::Minus => {
                let expr = self.expr_(usize::MAX).context("- expr")?;
                expr!(PreOp { kind: PreOpKind::Neg, expr }, span)
            },
            TokenKind::Arrow => return self.function_tail(Box::new([]), span),
            TokenKind::Asterisk => todo!("TokenKind::Asterisk"),
            TokenKind::Ampersand => todo!("TokenKind::Ampersand"),
            //TokenKind::Pipe => todo!("TokenKind::Pipe"),
            //TokenKind::PipePipe => todo!("TokenKind::PipePipe"),
            //TokenKind::PipeEq => todo!("TokenKind::PipeEq"),
            //TokenKind::Caret => todo!("TokenKind::Caret"),
            //TokenKind::CaretEq => todo!("TokenKind::CaretEq"),
            TokenKind::Dot => todo!("TokenKind::Dot"),
            //TokenKind::DotAsterisk => todo!("TokenKind::DotAsterisk"),
            //TokenKind::DotAmpersand => todo!("TokenKind::DotAmpersand"),
            //TokenKind::Comma => todo!("TokenKind::Comma"),
            TokenKind::Colon => todo!("TokenKind::Colon"),
            //TokenKind::ColonColon => todo!("TokenKind::ColonColon"),
            //TokenKind::ColonEq => todo!("TokenKind::ColonEq"),
            TokenKind::Question => {
                let ty = self.expr().expect("type after ?");
                expr!(OptionShort(ty), span.join(unsafe { ty.as_ref().span }))
            },
            TokenKind::Pound => todo!("TokenKind::Pound"),
            TokenKind::Dollar => todo!("TokenKind::Dollar"),
            TokenKind::At => todo!("TokenKind::At"),
            TokenKind::Tilde => todo!("TokenKind::Tilde"),
            TokenKind::BackSlash => todo!("TokenKind::BackSlash"),
            TokenKind::BackTick => todo!("TokenKind::BackTick"),

            TokenKind::CloseParenthesis
            | TokenKind::CloseBracket
            | TokenKind::CloseBrace
            | TokenKind::Semicolon => return err!(NoInput, self.lex.pos_span()),
            t => return err!(UnexpectedToken2(t), span),
        };
        self.alloc(expr)
    }

    /// parsing starts after the '->'
    pub fn function_tail(
        &mut self,
        params: Box<[(Ident, Option<NonNull<Expr>>)]>,
        start_span: Span,
    ) -> ParseResult<NonNull<Expr>> {
        let expr = self.expr().context("fn body")?;
        let (ret_type, body) = match self.lex.next_if(|t| t.kind == TokenKind::OpenBrace) {
            Some(brace) => (Some(expr), self.block(brace.span).context("fn body")?),
            None => (None, expr),
        };
        self.alloc(Expr::new(
            ExprKind::Fn { params, ret_type, body },
            start_span.join(unsafe { body.as_ref().span }),
        ))
    }

    pub fn if_after_cond(
        &mut self,
        condition: NonNull<Expr>,
        start_span: Span,
    ) -> ParseResult<NonNull<Expr>> {
        let then_body = self.expr_(IF_PRECEDENCE).context("then body")?;
        self.ws0();
        let else_body = self
            .lex
            .peek()
            .filter(|t| t.kind == TokenKind::Keyword(Keyword::Else))
            .inspect(|_| self.lex.advance())
            .map(|_| self.expr_(IF_PRECEDENCE).context("else body"))
            .transpose()?;
        self.alloc(Expr::new(
            ExprKind::If { condition, then_body, else_body },
            start_span, // TODO
        ))
    }

    /// `... ( ... )`
    /// `     ^` starts here
    pub fn call(
        &mut self,
        func: NonNull<Expr>,
        open_paren_span: Span,
        mut args: Vec<NonNull<Expr>>,
    ) -> ParseResult<NonNull<Expr>> {
        let closing_paren_span = loop {
            self.ws0();
            if let Some(closing_paren) = self.lex.next_if(|t| t.kind == TokenKind::CloseParenthesis)
            {
                break closing_paren.span;
            }
            args.push(self.expr()?);
            match self.ws0_and_next_tok()? {
                Token { kind: TokenKind::CloseParenthesis, span } => break span,
                Token { kind: TokenKind::Comma, .. } => continue,
                Token { kind, span } => {
                    return err!(UnexpectedToken2(kind), span).context("expect ',' or ')'");
                },
            };
        };
        self.alloc(Expr::new(
            ExprKind::Call { func, args: args.into_boxed_slice() },
            open_paren_span.join(closing_paren_span),
        ))
    }

    /// parses block context and '}', doesn't parse the '{'
    pub fn block(&mut self, open_brace_span: Span) -> ParseResult<NonNull<Expr>> {
        let mut stmts = Vec::new();
        let (has_trailing_semicolon, closing_brace_span) = loop {
            self.ws0();
            if let Some(closing_brace) = self.lex.next_if(|t| t.kind == TokenKind::CloseBrace) {
                break (true, closing_brace.span);
            }
            stmts.push(self.expr()?);
            match self.ws0_and_next_tok()? {
                Token { kind: TokenKind::CloseBrace, span } => break (false, span),
                Token { kind: TokenKind::Semicolon, .. } => continue,
                Token { kind, span } => {
                    return err!(UnexpectedToken2(kind), span).context("expect ';' or '}'");
                },
            };
        };
        let stmts = stmts.into_boxed_slice();
        self.alloc(Expr::new(
            ExprKind::Block { stmts, has_trailing_semicolon },
            open_brace_span.join(closing_brace_span),
        ))
    }

    /// starts parsing after the colon:
    /// `mut a: int = 0;`
    /// `      ^`
    pub fn typed_decl(
        &mut self,
        markers: DeclMarkers,
        ident: NonNull<Expr>,
    ) -> ParseResult<NonNull<Expr>> {
        let ty = self.expr().context("decl type")?;
        let t = self.lex.peek().ok_or(err!(x NoInput, self.lex.pos_span()))?;
        let expr = match t.kind {
            TokenKind::Eq => {
                self.lex.advance();
                let init = Some(self.expr().context(": ty = init")?);
                Expr::new(
                    ExprKind::VarDecl { markers, ident, kind: VarDeclKind::WithTy { ty, init } },
                    t.span, // TODO
                )
            },
            TokenKind::Colon => {
                self.lex.advance();
                let init = self.expr().context(": ty : init")?;
                Expr::new(ExprKind::ConstDecl { markers, ident, ty: Some(ty), init }, t.span) // TODO
            },
            TokenKind::Semicolon => Expr::new(
                ExprKind::VarDecl { markers, ident, kind: VarDeclKind::WithTy { ty, init: None } },
                t.span, // TODO
            ),
            kind => return err!(UnexpectedToken2(kind), t.span),
        };
        self.alloc(expr)
    }

    pub fn ident(&mut self) -> ParseResult<Ident> {
        self.tok(TokenKind::Ident).map(|t| Ident { span: t.span })
    }

    // -------

    /// 0+ whitespace
    pub fn ws0(&mut self) {
        self.lex.advance_while(|t| t.kind.is_whitespace());
    }

    /// 1+ whitespace
    pub fn ws1(&mut self) -> ParseResult<()> {
        self.tok_where(|t| t.kind.is_whitespace())?;
        self.ws0();
        Ok(())
    }

    pub fn tok(&mut self, tok: TokenKind) -> ParseResult<Token> {
        self.tok_where(|t| t.kind == tok)
    }

    pub fn tok_where(&mut self, cond: impl FnOnce(Token) -> bool) -> ParseResult<Token> {
        let t = self.next_tok()?;
        if cond(t) {
            Ok(t)
        } else {
            err!(UnexpectedToken2(t.kind), self.lex.pos_span())
        }
    }

    #[inline]
    pub fn ws0_and_next_tok(&mut self) -> ParseResult<Token> {
        self.ws0();
        self.next_tok()
    }

    #[inline]
    pub fn next_tok(&mut self) -> ParseResult<Token> {
        self.lex.next().ok_or(err!(x NoInput, self.lex.pos_span()))
    }

    // helpers:

    #[inline]
    pub fn alloc<T>(&self, val: T) -> ParseResult<NonNull<T>> {
        match self.alloc.try_alloc(val) {
            Result::Ok(ok) => Ok(NonNull::from(ok)),
            Result::Err(err) => err!(AllocErr(err), self.lex.pos_span()),
        }
    }

    /// # Source
    ///
    /// see [`bumpalo::Bump::alloc_slice_copy`]
    #[inline]
    pub fn alloc_slice<T: Copy>(&self, slice: &[T]) -> ParseResult<NonNull<[T]>> {
        let layout = core::alloc::Layout::for_value(slice);
        let dst = self
            .alloc
            .try_alloc_layout(layout)
            .map_err(|err| err!(x AllocErr(err), self.lex.pos_span()))?
            .cast::<T>();

        Ok(NonNull::from(unsafe {
            core::ptr::copy_nonoverlapping(slice.as_ptr(), dst.as_ptr(), slice.len());
            core::slice::from_raw_parts_mut(dst.as_ptr(), slice.len())
        }))
    }

    pub fn get_pos(&self) -> usize {
        self.lex.get_pos()
    }

    pub fn set_pos(&mut self, pos: usize) {
        self.lex.set_pos(pos);
    }

    pub fn get_text(&self, range: Range<usize>) -> &str {
        &self.lex.get_code().0[range]
    }

    pub fn advance_if_ok<T>(
        &mut self,
        f: impl FnOnce(&mut Self) -> ParseResult<T>,
    ) -> ParseResult<T> {
        let pos = self.get_pos();
        f(self).inspect_err(|_| self.set_pos(pos))
    }
}

pub enum FollowingOperator {
    /// a.b
    /// `^`
    Dot,
    /// a(b)
    /// `^`
    Call,
    /// a[b]
    /// `^`
    Index,

    /// `alloc(MyStruct).{ a = 1, b = "asdf" }`
    /// `               ^^`
    Initializer,

    /// `arg -> ...`
    /// `    ^^`
    SingleArgNoParenFn,

    /// `a op`
    /// `  ^^`
    PostOp(PostOpKind),

    /// `a op b`
    /// `  ^^`
    BinOp(BinOpKind),

    /// `a = b`
    /// `  ^`
    Assign,
    /// `a op= b`
    /// `  ^^^`
    BinOpAssign(BinOpKind),

    /// `a := b`
    /// `  ^^`
    VarDecl,
    /// `a :: b`
    /// `  ^^`
    ConstDecl,
    /// `a: ty = b` or `a: ty = b`
    /// ` ^`         `   ^`
    TypedDecl,

    Pipe,
}

impl FollowingOperator {
    pub fn new(token_kind: TokenKind) -> Option<FollowingOperator> {
        Some(match token_kind {
            //TokenKind::Ident => todo!("TokenKind::Ident"),
            //TokenKind::Keyword(_) => todo!("TokenKind::Keyword(_)"),
            //TokenKind::Literal(_) => todo!("TokenKind::Literal(_)"),
            TokenKind::OpenParenthesis => FollowingOperator::Call,
            TokenKind::OpenBracket => FollowingOperator::Index,
            TokenKind::Eq => FollowingOperator::Assign,
            TokenKind::EqEq => FollowingOperator::BinOp(BinOpKind::Eq),
            TokenKind::FatArrow => todo!("TokenKind::FatArrow"),
            TokenKind::Bang => todo!("TokenKind::Bang"),
            TokenKind::BangEq => FollowingOperator::BinOp(BinOpKind::Ne),
            TokenKind::Lt => FollowingOperator::BinOp(BinOpKind::Lt),
            TokenKind::LtEq => FollowingOperator::BinOp(BinOpKind::Le),
            TokenKind::LtLt => FollowingOperator::BinOp(BinOpKind::ShiftL),
            TokenKind::LtLtEq => FollowingOperator::BinOpAssign(BinOpKind::ShiftL),
            TokenKind::Gt => FollowingOperator::BinOp(BinOpKind::Gt),
            TokenKind::GtEq => FollowingOperator::BinOp(BinOpKind::Ge),
            TokenKind::GtGt => FollowingOperator::BinOp(BinOpKind::ShiftR),
            TokenKind::GtGtEq => FollowingOperator::BinOpAssign(BinOpKind::ShiftR),
            TokenKind::Plus => FollowingOperator::BinOp(BinOpKind::Add),
            TokenKind::PlusEq => FollowingOperator::BinOpAssign(BinOpKind::Add),
            TokenKind::Minus => FollowingOperator::BinOp(BinOpKind::Sub),
            TokenKind::MinusEq => FollowingOperator::BinOpAssign(BinOpKind::Sub),
            TokenKind::Arrow => FollowingOperator::SingleArgNoParenFn,
            TokenKind::Asterisk => FollowingOperator::BinOp(BinOpKind::Mul),
            TokenKind::AsteriskEq => FollowingOperator::BinOpAssign(BinOpKind::Mul),
            TokenKind::Slash => FollowingOperator::BinOp(BinOpKind::Div),
            TokenKind::SlashEq => FollowingOperator::BinOpAssign(BinOpKind::Div),
            TokenKind::Percent => FollowingOperator::BinOp(BinOpKind::Mod),
            TokenKind::PercentEq => FollowingOperator::BinOpAssign(BinOpKind::Mod),
            TokenKind::Ampersand => FollowingOperator::BinOp(BinOpKind::BitAnd),
            TokenKind::AmpersandAmpersand => FollowingOperator::BinOp(BinOpKind::And),
            TokenKind::AmpersandEq => FollowingOperator::BinOpAssign(BinOpKind::BitAnd),
            TokenKind::Pipe => FollowingOperator::Pipe, /* TODO: find a solution for pipe vs */
            // bitor (currently bitand, bitxor and
            // bitor are ignored)
            TokenKind::PipePipe => FollowingOperator::BinOp(BinOpKind::Or),
            TokenKind::PipeEq => FollowingOperator::BinOpAssign(BinOpKind::BitOr),
            TokenKind::Caret => FollowingOperator::BinOp(BinOpKind::BitXor),
            TokenKind::CaretEq => FollowingOperator::BinOpAssign(BinOpKind::BitXor),
            TokenKind::Dot => FollowingOperator::Dot,
            TokenKind::DotDot => FollowingOperator::BinOp(BinOpKind::Range),
            TokenKind::DotDotEq => FollowingOperator::BinOp(BinOpKind::RangeInclusive),
            TokenKind::DotAsterisk => FollowingOperator::PostOp(PostOpKind::Deref),
            TokenKind::DotAmpersand => FollowingOperator::PostOp(PostOpKind::AddrOf),
            TokenKind::DotOpenBrace => FollowingOperator::Initializer,
            TokenKind::Colon => FollowingOperator::TypedDecl,
            TokenKind::ColonColon => FollowingOperator::ConstDecl,
            TokenKind::ColonEq => FollowingOperator::VarDecl,
            //TokenKind::Semicolon => todo!("TokenKind::Semicolon"),
            TokenKind::Question => FollowingOperator::PostOp(PostOpKind::Try),
            TokenKind::Pound => todo!("TokenKind::Pound"),
            TokenKind::Dollar => todo!("TokenKind::Dollar"),
            TokenKind::At => todo!("TokenKind::At"),
            TokenKind::Tilde => todo!("TokenKind::Tilde"),
            TokenKind::BackSlash => todo!("TokenKind::BackSlash"),
            TokenKind::BackTick => todo!("TokenKind::BackTick"),
            _ => return None,
        })
    }
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

    /// TODO: find a solution for pipe vs bitor (currently bitand, bitxor and
    /// bitor are ignored)
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

const MIN_PRECEDENCE: usize = 0;
const PIPE_PRECEDENCE: usize = 1;
const IF_PRECEDENCE: usize = 2;

impl BinOpKind {
    pub fn precedence(self) -> usize {
        match self {
            BinOpKind::Mul | BinOpKind::Div | BinOpKind::Mod => 19,
            BinOpKind::Add | BinOpKind::Sub => 18,
            BinOpKind::ShiftL | BinOpKind::ShiftR => 17,
            BinOpKind::BitAnd => 16,
            BinOpKind::BitXor => 15,
            BinOpKind::BitOr => 14,
            BinOpKind::Eq
            | BinOpKind::Ne
            | BinOpKind::Lt
            | BinOpKind::Le
            | BinOpKind::Gt
            | BinOpKind::Ge => 13,
            BinOpKind::And => 12,
            BinOpKind::Or => 11,
            BinOpKind::Range | BinOpKind::RangeInclusive => 10,
        }
    }

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

#[derive(Debug, Clone)]
pub struct StmtIter<'code, 'alloc> {
    parser: ExprParser<'code, 'alloc>,
}

impl<'code, 'alloc> Iterator for StmtIter<'code, 'alloc> {
    type Item = ParseResult<NonNull<Expr>>;

    fn next(&mut self) -> Option<Self::Item> {
        match self.parser.top_level_item() {
            Err(ParseError { kind: ParseErrorKind::Finished, .. }) => return None,
            res => Some(res),
        }
    }
}

impl<'code, 'alloc> StmtIter<'code, 'alloc> {
    /// Parses top-level items until the end of the [`Code`] or until an
    /// [`PError`] occurs.
    pub fn parse(code: &'code Code, alloc: &'alloc bumpalo::Bump) -> Self {
        let mut parser = ExprParser::new(Lexer::new(code), alloc);
        parser.ws0();
        Self { parser }
    }
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeclMarkerKind {
    Pub,
    Mut,
    Rec,
}

#[derive(Debug, Clone)]
pub enum VarDeclKind {
    /// `<name>: <ty>;`
    /// `<name>: <ty> = <init>;`
    WithTy { ty: NonNull<Expr>, init: Option<NonNull<Expr>> },
    /// `<name> := <init>;`
    InferTy { init: NonNull<Expr> },
}

impl VarDeclKind {
    pub fn into_ty_val(self) -> (Option<NonNull<Expr>>, Option<NonNull<Expr>>) {
        match self {
            VarDeclKind::WithTy { ty, init } => (Some(ty), init),
            VarDeclKind::InferTy { init } => (None, Some(init)),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Ident {
    pub span: Span,
}

impl Ident {
    pub fn try_from_tok(t: Token, lex: &Lexer<'_>) -> ParseResult<Ident> {
        if Keyword::from_str(&&lex.get_code()[t.span]).is_ok() {
            return err!(NotAnIdent, lex.pos_span());
        }
        Ok(Ident { span: t.span })
    }

    pub fn into_expr(self) -> Expr {
        Expr { kind: ExprKind::Ident, span: self.span }
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

impl Expr {
    /// This generates new Code and doesn't read the orignal code!
    pub fn to_text(&self, code: &Code) -> String {
        unsafe {
            #[allow(unused)]
            match &self.kind {
                ExprKind::Ident => code[self.span].to_string(),
                ExprKind::ArraySemi { val, count } => {
                    format!("[{};{}]", val.as_ref().to_text(code), count.as_ref().to_text(code))
                },
                ExprKind::ArrayComma { elements } => format!(
                    "[{}]",
                    elements
                        .as_ref()
                        .iter()
                        .map(|e| e.to_text(code))
                        .intersperse(",".to_string())
                        .collect::<String>()
                ),
                ExprKind::Tuple { elements } => format!(
                    "({})",
                    elements
                        .as_ref()
                        .iter()
                        .map(|e| e.to_text(code))
                        .intersperse(",".to_string())
                        .collect::<String>()
                ),
                ExprKind::Parenthesis { expr } => format!("({})", expr.as_ref().to_text(code)),
                ExprKind::Literal(_) | ExprKind::BoolLit(_) => code[self.span].to_string(),
                ExprKind::Fn { params, ret_type, body } => format!(
                    "({})->{}",
                    params
                        .as_ref()
                        .iter()
                        .map(|(ident, ty)| {
                            let ty = ty
                                .as_ref()
                                .map(|e| format!(":{}", e.as_ref().to_text(code)))
                                .unwrap_or_default();
                            format!("{}{}", &code[ident.span], ty)
                        })
                        .intersperse(",".to_string())
                        .collect::<String>(),
                    {
                        let body = body.as_ref().to_text(code);
                        match ret_type {
                            Some(ret_type) => {
                                format!("{} {{{}}}", ret_type.as_ref().to_text(code), body)
                            },
                            None => body,
                        }
                    }
                ),
                ExprKind::StructDef(..) => panic!(),
                ExprKind::TupleStructDef(..) => panic!(),
                ExprKind::Union {} => panic!(),
                ExprKind::Enum {} => panic!(),
                ExprKind::OptionShort(ty) => {
                    format!("?{}", ty.as_ref().to_text(code))
                },
                ExprKind::Ptr { is_mut, ty } => {
                    format!("*{}{}", if *is_mut { "mut " } else { "" }, ty.as_ref().to_text(code))
                },
                ExprKind::Initializer { lhs, fields } => panic!(),
                ExprKind::Block { stmts, has_trailing_semicolon } => {
                    format!(
                        "{{{}{}}}",
                        String::join(";", stmts.as_ref().iter().map(|a| a.as_ref().to_text(code))),
                        if *has_trailing_semicolon { ";" } else { "" }
                    )
                },
                ExprKind::Dot { lhs, rhs } => {
                    format!("{}.{}", lhs.as_ref().to_text(code), &code[rhs.span])
                },
                //ExprKind::Colon { lhs, rhs } => { format!("{}:{}", lhs.as_ref().to_text(code),
                // rhs.as_ref().to_text(code)) },
                ExprKind::PostOp { kind, expr } => panic!(),
                ExprKind::Index { lhs, idx } => panic!(),
                //ExprKind::CompCall { func, args } => panic!(),
                ExprKind::Call { func, args } => format!(
                    "{}({})",
                    func.as_ref().to_text(code),
                    args.as_ref()
                        .iter()
                        .map(|e| e.as_ref().to_text(code))
                        .intersperse(",".to_string())
                        .collect::<String>()
                ),
                ExprKind::PreOp { kind, expr } => format!(
                    "{}{}",
                    match kind {
                        PreOpKind::AddrOf => "&",
                        PreOpKind::AddrMutOf => "&mut ",
                        PreOpKind::Deref => "*",
                        PreOpKind::Not => "!",
                        PreOpKind::Neg => "- ",
                    },
                    expr.as_ref().to_text(code)
                ),
                ExprKind::BinOp { lhs, op, rhs } => {
                    format!(
                        "{}{}{}",
                        lhs.as_ref().to_text(code),
                        op.to_binop_text(),
                        rhs.as_ref().to_text(code)
                    )
                },
                ExprKind::Assign { lhs, rhs } => todo!("ExprKind::Assign"),
                ExprKind::BinOpAssign { lhs, op, rhs } => {
                    format!(
                        "{}{}{}",
                        lhs.as_ref().to_text(code),
                        op.to_binop_assign_text(),
                        rhs.as_ref().to_text(code)
                    )
                },
                ExprKind::VarDecl { markers, ident, kind } => {
                    let DeclMarkers { is_pub, is_mut, is_rec } = markers;
                    format!(
                        "{}{}{}{}{}",
                        if *is_pub { "pub " } else { "" },
                        if *is_mut { "mut " } else { "" },
                        if *is_rec { "rec " } else { "" },
                        unsafe { ident.as_ref().to_text(code) },
                        match kind {
                            VarDeclKind::WithTy { ty, init: Some(init) } => format!(
                                ": {} = {}",
                                ty.as_ref().to_text(code),
                                init.as_ref().to_text(code)
                            ),
                            VarDeclKind::WithTy { ty, init: None } =>
                                format!(": {}", ty.as_ref().to_text(code)),
                            VarDeclKind::InferTy { init } =>
                                format!(" := {}", init.as_ref().to_text(code)),
                            // VarDeclKind::Const { ty: Some(ty), init } =>
                            //     format!(": {} : {}", ty.as_ref().to_text(code),
                            // init.as_ref().to_text(code)),
                            // VarDeclKind::Const { ty: None, init } =>
                            //     format!(" :: {}", init.as_ref().to_text(code)),
                        },
                    )
                },
                ExprKind::ConstDecl { markers, ident, ty, init } => {
                    let DeclMarkers { is_pub, is_mut, is_rec } = markers;
                    format!(
                        "{}{}{}{}{}",
                        if *is_pub { "pub " } else { "" },
                        if *is_mut { "mut " } else { "" },
                        if *is_rec { "rec " } else { "" },
                        unsafe { ident.as_ref().to_text(code) },
                        if let Some(ty) = ty {
                            format!(
                                ": {} : {}",
                                ty.as_ref().to_text(code),
                                init.as_ref().to_text(code)
                            )
                        } else {
                            format!(" :: {}", init.as_ref().to_text(code))
                        }
                    )
                },
                ExprKind::If { condition, then_body, else_body } => {
                    format!(
                        "if {} {}{}",
                        condition.as_ref().to_text(code),
                        then_body.as_ref().to_text(code),
                        else_body
                            .as_ref()
                            .map(|e| format!(" else {}", e.as_ref().to_text(code)))
                            .unwrap_or_default()
                    )
                },
                ExprKind::Match { val, else_body } => todo!(),
                ExprKind::For { source, iter_var, body } => todo!(),
                ExprKind::While { condition, body } => todo!(),
                ExprKind::Catch { lhs } => todo!(),
                ExprKind::Pipe { lhs } => todo!(),
                ExprKind::Return { expr } => format!(
                    "return{}",
                    expr.map(|expr| format!(" {}", expr.as_ref().to_text(code)))
                        .unwrap_or_default()
                ),
                ExprKind::Semicolon(expr) => format!("{}", expr.as_ref().to_text(code)),
            }
        }
    }

    pub fn write_tree(&self, lines: &mut TreeLines, code: &Code) {
        unsafe {
            match &self.kind {
                ExprKind::Ident
                | ExprKind::Literal(LitKind::Char)
                | ExprKind::Literal(LitKind::BChar)
                | ExprKind::Literal(LitKind::Int)
                | ExprKind::Literal(LitKind::Float)
                | ExprKind::Literal(LitKind::Str)
                | ExprKind::BoolLit(_) => lines.write(&self.to_text(code)),
                /*
                ExprKind::ArrayShort { val, count } => {
                    format!("[{};{}]", val.to_text(code), count.to_text(code))
                },
                ExprKind::ArrayInit { elements } => format!(
                    "[{}]",
                    elements
                        .iter()
                        .map(|e| e.to_text(code))
                        .intersperse(",".to_string())
                        .collect::<String>()
                ),
                */
                ExprKind::Tuple { elements } => {
                    lines.write("(");
                    for (idx, e) in elements.as_ref().into_iter().enumerate() {
                        if idx != 0 {
                            lines.write(",");
                        }

                        lines.scope_next_line(|l| e.write_tree(l, code));
                        lines.write_minus(e.to_text(code).len());
                    }
                    lines.write(")");
                },
                ExprKind::Parenthesis { expr } => {
                    lines.write("(");
                    lines.scope_next_line(|l| expr.as_ref().write_tree(l, code));
                    lines.write_minus(expr.as_ref().to_text(code).len());
                    lines.write(")");
                },
                ExprKind::Fn { params, ret_type, body } => {
                    let body = body.as_ref();
                    lines.write("(");

                    for (idx, (ident, ty)) in params.as_ref().into_iter().enumerate() {
                        if idx != 0 {
                            lines.write(",");
                        }

                        lines.scope_next_line(|l| l.write(&code[ident.span]));
                        lines.write_minus(ident.span.len());
                        if let Some(ty) = ty {
                            let ty = ty.as_ref();
                            lines.write(":");
                            lines.scope_next_line(|l| ty.write_tree(l, code));
                            lines.write_minus(ty.to_text(code).len());
                        }
                    }
                    lines.write(")->");
                    match ret_type {
                        Some(ret_type) => {
                            let ret_type = ret_type.as_ref();
                            lines.scope_next_line(|l| ret_type.write_tree(l, code));
                            lines.write_minus(ret_type.to_text(code).len());
                            lines.write("{");
                            lines.scope_next_line(|l| body.write_tree(l, code));
                            lines.write_minus(body.to_text(code).len());
                            lines.write("}");
                        },
                        None => {
                            lines.scope_next_line(|l| body.write_tree(l, code));
                            lines.write_minus(body.to_text(code).len());
                        },
                    }
                },
                /*
                ExprKind::StructDef { fields } => panic!(),
                ExprKind::StructInit { fields, span } => panic!(),
                ExprKind::TupleStructDef { fields, span } => panic!(),
                ExprKind::Union { span } => panic!(),
                ExprKind::Enum { span } => panic!(),
                */
                ExprKind::OptionShort(ty) => {
                    lines.write("?");
                    lines.scope_next_line(|l| ty.as_ref().write_tree(l, code));
                    lines.write_minus(ty.as_ref().to_text(code).len());
                },
                ExprKind::Ptr { is_mut, ty } => {
                    lines.write("*");
                    if *is_mut {
                        lines.write("mut ");
                    }
                    lines.scope_next_line(|l| ty.as_ref().write_tree(l, code));
                    lines.write_minus(ty.as_ref().to_text(code).len());
                },
                ExprKind::Block { stmts, has_trailing_semicolon } => {
                    lines.write("{");
                    let len = stmts.len();
                    for (idx, s) in stmts.as_ref().iter().enumerate() {
                        lines.scope_next_line(|l| s.as_ref().write_tree(l, code));
                        lines.write_minus(s.as_ref().to_text(code).len());
                        if idx + 1 < len || *has_trailing_semicolon {
                            lines.write(";");
                        }
                    }
                    lines.write("}");
                },
                ExprKind::Dot { lhs, rhs } => {
                    let lhs = lhs.as_ref();
                    lines.scope_next_line(|l| lhs.write_tree(l, code));
                    lines.write_minus(lhs.to_text(code).len());
                    lines.write(".");
                    lines.scope_next_line(|l| l.write(&code[rhs.span]));
                    lines.write_minus(rhs.span.len());
                },
                /*
                ExprKind::Colon { lhs, rhs } => {
                    lines.scope_next_line(|l| lhs.write_tree(l, code));
                    lines.write_minus(lhs.to_text(code).len());
                    lines.write(":");
                    lines.scope_next_line(|l| rhs.write_tree(l, code));
                    lines.write_minus(rhs.span.len());
                },
                ExprKind::PostOp { kind, expr, span } => panic!(),
                ExprKind::Index { lhs, idx, span } => panic!(),
                ExprKind::CompCall { func, args } => panic!(),
                */
                ExprKind::Call { func, args } => {
                    let func = func.as_ref();
                    lines.scope_next_line(|l| func.write_tree(l, code));
                    lines.write_minus(func.to_text(code).len());
                    lines.write("(");
                    let len = args.as_ref().len();
                    for (idx, arg) in args.as_ref().into_iter().enumerate() {
                        let arg = arg.as_ref();
                        lines.scope_next_line(|l| arg.write_tree(l, code));
                        lines.write_minus(arg.to_text(code).len());
                        if idx + 1 != len {
                            lines.write(",");
                        }
                    }
                    lines.write(")");
                },
                ExprKind::PreOp { kind, expr } => {
                    let expr = expr.as_ref();
                    lines.write(match kind {
                        PreOpKind::AddrOf => "&",
                        PreOpKind::AddrMutOf => "&mut ",
                        PreOpKind::Deref => "*",
                        PreOpKind::Not => "!",
                        PreOpKind::Neg => "- ",
                    });
                    lines.scope_next_line(|l| expr.write_tree(l, code));
                    lines.write_minus(expr.to_text(code).len());
                },
                ExprKind::BinOp { lhs, op, rhs } => {
                    let lhs = lhs.as_ref();
                    let rhs = rhs.as_ref();
                    lines.scope_next_line(|l| lhs.write_tree(l, code));
                    lines.write_minus(lhs.to_text(code).len());
                    lines.write(op.to_binop_text());
                    lines.scope_next_line(|l| rhs.write_tree(l, code));
                    lines.write_minus(rhs.to_text(code).len());
                },
                ExprKind::Assign { lhs, rhs } => {
                    let lhs = lhs.as_ref();
                    let rhs = rhs.as_ref();
                    lines.scope_next_line(|l| lhs.write_tree(l, code));
                    lines.write_minus(lhs.to_text(code).len());
                    lines.write("=");
                    lines.scope_next_line(|l| rhs.write_tree(l, code));
                    lines.write_minus(rhs.to_text(code).len());
                },
                ExprKind::BinOpAssign { lhs, op, rhs } => {
                    let lhs = lhs.as_ref();
                    let rhs = rhs.as_ref();
                    lines.scope_next_line(|l| lhs.write_tree(l, code));
                    lines.write_minus(lhs.to_text(code).len());
                    lines.write(op.to_binop_assign_text());
                    lines.scope_next_line(|l| rhs.write_tree(l, code));
                    lines.write_minus(rhs.to_text(code).len());
                },
                ExprKind::If { condition, then_body, else_body } => {
                    let condition = condition.as_ref();
                    let then_body = then_body.as_ref();
                    lines.write("if ");
                    lines.scope_next_line(|l| condition.write_tree(l, code));
                    lines.write_minus(condition.to_text(code).len());
                    lines.write(" ");
                    lines.scope_next_line(|l| then_body.write_tree(l, code));
                    lines.write_minus(then_body.to_text(code).len());
                    if let Some(else_body) = else_body {
                        let else_body = else_body.as_ref();
                        lines.write(" else ");
                        lines.scope_next_line(|l| else_body.write_tree(l, code));
                        lines.write_minus(else_body.to_text(code).len());
                    }
                },
                ExprKind::VarDecl { markers, ident, kind } => {
                    let ident = ident.as_ref();
                    let DeclMarkers { is_pub, is_mut, is_rec } = markers;
                    lines.write(&format!(
                        "{}{}{}",
                        if *is_pub { "pub " } else { "" },
                        if *is_mut { "mut " } else { "" },
                        if *is_rec { "rec " } else { "" }
                    ));

                    lines.scope_next_line(|l| ident.write_tree(l, code));
                    lines.write_minus(ident.span.len());

                    match kind {
                        VarDeclKind::WithTy { ty, init } => {
                            let ty = ty.as_ref();
                            lines.write(": ");
                            lines.scope_next_line(|l| ty.write_tree(l, code));
                            lines.write_minus(ty.span.len());
                            if let Some(init) = init {
                                let init = init.as_ref();
                                lines.write(" = ");
                                lines.scope_next_line(|l| init.write_tree(l, code));
                                lines.write_minus(init.span.len());
                            }
                        },
                        VarDeclKind::InferTy { init } => {
                            let init = init.as_ref();
                            lines.write(" := ");
                            lines.scope_next_line(|l| init.write_tree(l, code));
                            lines.write_minus(init.to_text(code).len());
                        },
                        /*
                        VarDeclKind::Const { ty: Some(ty), init } => {
                            lines.write(": ");
                            lines.scope_next_line(|l| ty.write_tree(l, code));
                            lines.write_minus(ty.span.len());
                            lines.write(" : ");
                            lines.scope_next_line(|l| init.write_tree(l, code));
                            lines.write_minus(init.span.len());
                        },
                        VarDeclKind::Const { ty: None, init } => {
                            lines.write(" :: ");
                            lines.scope_next_line(|l| init.write_tree(l, code));
                            lines.write_minus(init.span.len());
                        },
                        */
                    }
                },
                ExprKind::ConstDecl { markers, ident, ty, init } => {
                    let ident = ident.as_ref();
                    let init = init.as_ref();
                    let init_len = init.to_text(code).len();
                    let DeclMarkers { is_pub, is_mut, is_rec } = markers;
                    lines.write(&format!(
                        "{}{}{}",
                        if *is_pub { "pub " } else { "" },
                        if *is_mut { "mut " } else { "" },
                        if *is_rec { "rec " } else { "" }
                    ));

                    lines.scope_next_line(|l| ident.write_tree(l, code));
                    lines.write_minus(ident.span.len());

                    if let Some(ty) = ty {
                        let ty = ty.as_ref();
                        lines.write(": ");
                        lines.scope_next_line(|l| ty.write_tree(l, code));
                        lines.write_minus(ty.span.len());
                        lines.write(" : ");
                        lines.scope_next_line(|l| init.write_tree(l, code));
                        lines.write_minus(init_len);
                    } else {
                        lines.write(" :: ");
                        lines.scope_next_line(|l| init.write_tree(l, code));
                        lines.write_minus(init_len);
                    }
                    /*
                       if let Some(ty) = ty {
                           format!(": {} : {}", ty.as_ref().to_text(code), init.as_ref().to_text(code))
                       } else {
                           format!(" :: {}", init.as_ref().to_text(code))
                       }
                    */
                },
                ExprKind::Return { expr } => {
                    lines.write("return");
                    if let Some(expr) = expr {
                        let expr = expr.as_ref();
                        lines.write(" ");
                        lines.scope_next_line(|l| expr.write_tree(l, code));
                        lines.write_minus(expr.to_text(code).len());
                    }
                },
                ExprKind::Semicolon(expr) => {
                    let expr = expr.as_ref();
                    lines.scope_next_line(|l| expr.write_tree(l, code));
                    lines.write_minus(expr.to_text(code).len());
                    lines.write(";");
                },

                k => panic!("{:?}", k),
            }
        }
    }

    pub fn print_tree(&self, code: &Code) {
        let mut lines = TreeLines::default();
        self.write_tree(&mut lines, code);
        for l in lines.lines {
            println!("| {}", l.0);
        }
    }
}
