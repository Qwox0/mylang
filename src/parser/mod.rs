use crate::scratch_pool::ScratchPool;
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

#[derive(Debug, Clone, Copy)]
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
        elements: NonNull<[Expr]>,
    },
    /// `(<expr>, <expr>, ..., <expr>,)`
    /// both for types and literals
    Tuple {
        elements: NonNull<[Expr]>,
    },
    /// `(<ident>, <ident>: <ty>, ..., <ident>,) -> <type> { <body> }`
    /// `(<ident>, <ident>: <ty>, ..., <ident>,) -> <body>`
    /// `-> <type> { <body> }`
    /// `-> <body>`
    Fn {
        params: NonNull<[VarDecl]>,
        ret_type: Option<NonNull<Expr>>,
        body: NonNull<Expr>,
    },
    /// `( <expr> )`
    Parenthesis {
        expr: NonNull<Expr>,
    },
    /// `{ <stmt>`*` }`
    Block {
        stmts: NonNull<[NonNull<Expr>]>,
        has_trailing_semicolon: bool,
    },

    /// `struct { a: int, b: String, c: (u8, u32) }`
    StructDef(NonNull<[VarDecl]>),
    /// `union { a: int, b: String, c: (u8, u32) }`
    UnionDef(NonNull<[VarDecl]>),
    /// `enum { ... }`
    EnumDef {},
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
        fields: NonNull<[(Ident, Option<NonNull<Expr>>)]>,
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
        args: NonNull<[NonNull<Expr>]>,
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
    /// `mut rec <name>: <ty> : <init>`
    /// `mut rec <name> := <init>`
    /// `mut rec <name> :: <init>`
    VarDecl(VarDecl),

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

    pub fn try_to_ident(&self) -> ParseResult<Ident> {
        match self.kind {
            ExprKind::Ident => Ok(Ident { span: self.span }),
            _ => err!(NotAnIdent, self.span),
        }
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
            ($kind:ident($( $val:expr ),* $(,)?), $span:expr ) => {
                Expr::new(ExprKind::$kind($($val),*), $span)
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
            FollowingOperator::Call => return self.call(lhs, span, ScratchPool::new()),
            FollowingOperator::Index => {
                let idx = self.expr()?;
                let close = self.tok(TokenKind::CloseBracket)?;
                expr!(Index { lhs, idx }, span.join(close.span))
            },
            FollowingOperator::Initializer => {
                let mut fields = ScratchPool::new();
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
                    fields
                        .push((ident, init))
                        .map_err(|e| err!(x AllocErr(e), self.lex.pos_span()))?;

                    match t.kind {
                        TokenKind::Comma => continue,
                        TokenKind::CloseBrace => break,
                        _ => ParseError::unexpected_token(t).context("expected '=', ',' or '}'")?,
                    }
                }
                let close = self.tok(TokenKind::CloseBrace)?;
                let fields = self.clone_slice_from_scratch_pool(fields)?;
                expr!(Initializer { lhs: Some(lhs), fields }, span.join(close.span))
            },
            FollowingOperator::SingleArgNoParenFn => {
                let lhs = unsafe { lhs.as_ref() };
                let Expr { kind: ExprKind::Ident, span: param_span } = lhs else {
                    panic!("SingleArgFn: unknown rhs")
                };
                let param = VarDecl {
                    markers: DeclMarkers::default(),
                    ident: Ident { span: *param_span },
                    ty: None,
                    default: None,
                    is_const: false,
                };
                let params = ScratchPool::new_with_first_val(param)
                    .map_err(|e| err!(x AllocErr(e), self.lex.pos_span()))?;
                let params = self.clone_slice_from_scratch_pool(params)?;
                return self.function_tail(params, span);
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
                let ident = unsafe { lhs.as_ref() }.try_to_ident().context("var decl ident")?;
                let init = self.expr().context(":= init")?;
                let decl =
                    VarDecl { markers, ident, ty: None, default: Some(init), is_const: false };
                expr!(VarDecl(decl), span)
            },
            FollowingOperator::ConstDecl => {
                let markers = DeclMarkers::default();
                let ident = unsafe { lhs.as_ref() }.try_to_ident().context("const decl ident")?;
                let init = self.expr().context(":: init")?;
                let decl =
                    VarDecl { markers, ident, ty: None, default: Some(init), is_const: true };
                expr!(VarDecl(decl), span)
            },
            FollowingOperator::TypedDecl => {
                let decl = self.typed_decl(
                    DeclMarkers::default(),
                    unsafe { lhs.as_ref() }.try_to_ident().context("const decl ident")?,
                )?;
                expr!(VarDecl(decl), span)
            },
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
                        let args = ScratchPool::new_with_first_val(lhs)
                            .map_err(|e| err!(x AllocErr(e), self.lex.pos_span()))?;
                        self.call(func, open_paren.span, args)
                    },
                    _ => ParseError::unexpected_token(t)
                        .context("expected fn call, 'if', 'match', 'for' or 'while'")?,
                };
            },
        };
        self.alloc(expr)
    }

    /// anything which has higher precedence than any operator
    pub fn value(&mut self, min_precedence: usize) -> ParseResult<NonNull<Expr>> {
        let Token { kind, span } = self.next_tok().context("expected value")?;

        macro_rules! expr {
            ($kind:ident) => {
                Expr::new(ExprKind::$kind, span)
            };
            ($kind:ident, $span:expr) => {
                Expr::new(ExprKind::$kind, $span)
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
            TokenKind::Keyword(Keyword::Mut) => {
                self.ws0();
                let markers = DeclMarkers { is_pub: false, is_mut: true, is_rec: false };
                expr!(VarDecl(self.var_decl_with_markers(markers)?), span)
            },
            TokenKind::Keyword(Keyword::Rec) => {
                self.ws0();
                let markers = DeclMarkers { is_pub: false, is_mut: false, is_rec: true };
                expr!(VarDecl(self.var_decl_with_markers(markers)?), span)
            },
            TokenKind::Keyword(Keyword::Pub) => {
                self.ws0();
                let markers = DeclMarkers { is_pub: true, is_mut: false, is_rec: false };
                expr!(VarDecl(self.var_decl_with_markers(markers)?), span)
            },
            TokenKind::Keyword(Keyword::Struct) => {
                self.ws0();
                self.tok(TokenKind::OpenBrace).context("struct '{'")?;
                self.ws0();
                let fields = self.struct_fields()?;
                let close_b = self.tok(TokenKind::CloseBrace).context("struct '}'")?;
                expr!(StructDef(fields), span.join(close_b.span))
            },
            TokenKind::Keyword(Keyword::Union) => {
                self.ws0();
                self.tok(TokenKind::OpenBrace).context("struct '{'")?;
                self.ws0();
                let fields = self.struct_fields()?;
                let close_b = self.tok(TokenKind::CloseBrace).context("struct '}'")?;
                expr!(UnionDef(fields), span.join(close_b.span))
            },
            TokenKind::Keyword(Keyword::Enum) => todo!("enum"),
            TokenKind::Keyword(Keyword::Unsafe) => todo!("unsafe"),
            TokenKind::Keyword(Keyword::If) => {
                let condition = self.expr().context("if condition")?;
                return self.if_after_cond(condition, span).context("if");
            },
            //TokenKind::Keyword(Keyword::Else) => todo!("else"),
            TokenKind::Keyword(Keyword::Match) => {
                todo!("match body");
                let val = self.expr().context("match value")?;
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
                todo!("for");
                // let (source, iter_var, body) = todo!();
                // expr!(For { source, iter_var, body }, span)
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
                // TODO: currently no tuples allowed!
                // () -> ...
                if self.lex.advance_if(|t| t.kind == TokenKind::CloseParenthesis) {
                    self.ws0();
                    self.tok(TokenKind::Arrow).context("'->'")?;
                    return self.function_tail(self.alloc_empty_slice(), span);
                }
                let start_state = self.lex.get_state();
                let first_expr = self.expr().context("expr in (...)")?; // this assumes the parameter syntax is also a valid expression
                let t = self.ws0_and_next_tok().context("missing ')'")?;
                self.ws0();
                match t.kind {
                    // (expr)
                    TokenKind::CloseParenthesis
                        if !self.lex.peek().is_some_and(|t| t.kind == TokenKind::Arrow) =>
                    {
                        return self
                            .alloc(expr!(Parenthesis { expr: first_expr }, span.join(t.span)));
                    },
                    // (expr) -> ...
                    // (params...) -> ...
                    TokenKind::CloseParenthesis | TokenKind::Comma => {
                        self.lex.set_state(start_state) // TODO: maybe no resetting in the future
                    },
                    _ => ParseError::unexpected_token(t).context("expected ',' or ')'")?,
                };
                let mut params = ScratchPool::new();
                loop {
                    let param = self.var_decl().context("function parameter")?;
                    params.push(param).map_err(|e| err!(x AllocErr(e), self.lex.pos_span()))?;
                    self.ws0();
                    let t = self.next_tok().context("expected ',' or ')'")?;
                    self.ws0();
                    match t.kind {
                        TokenKind::Comma if self.lex.advance_if(is_close_paren) => break,
                        TokenKind::Comma => self.ws0(),
                        TokenKind::CloseParenthesis => break,
                        _ => ParseError::unexpected_token(t).context("expected ',' or ')'")?,
                    }
                }
                let params = self.clone_slice_from_scratch_pool(params)?;
                self.ws0();
                self.tok(TokenKind::Arrow).context("'->'")?;
                return self.function_tail(params, span);
            },
            TokenKind::OpenBracket => {
                self.expr().context("[...]")?;
                let Ok(Token { .. }) = self.ws0_and_next_tok() else {
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
            TokenKind::Arrow => return self.function_tail(self.alloc_empty_slice(), span),
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
            t => return err!(UnexpectedToken(t), span).context("expected valid value"),
        };
        self.alloc(expr)
    }

    /// `struct { ... }`
    /// `         ^^^` Parses this
    pub fn struct_fields(&mut self) -> ParseResult<NonNull<[VarDecl]>> {
        todo!("struct_fields");
        let mut fields = ScratchPool::new();
        /*
        loop {
            fields
                .push(self.var_decl()?)
                .map_err(|e| err!(x AllocErr(e), self.lex.pos_span()))?;
        }
        */
        self.clone_slice_from_scratch_pool(fields)
    }

    /// parsing starts after the '->'
    pub fn function_tail(
        &mut self,
        params: NonNull<[VarDecl]>,
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
        mut args: ScratchPool<NonNull<Expr>>,
    ) -> ParseResult<NonNull<Expr>> {
        let closing_paren_span = loop {
            self.ws0();
            if let Some(closing_paren) = self.lex.next_if(is_close_paren) {
                break closing_paren.span;
            }
            args.push(self.expr()?).map_err(|e| err!(x AllocErr(e), self.lex.pos_span()))?;
            match self.ws0_and_next_tok()? {
                Token { kind: TokenKind::CloseParenthesis, span } => break span,
                Token { kind: TokenKind::Comma, .. } => continue,
                t => ParseError::unexpected_token(t).context("expect ',' or ')'")?,
            };
        };
        let args = self.clone_slice_from_scratch_pool(args)?;
        self.alloc(Expr::new(
            ExprKind::Call { func, args },
            open_paren_span.join(closing_paren_span),
        ))
    }

    /// parses block context and '}', doesn't parse the '{'
    pub fn block(&mut self, open_brace_span: Span) -> ParseResult<NonNull<Expr>> {
        let mut stmts = ScratchPool::new();
        let (has_trailing_semicolon, closing_brace_span) = loop {
            self.ws0();
            if let Some(closing_brace) = self.lex.next_if(|t| t.kind == TokenKind::CloseBrace) {
                break (true, closing_brace.span);
            }
            stmts.push(self.expr()?).map_err(|e| err!(x AllocErr(e), self.lex.pos_span()))?;
            match self.ws0_and_next_tok()? {
                Token { kind: TokenKind::CloseBrace, span } => break (false, span),
                Token { kind: TokenKind::Semicolon, .. } => continue,
                t => ParseError::unexpected_token(t).context("expect ';' or '}'")?,
            };
        };
        let stmts = self.clone_slice_from_scratch_pool(stmts)?;
        self.alloc(Expr::new(
            ExprKind::Block { stmts, has_trailing_semicolon },
            open_brace_span.join(closing_brace_span),
        ))
    }

    /// These options are allowed in a lot of places:
    /// * `[markers] ident`
    /// * `[markers] ident: ty`
    /// * `[markers] ident: ty = default`
    /// * `[markers] ident := default`
    ///
    /// That is the reason why this doesn't return a [`NonNull<Expr>`].
    pub fn var_decl(&mut self) -> ParseResult<VarDecl> {
        self.var_decl_with_markers(DeclMarkers::default())
    }

    /// This will still parse more markers, if they exists.
    pub fn var_decl_with_markers(&mut self, mut markers: DeclMarkers) -> ParseResult<VarDecl> {
        let mut t = self.next_tok().context("expected variable marker or ident")?;

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
                TokenKind::Ident => break Ident { span: t.span },
                TokenKind::Keyword(Keyword::Mut) => set_marker!(Mut is_mut),
                TokenKind::Keyword(Keyword::Rec) => set_marker!(Rec is_rec),
                TokenKind::Keyword(Keyword::Pub) => set_marker!(Pub is_pub),
                _ => ParseError::unexpected_token(t).context("expected decl marker or ident")?,
            }
            self.ws1()?;
            t = self.next_tok().context("expected variable marker or ident")?;
        };

        self.ws0();
        let t = self.lex.peek();
        let init = match t.map(|t| t.kind) {
            Some(TokenKind::Colon) => {
                self.lex.advance();
                return self.typed_decl(markers, ident);
            },
            Some(TokenKind::ColonEq | TokenKind::ColonColon) => {
                self.lex.advance();
                Some(self.expr().context("variable initialization")?)
            },
            _ => None,
        };
        let is_const = t.is_some_and(|t| t.kind == TokenKind::ColonColon);
        Ok(VarDecl { markers, ident, ty: None, default: init, is_const })
    }

    /// starts parsing after the colon:
    /// `mut a: int = 0;`
    /// `      ^`
    pub fn typed_decl(&mut self, markers: DeclMarkers, ident: Ident) -> ParseResult<VarDecl> {
        let ty = Some(self.expr().context("decl type")?);
        self.ws0();
        let t = self.lex.peek();
        let init = t
            .filter(|t| matches!(t.kind, TokenKind::Eq | TokenKind::Colon))
            .map(|_| self.expr().context("variable initialization"))
            .transpose()?;
        let is_const = t.is_some_and(|t| t.kind == TokenKind::Colon);
        Ok(VarDecl { markers, ident, ty, default: init, is_const })
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
        self.tok_where(|t| t.kind == tok).with_context(|| format!("token ({:?})", tok))
    }

    pub fn tok_where(&mut self, cond: impl FnOnce(Token) -> bool) -> ParseResult<Token> {
        let t = self.next_tok()?;
        if cond(t) { Ok(t) } else { ParseError::unexpected_token(t)? }
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
    fn alloc<T>(&self, val: T) -> ParseResult<NonNull<T>> {
        match self.alloc.try_alloc(val) {
            Result::Ok(ok) => Ok(NonNull::from(ok)),
            Result::Err(err) => err!(AllocErr(err), self.lex.pos_span()),
        }
    }

    /// # Source
    ///
    /// see [`bumpalo::Bump::alloc_slice_copy`]
    #[inline]
    fn alloc_slice<T: Copy>(&self, slice: &[T]) -> ParseResult<NonNull<[T]>> {
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

    #[inline]
    fn clone_slice_from_scratch_pool<T: Clone>(
        &self,
        scratch_pool: ScratchPool<T>,
    ) -> ParseResult<NonNull<[T]>> {
        scratch_pool
            .clone_to_slice_in_bump(&self.alloc)
            .map_err(|e| err!(x AllocErr(e), self.lex.pos_span()))
    }

    #[inline]
    fn alloc_empty_slice<T>(&self) -> NonNull<[T]> {
        NonNull::from(&[])
    }

    pub fn get_text(&self, range: Range<usize>) -> &str {
        &self.lex.get_code().0[range]
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
    /// `a: ty = b` or `a: ty : b`
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

#[derive(Debug, Clone, Copy)]
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

#[derive(Debug, Clone, Copy)]
pub struct VarDecl {
    pub markers: DeclMarkers,
    pub ident: Ident,
    pub ty: Option<NonNull<Expr>>,
    /// * default value for fn params, struct fields, ...
    /// * init for local veriable declarations
    pub default: Option<NonNull<Expr>>,
    pub is_const: bool,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct DeclMarkers {
    pub is_pub: bool,
    pub is_mut: bool,
    pub is_rec: bool,
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

/*
#[derive(Debug, Clone, Copy)]
pub enum VarDeclKind {
    /// `<name>: <ty>;`
    /// `<name>: <ty> = <init>;`
    WithTy { ty: NonNull<Expr>, init: Option<NonNull<Expr>> },
    /// `<name> := <init>;`
    InferTy { init: NonNull<Expr> },
}

impl VarDeclKind {
    pub fn get_init(&self) -> Option<&NonNull<Expr>> {
        match self {
            VarDeclKind::WithTy { init, .. } => init.as_ref(),
            VarDeclKind::InferTy { init } => Some(init),
        }
    }
}
*/

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

#[derive(Debug, Clone, Copy)]
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

#[inline]
fn is_close_paren(t: Token) -> bool {
    t.kind == TokenKind::CloseParenthesis
}

pub mod debug {
    use super::{lexer::Code, Expr, Ident};

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

        pub fn write_expr_tree(&mut self, expr: &Expr, code: &Code) {
            self.scope_next_line(|l| expr.write_tree(l, code));
            self.write_minus(expr.to_text(code).len());
        }

        pub fn write_ident(&mut self, ident: &Ident, code: &Code) {
            self.scope_next_line(|l| l.write(&code[ident.span]));
            self.write_minus(ident.span.len());
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
                        .map(|decl| Expr::var_decl_to_text(decl, code))
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
                ExprKind::UnionDef(..) => panic!(),
                ExprKind::EnumDef {} => panic!(),
                ExprKind::OptionShort(ty) => {
                    format!("?{}", ty.as_ref().to_text(code))
                },
                ExprKind::Ptr { is_mut, ty } => {
                    format!("*{}{}", mut_marker(*is_mut), ty.as_ref().to_text(code))
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
                ExprKind::Assign { lhs, rhs } => {
                    format!("{}={}", lhs.as_ref().to_text(code), rhs.as_ref().to_text(code))
                },
                ExprKind::BinOpAssign { lhs, op, rhs } => {
                    format!(
                        "{}{}{}",
                        lhs.as_ref().to_text(code),
                        op.to_binop_assign_text(),
                        rhs.as_ref().to_text(code)
                    )
                },
                ExprKind::VarDecl(decl) => Expr::var_decl_to_text(decl, code),
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

                        lines.write_expr_tree(e, code);
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

                    for (idx, decl) in params.as_ref().into_iter().enumerate() {
                        if idx != 0 {
                            lines.write(",");
                        }

                        Expr::var_decl_write_tree(decl, lines, code)
                    }
                    lines.write(")->");
                    match ret_type {
                        Some(ret_type) => {
                            let ret_type = ret_type.as_ref();
                            lines.write_expr_tree(ret_type, code);
                            lines.write("{");
                            lines.write_expr_tree(body, code);
                            lines.write("}");
                        },
                        None => {
                            lines.write_expr_tree(body, code);
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
                    lines.write_expr_tree(lhs, code);
                    lines.write(".");
                    lines.scope_next_line(|l| l.write(&code[rhs.span]));
                    lines.write_minus(rhs.span.len());
                },
                /*
                ExprKind::Colon { lhs, rhs } => {
                    lines.write_expr_tree(lhs, code);
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
                    lines.write_expr_tree(func, code);
                    lines.write("(");
                    let len = args.as_ref().len();
                    for (idx, arg) in args.as_ref().into_iter().enumerate() {
                        let arg = arg.as_ref();
                        lines.write_expr_tree(arg, code);
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
                    lines.write_expr_tree(expr, code);
                },
                ExprKind::BinOp { lhs, op, rhs } => {
                    let lhs = lhs.as_ref();
                    let rhs = rhs.as_ref();
                    lines.write_expr_tree(lhs, code);
                    lines.write(op.to_binop_text());
                    lines.write_expr_tree(rhs, code);
                },
                ExprKind::Assign { lhs, rhs } => {
                    let lhs = lhs.as_ref();
                    let rhs = rhs.as_ref();
                    lines.write_expr_tree(lhs, code);
                    lines.write("=");
                    lines.write_expr_tree(rhs, code);
                },
                ExprKind::BinOpAssign { lhs, op, rhs } => {
                    let lhs = lhs.as_ref();
                    let rhs = rhs.as_ref();
                    lines.write_expr_tree(lhs, code);
                    lines.write(op.to_binop_assign_text());
                    lines.write_expr_tree(rhs, code);
                },
                ExprKind::If { condition, then_body, else_body } => {
                    let condition = condition.as_ref();
                    let then_body = then_body.as_ref();
                    lines.write("if ");
                    lines.write_expr_tree(condition, code);
                    lines.write(" ");
                    lines.write_expr_tree(then_body, code);
                    if let Some(else_body) = else_body {
                        let else_body = else_body.as_ref();
                        lines.write(" else ");
                        lines.write_expr_tree(else_body, code);
                    }
                },
                ExprKind::VarDecl(decl) => Expr::var_decl_write_tree(decl, lines, code),
                ExprKind::Return { expr } => {
                    lines.write("return");
                    if let Some(expr) = expr {
                        let expr = expr.as_ref();
                        lines.write(" ");
                        lines.write_expr_tree(expr, code);
                    }
                },
                ExprKind::Semicolon(expr) => {
                    lines.write_expr_tree(expr.as_ref(), code);
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

    fn var_decl_to_text(
        VarDecl { markers, ident, ty, default, is_const }: &VarDecl,
        code: &Code,
    ) -> String {
        unsafe {
            format!(
                "{}{}{}{}{}{}",
                if markers.is_pub { "pub " } else { "" },
                mut_marker(markers.is_mut),
                if markers.is_rec { "rec " } else { "" },
                &code[ident.span],
                ty.map(|ty| format!(": {}", ty.as_ref().to_text(code))).unwrap_or_default(),
                default
                    .map(|default| format!(
                        " {}{} {}",
                        if ty.is_none() { ":" } else { "" },
                        if *is_const { ":" } else { "=" },
                        default.as_ref().to_text(code)
                    ))
                    .unwrap_or_default()
            )
        }
    }

    fn var_decl_write_tree(
        VarDecl { markers, ident, ty, default, is_const }: &VarDecl,
        lines: &mut TreeLines,
        code: &Code,
    ) {
        lines.write(&format!(
            "{}{}{}",
            if markers.is_pub { "pub " } else { "" },
            mut_marker(markers.is_mut),
            if markers.is_rec { "rec " } else { "" }
        ));

        lines.write_ident(ident, code);

        if let Some(ty) = ty {
            let ty = unsafe { ty.as_ref() };
            lines.write(": ");
            lines.write_expr_tree(ty, code);
        }

        if let Some(default) = default {
            let default = unsafe { default.as_ref() };
            lines.write(&format!(
                " {}{} ",
                if ty.is_none() { ":" } else { "" },
                if *is_const { ":" } else { "=" },
            ));
            lines.write_expr_tree(default, code);
        }
    }
}

#[inline]
fn mut_marker(is_mut: bool) -> &'static str {
    if is_mut { "mut " } else { "" }
}
