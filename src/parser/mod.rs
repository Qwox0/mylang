//! # Parser module
//!
//! The parser allocates the AST ([`Expr`]) and the stored [`Type`]s to
//! [`Type::Unset`] or [`Type::Unevaluated`]

use crate::{
    ast::{
        BinOpKind, DeclMarkerKind, DeclMarkers, Expr, ExprKind, ExprWithTy, Fn, Ident, UnaryOpKind,
        VarDecl, VarDeclList, debug::DebugAst,
    },
    ptr::Ptr,
    scratch_pool::ScratchPool,
    type_::Type,
    util::{collect_all_result_errors, display_spanned_error},
};
use error::ParseErrorKind::*;
pub use error::*;
use lexer::{Code, Keyword, Lexer, Span, Token, TokenKind};
use parser_helper::ParserInterface;
use std::str::FromStr;

pub mod error;
pub mod lexer;
pub mod parser_helper;

macro_rules! expr {
    ($kind:ident, $span:expr) => {
        Expr::new(ExprKind::$kind, $span)
    };
    ($kind:ident($( $val:expr ),* $(,)?), $span:expr ) => {
        Expr::new(ExprKind::$kind($($val),*), $span)
    };
    ($kind:ident { $( $field:ident $( : $val:expr )? ),* $(,)? }, $span:expr $(,)? ) => {
        Expr::new(ExprKind::$kind{$($field $(:$val)?),*}, $span)
    };
}
use expr as expr_;

impl Expr {
    pub fn try_to_ident(&self) -> ParseResult<Ident> {
        match self.kind {
            ExprKind::Ident(text) => Ok(Ident { text, span: self.span }),
            _ => err(NotAnIdent, self.span),
        }
    }
}

pub fn ty(ty_expr: Ptr<Expr>) -> Type {
    Type::Unevaluated(ty_expr)
}

#[derive(Debug, Clone)]
pub struct Parser<'code, 'alloc> {
    lex: Lexer<'code>,
    alloc: &'alloc bumpalo::Bump,
}

impl<'code, 'alloc> Parser<'code, 'alloc> {
    pub fn new(lex: Lexer<'code>, alloc: &'alloc bumpalo::Bump) -> Parser<'code, 'alloc> {
        Self { lex, alloc }
    }

    pub fn top_level_item(&mut self) -> ParseResult<Ptr<Expr>> {
        let res = match self.expr() {
            Err(e @ ParseError { kind: ParseErrorKind::NoInput, .. }) => {
                Err(ParseError { kind: ParseErrorKind::Finished, ..e })
            },
            res @ Err(ParseError { kind: ParseErrorKind::UnexpectedToken(_), .. }) => {
                // skip over the unexpected token to prevent an infinite loop
                self.lex.advance();
                res
            },
            res => res,
        };
        self.lex
            .advance_while(|t| t.kind.is_whitespace() || t.kind == TokenKind::Semicolon);
        res
    }

    pub fn expr(&mut self) -> ParseResult<Ptr<Expr>> {
        self.expr_(MIN_PRECEDENCE)
    }

    pub fn expr_(&mut self, min_precedence: u8) -> ParseResult<Ptr<Expr>> {
        let mut lhs = self.ws0().value(min_precedence).context("expr first val")?;
        loop {
            match self.op_chain(lhs, min_precedence) {
                Ok(node) => lhs = node,
                Err(ParseError { kind: ParseErrorKind::Finished, .. }) => return Ok(lhs),
                err => return err.context("expr op chain element"),
            };
        }
    }

    pub fn op_chain(&mut self, lhs: Ptr<Expr>, min_precedence: u8) -> ParseResult<Ptr<Expr>> {
        let Some(Token { kind, span }) = self.ws0().lex.peek() else {
            return err(Finished, self.lex.pos_span());
        };

        let op = match FollowingOperator::new(kind) {
            Some(op) if op.precedence() > min_precedence => op,
            _ => return err(Finished, span),
        };
        self.lex.advance();

        let expr = match op {
            FollowingOperator::Dot => {
                let rhs = self.ws0().ident().context("dot rhs")?;
                expr!(Dot { lhs: Some(lhs), lhs_ty: Type::Unset, rhs }, span)
            },
            FollowingOperator::Call => return self.call(lhs, ScratchPool::new(), None),
            FollowingOperator::Index => {
                let lhs = ExprWithTy::untyped(lhs);
                let idx = ExprWithTy::untyped(self.expr()?);
                let close = self.tok(TokenKind::CloseBracket)?;
                expr!(Index { lhs, idx }, close.span)
            },
            FollowingOperator::PositionalInitializer => {
                let args = self.parse_call_args(ScratchPool::new())?;
                let close_p = self.tok(TokenKind::CloseParenthesis)?;
                expr!(
                    PositionalInitializer { lhs: Some(lhs), lhs_ty: Type::Unset, args },
                    span.join(close_p.span)
                )
            },
            FollowingOperator::NamedInitializer => {
                let (fields, close_b_span) = self.parse_initializer_fields()?;
                expr!(
                    NamedInitializer { lhs: Some(lhs), lhs_ty: Type::Unset, fields },
                    span.join(close_b_span)
                )
            },
            FollowingOperator::SingleArgNoParenFn => {
                let Ok(lhs) = lhs.try_to_ident() else { panic!("SingleArgFn: unknown rhs") };
                let param = VarDecl {
                    markers: DeclMarkers::default(),
                    ident: lhs,
                    ty: Type::Unset,
                    default: None,
                    is_const: false,
                };
                let params = self.alloc_one_val_slice(param)?.into();
                return self.function_tail(params, span);
            },
            FollowingOperator::PostOp(kind) => {
                expr!(UnaryOp { kind, expr: lhs, is_postfix: true }, span)
            },
            FollowingOperator::BinOp(op) => {
                let rhs = self.expr_(op.precedence())?;
                expr!(BinOp { lhs, op, rhs, arg_ty: Type::Unset }, span)
            },
            FollowingOperator::Assign => {
                let lhs = ExprWithTy::untyped(lhs);
                let rhs = self.expr()?;
                expr!(Assign { lhs, rhs }, span)
            },
            FollowingOperator::BinOpAssign(op) => {
                let lhs = ExprWithTy::untyped(lhs);
                let rhs = self.expr()?;
                expr!(BinOpAssign { lhs, op, rhs }, span)
            },
            FollowingOperator::VarDecl => {
                let markers = DeclMarkers::default();
                let ident = lhs.try_to_ident().context("var decl ident")?;
                let init = self.expr().context(":= init")?;
                let decl = VarDecl {
                    markers,
                    ident,
                    ty: Type::Unset,
                    default: Some(init),
                    is_const: false,
                };
                expr!(VarDecl(decl), ident.span)
            },
            FollowingOperator::ConstDecl => {
                let markers = DeclMarkers::default();
                let ident = lhs.try_to_ident().context("const decl ident")?;
                let init = self.expr().context(":: init")?;
                let decl = VarDecl {
                    markers,
                    ident,
                    ty: Type::Unset,
                    default: Some(init),
                    is_const: true,
                };
                expr!(VarDecl(decl), ident.span)
            },
            FollowingOperator::TypedDecl => {
                let ident = lhs.try_to_ident().context("const decl ident")?;
                let (decl, span) = self.typed_decl(DeclMarkers::default(), ident)?;
                expr!(VarDecl(decl), ident.span.join(span))
            },
            FollowingOperator::Pipe => {
                let t = self.ws0().next_tok()?;
                return match t.kind {
                    TokenKind::Keyword(Keyword::If) => {
                        self.if_after_cond(lhs, span, true).context("if")
                    },
                    TokenKind::Keyword(Keyword::Match) => {
                        todo!("| match")
                    },
                    TokenKind::Keyword(Keyword::For) => {
                        let source = ExprWithTy::untyped(lhs);
                        let iter_var = self.ws0().ident().context("for iteration variable");
                        // TODO: better compiler errors
                        // println!("{:?}", iter_var);
                        // println!("{:?}", self.lex.peek());
                        // println!("{:?}", self.lex.pos_span());
                        // display_span_in_code_with_label(
                        //     self.lex.pos_span(),
                        //     self.lex.get_code(),
                        //     "pos",
                        // );
                        // todo!();
                        let iter_var = iter_var?;

                        self.ws0().opt_do();
                        let body = self.ws0().expr().context("for body")?;
                        self.alloc(expr!(For { source, iter_var, body, was_piped: true }, t.span))
                    },
                    TokenKind::Keyword(Keyword::While) => {
                        self.ws0().opt_do();
                        let body = self.expr().context("while body")?;
                        self.alloc(expr!(While { condition: lhs, body, was_piped: true }, t.span))
                    },
                    TokenKind::Ident => {
                        let func = self.alloc(self.ident_from_span(t.span).into_expr())?;
                        self.tok(TokenKind::OpenParenthesis).context("pipe call: expect '('")?;
                        let args = self.scratch_pool_with_first_val(lhs)?;
                        self.call(func, args, Some(0))
                    },
                    _ => ParseError::unexpected_token(t)
                        .context("expected fn call, 'if', 'match', 'for' or 'while'")?,
                };
            },
        };
        self.alloc(expr)
    }

    /// anything which has higher precedence than any operator
    pub fn value(&mut self, min_precedence: u8) -> ParseResult<Ptr<Expr>> {
        let Token { kind, span } = self.peek_tok().context("expected value")?;

        macro_rules! expr {
            ($kind:ident) => {
                expr_!($kind, span)
            };
            ($kind:ident ( $( $val:expr ),* $(,)? ) ) => {
                expr_!($kind($($val)*), span)
            };
            ($kind:ident { $( $field:ident $( : $val:expr )? ),* $(,)? } ) => {
                expr_!($kind{$($field $(:$val)?),*}, span)
            };
            ($($t:tt)*) => {
                expr_!($($t)*)
            }
        }

        let expr = match kind {
            TokenKind::Ident => expr!(Ident(self.advanced().get_text_from_span(span))),
            TokenKind::Keyword(Keyword::Mut | Keyword::Rec | Keyword::Pub) => {
                let (decl, span_end) = self.var_decl()?;
                expr!(VarDecl(decl), span.join(span_end))
            },
            TokenKind::Keyword(Keyword::Struct) => {
                self.advanced().ws0().tok(TokenKind::OpenBrace).context("struct '{'")?;
                let fields = self.ws0().var_decl_list(TokenKind::Comma)?;
                let close_b = self.tok(TokenKind::CloseBrace).context("struct '}'")?;
                expr!(StructDef(fields), span.join(close_b.span))
            },
            TokenKind::Keyword(Keyword::Union) => {
                self.advanced().ws0().tok(TokenKind::OpenBrace).context("struct '{'")?;
                let fields = self.ws0().var_decl_list(TokenKind::Comma)?;
                let close_b = self.tok(TokenKind::CloseBrace).context("union '}'")?;
                expr!(UnionDef(fields), span.join(close_b.span))
            },
            TokenKind::Keyword(Keyword::Enum) => {
                self.advanced().ws0().tok(TokenKind::OpenBrace).context("enum '{'")?;
                let mut variants = ScratchPool::new();
                loop {
                    if self.ws0().peek_tok()?.is_invalid_start() {
                        break;
                    }
                    let variant_ident = self.ident().context("enum variant ident")?;
                    let ty = if self.ws0().lex.advance_if_kind(TokenKind::OpenParenthesis) {
                        let ty = ty(self.expr().context("variant type")?);
                        self.ws0().tok(TokenKind::CloseParenthesis)?;
                        ty
                    } else {
                        Type::Void
                    };
                    let decl = VarDecl {
                        markers: DeclMarkers::default(),
                        ident: variant_ident,
                        default: None,
                        ty,
                        is_const: false,
                    };
                    variants.push(decl).map_err(|e| self.wrap_alloc_err(e))?;
                    if !self.ws0().lex.advance_if_kind(TokenKind::Comma) {
                        break;
                    }
                }
                let variants = self.clone_slice_from_scratch_pool(variants)?;
                let close_b = self.tok(TokenKind::CloseBrace).context("union '}'")?;
                expr!(EnumDef(VarDeclList(variants)), span.join(close_b.span))
            },
            TokenKind::Keyword(Keyword::Unsafe) => todo!("unsafe"),
            TokenKind::Keyword(Keyword::If) => {
                let condition = self.advanced().expr().context("if condition")?;
                return self.if_after_cond(condition, span, false).context("if");
            },
            TokenKind::Keyword(Keyword::Match) => {
                todo!("match body");
                let val = self.advanced().expr().context("match value")?;
                let else_body = self
                    .ws0()
                    .lex
                    .next_if_kind(TokenKind::Keyword(Keyword::Else))
                    .map(|_else| {
                        self.ws1()?;
                        self.expr().context("match else body")
                    })
                    .transpose()?;
                expr!(Match { val, else_body, was_piped: false }, span)
            },
            TokenKind::Keyword(Keyword::For) => {
                todo!("for");
                // let (source, iter_var, body) = todo!();
                // expr!(For { source, iter_var, body }, span)
            },
            TokenKind::Keyword(Keyword::While) => {
                let condition = self.advanced().expr().context("while condition")?;
                self.ws0().opt_do();
                let body = self.expr().context("while body")?;
                expr!(While { condition, body, was_piped: false }, span)
            },
            TokenKind::Keyword(Keyword::Return) => {
                let expr =
                    self.advanced().expr().opt().context("return expr")?.map(ExprWithTy::untyped);
                expr!(Return { expr }, span)
            },
            TokenKind::Keyword(Keyword::Break) => {
                let expr =
                    self.advanced().expr().opt().context("break expr")?.map(ExprWithTy::untyped);
                expr!(Break { expr }, span)
            },
            TokenKind::Keyword(Keyword::Continue) => {
                self.lex.advance();
                expr!(Continue, span)
            },
            TokenKind::Keyword(Keyword::Defer) => {
                let expr = self.advanced().expr().context("defer expr")?;
                expr!(Defer(expr), span)
            },
            TokenKind::Literal(kind) => {
                expr!(Literal { kind, code: self.advanced().get_text_from_span(span) })
            },
            TokenKind::BoolLit(b) => {
                self.lex.advance();
                expr!(BoolLit(b))
            },
            TokenKind::OpenParenthesis => {
                self.advanced().ws0();
                // TODO: currently no tuples allowed!
                // () -> ...
                if self.lex.advance_if_kind(TokenKind::CloseParenthesis) {
                    self.ws0().tok(TokenKind::Arrow).context("'->'")?;
                    return self.function_tail(self.alloc_empty_slice().into(), span);
                }
                let start_state = self.lex.get_state();
                let first_expr = self.expr().context("expr in (...)")?; // this assumes the parameter syntax is also a valid expression
                let t = self.ws0().next_tok().context("missing ')'")?;
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
                let params =
                    self.var_decl_list(TokenKind::Comma).context("function parameter list")?;
                self.ws0().tok(TokenKind::CloseParenthesis).context("')'")?;
                self.ws0().tok(TokenKind::Arrow).context("'->'")?;
                return self.function_tail(params, span);
            },
            TokenKind::OpenBracket => {
                let Some(first_expr) =
                    self.advanced().ws0().expr().opt().context("first expr in [...]")?
                else {
                    // `[]` or `[]ty`
                    let close_b = self.tok(TokenKind::CloseBracket)?;
                    let arr = match self.expr_(min_precedence).opt()? {
                        Some(ty) => expr!(ArrayTy2 { ty }),
                        None => expr!(
                            ArrayLit { elements: self.alloc_empty_slice() },
                            span.join(close_b.span)
                        ),
                    };
                    return self.alloc(arr);
                };

                let t = self.ws0().next_tok()?;
                let kind = match t.kind {
                    // `[count]ty` or `[expr]`
                    TokenKind::CloseBracket => {
                        let arr = match self.expr_(min_precedence).opt()? {
                            Some(ty) => expr!(ArrayTy { count: first_expr, ty }),
                            None => {
                                let elements = self.scratch_pool_with_first_val(first_expr)?;
                                let elements = self.clone_slice_from_scratch_pool(elements)?;
                                let expr = expr!(ArrayLit { elements }, span.join(t.span));
                                expr
                            },
                        };
                        return self.alloc(arr);
                    },
                    // `[expr; count]`
                    TokenKind::Semicolon => {
                        let count = self.expr().context("array literal short count")?;
                        ExprKind::ArrayLitShort { val: first_expr, count }
                    },
                    // `[expr,]` or `[expr, expr, ...]`
                    TokenKind::Comma => {
                        let elems = self.scratch_pool_with_first_val(first_expr)?;
                        let (elements, _) = self.expr_list(TokenKind::Comma, elems)?;
                        ExprKind::ArrayLit { elements }
                    },
                    _ => {
                        return ParseError::unexpected_token(t).context("expected ']', ';' or ','");
                    },
                };

                let close_b = self.tok(TokenKind::CloseBracket)?;
                Expr::new(kind, span.join(close_b.span))
            },
            TokenKind::OpenBrace => return self.advanced().block(span),
            TokenKind::Bang => {
                let expr = self.advanced().expr_(PREOP_PRECEDENCE).context("! expr")?;
                expr!(UnaryOp { kind: UnaryOpKind::Not, expr, is_postfix: false }, span)
            },
            TokenKind::Plus => todo!("TokenKind::Plus"),
            TokenKind::Minus => {
                let expr = self.advanced().expr_(PREOP_PRECEDENCE).context("- expr")?;
                expr!(UnaryOp { kind: UnaryOpKind::Neg, expr, is_postfix: false }, span)
            },
            TokenKind::Arrow => {
                self.lex.advance();
                return self.function_tail(self.alloc_empty_slice().into(), span);
            },
            TokenKind::Asterisk => {
                // TODO: deref prefix
                self.advanced().ws0();
                let is_mut = self.lex.advance_if_kind(TokenKind::Keyword(Keyword::Mut));
                let pointee = self.expr().context("pointee type")?;
                expr!(Ptr { is_mut, ty: ty(pointee) })
            },
            TokenKind::Ampersand => {
                let is_mut =
                    self.advanced().ws0().lex.advance_if_kind(TokenKind::Keyword(Keyword::Mut));
                let kind = if is_mut { UnaryOpKind::AddrMutOf } else { UnaryOpKind::AddrOf };
                let expr = self.expr_(PREOP_PRECEDENCE).context("& <expr>")?;
                expr!(UnaryOp { kind, expr, is_postfix: false })
            },
            //TokenKind::Pipe => todo!("TokenKind::Pipe"),
            //TokenKind::PipePipe => todo!("TokenKind::PipePipe"),
            //TokenKind::PipeEq => todo!("TokenKind::PipeEq"),
            //TokenKind::Caret => todo!("TokenKind::Caret"),
            //TokenKind::CaretEq => todo!("TokenKind::CaretEq"),
            TokenKind::Dot => {
                let rhs = self.advanced().ws0().ident().context("dot rhs")?;
                expr!(Dot { lhs: None, lhs_ty: Type::Unset, rhs }, span)
            },
            //TokenKind::DotAsterisk => todo!("TokenKind::DotAsterisk"),
            //TokenKind::DotAmpersand => todo!("TokenKind::DotAmpersand"),
            TokenKind::DotOpenBrace => {
                let (fields, close_b_span) = self.advanced().parse_initializer_fields()?;
                expr!(
                    NamedInitializer { lhs: None, lhs_ty: Type::Unset, fields },
                    span.join(close_b_span)
                )
            },
            //TokenKind::Comma => todo!("TokenKind::Comma"),
            TokenKind::Colon => todo!("TokenKind::Colon"),
            //TokenKind::ColonColon => todo!("TokenKind::ColonColon"),
            //TokenKind::ColonEq => todo!("TokenKind::ColonEq"),
            TokenKind::Question => {
                let type_ = self.advanced().expr().expect("type after ?");
                expr!(OptionShort(ty(type_)), span.join(type_.span))
            },
            TokenKind::Pound => todo!("TokenKind::Pound"),
            TokenKind::Dollar => todo!("TokenKind::Dollar"),
            TokenKind::At => todo!("TokenKind::At"),
            TokenKind::Tilde => todo!("TokenKind::Tilde"),
            TokenKind::Backslash => todo!("TokenKind::BackSlash"),
            TokenKind::Backtick => todo!("TokenKind::BackTick"),
            t => return err(UnexpectedToken(t), span).context("expected valid value"),
        };

        self.alloc(expr)
    }

    /// also parses the `}`
    pub fn parse_initializer_fields(
        &mut self,
    ) -> ParseResult<(Ptr<[(Ident, Option<Ptr<Expr>>)]>, Span)> {
        let mut fields = ScratchPool::new();
        let close_b_span = loop {
            if let Some(t) = self.ws0().lex.next_if_kind(TokenKind::CloseBrace) {
                break t.span;
            }
            let ident = self.ident().context("initializer field ident")?;
            let init = self
                .ws0()
                .lex
                .next_if_kind(TokenKind::Eq)
                .map(|_| self.expr().context("init expr"))
                .transpose()?;
            fields.push((ident, init)).map_err(|e| self.wrap_alloc_err(e))?;

            match self.next_tok() {
                Ok(Token { kind: TokenKind::Comma, .. }) => {},
                Ok(Token { kind: TokenKind::CloseBrace, span }) => break span,
                t => {
                    t.and_then(ParseError::unexpected_token).context("expected '=', ',' or '}'")?
                },
            }
        };
        Ok((self.clone_slice_from_scratch_pool(fields)?, close_b_span))
    }

    /// parsing starts after the '->'
    pub fn function_tail(
        &mut self,
        params: VarDeclList,
        start_span: Span,
    ) -> ParseResult<Ptr<Expr>> {
        let expr = self.expr().context("fn return type or body")?;
        let (ret_type, body) = match self.lex.next_if_kind(TokenKind::OpenBrace) {
            Some(brace) => (ty(expr), self.block(brace.span).context("fn body")?),
            None => (Type::Unset, expr),
        };
        self.alloc(expr!(Fn(Fn { params, ret_type, body }), start_span))
    }

    pub fn if_after_cond(
        &mut self,
        condition: Ptr<Expr>,
        start_span: Span,
        was_piped: bool,
    ) -> ParseResult<Ptr<Expr>> {
        self.ws0().opt_do();
        let then_body = self.expr_(IF_PRECEDENCE).context("then body")?;
        let else_body = self
            .ws0()
            .lex
            .next_if_kind(TokenKind::Keyword(Keyword::Else))
            .map(|_| self.expr_(IF_PRECEDENCE).context("else body"))
            .transpose()?;
        self.alloc(expr!(If { condition, then_body, else_body, was_piped }, start_span))
    }

    /// `... ( ... )`
    /// `     ^` starts here
    /// TODO: `... ( <expr>, ..., param=<expr>, ... )`
    pub fn parse_call_args(
        &mut self,
        args: ScratchPool<Ptr<Expr>>,
    ) -> ParseResult<Ptr<[Ptr<Expr>]>> {
        Ok(self.expr_list(TokenKind::Comma, args).context("call args")?.0)
    }

    /// `... ( ... )`
    /// `     ^` starts here
    /// TODO: `... ( <expr>, ..., param=<expr>, ... )`
    pub fn call(
        &mut self,
        func: Ptr<Expr>,
        args: ScratchPool<Ptr<Expr>>,
        pipe_idx: Option<usize>,
    ) -> ParseResult<Ptr<Expr>> {
        let func = ExprWithTy::untyped(func);
        let args = self.parse_call_args(args)?;
        let closing_paren_span =
            self.tok(TokenKind::CloseParenthesis).context("expected ',' or ')'")?.span;
        self.alloc(expr!(Call { func, args, pipe_idx }, closing_paren_span))
    }

    /// parses block context and '}', doesn't parse the '{'
    pub fn block(&mut self, open_brace_span: Span) -> ParseResult<Ptr<Expr>> {
        let (stmts, has_trailing_semicolon) =
            self.expr_with_ty_list(TokenKind::Semicolon, ScratchPool::new())?;
        let closing_brace_span =
            self.tok(TokenKind::CloseBrace).context("expected ';' or '}'")?.span;
        let span = open_brace_span.join(closing_brace_span);
        self.alloc(expr!(Block { stmts, has_trailing_semicolon }, span))
    }

    /// Also returns a `has_trailing_sep` [`bool`].
    pub fn expr_list(
        &mut self,
        sep: TokenKind,
        mut list_pool: ScratchPool<Ptr<Expr>>,
    ) -> ParseResult<(Ptr<[Ptr<Expr>]>, bool)> {
        let mut has_trailing_sep = false;
        loop {
            let Some(expr) = self.expr().opt().context("expr in list")? else { break };
            list_pool.push(expr).map_err(|e| self.wrap_alloc_err(e))?;
            has_trailing_sep = self.ws0().lex.advance_if_kind(sep);
            if !has_trailing_sep {
                break;
            }
            self.ws0();
        }
        Ok((self.clone_slice_from_scratch_pool(list_pool)?, has_trailing_sep))
    }

    pub fn expr_with_ty_list(
        &mut self,
        sep: TokenKind,
        mut list_pool: ScratchPool<ExprWithTy>,
    ) -> ParseResult<(Ptr<[ExprWithTy]>, bool)> {
        let mut has_trailing_sep = false;
        loop {
            let Some(expr) = self.expr().opt().context("expr in list")? else { break };
            list_pool.push(ExprWithTy::untyped(expr)).map_err(|e| self.wrap_alloc_err(e))?;
            has_trailing_sep = self.ws0().lex.advance_if_kind(sep);
            if !has_trailing_sep {
                break;
            }
            self.ws0();
        }
        Ok((self.clone_slice_from_scratch_pool(list_pool)?, has_trailing_sep))
    }

    /// Parses [`Parser::var_decl`] multiple times, seperated by `sep`. Also
    /// allows a trailing `sep`.
    pub fn var_decl_list(&mut self, sep: TokenKind) -> ParseResult<VarDeclList> {
        let mut list = ScratchPool::new();
        loop {
            let Some((decl, _end_span)) = self.var_decl().opt().context("var_decl")? else {
                break;
            };
            list.push(decl).map_err(|e| self.wrap_alloc_err(e))?;
            if !self.ws0().lex.advance_if_kind(sep) {
                break;
            }
            self.ws0();
        }
        self.clone_slice_from_scratch_pool(list).map(VarDeclList)
    }

    pub fn var_decl(&mut self) -> ParseResult<(VarDecl, Span)> {
        let mut markers = DeclMarkers::default();
        let mut t = self.peek_tok().context("expected variable marker or ident")?;

        macro_rules! set_marker {
            ($variant:ident $field:ident) => {
                if markers.$field {
                    return err(DuplicateDeclMarker(DeclMarkerKind::$variant), t.span);
                } else {
                    markers.$field = true
                }
            };
        }

        let ident = loop {
            match t.kind {
                TokenKind::Ident => break self.advanced().ident_from_span(t.span),
                TokenKind::Keyword(Keyword::Mut) => set_marker!(Mut is_mut),
                TokenKind::Keyword(Keyword::Rec) => set_marker!(Rec is_rec),
                TokenKind::Keyword(Keyword::Pub) => set_marker!(Pub is_pub),
                _ => ParseError::unexpected_token(t).context("expected decl marker or ident")?,
            }
            self.advanced().ws1()?;
            t = self.peek_tok().context("expected variable marker or ident")?;
        };

        let t = self.ws0().lex.peek();
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
        let span = ident.span;
        Ok((VarDecl { markers, ident, ty: Type::Unset, default: init, is_const }, span))
    }

    /// starts parsing after the colon:
    /// `mut a: int = 0;`
    /// `      ^`
    ///
    /// Also returns the [`Span`] of the parsed type.
    pub fn typed_decl(
        &mut self,
        markers: DeclMarkers,
        ident: Ident,
    ) -> ParseResult<(VarDecl, Span)> {
        let ty_expr = self.expr_(DECL_TYPE_PRECEDENCE).context("decl type")?;
        let t = self.ws0().lex.next_if(|t| matches!(t.kind, TokenKind::Eq | TokenKind::Colon));
        let init = t.map(|_| self.expr().context("variable initialization")).transpose()?;
        let is_const = t.is_some_and(|t| t.kind == TokenKind::Colon);
        let ty = ty(ty_expr);
        Ok((VarDecl { markers, ident, ty, default: init, is_const }, ty_expr.span))
    }

    pub fn ident(&mut self) -> ParseResult<Ident> {
        self.tok(TokenKind::Ident).map(|t| self.ident_from_span(t.span))
    }

    /// this doesn't check if the text at span is valid
    pub fn ident_from_span(&self, span: Span) -> Ident {
        Ident { text: self.lex.get_code()[span].into(), span }
    }

    /// Parses the `do` keyword 0 or 1 times.
    fn opt_do(&mut self) {
        #[allow(unused_must_use)] // `do` is optional
        self.tok(TokenKind::Keyword(Keyword::Do));
    }

    // -------

    /// 0+ whitespace
    pub fn ws0(&mut self) -> &mut Self {
        self.lex.advance_while(|t| t.kind.is_whitespace());
        self
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
        let t = self.peek_tok()?;
        if cond(t) {
            self.lex.advance();
            Ok(t)
        } else {
            ParseError::unexpected_token(t)?
        }
    }

    pub fn advanced(&mut self) -> &mut Self {
        self.lex.advance();
        self
    }

    #[inline]
    pub fn next_tok(&mut self) -> ParseResult<Token> {
        self.lex.next().ok_or(err_val(NoInput, self.lex.pos_span()))
    }

    #[inline]
    pub fn peek_tok(&mut self) -> ParseResult<Token> {
        self.lex.peek().ok_or(err_val(NoInput, self.lex.pos_span()))
    }

    // helpers:

    #[inline]
    fn alloc<T>(&self, val: T) -> ParseResult<Ptr<T>> {
        match self.alloc.try_alloc(val) {
            Result::Ok(ok) => Ok(Ptr::from(ok)),
            Result::Err(e) => err(AllocErr(e), self.lex.pos_span()),
        }
    }

    /// # Source
    ///
    /// see [`bumpalo::Bump::alloc_slice_copy`]
    #[inline]
    #[allow(unused)]
    fn alloc_slice<T: Copy>(&self, slice: &[T]) -> ParseResult<Ptr<[T]>> {
        let layout = core::alloc::Layout::for_value(slice);
        let dst = self
            .alloc
            .try_alloc_layout(layout)
            .map_err(|err| self.wrap_alloc_err(err))?
            .cast::<T>();

        Ok(Ptr::from(unsafe {
            core::ptr::copy_nonoverlapping(slice.as_ptr(), dst.as_ptr(), slice.len());
            core::slice::from_raw_parts_mut(dst.as_ptr(), slice.len())
        }))
    }

    fn scratch_pool_with_first_val<'bump, T>(
        &self,
        first: T,
    ) -> ParseResult<ScratchPool<'bump, T>> {
        ScratchPool::new_with_first_val(first).map_err(|e| self.wrap_alloc_err(e))
    }

    /// Clones all values from a [`ScratchPool`] to `self.alloc`.
    #[inline]
    fn clone_slice_from_scratch_pool<T: Clone>(
        &self,
        scratch_pool: ScratchPool<T>,
    ) -> ParseResult<Ptr<[T]>> {
        scratch_pool
            .clone_to_slice_in_bump(&self.alloc)
            .map_err(|e| self.wrap_alloc_err(e))
    }

    #[inline]
    fn wrap_alloc_err(&self, err: bumpalo::AllocErr) -> ParseError {
        err_val(AllocErr(err), self.lex.pos_span())
    }

    #[inline]
    fn alloc_empty_slice<T>(&self) -> Ptr<[T]> {
        Ptr::from(&[] as &[T])
    }

    #[inline]
    fn alloc_one_val_slice<T>(&self, val: T) -> ParseResult<Ptr<[T]>> {
        let mut ptr = self.alloc(val)?;
        Ok(unsafe { core::slice::from_raw_parts_mut(ptr.as_mut() as *mut T, 1).into() })
    }

    fn get_text_from_span(&self, span: Span) -> Ptr<str> {
        self.lex.get_code()[span].into()
    }
}

#[derive(Debug)]
pub enum FollowingOperator {
    /// `a.b`
    /// ` ^`
    Dot,
    /// `a(b)`
    /// ` ^`
    Call,
    /// `a[b]`
    /// ` ^`
    Index,

    /// `alloc(MyStruct).(1, "asdf")`
    /// `               ^^`
    PositionalInitializer,
    /// `alloc(MyStruct).{ a = 1, b = "asdf" }`
    /// `               ^^`
    NamedInitializer,

    /// `arg -> ...`
    /// `    ^^`
    SingleArgNoParenFn,

    /// `a op`
    /// `  ^^`
    PostOp(UnaryOpKind),

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
            TokenKind::AmpersandAmpersandEq => FollowingOperator::BinOpAssign(BinOpKind::And),
            TokenKind::AmpersandEq => FollowingOperator::BinOpAssign(BinOpKind::BitAnd),
            TokenKind::Pipe => {
                // TODO: find a solution for pipe vs bitor (currently bitand, bitxor and bitor
                // are ignored)
                FollowingOperator::Pipe
            },
            TokenKind::PipePipe => FollowingOperator::BinOp(BinOpKind::Or),
            TokenKind::PipePipeEq => FollowingOperator::BinOpAssign(BinOpKind::Or),
            TokenKind::PipeEq => FollowingOperator::BinOpAssign(BinOpKind::BitOr),
            TokenKind::Caret => FollowingOperator::BinOp(BinOpKind::BitXor),
            TokenKind::CaretEq => FollowingOperator::BinOpAssign(BinOpKind::BitXor),
            TokenKind::Dot => FollowingOperator::Dot,
            TokenKind::DotDot => FollowingOperator::BinOp(BinOpKind::Range),
            TokenKind::DotDotEq => FollowingOperator::BinOp(BinOpKind::RangeInclusive),
            TokenKind::DotAsterisk => FollowingOperator::PostOp(UnaryOpKind::Deref),
            TokenKind::DotAmpersand => FollowingOperator::PostOp(UnaryOpKind::AddrOf),
            TokenKind::DotOpenParenthesis => FollowingOperator::PositionalInitializer,
            TokenKind::DotOpenBrace => FollowingOperator::NamedInitializer,
            TokenKind::Colon => FollowingOperator::TypedDecl,
            TokenKind::ColonColon => FollowingOperator::ConstDecl,
            TokenKind::ColonEq => FollowingOperator::VarDecl,
            //TokenKind::Semicolon => todo!("TokenKind::Semicolon"),
            TokenKind::Question => FollowingOperator::PostOp(UnaryOpKind::Try),
            TokenKind::Pound => todo!("TokenKind::Pound"),
            TokenKind::Dollar => todo!("TokenKind::Dollar"),
            TokenKind::At => todo!("TokenKind::At"),
            TokenKind::Tilde => todo!("TokenKind::Tilde"),
            TokenKind::Backslash => todo!("TokenKind::BackSlash"),
            TokenKind::Backtick => todo!("TokenKind::BackTick"),
            _ => return None,
        })
    }

    fn precedence(&self) -> u8 {
        match self {
            FollowingOperator::Dot
            | FollowingOperator::Call
            | FollowingOperator::Index
            | FollowingOperator::PositionalInitializer
            | FollowingOperator::NamedInitializer
            | FollowingOperator::SingleArgNoParenFn
            | FollowingOperator::PostOp(_) => 21,
            FollowingOperator::BinOp(k) => k.precedence(),
            FollowingOperator::Assign | FollowingOperator::BinOpAssign(_) => 3,
            FollowingOperator::VarDecl
            | FollowingOperator::ConstDecl
            | FollowingOperator::TypedDecl => 2,
            FollowingOperator::Pipe => 4, // TODO: higher or lower then decl?
        }
    }
}

const MIN_PRECEDENCE: u8 = 0;
const IF_PRECEDENCE: u8 = 1;
const PREOP_PRECEDENCE: u8 = 20;
/// `a: ty = init`
/// `   ^^`
/// must be higher than [`FollowingOperator::Assign`]!
const DECL_TYPE_PRECEDENCE: u8 = 4;

impl BinOpKind {
    pub fn precedence(self) -> u8 {
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
}

#[derive(Debug, Clone)]
pub struct StmtIter<'code, 'alloc> {
    parser: Parser<'code, 'alloc>,
}

impl<'code, 'alloc> Iterator for StmtIter<'code, 'alloc> {
    type Item = ParseResult<Ptr<Expr>>;

    fn next(&mut self) -> Option<Self::Item> {
        match self.parser.top_level_item() {
            Err(ParseError { kind: ParseErrorKind::Finished, .. }) => None,
            res => Some(res),
        }
    }
}

impl<'code, 'alloc> StmtIter<'code, 'alloc> {
    /// Parses top-level items until the end of the [`Code`] or until an
    /// [`PError`] occurs.
    #[inline]
    pub fn parse(code: &'code Code, alloc: &'alloc bumpalo::Bump) -> Self {
        let mut parser = Parser::new(Lexer::new(code), alloc);
        parser.ws0();
        Self { parser }
    }

    #[inline]
    pub fn collect_or_fail(self, code: &'code Code) -> Vec<Ptr<Expr>> {
        collect_all_result_errors(self).unwrap_or_else(|errors| {
            for e in errors {
                display_spanned_error(&e, code);
            }
            std::process::exit(1);
        })
    }

    #[inline]
    pub fn parse_all_or_fail(code: &'code Code, alloc: &'alloc bumpalo::Bump) -> Vec<Ptr<Expr>> {
        Self::parse(code, alloc).collect_or_fail(code)
    }

    pub fn try_parse_all(
        code: &'code Code,
        alloc: &'alloc bumpalo::Bump,
    ) -> Result<Vec<Ptr<Expr>>, Vec<ParseError>> {
        collect_all_result_errors(Self::parse(code, alloc))
    }

    pub fn parse_and_debug(code: &'code Code) -> Result<(), ()> {
        let mut has_err = false;
        for s in StmtIter::parse(code, &bumpalo::Bump::new()) {
            match s {
                Ok(s) => {
                    println!("stmt @ {:?}", s);
                    s.print_tree();
                },
                Err(e) => {
                    display_spanned_error(&e, code);
                    has_err = true;
                },
            }
        }
        if has_err { Err(()) } else { Ok(()) }
    }
}

impl Ident {
    pub fn try_from_tok(t: Token, lex: &Lexer<'_>) -> ParseResult<Ident> {
        let text = &lex.get_code()[t.span];
        if Keyword::from_str(text).is_ok() {
            return err(NotAnIdent, lex.pos_span());
        }
        Ok(Ident { text: text.into(), span: t.span })
    }
}
