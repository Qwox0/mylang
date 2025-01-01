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
    util::{UnwrapDebug, collect_all_result_errors, display_spanned_error, replace_escape_chars},
};
use core::str;
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
            _ => err(NotAnIdent, self.full_span()),
        }
    }
}

pub fn ty(mut ty_expr: Ptr<Expr>) -> Type {
    if let ExprKind::Fn(f) = &mut ty_expr.kind {
        debug_assert_eq!(f.ret_type, Type::Unset);
        f.ret_type = Type::Unevaluated(f.body.take().unwrap_debug());
    }
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

    pub fn top_level_item(&mut self) -> ParseResult<Option<Ptr<Expr>>> {
        if self.lex.is_empty() {
            return Ok(None);
        }
        let res = self.expr();
        if let Err(ParseError { kind: ParseErrorKind::UnexpectedToken(_), .. }) = res {
            // skip over the unexpected token to prevent an infinite loop
            self.lex.advance();
        }
        self.lex
            .advance_while(|t| t.kind.is_whitespace() || t.kind == TokenKind::Semicolon);
        res.map(Some)
    }

    pub fn expr(&mut self) -> ParseResult<Ptr<Expr>> {
        self.expr_(MIN_PRECEDENCE)
    }

    pub fn expr_(&mut self, min_precedence: u8) -> ParseResult<Ptr<Expr>> {
        let mut lhs = self.ws0().value(/*min_precedence*/).context("expr first val")?;
        loop {
            match self.op_chain(lhs, min_precedence) {
                Ok(Some(node)) => lhs = node,
                Ok(None) => return Ok(lhs),
                Err(err) => return Err(err).context("expr op chain element"),
            };
        }
    }

    pub fn op_chain(
        &mut self,
        lhs: Ptr<Expr>,
        min_precedence: u8,
    ) -> ParseResult<Option<Ptr<Expr>>> {
        let Some(Token { kind, span }) = self.ws0().lex.peek() else { return Ok(None) };

        let op = match FollowingOperator::new(kind) {
            Some(op) if op.precedence() > min_precedence => op,
            _ => return Ok(None),
        };
        self.lex.advance();

        let expr = match op {
            FollowingOperator::Dot => {
                let rhs = self.ws0().ident().context("dot rhs")?;
                if &*rhs.text == "as" {
                    self.ws0().tok(TokenKind::OpenParenthesis).context("expected '('")?;
                    let target_ty = self.ws0().expr().context("cast target type")?;
                    let close_p =
                        self.ws0().tok(TokenKind::CloseParenthesis).context("expected ')'")?;
                    let span = span.join(close_p.span);
                    expr!(Cast { lhs: ExprWithTy::untyped(lhs), target_ty: ty(target_ty) }, span)
                } else {
                    expr!(Dot { lhs: Some(lhs), lhs_ty: Type::Unset, rhs }, span)
                }
            },
            FollowingOperator::Call => return self.call(lhs, ScratchPool::new(), None).map(Some),
            FollowingOperator::Index => {
                let lhs = ExprWithTy::untyped(lhs);
                let idx = ExprWithTy::untyped(self.expr()?);
                let close = self.tok(TokenKind::CloseBracket)?;
                expr!(Index { lhs, idx }, close.span)
            },
            FollowingOperator::PositionalInitializer => {
                let (args, close_p_span) = self.parse_call(ScratchPool::new())?;
                expr!(
                    PositionalInitializer { lhs: Some(lhs), lhs_ty: Type::Unset, args },
                    span.join(close_p_span)
                )
            },
            FollowingOperator::NamedInitializer => {
                let (fields, close_b_span) = self.parse_initializer_fields()?;
                expr!(
                    NamedInitializer { lhs: Some(lhs), lhs_ty: Type::Unset, fields },
                    span.join(close_b_span)
                )
            },
            FollowingOperator::ArrayInitializer => todo!("ArrayInitializer"),
            FollowingOperator::SingleArgNoParenFn => {
                let Ok(lhs) = lhs.try_to_ident() else { panic!("SingleArgFn: unknown rhs") };
                let param = VarDecl::new_basic(lhs, Type::Unset);
                let params = self.alloc_one_val_slice(param)?.into();
                return self.function_tail(params, span).map(Some);
            },
            FollowingOperator::PostOp(kind) => {
                let kind = if kind == UnaryOpKind::AddrOf
                    && self.ws0().lex.advance_if_kind(TokenKind::Keyword(Keyword::Mut))
                {
                    UnaryOpKind::AddrMutOf
                } else {
                    kind
                };
                expr!(UnaryOp { kind, expr: lhs, is_postfix: true }, span)
            },
            FollowingOperator::BinOp(op) => {
                let rhs = self.expr_(op.precedence())?;
                expr!(BinOp { lhs, op, rhs, arg_ty: Type::Unset }, span)
            },
            FollowingOperator::Range { is_inclusive } => {
                let end = self.expr_(op.precedence()).opt().context("range end")?;
                expr!(Range { start: Some(lhs), end, is_inclusive }, span)
            },
            FollowingOperator::Pipe => {
                let t = self.ws0().next_tok()?;
                match t.kind {
                    TokenKind::Keyword(Keyword::If) => {
                        return self.if_after_cond(lhs, span, true).context("| if").map(Some);
                    },
                    TokenKind::Keyword(Keyword::Match) => {
                        todo!("| match")
                    },
                    TokenKind::Keyword(Keyword::For) => {
                        let source = ExprWithTy::untyped(lhs);
                        let iter_var = self.ws0().ident().context("for iteration variable")?;
                        self.ws0().opt_do();
                        let body = self.ws0().expr().context("for body")?;
                        expr!(For { source, iter_var, body, was_piped: true }, t.span)
                    },
                    TokenKind::Keyword(Keyword::While) => {
                        self.ws0().opt_do();
                        let body = self.expr().context("while body")?;
                        expr!(While { condition: lhs, body, was_piped: true }, t.span)
                    },
                    TokenKind::Ident => {
                        let func = self.alloc(self.ident_from_span(t.span).into_expr())?;
                        self.tok(TokenKind::OpenParenthesis).context("pipe call: expect '('")?;
                        let args = self.scratch_pool_with_first_val(lhs)?;
                        return self.call(func, args, Some(0)).map(Some);
                    },
                    _ => ParseError::unexpected_token(t)
                        .context("expected fn call, 'if', 'match', 'for' or 'while'")?,
                }
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
        };
        self.alloc(expr).map(Some)
    }

    /// anything which has higher precedence than any operator
    //pub fn value(&mut self, min_precedence: u8) -> ParseResult<Ptr<Expr>> {
    pub fn value(&mut self) -> ParseResult<Ptr<Expr>> {
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
                    let decl = VarDecl::new_basic(variant_ident, ty);
                    variants.push(decl).map_err(|e| self.wrap_alloc_err(e))?;
                    if !self.ws0().lex.advance_if_kind(TokenKind::Comma) {
                        break;
                    }
                }
                let variants = self.clone_slice_from_scratch_pool(variants)?;
                let close_b = self.tok(TokenKind::CloseBrace).context("union '}'")?;
                expr!(EnumDef(variants), span.join(close_b.span))
            },
            TokenKind::Keyword(Keyword::Unsafe) => todo!("unsafe"),
            TokenKind::Keyword(Keyword::Extern) => {
                self.advanced().ws1()?;
                let ident = self.ident()?;
                self.ws0().tok(TokenKind::Colon)?;
                let ty_expr = self.expr().context("type of extern decl")?;
                expr!(Extern { ident, ty: ty(ty_expr) }, span.join(ty_expr.full_span()))
            },
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
                let iter_var = self.advanced().ws0().ident()?;
                // "in" is not a global keyword
                match self.ws0().tok(TokenKind::Ident) {
                    Ok(i) if &*self.get_text_from_span(i.span) == "in" => Ok(()),
                    Ok(t) => ParseError::unexpected_token(t),
                    Err(e) => Err(e),
                }
                .context("expected `in`")?;
                let source = ExprWithTy::untyped(self.ws0().expr().context("for .. in source")?);
                self.ws0().opt_do();
                let body = self.expr().context("for .. in body")?;
                expr!(For { source, iter_var, body, was_piped: false })
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
            TokenKind::Keyword(Keyword::Autocast) => {
                let expr =
                    self.advanced().expr().context("autocast expr").map(ExprWithTy::untyped)?;
                expr!(Autocast { expr }, span)
            },
            TokenKind::Keyword(Keyword::Defer) => {
                let expr = self.advanced().expr().context("defer expr")?;
                expr!(Defer(expr), span)
            },
            TokenKind::IntLit => expr!(IntLit(self.advanced().get_text_from_span(span))),
            TokenKind::FloatLit => expr!(FloatLit(self.advanced().get_text_from_span(span))),
            TokenKind::BoolLitTrue => {
                self.lex.advance();
                expr!(BoolLit(true))
            },
            TokenKind::BoolLitFalse => {
                self.lex.advance();
                expr!(BoolLit(false))
            },
            TokenKind::CharLit => {
                let code = replace_escape_chars(&self.advanced().lex.get_code()[span]);
                let mut chars = code.chars();

                let start = chars.next();
                debug_assert_eq!(start, Some('\''));
                let end = chars.next_back();
                debug_assert_eq!(end, Some('\''));

                let Some(c) = chars.next() else { return err(InvalidCharLit, span) };
                if chars.next().is_some() {
                    return err(InvalidCharLit, span);
                }
                expr!(CharLit(c))
            },
            TokenKind::BCharLit => {
                let code = replace_escape_chars(&self.advanced().lex.get_code()[span]);
                let mut bytes = code.bytes();

                let prefix = bytes.next();
                debug_assert_eq!(prefix, Some(b'b'));
                let start = bytes.next();
                debug_assert_eq!(start, Some(b'\''));
                let end = bytes.next_back();
                debug_assert_eq!(end, Some(b'\''));

                let Some(byte) = bytes.next() else { return err(InvalidBCharLit, span) };
                if bytes.next().is_some() {
                    return err(InvalidCharLit, span);
                }
                expr!(BCharLit(byte))
            },
            TokenKind::StrLit => {
                let lit = self.advanced().get_text_from_span(span);
                expr!(StrLit(Ptr::from(&lit[1..lit.len().saturating_sub(1)])))
            },
            TokenKind::MultilineStrLitLine => {
                // Note: bumpalo::Bump allocates in the wrong direction
                let mut scratch = Vec::with_capacity(1024);
                while let Some(t) = self.ws0().lex.next_if_kind(TokenKind::MultilineStrLitLine) {
                    scratch.extend_from_slice(self.get_text_from_span(t.span)[2..].as_bytes());
                }
                let bytes = self.alloc_slice(&scratch)?;
                let text = unsafe { std::str::from_utf8_unchecked(&bytes) };
                debug_assert!(text.ends_with('\n'));
                expr!(StrLit(Ptr::from(&text[0..text.len().saturating_sub(1)])))
            },
            TokenKind::OpenParenthesis => {
                self.advanced().ws0();
                // TODO: currently no tuples allowed!
                // () -> ...
                if self.lex.advance_if_kind(TokenKind::CloseParenthesis) {
                    self.ws0().tok(TokenKind::Arrow).context("'->'")?;
                    return self.function_tail(self.alloc_empty_slice().into(), span);
                }
                let first_expr = self.expr().context("expr in (...)")?; // this assumes the parameter syntax is also a valid expression
                let t = self.ws0().next_tok().context("missing ')'")?;
                self.ws0();
                let params = match t.kind {
                    // (expr)
                    TokenKind::CloseParenthesis if !self.lex.advance_if_kind(TokenKind::Arrow) => {
                        return self
                            .alloc(expr!(Parenthesis { expr: first_expr }, span.join(t.span)));
                    },
                    // (expr) -> ...
                    TokenKind::CloseParenthesis => {
                        let Some(decl) = first_expr.as_var_decl() else { todo!("better error") };
                        self.alloc(decl)?.as_slice1()
                    },
                    // (params...) -> ...
                    TokenKind::Comma => {
                        let Some(decl) = first_expr.as_var_decl() else { todo!("better error") };
                        let params = ScratchPool::new_with_first_val(decl)
                            .map_err(|e| self.wrap_alloc_err(e))?;
                        let params = self
                            .var_decl_list_with_start_list(TokenKind::Comma, params)
                            .context("function parameter list")?;
                        self.ws0().tok(TokenKind::CloseParenthesis).context("')'")?;
                        self.ws0().tok(TokenKind::Arrow).context("'->'")?;
                        params
                    },
                    _ => return ParseError::unexpected_token(t).context("expected ',' or ')'"),
                };
                return self.function_tail(params, span);
            },
            TokenKind::OpenBracket => {
                let count = self.advanced().ws0().expr().opt().context("array type count")?;
                self.tok(TokenKind::CloseBracket).context("array ty ']'")?;
                let is_mut = self.ws0().lex.advance_if_kind(TokenKind::Keyword(Keyword::Mut));
                let ty_expr = self.ws0().expr_(PREOP_PRECEDENCE)?;
                let ty_span = ty_expr.full_span();
                let ty = ty(ty_expr);
                match count {
                    Some(count) => expr!(ArrayTy { count, ty }, span.join(ty_span)),
                    None => expr!(SliceTy { ty, is_mut }, span.join(ty_span)),
                }
            },
            TokenKind::OpenBrace => return self.advanced().block(span).context("block"),
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
                return self.function_tail(self.alloc_empty_slice(), span);
            },
            TokenKind::Asterisk => {
                // TODO: deref prefix
                let is_mut =
                    self.advanced().ws0().lex.advance_if_kind(TokenKind::Keyword(Keyword::Mut));
                let pointee = self.ws0().expr_(PREOP_PRECEDENCE).context("pointee type")?;
                expr!(PtrTy { ty: ty(pointee), is_mut }, span.join(pointee.full_span()))
            },
            TokenKind::Ampersand => {
                let is_mut =
                    self.advanced().ws0().lex.advance_if_kind(TokenKind::Keyword(Keyword::Mut));
                let kind = if is_mut { UnaryOpKind::AddrMutOf } else { UnaryOpKind::AddrOf };
                let expr = self.expr_(PREOP_PRECEDENCE).context("& <expr>")?;
                expr!(UnaryOp { kind, expr, is_postfix: false })
            },
            TokenKind::Dot => {
                let rhs = self.advanced().ws0().ident().context("dot rhs")?;
                expr!(Dot { lhs: None, lhs_ty: Type::Unset, rhs }, span)
            },
            TokenKind::DotDot => {
                let end =
                    self.advanced().ws0().expr_(RANGE_PRECEDENCE).opt().context("range end")?;
                expr!(Range { start: None, end, is_inclusive: false })
            },
            TokenKind::DotDotEq => {
                let end =
                    self.advanced().ws0().expr_(RANGE_PRECEDENCE).opt().context("range end")?;
                if end.is_none() {
                    return err(ParseErrorKind::RangeInclusiveWithoutEnd, span);
                }
                expr!(Range { start: None, end, is_inclusive: true })
            },
            TokenKind::DotOpenParenthesis => {
                let (args, close_p_span) = self.advanced().ws0().parse_call(ScratchPool::new())?;
                expr!(
                    PositionalInitializer { lhs: None, lhs_ty: Type::Unset, args },
                    span.join(close_p_span)
                )
            },
            TokenKind::DotOpenBrace => {
                let (fields, close_b_span) = self.advanced().parse_initializer_fields()?;
                expr!(
                    NamedInitializer { lhs: None, lhs_ty: Type::Unset, fields },
                    span.join(close_b_span)
                )
            },
            TokenKind::DotOpenBracket => {
                macro_rules! new_arr_init {
                    ($elements:expr) => {{
                        let elements = $elements;
                        ExprKind::ArrayInitializer { lhs: None, lhs_ty: Type::Unset, elements }
                    }};
                }

                let Some(first_expr) =
                    self.advanced().ws0().expr().opt().context("first expr in .[...]")?
                else {
                    // `.[]`
                    let kind = new_arr_init!(self.alloc_empty_slice());
                    let close_b = self.tok(TokenKind::CloseBracket)?;
                    return self.alloc(Expr::new(kind, span.join(close_b.span)));
                };
                let t = self.ws0().peek_tok()?;
                let kind = match t.kind {
                    // `.[expr]`
                    TokenKind::CloseBracket => {
                        let elements = self.scratch_pool_with_first_val(first_expr)?;
                        new_arr_init!(self.clone_slice_from_scratch_pool(elements)?)
                    },
                    // `.[expr; count]`
                    TokenKind::Semicolon => {
                        self.advanced().ws0();
                        let count = self.expr().context("array literal short count")?;
                        ExprKind::ArrayInitializerShort {
                            lhs: None,
                            lhs_ty: Type::Unset,
                            val: first_expr,
                            count,
                        }
                    },
                    // `.[expr,]` or `.[expr, expr, ...]`
                    TokenKind::Comma => {
                        self.advanced().ws0();
                        let elems = self.scratch_pool_with_first_val(first_expr)?;
                        new_arr_init!(self.expr_list(TokenKind::Comma, elems)?.0)
                    },
                    _ => {
                        return ParseError::unexpected_token(t).context("expected ']', ';' or ','");
                    },
                };
                let close_b = self.tok(TokenKind::CloseBracket)?;
                Expr::new(kind, span.join(close_b.span))
            },
            TokenKind::Colon => todo!("TokenKind::Colon"),
            TokenKind::Question => {
                let type_ = self.advanced().expr_(PREOP_PRECEDENCE).expect("type after ?");
                expr!(OptionShort(ty(type_)), span.join(type_.full_span()))
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
        self.alloc(expr!(Fn(Fn { params, ret_type, body: Some(body) }), start_span))
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
    pub fn parse_call(
        &mut self,
        args: ScratchPool<Ptr<Expr>>,
    ) -> ParseResult<(Ptr<[Ptr<Expr>]>, Span)> {
        let res: ParseResult<_> = try {
            //let args = self.parse_call_args(args)?;
            let args = self.expr_list(TokenKind::Comma, args).context("call args")?.0;
            let closing_paren_span =
                self.tok(TokenKind::CloseParenthesis).context("expected ',' or ')'")?.span;
            (args, closing_paren_span)
        };
        if res.is_err() {
            let mut depth: usize = 0;
            self.lex.advance_while(|t| {
                match t.kind {
                    TokenKind::OpenParenthesis => depth += 1,
                    TokenKind::CloseParenthesis if depth == 0 => return false,
                    TokenKind::CloseParenthesis => depth -= 1,
                    _ => {},
                };
                true
            });
            self.lex.advance();
        }
        res
    }

    pub fn call(
        &mut self,
        func: Ptr<Expr>,
        args: ScratchPool<Ptr<Expr>>,
        pipe_idx: Option<usize>,
    ) -> ParseResult<Ptr<Expr>> {
        let func = ExprWithTy::untyped(func);
        let (args, closing_paren_span) = self.parse_call(args)?;
        self.alloc(expr!(Call { func, args, pipe_idx }, closing_paren_span))
    }

    /// parses block context and '}', doesn't parse the '{'
    pub fn block(&mut self, open_brace_span: Span) -> ParseResult<Ptr<Expr>> {
        let res = self._block_inner(open_brace_span);
        if res.is_err() {
            // skip the remaining block
            let mut depth: usize = 0;
            self.lex.advance_while(|t| {
                match t.kind {
                    TokenKind::OpenBrace => depth += 1,
                    TokenKind::CloseBrace if depth == 0 => return false,
                    TokenKind::CloseBrace => depth -= 1,
                    _ => {},
                };
                true
            });
            self.lex.advance();
        }
        res
    }

    #[inline]
    fn _block_inner(&mut self, open_brace_span: Span) -> ParseResult<Ptr<Expr>> {
        let mut list_pool = ScratchPool::new();
        let mut has_trailing_semicolon = false;
        loop {
            if matches!(
                self.ws0().lex.peek(),
                None | Some(Token { kind: TokenKind::CloseBrace, .. })
            ) {
                break;
            }
            let expr = self.expr().context("expr in block")?;
            list_pool.push(ExprWithTy::untyped(expr)).map_err(|e| self.wrap_alloc_err(e))?;
            has_trailing_semicolon = self.ws0().lex.advance_if_kind(TokenKind::Semicolon);
            if !has_trailing_semicolon && expr.kind.block_expects_trailing_semicolon() {
                break;
            }
        }
        let stmts = self.clone_slice_from_scratch_pool(list_pool)?;
        let closing_brace = self.tok(TokenKind::CloseBrace).context("expected ';' or '}'")?;
        let span = open_brace_span.join(closing_brace.span);
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

    /// Parses [`Parser::var_decl`] multiple times, seperated by `sep`. Also
    /// allows a trailing `sep`.
    pub fn var_decl_list(&mut self, sep: TokenKind) -> ParseResult<VarDeclList> {
        self.var_decl_list_with_start_list(sep, ScratchPool::new())
    }

    pub fn var_decl_list_with_start_list(
        &mut self,
        sep: TokenKind,
        mut list: ScratchPool<VarDecl>,
    ) -> ParseResult<VarDeclList> {
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
        self.clone_slice_from_scratch_pool(list)
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

        match self.ws0().lex.peek() {
            Some(Token { kind: TokenKind::Colon, .. }) => {
                self.advanced().typed_decl(markers, ident)
            },
            Some(t @ Token { kind: TokenKind::ColonEq | TokenKind::ColonColon, .. }) => {
                let init = Some(self.advanced().expr().context("variable initialization")?);
                let is_const = t.kind == TokenKind::ColonColon;
                let span = ident.span;
                Ok((VarDecl { markers, ident, ty: Type::Unset, default: init, is_const }, span))
            },
            Some(t) => ParseError::unexpected_token(t).context("expected ':', ':=' or '::'"),
            None => err(NoInput, self.lex.pos_span()),
        }
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
        Ident { text: self.get_text_from_span(span), span }
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

    /// [`ExprKind::PositionalInitializer`]
    /// `alloc(MyStruct).(1, "asdf")`
    /// `               ^^`
    PositionalInitializer,
    /// `alloc(MyStruct).{ a = 1, b = "asdf" }`
    /// `               ^^`
    /// [`ExprKind::NamedInitializer`]
    NamedInitializer,
    /// [`ExprKind::ArrayInitializer`]
    ArrayInitializer,

    /// `arg -> ...`
    /// `    ^^`
    SingleArgNoParenFn,

    /// `a op`
    /// `  ^^`
    PostOp(UnaryOpKind),

    /// `a op b`
    /// `  ^^`
    BinOp(BinOpKind),

    Range {
        is_inclusive: bool,
    },

    /// `a |> b`
    Pipe,

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
            TokenKind::Pipe => FollowingOperator::BinOpAssign(BinOpKind::BitOr),
            TokenKind::PipePipe => FollowingOperator::BinOp(BinOpKind::Or),
            TokenKind::PipePipeEq => FollowingOperator::BinOpAssign(BinOpKind::Or),
            TokenKind::PipeEq => FollowingOperator::BinOpAssign(BinOpKind::BitOr),
            TokenKind::PipeGt => FollowingOperator::Pipe,
            TokenKind::Caret => FollowingOperator::BinOp(BinOpKind::BitXor),
            TokenKind::CaretEq => FollowingOperator::BinOpAssign(BinOpKind::BitXor),
            TokenKind::Dot => FollowingOperator::Dot,
            TokenKind::DotDot => FollowingOperator::Range { is_inclusive: false },
            TokenKind::DotDotEq => FollowingOperator::Range { is_inclusive: true },
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
            | FollowingOperator::PostOp(_)
            | FollowingOperator::SingleArgNoParenFn => 22,
            FollowingOperator::PositionalInitializer
            | FollowingOperator::NamedInitializer
            | FollowingOperator::ArrayInitializer => 20,
            FollowingOperator::BinOp(k) => k.precedence(),
            FollowingOperator::Range { .. } => RANGE_PRECEDENCE,
            FollowingOperator::Pipe => 4,
            FollowingOperator::Assign | FollowingOperator::BinOpAssign(_) => 3,
            FollowingOperator::VarDecl
            | FollowingOperator::ConstDecl
            | FollowingOperator::TypedDecl => 2,
        }
    }
}

const MIN_PRECEDENCE: u8 = 0;
const IF_PRECEDENCE: u8 = 1;
/// also for `*ty`, `[]ty`, `?ty`
const PREOP_PRECEDENCE: u8 = 21;
const RANGE_PRECEDENCE: u8 = 10;
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
        self.parser.top_level_item().transpose()
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
