//! # Parser module
//!
//! The parser allocates the AST ([`Expr`]) and the stored [`Type`]s to
//! [`Type::Unset`] or [`Type::Unevaluated`]

use crate::{
    arena_allocator::Arena,
    ast::{
        self, Ast, BinOpKind, Decl, DeclList, DeclMarkerKind, DeclMarkers, Ident, UnaryOpKind,
        UpcastToAst, ast_new,
    },
    context::{CompilationContextInner, CtxDiagnosticReporter},
    diagnostic_reporter::DiagnosticReporter,
    literals::{self, replace_escape_chars},
    ptr::{OPtr, Ptr},
    scratch_pool::ScratchPool,
    util::then,
};
use core::str;
use error::ParseErrorKind::*;
pub use error::*;
use lexer::{Keyword, Lexer, Span, Token, TokenKind};
use parser_helper::ParserInterface;

pub mod error;
pub mod lexer;
pub mod parser_helper;

macro_rules! expr_ {
    ($kind:ident { $( $field:ident $( : $val:expr )? ),* $(,)? }, $span:expr $(,)? ) => {
        ast_new!($kind { span: $span, $($field $(:$val)?),* })
    };
}

impl Ptr<Ast> {
    pub fn try_to_ident(self) -> ParseResult<Ptr<ast::Ident>> {
        self.try_downcast::<ast::Ident>()
            .ok_or_else(|| err_val(NotAnIdent, self.full_span()))
    }
}

pub fn parse(cctx: &mut CompilationContextInner) -> OPtr<ast::Block> {
    let mut parser = Parser::new(cctx);
    parser.ws0();
    parser.top_level_scope()
}

pub struct Parser<'cctx> {
    lex: Lexer<'cctx>,
    // scope: OPtr<ast::Block>,
    dr: &'cctx mut CtxDiagnosticReporter,
    alloc: &'cctx Arena,
}

impl<'cctx> Parser<'cctx> {
    fn new(cctx: &'cctx mut CompilationContextInner) -> Parser<'cctx> {
        Self { lex: Lexer::new(&cctx.code), dr: &mut cctx.diagnostic_reporter, alloc: &cctx.alloc }
    }

    fn top_level_scope(&mut self) -> OPtr<ast::Block> {
        match self.block_inner() {
            Ok(block) => Some(block),
            Err(e) => {
                self.dr.error(&e);
                None
            },
        }
    }

    fn expr(&mut self) -> ParseResult<Ptr<Ast>> {
        self.expr_(MIN_PRECEDENCE)
    }

    fn expr_(&mut self, min_precedence: u8) -> ParseResult<Ptr<Ast>> {
        let mut lhs = self.ws0().value(/*min_precedence*/).context("expr first val")?;
        loop {
            match self.op_chain(lhs, min_precedence) {
                Ok(Some(node)) => lhs = node,
                Ok(None) => return Ok(lhs),
                Err(err) => return Err(err).context("expr op chain element"),
            };
        }
    }

    fn op_chain(&mut self, lhs: Ptr<Ast>, min_precedence: u8) -> ParseResult<Option<Ptr<Ast>>> {
        let Some(Token { kind, span }) = self.ws0().lex.peek() else { return Ok(None) };

        let op = match FollowingOperator::new(kind) {
            Some(op) if op.precedence() > min_precedence => op,
            _ => return Ok(None),
        };
        self.lex.advance();

        macro_rules! expr {
            ($kind:ident { $( $field:ident $( : $val:expr )? ),* $(,)? }, $span:expr $(,)? ) => {
                self.alloc(expr_!($kind { $($field $(:$val)?),* }, $span))?.upcast()
            };
            ($expr:expr) => {
                self.alloc($expr)?.upcast()
            };
        }

        Ok(Some(match op {
            FollowingOperator::Dot => {
                let rhs = self.ws0().ident().context("dot rhs")?;
                if &*rhs.text == "as" {
                    self.ws0().tok(TokenKind::OpenParenthesis).context("expected '('")?;
                    let target_ty = self.ws0().expr().context("cast target type")?;
                    let close_p =
                        self.ws0().tok(TokenKind::CloseParenthesis).context("expected ')'")?;
                    let span = span.join(close_p.span);
                    expr!(Cast { expr: lhs, target_ty }, span)
                } else {
                    expr!(Dot { lhs: Some(lhs), has_lhs: true, rhs }, span)
                }
            },
            FollowingOperator::Call => self.call(lhs, ScratchPool::new(), None)?.upcast(),
            FollowingOperator::Index => {
                let idx = self.expr()?;
                let close = self.tok(TokenKind::CloseBracket)?;
                expr!(Index { lhs, idx }, close.span)
            },
            FollowingOperator::PositionalInitializer => {
                let (args, close_p_span) = self.parse_call(ScratchPool::new())?;
                let span = span.join(close_p_span);
                expr!(PositionalInitializer { lhs: Some(lhs), args }, span)
            },
            FollowingOperator::NamedInitializer => {
                let (fields, close_b_span) = self.parse_initializer_fields()?;
                let span = span.join(close_b_span);
                expr!(NamedInitializer { lhs: Some(lhs), fields }, span)
            },
            FollowingOperator::ArrayInitializer => todo!("ArrayInitializer"),
            FollowingOperator::SingleArgNoParenFn => {
                let Ok(lhs) = lhs.try_to_ident() else { panic!("SingleArgFn: unknown rhs") };
                let param = self.alloc(ast::Decl::from_ident(lhs))?;
                let params = self.alloc_one_val_slice(param)?;
                self.function_tail(params, span)?.upcast()
            },
            FollowingOperator::PostOp(mut op) => {
                if op == UnaryOpKind::AddrOf
                    && self.ws0().lex.advance_if_kind(TokenKind::Keyword(Keyword::Mut))
                {
                    op = UnaryOpKind::AddrMutOf
                }
                expr!(UnaryOp { op, expr: lhs, is_postfix: true }, span)
            },
            FollowingOperator::BinOp(op) => {
                let rhs = self.expr_(op.precedence())?;
                expr!(BinOp { lhs, op, rhs }, span)
            },
            FollowingOperator::Range { is_inclusive } => {
                let end = self.expr_(op.precedence()).opt().context("range end")?;
                expr!(Range { start: Some(lhs), end, is_inclusive }, span)
            },
            FollowingOperator::Pipe => {
                let t = self.ws0().next_tok()?;
                match t.kind {
                    TokenKind::Keyword(Keyword::If) => {
                        self.if_after_cond(lhs, span, true).context("| if")?.upcast()
                    },
                    TokenKind::Keyword(Keyword::Match) => {
                        todo!("| match")
                    },
                    TokenKind::Keyword(Keyword::For) => {
                        let iter_var = self.ws0().ident().context("for iteration variable")?;
                        self.ws0().opt_do();
                        let body = self.ws0().expr().context("for body")?;
                        expr!(For { source: lhs, iter_var, body, was_piped: true }, t.span)
                    },
                    TokenKind::Keyword(Keyword::While) => {
                        self.ws0().opt_do();
                        let body = self.expr().context("while body")?;
                        expr!(While { condition: lhs, body, was_piped: true }, t.span)
                    },
                    TokenKind::Ident => {
                        let text = self.get_text_from_span(t.span);
                        let func = self.alloc(ast::Ident::new(text, t.span))?.upcast();
                        self.tok(TokenKind::OpenParenthesis).context("pipe call: expect '('")?;
                        let args = self.scratch_pool_with_first_val(lhs)?;
                        self.call(func, args, Some(0))?.upcast()
                    },
                    _ => ParseError::unexpected_token(t)
                        .context("expected fn call, 'if', 'match', 'for' or 'while'")?,
                }
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
                let ident = lhs.try_to_ident().context("var decl ident")?;
                let mut d = ast::Decl::from_ident(ident);
                d.init = Some(self.expr().context(":= init")?);
                expr!(d)
            },
            FollowingOperator::ConstDecl => {
                let ident = lhs.try_to_ident().context("const decl ident")?;
                let mut d = ast::Decl::from_ident(ident);
                d.init = Some(self.expr().context(":: init")?);
                d.is_const = true;
                expr!(d)
            },
            FollowingOperator::TypedDecl => {
                let ident = lhs.try_to_ident().context("const decl ident")?;
                let mut d = self.typed_decl(DeclMarkers::default(), ident)?;
                d.span = ident.span.join(d.span);
                d.upcast()
            },
        }))
    }

    /// anything which has higher precedence than any operator
    //pub fn value(&mut self, min_precedence: u8) -> ParseResult<Ptr<Ast>> {
    fn value(&mut self) -> ParseResult<Ptr<Ast>> {
        let Token { kind, span } = self.peek_tok().context("expected value")?;

        macro_rules! expr {
            ($kind:ident { $( $field:ident $( : $val:expr )? ),* $(,)? }) => { {
                let expr = expr_!($kind { $($field $(:$val)?),* }, span);
                self.alloc(expr)?.upcast()
            } };
            ($kind:ident { $( $field:ident $( : $val:expr )? ),* $(,)? }, $span:expr $(,)? ) => { {
                let expr = expr_!($kind { $($field $(:$val)?),* }, $span);
                self.alloc(expr)?.upcast()
            } };
            ($expr:expr) => {
                self.alloc($expr)?.upcast()
            };
        }

        Ok(match kind {
            TokenKind::Ident => {
                let i = Ident::new(self.advanced().get_text_from_span(span), span);
                self.alloc(i)?.upcast()
            },
            TokenKind::Keyword(Keyword::Mut | Keyword::Rec | Keyword::Pub) => {
                self.var_decl()?.upcast()
            },
            TokenKind::Keyword(Keyword::Struct) => {
                self.advanced().ws0().tok(TokenKind::OpenBrace).context("struct '{'")?;
                let fields = self.ws0().var_decl_list(TokenKind::Comma)?;
                let close_b = self.tok(TokenKind::CloseBrace).context("struct '}'")?;
                expr!(StructDef { fields }, span.join(close_b.span))
            },
            TokenKind::Keyword(Keyword::Union) => {
                self.advanced().ws0().tok(TokenKind::OpenBrace).context("struct '{'")?;
                let fields = self.ws0().var_decl_list(TokenKind::Comma)?;
                let close_b = self.tok(TokenKind::CloseBrace).context("union '}'")?;
                expr!(UnionDef { fields }, span.join(close_b.span))
            },
            TokenKind::Keyword(Keyword::Enum) => {
                self.advanced().ws0().tok(TokenKind::OpenBrace).context("enum '{'")?;
                let mut variants = ScratchPool::new();
                loop {
                    if self.ws0().peek_tok()?.is_invalid_start() {
                        break;
                    }
                    let variant_ident = self.ident().context("enum variant ident")?;
                    let ty = then!(
                        self.ws0().lex.advance_if_kind(TokenKind::OpenParenthesis) => {
                        let ty_expr = self.expr().context("variant type")?;
                        self.ws0().tok(TokenKind::CloseParenthesis)?;
                        ty_expr
                    });
                    let mut decl = self.alloc(ast::Decl::new(variant_ident, span))?;
                    decl.var_ty_expr = ty;
                    variants.push(decl).map_err(|e| self.wrap_alloc_err(e))?;
                    if !self.ws0().lex.advance_if_kind(TokenKind::Comma) {
                        break;
                    }
                }
                let variants = self.clone_slice_from_scratch_pool(variants)?;
                let close_b = self.tok(TokenKind::CloseBrace).context("union '}'")?;
                expr!(EnumDef { variants }, span.join(close_b.span))
            },
            TokenKind::Keyword(Keyword::Unsafe) => todo!("unsafe"),
            TokenKind::Keyword(Keyword::Extern) => {
                self.advanced().ws1()?;
                let ident = self.ident()?;
                let mut d = ast::Decl::from_ident(ident);
                d.span = span;
                d.is_extern = true;
                self.ws0().tok(TokenKind::Colon)?;
                d.var_ty_expr = Some(self.expr().context("type of extern decl")?);
                expr!(d)
            },
            TokenKind::Keyword(Keyword::If) => {
                let condition = self.advanced().expr().context("if condition")?;
                self.if_after_cond(condition, span, false).context("if")?.upcast()
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
                let source = self.ws0().expr().context("for .. in source")?;
                self.ws0().opt_do();
                let body = self.expr().context("for .. in body")?;
                expr!(For { source, iter_var, body, was_piped: false }, span)
            },
            TokenKind::Keyword(Keyword::While) => {
                let condition = self.advanced().expr().context("while condition")?;
                self.ws0().opt_do();
                let body = self.expr().context("while body")?;
                expr!(While { condition, body, was_piped: false }, span)
            },
            TokenKind::Keyword(Keyword::Return) => {
                let expr = self.advanced().expr().opt().context("return expr")?;
                expr!(Return { expr, parent_fn: None }, span)
            },
            TokenKind::Keyword(Keyword::Break) => {
                let expr = self.advanced().expr().opt().context("break expr")?;
                expr!(Break { expr }, span)
            },
            TokenKind::Keyword(Keyword::Continue) => {
                self.lex.advance();
                expr!(Continue {}, span)
            },
            TokenKind::Keyword(Keyword::Autocast) => {
                let expr = self.advanced().expr().context("autocast expr")?;
                expr!(Autocast { expr }, span)
            },
            TokenKind::Keyword(Keyword::Defer) => {
                let expr = self.advanced().expr().context("defer expr")?;
                expr!(Defer { expr }, span)
            },
            TokenKind::IntLit => {
                let res = literals::parse_int_lit(&self.advanced().get_text_from_span(span));
                expr!(IntVal { val: res.map_err(|e| err_val(InvalidIntLit(e), span))? })
            },
            TokenKind::FloatLit => {
                let res = literals::parse_float_lit(&self.advanced().get_text_from_span(span));
                expr!(FloatVal { val: res.map_err(|e| err_val(InvalidFloatLit(e), span))? })
            },
            TokenKind::BoolLitTrue | TokenKind::BoolLitFalse => {
                self.lex.advance();
                expr!(BoolVal { val: kind == TokenKind::BoolLitTrue })
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
                expr!(CharVal { val: c })
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
                //expr!(BCharLit { val: byte })
                expr!(IntVal { val: byte as i128 })
            },
            TokenKind::StrLit => {
                let lit = self.advanced().get_text_from_span(span);
                expr!(StrVal { text: Ptr::from(&lit[1..lit.len().saturating_sub(1)]) })
            },
            TokenKind::MultilineStrLitLine => {
                // Note: Arena allocates in the wrong direction
                let mut scratch = Vec::with_capacity(1024);
                while let Some(t) = self.ws0().lex.next_if_kind(TokenKind::MultilineStrLitLine) {
                    scratch.extend_from_slice(self.get_text_from_span(t.span)[2..].as_bytes());
                }
                let bytes = self.alloc_slice(&scratch)?;
                let text = unsafe { std::str::from_utf8_unchecked(&bytes) };
                debug_assert!(text.ends_with('\n'));
                expr!(StrVal { text: Ptr::from(&text[0..text.len().saturating_sub(1)]) })
            },
            TokenKind::OpenParenthesis => {
                self.advanced().ws0();
                // TODO: currently no tuples allowed!
                // () -> ...
                if self.lex.advance_if_kind(TokenKind::CloseParenthesis) {
                    self.ws0().tok(TokenKind::Arrow).context("'->'")?;
                    return Ok(self.function_tail(self.alloc_empty_slice(), span)?.upcast());
                }
                let mut first_expr = self.expr().context("expr in (...)")?; // this assumes the parameter syntax is also a valid expression
                let t = self.ws0().next_tok().context("missing ')'")?;
                self.ws0();
                let params = match t.kind {
                    // (expr)
                    TokenKind::CloseParenthesis if !self.lex.advance_if_kind(TokenKind::Arrow) => {
                        first_expr.parenthesis_count += 1;
                        return Ok(first_expr);
                    },
                    // (expr) -> ...
                    TokenKind::CloseParenthesis => {
                        let decl = if let Some(i) = first_expr.try_downcast::<ast::Ident>() {
                            self.alloc(ast::Decl::from_ident(i))?
                        } else if let Some(decl) = first_expr.try_downcast::<ast::Decl>() {
                            decl
                        } else {
                            todo!("better error")
                        };
                        self.alloc_slice(&[decl])?
                    },
                    // (params...) -> ...
                    TokenKind::Comma => {
                        let first_decl = if let Some(i) = first_expr.try_downcast::<ast::Ident>() {
                            self.alloc(ast::Decl::from_ident(i))?
                        } else if let Some(decl) = first_expr.try_downcast::<ast::Decl>() {
                            decl
                        } else {
                            todo!("better error")
                        };
                        let params = ScratchPool::new_with_first_val(first_decl)
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
                self.function_tail(params, span)?.upcast()
            },
            TokenKind::OpenBracket => {
                let len = self.advanced().ws0().expr().opt().context("array type count")?;
                self.tok(TokenKind::CloseBracket).context("array ty ']'")?;
                let is_mut = self.ws0().lex.advance_if_kind(TokenKind::Keyword(Keyword::Mut));
                let elem_ty = self.ws0().expr_(PREOP_PRECEDENCE)?;
                let span = span.join(elem_ty.span);
                match len {
                    Some(len) => expr!(ArrayTy { len, elem_ty }, span),
                    None => expr!(SliceTy { elem_ty, is_mut }, span),
                }
            },
            TokenKind::OpenBrace => self.advanced().block(span).context("block")?.upcast(),
            TokenKind::Bang => {
                let expr = self.advanced().expr_(PREOP_PRECEDENCE).context("! expr")?;
                expr!(UnaryOp { op: UnaryOpKind::Not, expr, is_postfix: false }, span)
            },
            TokenKind::Plus => todo!("TokenKind::Plus"),
            TokenKind::Minus => {
                let expr = self.advanced().expr_(PREOP_PRECEDENCE).context("- expr")?;
                expr!(UnaryOp { op: UnaryOpKind::Neg, expr, is_postfix: false }, span)
            },
            TokenKind::Arrow => {
                self.lex.advance();
                self.function_tail(self.alloc_empty_slice(), span)?.upcast()
            },
            TokenKind::Asterisk => {
                // TODO: deref prefix
                let is_mut =
                    self.advanced().ws0().lex.advance_if_kind(TokenKind::Keyword(Keyword::Mut));
                let pointee = self.ws0().expr_(PREOP_PRECEDENCE).context("pointee type")?;
                expr!(PtrTy { pointee, is_mut }, span.join(pointee.full_span()))
            },
            TokenKind::Ampersand => {
                let is_mut =
                    self.advanced().ws0().lex.advance_if_kind(TokenKind::Keyword(Keyword::Mut));
                let op = if is_mut { UnaryOpKind::AddrMutOf } else { UnaryOpKind::AddrOf };
                let expr = self.expr_(PREOP_PRECEDENCE).context("& <expr>")?;
                expr!(UnaryOp { op, expr, is_postfix: false })
            },
            TokenKind::Dot => {
                let rhs = self.advanced().ws0().ident().context("dot rhs")?;
                expr!(ast::Dot::new(None, rhs, span))
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
                let span = span.join(close_p_span);
                expr!(PositionalInitializer { lhs: None, args }, span)
            },
            TokenKind::DotOpenBrace => {
                let (fields, close_b_span) = self.advanced().parse_initializer_fields()?;
                let span = span.join(close_b_span);
                expr!(NamedInitializer { lhs: None, fields }, span)
            },
            TokenKind::DotOpenBracket => {
                macro_rules! new_arr_init {
                    ($elements:expr, $span:expr) => {
                        expr!(ArrayInitializer { lhs: None, elements: $elements }, $span)
                    };
                }

                let Some(first_expr) =
                    self.advanced().ws0().expr().opt().context("first expr in .[...]")?
                else {
                    // `.[]`
                    let close_b = self.tok(TokenKind::CloseBracket)?;
                    return Ok(new_arr_init!(self.alloc_empty_slice(), span.join(close_b.span)));
                };
                let t = self.ws0().peek_tok()?;
                let mut kind = match t.kind {
                    // `.[expr]`
                    TokenKind::CloseBracket => {
                        let elements = self.scratch_pool_with_first_val(first_expr)?;
                        new_arr_init!(self.clone_slice_from_scratch_pool(elements)?, Span::ZERO)
                    },
                    // `.[expr; count]`
                    TokenKind::Semicolon => {
                        self.advanced().ws0();
                        let count = self.expr().context("array literal short count")?;
                        expr!(
                            ArrayInitializerShort { lhs: None, val: first_expr, count },
                            Span::ZERO
                        )
                    },
                    // `.[expr,]` or `.[expr, expr, ...]`
                    TokenKind::Comma => {
                        self.advanced().ws0();
                        let elems = self.scratch_pool_with_first_val(first_expr)?;
                        new_arr_init!(self.expr_list(TokenKind::Comma, elems)?.0, Span::ZERO)
                    },
                    _ => {
                        return ParseError::unexpected_token(t).context("expected ']', ';' or ','");
                    },
                };
                let close_b = self.tok(TokenKind::CloseBracket)?;
                kind.span = span.join(close_b.span);
                kind
            },
            TokenKind::Colon => todo!("TokenKind::Colon"),
            TokenKind::Question => {
                let inner_ty = self.advanced().expr_(PREOP_PRECEDENCE).expect("type after ?");
                expr!(OptionTy { inner_ty }, span.join(inner_ty.full_span()))
            },
            TokenKind::Pound => todo!("TokenKind::Pound"),
            TokenKind::Dollar => todo!("TokenKind::Dollar"),
            TokenKind::At => todo!("TokenKind::At"),
            TokenKind::Tilde => todo!("TokenKind::Tilde"),
            TokenKind::Backslash => todo!("TokenKind::BackSlash"),
            TokenKind::Backtick => todo!("TokenKind::BackTick"),
            t => return err(UnexpectedToken(t), span).context("expected valid value"),
        })
    }

    /// also parses the `}`
    fn parse_initializer_fields(
        &mut self,
    ) -> ParseResult<(Ptr<[(Ptr<ast::Ident>, OPtr<Ast>)]>, Span)> {
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
    fn function_tail(&mut self, params: DeclList, start_span: Span) -> ParseResult<Ptr<ast::Fn>> {
        let expr = self.expr().context("fn return type or body")?;
        let (ret_ty_expr, body) = match self.lex.next_if_kind(TokenKind::OpenBrace) {
            Some(brace) => (Some(expr), self.block(brace.span).context("fn body")?.upcast()),
            None => (None, expr),
        };
        self.alloc(expr_!(Fn { params, ret_ty_expr, ret_ty: None, body: Some(body) }, start_span))
    }

    fn if_after_cond(
        &mut self,
        condition: Ptr<Ast>,
        start_span: Span,
        was_piped: bool,
    ) -> ParseResult<Ptr<ast::If>> {
        self.ws0().opt_do();
        let then_body = self.expr_(IF_PRECEDENCE).context("then body")?;
        let else_body = self
            .ws0()
            .lex
            .next_if_kind(TokenKind::Keyword(Keyword::Else))
            .map(|_| self.expr_(IF_PRECEDENCE).context("else body"))
            .transpose()?;
        self.alloc(expr_!(If { condition, then_body, else_body, was_piped }, start_span))
    }

    /// `... ( ... )`
    /// `     ^` starts here
    /// TODO: `... ( <expr>, ..., param=<expr>, ... )`
    fn parse_call(&mut self, args: ScratchPool<Ptr<Ast>>) -> ParseResult<(Ptr<[Ptr<Ast>]>, Span)> {
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

    fn call(
        &mut self,
        func: Ptr<Ast>,
        args: ScratchPool<Ptr<Ast>>,
        pipe_idx: Option<usize>,
    ) -> ParseResult<Ptr<ast::Call>> {
        let (args, closing_paren_span) = self.parse_call(args)?;
        self.alloc(expr_!(Call { func, args, pipe_idx }, closing_paren_span))
    }

    /// parses block context and '}', doesn't parse the '{'
    fn block(&mut self, open_brace_span: Span) -> ParseResult<Ptr<ast::Block>> {
        let mut res = self.block_inner();
        match res.as_mut() {
            Ok(block) => {
                let closing_brace =
                    self.tok(TokenKind::CloseBrace).context("expected ';' or '}'")?;
                block.span = open_brace_span.join(closing_brace.span);
            },
            Err(_) => {
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
            },
        }
        res
    }

    #[inline]
    fn block_inner(&mut self) -> ParseResult<Ptr<ast::Block>> {
        let mut list_pool = ScratchPool::new();
        //let mut list_pool = Vec::new(); // TODO: compare
        let mut has_trailing_semicolon = false;
        self.ws0();
        loop {
            if self.lex.peek().is_none_or(|t| t.kind == TokenKind::CloseBrace) {
                break;
            }
            let expr = self.expr().context("expr in block")?;
            list_pool.push(expr).map_err(|e| self.wrap_alloc_err(e))?;
            // list_pool.push(expr);
            has_trailing_semicolon = self.ws0().lex.advance_if_kind(TokenKind::Semicolon);
            if !has_trailing_semicolon && expr.block_expects_trailing_semicolon() {
                break;
            }
            self.lex
                .advance_while(|t| t.kind.is_whitespace() || t.kind == TokenKind::Semicolon);
        }
        let stmts = self.clone_slice_from_scratch_pool(list_pool)?;
        Ok(self.alloc(ast_new!(Block { span: Span::ZERO, has_trailing_semicolon, stmts }))?)
    }

    /// Also returns a `has_trailing_sep` [`bool`].
    fn expr_list(
        &mut self,
        sep: TokenKind,
        mut list_pool: ScratchPool<Ptr<Ast>>,
    ) -> ParseResult<(Ptr<[Ptr<Ast>]>, bool)> {
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
    fn var_decl_list(&mut self, sep: TokenKind) -> ParseResult<DeclList> {
        self.var_decl_list_with_start_list(sep, ScratchPool::new())
    }

    fn var_decl_list_with_start_list(
        &mut self,
        sep: TokenKind,
        mut list: ScratchPool<Ptr<ast::Decl>>,
    ) -> ParseResult<DeclList> {
        loop {
            let Some(decl) = self.var_decl().opt().context("var_decl")? else { break };
            list.push(decl).map_err(|e| self.wrap_alloc_err(e))?;
            if !self.ws0().lex.advance_if_kind(sep) {
                break;
            }
            self.ws0();
        }
        self.clone_slice_from_scratch_pool(list)
    }

    fn var_decl(&mut self) -> ParseResult<Ptr<ast::Decl>> {
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
                TokenKind::Ident => break self.advanced().ident_from_span(t.span)?,
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
                let mut d = Decl::from_ident(ident);
                d.markers = markers;
                d.init = Some(self.advanced().expr().context("variable initialization")?);
                d.is_const = t.kind == TokenKind::ColonColon;
                self.alloc(d)
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
    fn typed_decl(
        &mut self,
        markers: DeclMarkers,
        ident: Ptr<ast::Ident>,
    ) -> ParseResult<Ptr<ast::Decl>> {
        let mut d = Decl::from_ident(ident);
        d.markers = markers;
        d.var_ty_expr = Some(self.expr_(DECL_TYPE_PRECEDENCE).context("decl type")?);
        let t = self.ws0().lex.next_if(|t| matches!(t.kind, TokenKind::Eq | TokenKind::Colon));
        d.init = t.map(|_| self.expr().context("variable initialization")).transpose()?;
        d.is_const = t.is_some_and(|t| t.kind == TokenKind::Colon);
        self.alloc(d)
    }

    fn ident(&mut self) -> ParseResult<Ptr<ast::Ident>> {
        let t = self.tok(TokenKind::Ident)?;
        self.ident_from_span(t.span)
    }

    /// this doesn't check if the text at span is valid
    fn ident_from_span(&self, span: Span) -> ParseResult<Ptr<ast::Ident>> {
        self.alloc(Ident::new(self.get_text_from_span(span), span))
    }

    /// Parses the `do` keyword 0 or 1 times.
    fn opt_do(&mut self) {
        #[allow(unused_must_use)] // `do` is optional
        self.tok(TokenKind::Keyword(Keyword::Do));
    }

    // -------

    /// 0+ whitespace
    fn ws0(&mut self) -> &mut Self {
        self.lex.advance_while(|t| t.kind.is_whitespace());
        self
    }

    /// 1+ whitespace
    fn ws1(&mut self) -> ParseResult<()> {
        self.tok_where(|t| t.kind.is_whitespace())?;
        self.ws0();
        Ok(())
    }

    fn tok(&mut self, tok: TokenKind) -> ParseResult<Token> {
        self.tok_where(|t| t.kind == tok).with_context(|| format!("token ({:?})", tok))
    }

    fn tok_where(&mut self, cond: impl FnOnce(Token) -> bool) -> ParseResult<Token> {
        let t = self.peek_tok()?;
        if cond(t) {
            self.lex.advance();
            Ok(t)
        } else {
            ParseError::unexpected_token(t)?
        }
    }

    fn advanced(&mut self) -> &mut Self {
        self.lex.advance();
        self
    }

    #[inline]
    fn next_tok(&mut self) -> ParseResult<Token> {
        self.lex.next().ok_or(err_val(NoInput, self.lex.pos_span()))
    }

    #[inline]
    fn peek_tok(&mut self) -> ParseResult<Token> {
        self.lex.peek().ok_or(err_val(NoInput, self.lex.pos_span()))
    }

    // helpers:

    #[inline]
    fn alloc<T>(&self, val: T) -> ParseResult<Ptr<T>> {
        self.alloc.alloc(val).map_err(|e| err_val(AllocErr(e), self.lex.pos_span()))
    }

    #[inline]
    fn alloc_slice<T: Copy>(&self, slice: &[T]) -> ParseResult<Ptr<[T]>> {
        self.alloc.alloc_slice(slice).map_err(|err| self.wrap_alloc_err(err))
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
            .clone_to_slice_into_arena(&self.alloc)
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
        let ptr = self.alloc(val)?;
        Ok(unsafe { core::slice::from_raw_parts_mut(ptr.as_mut() as *mut T, 1) }.into())
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
