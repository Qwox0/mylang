//! # Parser module
//!
//! The parser allocates the AST ([`Expr`]) and the stored [`Type`]s to
//! [`Type::Unset`] or [`Type::Unevaluated`]

use crate::{
    ast::{
        self, Ast, AstKind, BinOpKind, DeclFlags, EnumFlags, FnFlags, For, GenericSlotFlags,
        StructDef, StructFlags, SwitchCase, UnaryOpKind, UpcastToAst, ast_new,
    },
    context::{CompilationContextInner, ctx_mut, primitives},
    diagnostics::{cerror, cerror2, chint},
    literals::{self, replace_escape_chars},
    ptr::{OPtr, Ptr},
    scope::{Scope, ScopeAndAggregateInfo, ScopeKind},
    source_file::SourceFile,
    util::{BitFlags, OptionExt, UnwrapDebug, concat_arr, then},
};
use core::{fmt, str};
pub use error::*;
use lexer::{Keyword, Lexer, Span, Token, TokenKind};
use num::BigInt;
use parser_helper::ParserInterface;
use std::ops::DerefMut;

pub mod error;
pub mod lexer;
pub mod parser_helper;

fn peek_is_invalid_start(parser: &Parser, prec: u8) -> bool {
    parser.lex.peek_or_eof().kind.is_invalid_start(prec)
}

macro_rules! opt {
    ($self:expr, $method:ident($($arg:expr),* $(,)?), $prec:expr) => {{
        let _self: &mut Parser = $self;
        if peek_is_invalid_start(_self, $prec) {
            Ok(None)
        } else {
            _self.$method($($arg),*).map(Some)
        }
    }};
}

macro_rules! expr {
    ($kind:ident { $( $field:ident $( : $val:expr )? ),* $(,)? }, $span:expr $(,)? ) => {
        ast_new!($kind { $($field $(:$val)?),* }, $span).upcast()
    };
    ($expr:expr) => {
        crate::context::ctx().alloc.alloc($expr)?.upcast()
    };
}

/// This won't consume the token matching `$until_pat`
#[rustfmt::skip]
macro_rules! skip_tokens_until {
    (
        $p:expr,
        until = $until_pat:pat,
        scope_open = $open_scope_pat:pat,
        scope_close = $close_scope_pat:pat $(,)?
    ) => {
        let mut depth: usize = 0;
        while let Some(t) = $p.lex.peek() {
            match t.kind {
                $open_scope_pat => depth += 1,
                $until_pat if depth == 0 => break,
                $close_scope_pat => depth -= 1,
                _ => {},
            };
            $p.lex.advance();
        }
    };
}

macro_rules! parse_in_block {
    ($self:ident, sep = [ $( $sep:path ),* $(,)? ], in_block = $in_block:ident, $parse_item:expr) => {{
        let mut has_trailing_sep = false;
        const SEP: [TokenKind; [$($sep),*].len()] = [$($sep),*];
        loop {
            $self.lex.advance_while(|t| t.kind.is_ignored() || SEP.contains(&t.kind));
            if $self.lex.peek().is_none_or(|t| t.kind == TokenKind::CloseBrace) {
                break;
            }
            let res: ParseResult<Ptr<ast::Ast>> = try {
                $parse_item
            };
            let Ok(expr) = res else {
                if $self.lex.has_unclosed_block_comment() {
                    return Err(parse_err());
                }
                skip_tokens_until!(
                    $self,
                    until = TokenKind::CloseBrace $( | $sep )*,
                    scope_open = TokenKind::OpenBrace,
                    scope_close = TokenKind::CloseBrace,
                );
                continue;
            };
            has_trailing_sep = $self.lex.advance_if(|t| SEP.contains(&t.kind));
            let peek = $self.lex.peek();
            if !has_trailing_sep
                && (peek.is_some_and(|t| t.kind != TokenKind::CloseBrace)
                    || (!$in_block && peek.is_none()))
                //&& peek.kind != TokenKind::CloseBrace
                //&& (!$in_block || peek.kind != TokenKind::EOF)
                && expr.block_expects_trailing_sep()
            {
                if expr.kind == AstKind::Ident {
                    expected_token(expr.full_span().after(), &concat_arr!(DECL_TAIL_TOKENS, SEP));
                } else {
                    expected_token(expr.full_span().after(), &SEP);
                }
                continue;
            }
        }
        has_trailing_sep
    }};
}

pub fn parse_files(cctx: Ptr<CompilationContextInner>) -> impl DerefMut<Target = [Ptr<Ast>]> {
    debug_assert!(cctx.import_manager.start_file.is_some());
    let mut stmts = Vec::new();
    let has_prelude = parse_prelude_into(cctx, &mut stmts);

    // Note: this idx-based loop is needed because `ctx.files` might get mutated while the loop is
    // running.
    let mut idx = has_prelude as usize;
    while let Some(&file) = cctx.files().get(idx) {
        parse_file_into(file, cctx, &mut stmts);
        idx += 1;
    }
    #[cfg(not(test))]
    return stmts.into_boxed_slice();
    #[cfg(test)]
    return Ptr::from_slice(cctx.as_mut().stmts.set_once(stmts.into_boxed_slice()).as_mut());
}

pub fn parse_file_into(
    mut file: Ptr<SourceFile>,
    cctx: Ptr<CompilationContextInner>,
    stmts: &mut Vec<Ptr<Ast>>,
) -> Ptr<Scope> {
    debug_assert!(file.stmt_range.is_none());
    debug_assert!(cctx.files().contains(&file));

    let start_idx = stmts.len();
    let mut p = Parser { lex: Lexer::new(file), cctx };
    let _ = p.parse_stmts_into(stmts, false);
    while let Some(t) = p.lex.next()
        && !p.lex.has_unclosed_block_comment()
    {
        debug_assert_eq!(t.kind, TokenKind::CloseBrace);
        unexpected_token(t, &[]);
        let _ = p.parse_stmts_into(stmts, false);
    }
    //println!("{} -> {} LOC", file.path.display(), p.lex.cursor.code_lines_compact);
    cctx.as_mut().code_lines_compact += p.lex.cursor.code_lines_compact;
    debug_assert!(p.lex.is_empty());
    let stmt_range = start_idx..stmts.len();
    file.set_stmt_range(stmt_range.clone());
    file.scope.set_once(Scope::file(&stmts[stmt_range], cctx.global_scope)).into()
}

pub fn parse_prelude_into(cctx: Ptr<CompilationContextInner>, stmts: &mut Vec<Ptr<Ast>>) -> bool {
    let Some(prelude) = cctx.import_manager.prelude_file else { return false };
    debug_assert_eq!(prelude, cctx.files()[0]);
    debug_assert_eq!(cctx.global_scope.kind, ScopeKind::Root);
    cctx.as_mut().global_scope = parse_file_into(prelude, cctx, stmts);
    return true;
}

pub struct Parser {
    lex: Lexer,
    cctx: Ptr<CompilationContextInner>,
}

impl Parser {
    fn expr(&mut self) -> ParseResult<Ptr<Ast>> {
        self.expr_(MIN_PRECEDENCE)
    }

    fn expr_(&mut self, min_precedence: u8) -> ParseResult<Ptr<Ast>> {
        let mut lhs = self.value(min_precedence)?;
        loop {
            match self.op_chain(lhs, min_precedence) {
                Ok(node) if node != lhs => lhs = node,
                res => return res,
            };
        }
    }

    /// Returns `Ok(lhs)` iff no further valid [`FollowingOperator`] can be found.
    fn op_chain(&mut self, lhs: Ptr<Ast>, min_precedence: u8) -> ParseResult<Ptr<Ast>> {
        let Some(Token { kind, span }) = self.lex.peek() else { return Ok(lhs) };

        let op = match FollowingOperator::new(kind) {
            Some(op) if op.precedence() > min_precedence => op,
            _ => return Ok(lhs),
        };
        self.lex.advance();

        return Ok(match op {
            FollowingOperator::Dot => {
                let rhs = self.ident()?;
                if rhs.sym == primitives().as_sym {
                    self.tok(TokenKind::OpenParenthesis)?;
                    let target_ty = self.expr()?;
                    let close_p = self.tok(TokenKind::CloseParenthesis)?;
                    let span = span.join(close_p.span);
                    expr!(Cast { operand: lhs, target_ty }, span)
                } else {
                    expr!(Dot { lhs: Some(lhs), has_lhs: true, rhs }, span)
                }
            },
            FollowingOperator::Call => self.call(lhs, vec![], None)?.upcast(),
            FollowingOperator::Index => {
                let idx = self.expr()?;
                let close = self.tok(TokenKind::CloseBracket)?;
                let mut_t = self.lex.next_if_kind(TokenKind::Keyword(Keyword::Mut));
                expr!(Index { mut_access: mut_t.is_some(), lhs, idx }, mut_t.unwrap_or(close).span)
            },
            FollowingOperator::PositionalInitializer => {
                let mut args = vec![];
                let close_p_span = self.parse_call(&mut args)?;
                let span = span.join(close_p_span);
                let args = self.alloc_slice(&args)?;
                self.alloc(ast::PositionalInitializer::new(Some(lhs), args, span))?.upcast()
            },
            FollowingOperator::NamedInitializer => {
                let (fields, close_b_span) = self.parse_initializer_fields()?;
                let span = span.join(close_b_span);
                self.alloc(ast::NamedInitializer::new(Some(lhs), fields, span))?.upcast()
            },
            FollowingOperator::ArrayInitializer => self.parse_array_initializer(Some(lhs), span)?,
            FollowingOperator::SingleArgNoParenFn => {
                let Some(lhs) = lhs.try_downcast::<ast::Ident>() else {
                    return cerror2!(
                        lhs.full_span(),
                        "expected parameters, got an expression '{:?}'",
                        lhs.kind
                    );
                };
                let param = self.alloc(ast::Decl::from_ident(lhs))?;
                self.function_tail(vec![param], span, min_precedence)?.upcast()
            },
            FollowingOperator::PostOp(mut op) => {
                let mut span = span;
                if op == UnaryOpKind::AddrOf
                    && let Some(t) = self.lex.next_if_kind(TokenKind::Keyword(Keyword::Mut))
                {
                    op = UnaryOpKind::AddrMutOf;
                    span = span.join(t.span);
                }
                expr!(UnaryOp { op, operand: lhs, is_postfix: true }, span)
            },
            FollowingOperator::BinOp(op) => {
                let rhs = self.expr_(op.precedence())?;
                expr!(BinOp { lhs, op, rhs }, span)
            },
            FollowingOperator::Range { is_inclusive } => {
                let end = opt!(self, expr_(op.precedence()), min_precedence)?;
                if is_inclusive && end.is_none() {
                    return cerror2!(span, "an inclusive range must have an end bound");
                }
                expr!(Range { start: Some(lhs), end, is_inclusive }, span)
            },
            FollowingOperator::OrElse => {
                let rhs = self.expr_(op.precedence())?;
                expr!(OrElse { lhs, rhs }, span)
            },
            FollowingOperator::Pipe => {
                let t = self.lex.peek_or_eof();
                match t.kind {
                    TokenKind::Keyword(Keyword::If | Keyword::Then) => {
                        self.advanced().if_after_cond(lhs, span, true)?.upcast()
                    },
                    TokenKind::Keyword(Keyword::Switch) => {
                        self.advanced().switch_body(lhs, span, true)?.upcast()
                    },
                    TokenKind::Keyword(Keyword::For) => {
                        let iter_var = self.advanced().ident()?;
                        self.opt_do();
                        let body = self.expr()?;
                        For::new(lhs, iter_var, body, true, t.span, &self.cctx.alloc)?.upcast()
                    },
                    TokenKind::Keyword(Keyword::While) => {
                        self.advanced().opt_do();
                        let body = self.expr()?;
                        expr!(While { condition: lhs, body, was_piped: true }, t.span)
                    },
                    _ => {
                        let func = self.expr_(PIPE_TARGET_PRECEDENCE)?;
                        let mut args = vec![];
                        let pipe_idx = if let Some(dot) = func.try_downcast::<ast::Dot>()
                            && dot.lhs.is_none()
                        {
                            // For simplicity the following two cases are handled differently:
                            // `x |> XType.func(...)` is converted to `XType.func(x, ...)`
                            // `x |>      .func(...)` is converted to `x.func(...)`
                            // Problem: `func` might not be a method but a function
                            //     => the second syntax is converted to a method-like call
                            //     => omitting the type might produce different code
                            dot.as_mut().lhs = Some(lhs);
                            dot.as_mut().has_lhs = true;
                            None
                        } else {
                            // Note: This works for both `x |> Type.func()` and `x |> y.func()`!
                            args.push(lhs);
                            Some(0)
                        };
                        self.tok(TokenKind::OpenParenthesis)?;
                        self.call(func, args, pipe_idx)?.upcast()
                    },
                }
            },
            FollowingOperator::Assign => {
                let rhs = self.expr()?;
                expr!(Assign { lhs, rhs, is_explicit_generic_arg: false }, span)
            },
            FollowingOperator::BinOpAssign(op) => {
                let rhs = self.expr()?;
                expr!(BinOpAssign { lhs, op, rhs }, span)
            },
            FollowingOperator::Decl(kind) => {
                self.decl_tail(ast::Decl::from_lhs(lhs, &self.cctx.alloc)?, kind)?.upcast()
            },
        });
    }

    /// anything which has higher precedence than any operator
    fn value(&mut self, prec: u8) -> ParseResult<Ptr<Ast>> {
        let Token { kind, span } = self.lex.peek_or_eof();

        Ok(match kind {
            TokenKind::Ident => self.advanced().ident_from_span(span)?.upcast(),
            TokenKind::Keyword(Keyword::Mut | Keyword::Rec | Keyword::Pub | Keyword::Static) => {
                self.var_decl(false)?.upcast()
            },
            TokenKind::Keyword(Keyword::Struct) => {
                self.lex.advance();
                let mut flags = StructFlags::default();

                // generic parameters
                let mut generics_scope = None;
                if self.lex.advance_if_kind(TokenKind::OpenParenthesis) {
                    let mut generics = Vec::new();
                    self.parse_with_sep(
                        &[TokenKind::Comma],
                        &mut generics,
                        |self_| {
                            let dollar = self_.lex.peek_or_eof();
                            if dollar.kind != TokenKind::Dollar {
                                return Err(unexpected_token(dollar, &[TokenKind::Dollar]));
                            }
                            let decl = self_.var_decl_no_markers(true)?;
                            debug_assert!(decl.flags.get(DeclFlags::IS_GENERIC));
                            debug_assert!(decl.is_const);
                            decl.generic
                                .u()
                                .as_mut()
                                .flags
                                .set(GenericSlotFlags::WAS_ADDED_TO_SCOPE);
                            Ok(decl)
                        },
                        MIN_PRECEDENCE,
                        "parameter",
                    )?;
                    self.tok(TokenKind::CloseParenthesis)?;
                    if generics.len() > 0 {
                        flags.set(StructFlags::IS_GENERIC);
                        generics_scope =
                            Some(self.alloc(Scope::new(generics, ScopeKind::StructGenerics))?);
                    }
                }

                self.tok(TokenKind::OpenBrace)?;
                let decls = self.struct_body()?;
                let close_b = self.tok(TokenKind::CloseBrace)?;

                let span = span.join(close_b.span);
                StructDef::new(flags, decls, generics_scope, span, &self.cctx.alloc)?.upcast()
            },
            TokenKind::Keyword(Keyword::Union) => {
                self.advanced().tok(TokenKind::OpenBrace)?;
                let decls = self.struct_body()?;
                let ScopeAndAggregateInfo { scope, fields } =
                    Scope::for_aggregate(decls, &self.cctx.alloc, ScopeKind::Union)?;
                let close_b = self.tok(TokenKind::CloseBrace)?;
                expr!(
                    UnionDef {
                        scope,
                        sema_units: None,
                        fields,
                        external_consts: vec![],
                        finished_members: 0
                    },
                    span.join(close_b.span)
                )
            },
            TokenKind::Keyword(Keyword::Enum) => {
                self.advanced().tok(TokenKind::OpenBrace)?;
                let mut decls = Vec::new();
                parse_in_block!(
                    self,
                    sep = [TokenKind::Semicolon, TokenKind::Comma],
                    in_block = true,
                    {
                        let Some(variant_ident) = opt!(self, ident(), MIN_PRECEDENCE)? else {
                            break;
                        };
                        let decl = self.alloc(ast::Decl::from_ident(variant_ident))?;
                        let mut decl = self.decl_assign(decl, true)?;
                        if decl.var_ty_expr.is_none() && decl.init.is_none() {
                            let ty = then!(
                            self.lex.advance_if_kind(TokenKind::OpenParenthesis) => {
                                let ty_expr = self.expr()?;
                                let close_p = self.tok(TokenKind::CloseParenthesis)?;
                                decl.span = decl.span.join(close_p.span);
                                ty_expr
                            });
                            let variant_index =
                                then!(self.lex.advance_if_kind(TokenKind::Eq) => self.expr()?);
                            decl.var_ty_expr = ty;
                            decl.init = variant_index;
                            if decl.init.is_some() {
                                decl.flags.set(DeclFlags::HAS_INIT_EXPR);
                            }
                        } else if !decl.is_const {
                            return cerror2!(
                                decl.full_span(),
                                "expected variant or constant declaration; got variable \
                                 declaration"
                            );
                        }
                        decls.push(decl);
                        decl.upcast()
                    }
                );
                let close_b = self.tok(TokenKind::CloseBrace)?;
                let ScopeAndAggregateInfo { scope, fields } =
                    Scope::for_aggregate(decls, &self.cctx.alloc, ScopeKind::Enum)?;
                expr!(
                    EnumDef {
                        flags: EnumFlags::default(),
                        is_simple_enum: true,
                        scope,
                        generics_scope: None,
                        variants: fields,
                        external_consts: vec![],
                        tag_ty: None,
                        polymorphs: vec![],
                        sema_units: None,
                        finished_members: 0,
                    },
                    span.join(close_b.span)
                )
            },
            TokenKind::Keyword(Keyword::Unsafe) => todo!("unsafe"),
            TokenKind::Keyword(Keyword::If) => {
                let condition = self.advanced().expr()?;
                self.if_after_cond(condition, span, false)?.upcast()
            },
            TokenKind::Keyword(Keyword::Switch) => {
                let val = self.advanced().expr()?;
                self.switch_body(val, span, false)?.upcast()
            },
            TokenKind::Keyword(Keyword::For) => {
                let iter_var = self.advanced().ident()?;
                self.local_keyword("in")?;
                let source_expr = self.expr()?;
                self.opt_do();
                let body = self.expr()?;
                For::new(source_expr, iter_var, body, false, span, &self.cctx.alloc)?.upcast()
            },
            TokenKind::Keyword(Keyword::While) => {
                let condition = self.advanced().expr()?;
                self.opt_do();
                let body = self.expr()?;
                expr!(While { condition, body, was_piped: false }, span)
            },
            TokenKind::Keyword(Keyword::Loop) => {
                let body = self.advanced().expr()?;
                expr!(Loop { body, break_ty: None }, span)
            },
            TokenKind::Keyword(Keyword::Return) => {
                let val = opt!(self.advanced(), expr(), MIN_PRECEDENCE)?;
                expr!(Return { val, parent_fn: None }, span)
            },
            TokenKind::Keyword(Keyword::Break) => {
                let val = opt!(self.advanced(), expr(), MIN_PRECEDENCE)?;
                expr!(Break { val }, span)
            },
            TokenKind::Keyword(Keyword::Continue) => {
                self.lex.advance();
                expr!(Continue {}, span)
            },
            TokenKind::Keyword(Keyword::Autocast) => {
                let operand = self.advanced().expr()?;
                expr!(Autocast { operand }, span)
            },
            TokenKind::Keyword(Keyword::Defer) => {
                let stmt = self.advanced().expr()?;
                expr!(Defer { stmt }, span)
            },
            TokenKind::IntLit => {
                let val = literals::parse_int_lit(&self.advanced().get_text_from_span(span))
                    .ok_or_else(|| cerror!(span, "invalid integer literal"))?; // TODO: better error
                expr!(IntVal { val }, span)
            },
            TokenKind::FloatLit => {
                let val = literals::parse_float_lit(&self.advanced().get_text_from_span(span))
                    .map_err(|e| cerror!(span, "invalid float literal: {e}"))?;
                expr!(FloatVal { val }, span)
            },
            TokenKind::BoolLitTrue | TokenKind::BoolLitFalse => {
                self.lex.advance();
                expr!(BoolVal { val: kind == TokenKind::BoolLitTrue }, span)
            },
            TokenKind::CharLit => {
                let code = &self.advanced().lex.get_code()[span];
                debug_assert_eq!(code.as_bytes().first(), Some(&b'\''));
                debug_assert_eq!(code.as_bytes().last(), Some(&b'\''));
                let code = replace_escape_chars(&code[1..code.len() - 1]);
                let mut chars = code.chars();

                let Some(val) = chars.next() else {
                    return cerror2!(span, "character literals must not be empty");
                };
                if chars.next().is_some() {
                    return cerror2!(span, "character literal contains more than one character");
                }
                expr!(CharVal { val }, span)
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

                let Some(byte) = bytes.next() else {
                    return cerror2!(span, "byte character literals must not be empty");
                };
                if bytes.next().is_some() {
                    return cerror2!(span, "byte character literal contains more than one byte");
                }
                //expr!(BCharLit { val: byte }, span)
                expr!(IntVal { val: BigInt::from(byte) }, span)
            },
            TokenKind::StrLit => {
                let lit = self.advanced().get_text_from_span(span);
                expr!(StrVal { text: Ptr::from_ref(&lit[1..lit.len().saturating_sub(1)]) }, span)
            },
            TokenKind::MultilineStrLitLine => {
                // Note: Arena allocates in the wrong direction
                let mut scratch = Vec::with_capacity(1024);
                while let Some(t) = self.lex.next_if_kind(TokenKind::MultilineStrLitLine) {
                    let line_text = self.get_text_from_span(t.span);
                    debug_assert_eq!(&line_text[0..2], "\\\\");
                    scratch.extend_from_slice(line_text[2..].as_bytes());
                }
                let bytes = self.alloc_slice(&scratch)?;
                let text = unsafe { std::str::from_utf8_unchecked(&bytes) };
                debug_assert!(text.ends_with('\n'));
                expr!(StrVal { text: Ptr::from_ref(&text[0..text.len().saturating_sub(1)]) }, span)
            },
            TokenKind::OpenParenthesis => {
                // TODO: currently no tuples allowed!
                let first_expr = opt!(self.advanced(), expr(), MIN_PRECEDENCE)?; // this assumes the parameter syntax is also a valid expression
                let t = self.lex.next_or_eof()?;
                let params = match t.kind {
                    // (expr)
                    TokenKind::CloseParenthesis if !self.lex.advance_if_kind(TokenKind::Arrow) => {
                        let Some(mut expr) = first_expr else {
                            return Err(unexpected_token_expect1(t, "expression"));
                        };
                        expr.parenthesis_count += 1;
                        return Ok(expr);
                    },
                    // (expr) -> ...
                    TokenKind::CloseParenthesis if let Some(e) = first_expr => {
                        vec![e.try_to_decl()?.ok_or_else(|| unexpected_expr(e, "a parameter"))?]
                    },
                    // () -> ...
                    TokenKind::CloseParenthesis => Vec::new(),
                    // (params...) -> ...
                    TokenKind::Comma => {
                        let Some(expr) = first_expr else {
                            return Err(unexpected_token_expect1(t, "parameter"));
                        };
                        let Some(first_decl) = expr.try_to_decl()? else {
                            return Err(unexpected_expr(expr, "a parameter"));
                        };
                        let mut params = vec![first_decl];
                        self.parse_with_sep(
                            &[TokenKind::Comma],
                            &mut params,
                            |self_| self_.var_decl(true),
                            MIN_PRECEDENCE,
                            "parameter",
                        )?;
                        let expected_tok = if params.last().is_some_and(|d| d.is_lhs_only()) {
                            &EXPECTED_AFTER_IDENT_PARAM[..]
                        } else {
                            &EXPECTED_AFTER_PARAM[..]
                        };
                        self.tok_with_expected(TokenKind::CloseParenthesis, expected_tok)?;
                        self.tok(TokenKind::Arrow)?;
                        params
                    },
                    _ if first_expr.is_some_and(|e| e.kind == AstKind::Ident) => {
                        return Err(unexpected_token(t, &EXPECTED_AFTER_IDENT_PARAM));
                    },
                    _ => return Err(unexpected_token(t, &EXPECTED_AFTER_PARAM)),
                };
                self.function_tail(params, span, prec)?.upcast()
            },
            TokenKind::OpenBracket => {
                let len = opt!(self.advanced(), expr(), MIN_PRECEDENCE)?;
                self.tok(TokenKind::CloseBracket).inspect_err(|_| {
                    let t = self.lex.peek_or_eof();
                    if matches!(t.kind, TokenKind::Comma | TokenKind::Semicolon) {
                        chint!(
                            span.join(t.span),
                            "if you want to create an array value, consider using an array \
                             initializer `.[...]` instead"
                        )
                    }
                })?;
                let is_mut =
                    len.is_none() && self.lex.advance_if_kind(TokenKind::Keyword(Keyword::Mut));
                let elem_ty = self.expr_(TY_PREFIX_PRECEDENCE)?;
                match len {
                    Some(len) => expr!(ArrayTy { len, elem_ty }, span),
                    None => expr!(SliceTy { elem_ty, is_mut }, span),
                }
            },
            TokenKind::OpenBrace => self.block()?.upcast(),
            TokenKind::Bang | TokenKind::Keyword(Keyword::Not) => {
                let operand = self.advanced().expr_(PREOP_PRECEDENCE)?;
                expr!(UnaryOp { op: UnaryOpKind::Not, operand, is_postfix: false }, span)
            },
            TokenKind::Plus => todo!("TokenKind::Plus"),
            TokenKind::Minus => {
                let operand = self.advanced().expr_(PREOP_PRECEDENCE)?;
                expr!(UnaryOp { op: UnaryOpKind::Neg, operand, is_postfix: false }, span)
            },
            TokenKind::Arrow => self.advanced().function_tail(vec![], span, prec)?.upcast(),
            TokenKind::Asterisk => {
                // TODO: deref prefix
                let is_mut = self.advanced().lex.advance_if_kind(TokenKind::Keyword(Keyword::Mut));
                let pointee = self.expr_(TY_PREFIX_PRECEDENCE)?;
                expr!(PtrTy { pointee, is_mut }, span.join(pointee.full_span()))
            },
            TokenKind::Ampersand => {
                let is_mut = self.advanced().lex.advance_if_kind(TokenKind::Keyword(Keyword::Mut));
                let op = if is_mut { UnaryOpKind::AddrMutOf } else { UnaryOpKind::AddrOf };
                let operand = self.expr_(PREOP_PRECEDENCE)?;
                expr!(UnaryOp { op, operand, is_postfix: false }, span)
            },
            TokenKind::Dot => {
                let rhs = self.advanced().ident()?;
                expr!(ast::Dot::new(None, rhs, span))
            },
            TokenKind::DotDot => {
                let end = opt!(self.advanced(), expr_(RANGE_PRECEDENCE), prec)?;
                expr!(Range { start: None, end, is_inclusive: false }, span)
            },
            TokenKind::DotDotEq => {
                let end = opt!(self.advanced(), expr_(RANGE_PRECEDENCE), prec)?;
                if end.is_none() {
                    return cerror2!(span, "an inclusive range must have an end bound");
                }
                expr!(Range { start: None, end, is_inclusive: true }, span)
            },
            TokenKind::DotOpenParenthesis => {
                let mut args = vec![];
                let close_p_span = self.advanced().parse_call(&mut args)?;
                let span = span.join(close_p_span);
                let args = self.alloc_slice(&args)?;
                self.alloc(ast::PositionalInitializer::new(None, args, span))?.upcast()
            },
            TokenKind::DotOpenBrace => {
                let (fields, close_b_span) = self.advanced().parse_initializer_fields()?;
                let span = span.join(close_b_span);
                self.alloc(ast::NamedInitializer::new(None, fields, span))?.upcast()
            },
            TokenKind::DotOpenBracket => self.advanced().parse_array_initializer(None, span)?,
            TokenKind::Colon => todo!("TokenKind::Colon"),
            TokenKind::Question => {
                let inner_ty = self.advanced().expr_(TY_PREFIX_PRECEDENCE).expect("type after ?");
                expr!(OptionTy { inner_ty }, span.join(inner_ty.full_span()))
            },
            TokenKind::Pound => {
                let directive_ident = self.advanced().ident()?;
                let directive_name = directive_ident.sym.text(); // TODO?: also intern directive_names?

                let mut parse_str_lit_arg = |usage: &str| {
                    let arg = opt!(self, value(MAX_PRECEDENCE), MIN_PRECEDENCE)?;
                    arg.and_then(Ptr::<Ast>::try_downcast::<ast::StrVal>)
                        .ok_or_else::<ParseError, _>(|| {
                            cerror2!(
                                arg.map(|a| a.full_span()).unwrap_or(self.lex.pos_span()),
                                "Expected {} after the #{directive_name} directive",
                                usage
                            )
                        })
                };

                let p = primitives();

                // function-like directives:
                if directive_name == "import" {
                    let path = parse_str_lit_arg("a path string literal")?;
                    let idx = self.cctx.add_import(
                        &path.text,
                        Some(&self.lex.cursor.file.path),
                        path.span,
                    );
                    let mut i = expr!(ImportDirective { path, files_idx: idx.ok() }, span);
                    if idx.is_err() {
                        i.ty = Some(p.err_ty);
                        i.set_replacement(p.err_ty.upcast());
                    }
                    i
                } else if directive_name == "extern" {
                    expr!(ExternDirective { decl: None }, span.join(directive_ident.span))
                } else if directive_name == "intrinsic" {
                    let intrinsic_name = parse_str_lit_arg("an intrinsic name")?;
                    expr!(IntrinsicDirective { intrinsic_name, decl: None }, span)
                } else if directive_name == "library" {
                    let str_lit = parse_str_lit_arg("a library name")?;
                    self.cctx.add_library(str_lit)?;
                    // TODO: return `{library}` object
                    expr!(SimpleDirective { ret_ty: p.void_ty }, span.join(str_lit.span))
                } else if directive_name == "add_library_search_path" {
                    let str_lit = parse_str_lit_arg("a path string literal")?;
                    self.cctx.add_library_search_path(str_lit.text)?;
                    expr!(SimpleDirective { ret_ty: p.void_ty }, span.join(str_lit.span))
                } else if directive_name == "program_main" {
                    expr!(ProgramMainDirective {}, span.join(directive_ident.span))
                } else if directive_name == "sizeof" {
                    let type_ = self.value(MAX_PRECEDENCE)?;
                    expr!(SizeOfDirective { type_ }, span.join(directive_ident.span))
                } else if directive_name == "sizeof_val" {
                    let val = self.value(MAX_PRECEDENCE)?;
                    expr!(SizeOfValDirective { val }, span.join(directive_ident.span))
                } else if directive_name == "alignof" {
                    let type_ = self.value(MAX_PRECEDENCE)?;
                    expr!(AlignOfDirective { type_ }, span.join(directive_ident.span))
                } else if directive_name == "offsetof" {
                    self.tok(TokenKind::OpenParenthesis)?;
                    let type_ = self.value(MAX_PRECEDENCE)?;
                    self.tok(TokenKind::Comma)?;
                    let field = self.ident()?;
                    self.tok(TokenKind::CloseParenthesis)?;
                    expr!(OffsetOfDirective { type_, field }, span.join(directive_ident.span))
                }
                // annotation directives:
                else if directive_name == "varargs" {
                    let Some(func) = self.expr_(prec)?.try_downcast::<ast::Fn>() else {
                        return cerror2!(
                            span,
                            "Expected a function or a function type after #{directive_name} \
                             directive"
                        );
                    };
                    func.as_mut().flags.set(FnFlags::HAS_VARARGS);
                    func.upcast()
                } else if directive_name == "no_mangle" {
                    return cerror2!(
                        span.join(directive_ident.span),
                        "`#{directive_name}` is currently not implemented"
                    );
                } else if directive_name == "obj_symbol_name" {
                    // TODO: check for duplicates?
                    let name_lit = parse_str_lit_arg("a symbol name")?;
                    let Some(decl) = self.expr()?.try_downcast::<ast::Decl>() else {
                        return cerror2!(
                            span,
                            "Expected a declaration after #{directive_name} directive"
                        );
                    };
                    decl.as_mut().obj_symbol_name = Some(name_lit);
                    decl.upcast()
                } else if directive_name == "__runtime_entry_point" {
                    let func = self.expr()?;
                    let Some(decl) = func.try_downcast::<ast::Decl>() else {
                        return cerror2!(
                            span,
                            "Expected a function declaration after #{directive_name} directive"
                        );
                    };
                    if self.cctx.args.is_lib {
                        // Skip the entry_point `func` in libs to prevent name conflicts with "main"
                        expr!(Empty {}, span.join(directive_ident.span))
                    } else {
                        let main_ident = expr!(
                            Ident { sym: self.cctx.primitives.main_sym, decl: Some(decl) },
                            decl.ident.span
                        );
                        // not using `set_replacement` because it requires a type and this is just
                        // a temporary hack.
                        decl.as_mut().ident.replacement.set_once(main_ident);
                        func
                    }
                }
                //
                else {
                    return cerror2!(span.join(directive_ident.span), "Unknown compiler directive");
                }
            },
            TokenKind::Dollar => {
                let name = self.advanced().ident()?;
                self.alloc(ast::GenericSlot::new(name, span))?.upcast()
            },
            //TokenKind::At => todo!("TokenKind::At"),
            //TokenKind::Tilde => todo!("TokenKind::Tilde"),
            //TokenKind::Backslash => todo!("TokenKind::BackSlash"),
            //TokenKind::Backtick => todo!("TokenKind::BackTick"),
            kind => return Err(unexpected_token(Token { kind, span }, &[])),
        })
    }

    /// also parses the `}`
    fn parse_initializer_fields(
        &mut self,
    ) -> ParseResult<(Ptr<[(Ptr<ast::Ident>, OPtr<Ast>)]>, Span)> {
        let mut fields = vec![];
        let close_b_span = loop {
            if let Some(t) = self.lex.next_if_kind(TokenKind::CloseBrace) {
                break t.span;
            }
            let ident = self.ident()?;
            let init = then!(self.lex.advance_if_kind(TokenKind::Eq) => self.expr()?);
            fields.push((ident, init));

            match self.lex.next_or_eof()? {
                Token { kind: TokenKind::Comma, .. } => {},
                Token { kind: TokenKind::CloseBrace, span } => break span,
                t => {
                    return Err(unexpected_token(
                        t,
                        if init.is_none() {
                            &[TokenKind::Eq, TokenKind::Comma, TokenKind::CloseBrace]
                        } else {
                            &[TokenKind::Comma, TokenKind::CloseBrace]
                        },
                    ));
                },
            }
        };
        Ok((self.alloc_slice(&fields)?, close_b_span))
    }

    /// `.[1, 2, ..., 10]`
    /// `  ^^^^^^^^^^^^^^`
    fn parse_array_initializer(
        &mut self,
        lhs: OPtr<Ast>,
        open_b_span: Span,
    ) -> ParseResult<Ptr<Ast>> {
        macro_rules! new_arr_init {
            ($kind:ident { $( $field:ident $( : $val:expr )? ),* $(,)? } $(,)? ) => {{
                let close_b = self.tok(TokenKind::CloseBracket)?;
                expr!($kind { lhs, parsed_with_lhs: lhs.is_some(), $($field $(: $val )?),* }, open_b_span.join(close_b.span))
            }}
        }

        let Some(first_expr) = opt!(self, expr(), MIN_PRECEDENCE)? else {
            // `.[]`
            return Ok(new_arr_init!(ArrayInitializer { elements: Ptr::empty_slice() }));
        };
        let t = self.lex.peek_or_eof();
        Ok(match t.kind {
            // `.[expr]`
            TokenKind::CloseBracket => {
                let elements = self.alloc_slice(&[first_expr])?;
                new_arr_init!(ArrayInitializer { elements })
            },
            // `.[expr; count]`
            TokenKind::Semicolon => {
                let count = self.advanced().expr()?;
                new_arr_init!(ArrayInitializerShort { val: first_expr, count })
            },
            // `.[expr,]` or `.[expr, expr, ...]`
            TokenKind::Comma => {
                self.lex.advance();
                let mut elems = vec![first_expr];
                self.parse_with_sep(
                    &[TokenKind::Comma],
                    &mut elems,
                    |self_| self_.expr(),
                    MIN_PRECEDENCE,
                    "element",
                )?;
                let elements = self.alloc_slice(&elems)?;
                new_arr_init!(ArrayInitializer { elements })
            },
            _ => {
                return Err(unexpected_token(t, &[
                    TokenKind::Comma,
                    TokenKind::Semicolon,
                    TokenKind::CloseBracket,
                ]));
            },
        })
    }

    /// parsing starts after the '->'
    fn function_tail(
        &mut self,
        params: Vec<Ptr<ast::Decl>>,
        start_span: Span,
        mut outer_prec: u8,
    ) -> ParseResult<Ptr<ast::Fn>> {
        if !is_type_prec(outer_prec) {
            outer_prec = MIN_PRECEDENCE;
        }
        let expr = self.expr_(outer_prec)?;
        let between_expr_state = self.lex.get_state();
        let (ret_ty_expr, body) = if expr.kind != AstKind::Block
            && let Some(body) = opt!(self, expr(), outer_prec)?
            && {
                debug_assert!(!AstKind::Block.is_allowed_top_level());
                let is_invalid_body = body.kind.is_allowed_top_level();
                if is_invalid_body {
                    self.lex.set_state(between_expr_state); // causes "expected `;`" (see tests::function::error_missing_semicolon_after_fn)
                    // Note: If there are multiple functions without a trailing ';' between them,
                    // this function is called O(n^2) times.
                }
                !is_invalid_body
            } {
            (Some(expr), body)
        } else {
            (None, expr)
        };
        for p in params.iter() {
            p.as_mut().flags.set(DeclFlags::IS_PARAMETER);
        }
        Ok(ast::Fn::new(params, ret_ty_expr, Some(body), start_span, &self.cctx.alloc)?)
    }

    fn if_after_cond(
        &mut self,
        condition: Ptr<Ast>,
        start_span: Span,
        was_piped: bool,
    ) -> ParseResult<Ptr<ast::If>> {
        self.lex
            .advance_if(|t| matches!(t.kind, TokenKind::Keyword(Keyword::Then | Keyword::Do)));
        let then_body = self.expr_(IF_PRECEDENCE)?;
        let else_body = then!(self.lex.advance_if_kind(TokenKind::Keyword(Keyword::Else))
            => self.expr_(ELSE_PRECEDENCE)?);
        Ok(ast_new!(If { condition, then_body, else_body, was_piped }, start_span))
    }

    /// switch val { ... } else ...
    ///            ^
    fn switch_body(
        &mut self,
        val: Ptr<Ast>,
        start_span: Span,
        was_piped: bool,
    ) -> ParseResult<Ptr<ast::Switch>> {
        self.tok(TokenKind::OpenBrace)?;
        let mut cases = Vec::new();
        self.parse_with_sep(
            &[TokenKind::Comma, TokenKind::Semicolon],
            &mut cases,
            |self_| {
                let case = self_.expr()?;
                self_.tok(TokenKind::ColonGt)?;
                let body = self_.expr()?;
                Ok(SwitchCase::new(case, body, &self_.cctx.alloc)?)
            },
            MIN_PRECEDENCE,
            "case",
        )?;
        let close_b = self.tok(TokenKind::CloseBrace)?;
        let else_body = then!(self.lex.advance_if_kind(TokenKind::Keyword(Keyword::Else))
            => self.expr_(ELSE_PRECEDENCE)?);

        let cases = self.alloc_slice(&cases)?;
        let switch =
            ast_new!(Switch { val, cases, else_body, was_piped }, start_span.join(close_b.span));
        for c in cases.iter() {
            c.scope.as_mut().expr.set_once(switch.upcast());
        }
        Ok(switch)
    }

    /// `... ( ... )`
    /// `     ^` starts here
    /// TODO: `... ( <expr>, ..., param=<expr>, ... )`
    fn parse_call(&mut self, args: &mut Vec<Ptr<Ast>>) -> ParseResult<Span> {
        let res: ParseResult<_> = try {
            self.parse_with_sep(
                &[TokenKind::Comma],
                args,
                |self_| self_.expr(),
                MIN_PRECEDENCE,
                "expression",
            )?;
            let closing_paren_span =
                self.tok_with_expected(TokenKind::CloseParenthesis, &EXPECTED_AFTER_PARAM)?.span;
            closing_paren_span
        };
        if res.is_err() {
            skip_tokens_until!(
                self,
                until = TokenKind::CloseParenthesis,
                scope_open = TokenKind::OpenParenthesis,
                scope_close = TokenKind::CloseParenthesis,
            );
            self.lex.advance();
        }
        res
    }

    fn call(
        &mut self,
        func: Ptr<Ast>,
        start_args: Vec<Ptr<Ast>>,
        pipe_idx: Option<usize>,
    ) -> ParseResult<Ptr<ast::Call>> {
        let mut args = start_args;
        let closing_paren_span = self.parse_call(&mut args)?;
        let args = self.alloc_slice(&args)?;
        Ok(ast_new!(Call { func, args, pipe_idx, resolved_fn_inst: None }, closing_paren_span))
    }

    /// expects next token to be '{' and parses until and including the '}'
    fn block(&mut self) -> ParseResult<Ptr<ast::Block>> {
        let open_b = self.lex.next().u();
        debug_assert_eq!(open_b.kind, TokenKind::OpenBrace);

        let mut stmts = Vec::new();
        let has_trailing_semicolon = self.parse_stmts_into(&mut stmts, true)?;

        let close_b = self.tok(TokenKind::CloseBrace)?;

        let span = open_b.span.join(close_b.span);
        let stmts = self.alloc_slice(&stmts)?;
        Ok(ast::Block::new(stmts, has_trailing_semicolon, span, &self.cctx.alloc)?)
    }

    /// Parses the insides of a [`Parser::block`]: Expressions and statements seperated by ';'.
    ///
    /// If this returns [`Ok`] next token is [`None`] or a [`TokenKind::CloseBrace`].
    ///
    /// This handles errors.
    #[inline]
    fn parse_stmts_into(
        &mut self,
        stmts: &mut Vec<Ptr<Ast>>,
        in_block: bool,
    ) -> ParseResult<HasTrailingSemicolon> {
        Ok(parse_in_block!(self, sep = [TokenKind::Semicolon], in_block = in_block, {
            let expr = self.expr()?;
            stmts.push(expr);
            expr
        }))
    }

    fn parse_with_sep<T>(
        &mut self,
        sep: &[TokenKind],
        items: &mut Vec<T>,
        mut parse_item: impl FnMut(&mut Self) -> ParseResult<T>,
        opt_prec: u8,
        item_name: impl fmt::Display,
    ) -> ParseResult<bool> {
        let mut has_trailing_sep = false;
        loop {
            if peek_is_invalid_start(self, opt_prec) {
                break;
            }
            items.push(parse_item(self)?);
            has_trailing_sep = self.lex.advance_if(|t| sep.contains(&t.kind));
            if !has_trailing_sep {
                break;
            }
        }
        if let Some(t) = self.lex.peek()
            && sep.contains(&t.kind)
        {
            return Err(unexpected_token_expect1(t, item_name));
        }
        Ok(has_trailing_sep)
    }

    /// also returns the field_count
    fn struct_body(&mut self) -> ParseResult<Vec<Ptr<ast::Decl>>> {
        let mut decls = Vec::new();
        parse_in_block!(self, sep = [TokenKind::Semicolon, TokenKind::Comma], in_block = true, {
            let expr = self.expr()?;
            if let Some(decl) = expr.try_downcast::<ast::Decl>() {
                if let Some(on_ty_expr) = decl.on_type {
                    cerror!(on_ty_expr.full_span(), "currently not supported"); // TODO
                } else {
                    decls.push(decl);
                }
            } else {
                cerror!(expr.full_span(), "expected field or constant declaration");
            };
            expr
        });
        Ok(decls)
    }

    fn var_decl(&mut self, allow_ident_only: bool) -> ParseResult<Ptr<ast::Decl>> {
        let mut markers = DeclFlags::default();
        {
            let mut t = self.lex.peek_or_eof();

            macro_rules! set_marker {
                ($variant:ident $mask:ident) => {
                    if markers.get(DeclFlags::$mask) {
                        let marker_text = Keyword::$variant.as_str();
                        return cerror2!(t.span, "duplicate marker '{marker_text}' on declaration");
                    } else {
                        markers.set(DeclFlags::$mask)
                    }
                };
            }

            loop {
                match t.kind {
                    TokenKind::Ident | TokenKind::Dollar => break,
                    TokenKind::Keyword(Keyword::Mut) => set_marker!(Mut IS_MUT),
                    TokenKind::Keyword(Keyword::Rec) => set_marker!(Rec IS_REC),
                    TokenKind::Keyword(Keyword::Pub) => set_marker!(Pub IS_PUB),
                    TokenKind::Keyword(Keyword::Static) => set_marker!(Static IS_STATIC),
                    _ => {
                        return Err(unexpected_token(t, &[
                            TokenKind::Ident,
                            TokenKind::Keyword(Keyword::Mut),
                            TokenKind::Keyword(Keyword::Rec),
                            TokenKind::Keyword(Keyword::Pub),
                            TokenKind::Keyword(Keyword::Static),
                            TokenKind::Dollar,
                        ]));
                    },
                }
                t = self.advanced().lex.peek_or_eof();
            }
        }

        let mut decl = self.var_decl_no_markers(allow_ident_only)?;
        decl.flags.data |= markers.data;
        Ok(decl)
    }

    fn var_decl_no_markers(&mut self, allow_ident_only: bool) -> ParseResult<Ptr<ast::Decl>> {
        let lhs = self.expr_(ASSIGN_PRECEDENCE)?;
        let decl = ast::Decl::from_lhs(lhs, &self.cctx.alloc)?;
        self.decl_assign(decl, allow_ident_only)
    }

    fn decl_assign(
        &mut self,
        decl: Ptr<ast::Decl>,
        allow_ident_only: bool,
    ) -> ParseResult<Ptr<ast::Decl>> {
        let t = self.lex.peek_or_eof();
        let kind = match t.kind {
            TokenKind::Colon => DeclTailKind::Typed,
            TokenKind::ColonEq => DeclTailKind::Var,
            TokenKind::ColonColon => DeclTailKind::Const,
            _ if allow_ident_only => return Ok(decl),
            _ => return Err(unexpected_token(t, &DECL_TAIL_TOKENS)),
        };
        self.advanced().decl_tail(decl, kind)
    }

    /// `mut x : ...`
    /// `mut x := ...`
    /// `mut x :: ...`
    /// `      ^`
    /// [`FollowingOperator::Decl`]
    fn decl_tail(
        &mut self,
        mut decl: Ptr<ast::Decl>,
        kind: DeclTailKind,
    ) -> ParseResult<Ptr<ast::Decl>> {
        match kind {
            DeclTailKind::Var | DeclTailKind::Const => {
                decl.is_const = matches!(kind, DeclTailKind::Const);
                decl.init = Some(self.expr()?);
            },
            DeclTailKind::Typed => {
                decl.var_ty_expr = Some(self.expr_(DECL_TYPE_PRECEDENCE)?);
                let eq = self.lex.next_if(|t| matches!(t.kind, TokenKind::Eq | TokenKind::Colon));
                decl.is_const = eq.is_some_and(|t| t.kind == TokenKind::Colon);
                decl.init = then!(eq.is_some() => self.expr()?);
            },
        }
        if decl.init.is_some() {
            decl.flags.set(DeclFlags::HAS_INIT_EXPR);
        }
        Ok(decl)
    }

    fn ident(&mut self) -> ParseResult<Ptr<ast::Ident>> {
        let t = self.tok(TokenKind::Ident)?;
        self.ident_from_span(t.span)
    }

    /// this doesn't check if the text at span is valid
    fn ident_from_span(&self, span: Span) -> ParseResult<Ptr<ast::Ident>> {
        let sym = ctx_mut().symbols.get_or_intern(self.get_text_from_span(span));
        Ok(ast_new!(Ident { span, sym, decl: None }))
    }

    fn local_keyword(&mut self, local_keyword: &str) -> ParseResult<()> {
        let i = self.tok(TokenKind::Ident)?;
        if &*self.get_text_from_span(i.span) != local_keyword {
            return Err(unexpected_token_expect1(i, format_args!("`{local_keyword}`")));
        }
        Ok(())
    }

    /// Parses the `do` keyword 0 or 1 times.
    fn opt_do(&mut self) {
        self.lex.advance_if_kind(TokenKind::Keyword(Keyword::Do));
    }

    // -------

    fn tok(&mut self, tok: TokenKind) -> ParseResult<Token> {
        self.tok_with_expected(tok, &[tok])
    }

    fn tok_with_expected(&mut self, tok: TokenKind, expected: &[TokenKind]) -> ParseResult<Token> {
        debug_assert!(expected.contains(&tok));
        let t = self.lex.peek_or_eof();
        if t.kind == tok {
            self.lex.advance();
            Ok(t)
        } else {
            return Err(unexpected_token(t, expected));
        }
    }

    fn advanced(&mut self) -> &mut Self {
        self.lex.advance();
        self
    }

    // helpers:

    #[inline]
    fn alloc<T>(&self, val: T) -> ParseResult<Ptr<T>> {
        Ok(self.cctx.alloc.alloc(val)?)
    }

    #[inline]
    fn alloc_slice<T: Copy>(&self, slice: &[T]) -> ParseResult<Ptr<[T]>> {
        Ok(self.cctx.alloc.alloc_slice(slice)?)
    }

    fn get_text_from_span(&self, span: Span) -> Ptr<str> {
        Ptr::from_ref(&self.lex.get_code()[span])
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
    /// `a orelse b`, `a ?? b`
    OrElse,

    /// `a |> b`
    Pipe,

    /// `a = b`
    /// `  ^`
    Assign,
    /// `a op= b`
    /// `  ^^^`
    BinOpAssign(BinOpKind),

    Decl(DeclTailKind),
}

#[derive(Debug)]
pub enum DeclTailKind {
    /// `a := b`
    /// `  ^^`
    Var,
    /// `a :: b`
    /// `  ^^`
    Const,
    /// `a: ty = b` or `a: ty : b`
    /// ` ^`         `   ^`
    Typed,
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
            TokenKind::AmpersandAmpersand | TokenKind::Keyword(Keyword::And) => {
                FollowingOperator::BinOp(BinOpKind::And)
            },
            TokenKind::AmpersandAmpersandEq | TokenKind::Keyword(Keyword::AndEq) => {
                FollowingOperator::BinOpAssign(BinOpKind::And)
            },
            TokenKind::AmpersandEq => FollowingOperator::BinOpAssign(BinOpKind::BitAnd),
            TokenKind::Pipe => FollowingOperator::BinOp(BinOpKind::BitOr),
            TokenKind::PipePipe | TokenKind::Keyword(Keyword::Or) => {
                FollowingOperator::BinOp(BinOpKind::Or)
            },
            TokenKind::PipePipeEq | TokenKind::Keyword(Keyword::OrEq) => {
                FollowingOperator::BinOpAssign(BinOpKind::Or)
            },
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
            TokenKind::DotOpenBracket => FollowingOperator::ArrayInitializer,
            TokenKind::DotOpenBrace => FollowingOperator::NamedInitializer,
            TokenKind::Colon => FollowingOperator::Decl(DeclTailKind::Typed),
            TokenKind::ColonColon => FollowingOperator::Decl(DeclTailKind::Const),
            TokenKind::ColonEq => FollowingOperator::Decl(DeclTailKind::Var),
            //TokenKind::Semicolon => todo!("TokenKind::Semicolon"),
            TokenKind::Question => FollowingOperator::PostOp(UnaryOpKind::Try),
            //TokenKind::QuestionQuestion |
            TokenKind::Keyword(Keyword::OrElse) => FollowingOperator::OrElse,
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
            FollowingOperator::Dot => DOT_PRECEDENCE,
            FollowingOperator::Call => CALL_PRECEDENCE,
            FollowingOperator::Index
            | FollowingOperator::PositionalInitializer
            | FollowingOperator::NamedInitializer
            | FollowingOperator::ArrayInitializer => INITIALIZER_PRECEDENCE,
            FollowingOperator::SingleArgNoParenFn | FollowingOperator::PostOp(_) => {
                DIRECT_POSTOP_PRECEDENCE
            },
            FollowingOperator::BinOp(k) => k.precedence(),
            FollowingOperator::Range { .. } => RANGE_PRECEDENCE,
            FollowingOperator::OrElse => ORELSE_PRECEDENCE,
            FollowingOperator::Pipe => PIPE_PRECEDENCE,
            FollowingOperator::Assign
            | FollowingOperator::BinOpAssign(_)
            | FollowingOperator::Decl(_) => ASSIGN_PRECEDENCE,
        }
    }
}

const __PRECEDENCE_CHECKS: () = const {
    assert!(DOT_PRECEDENCE > PIPE_TARGET_PRECEDENCE); // ... |> (A.func)(...)
    assert!(PIPE_TARGET_PRECEDENCE > CALL_PRECEDENCE); // (... |> A.func)(...)

    assert!(TY_PREFIX_PRECEDENCE > INITIALIZER_PRECEDENCE); // ([]u8).{ ptr, len }

    assert!(ORELSE_PRECEDENCE > PIPE_PRECEDENCE); // (optional orelse default) |> do_something()

    assert!(DECL_TYPE_PRECEDENCE > ASSIGN_PRECEDENCE); // a: (ty) = init

    assert!(CALL_PRECEDENCE > TY_PREFIX_PRECEDENCE); // *(MyType(i32))
};

const MAX_PRECEDENCE: u8 = u8::MAX;

const DOT_PRECEDENCE: u8 = 25;
const PIPE_TARGET_PRECEDENCE: u8 = 24;
const CALL_PRECEDENCE: u8 = 23;

/// for `*ty`, `[]ty`, `?ty`
const TY_PREFIX_PRECEDENCE: u8 = 22;

const DIRECT_POSTOP_PRECEDENCE: u8 = 21;
const INITIALIZER_PRECEDENCE: u8 = DIRECT_POSTOP_PRECEDENCE;

const PREOP_PRECEDENCE: u8 = 20;
// BinOp precedence: 11..=19
const RANGE_PRECEDENCE: u8 = 10;

const ORELSE_PRECEDENCE: u8 = 5;
const PIPE_PRECEDENCE: u8 = 4;

/// `a: ty = init`
/// `   ^^`
const DECL_TYPE_PRECEDENCE: u8 = 3;

/// `a = 1`
/// `a := init`
/// `a : ty = init`
/// `  ^`
const ASSIGN_PRECEDENCE: u8 = 2;

const IF_PRECEDENCE: u8 = 1;
const ELSE_PRECEDENCE: u8 = 1;

const MIN_PRECEDENCE: u8 = 0;

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

// An explicit state flag might be better.
fn is_type_prec(prec: u8) -> bool {
    [DECL_TYPE_PRECEDENCE, TY_PREFIX_PRECEDENCE].contains(&prec)
}

type HasTrailingSemicolon = bool;

const EXPECTED_AFTER_IDENT_PARAM: [TokenKind; 4] =
    concat_arr!(VAR_DECL_TAIL_TOKENS, EXPECTED_AFTER_PARAM);
const EXPECTED_AFTER_PARAM: [TokenKind; 2] = [TokenKind::Comma, TokenKind::CloseParenthesis];
const VAR_DECL_TAIL_TOKENS: [TokenKind; 2] = [TokenKind::Colon, TokenKind::ColonEq];
const DECL_TAIL_TOKENS: [TokenKind; 3] = concat_arr!(VAR_DECL_TAIL_TOKENS, [TokenKind::ColonColon]);
