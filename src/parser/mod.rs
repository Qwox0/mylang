//! # Parser module
//!
//! The parser allocates the AST ([`Expr`]) and the stored [`Type`]s to
//! [`Type::Unset`] or [`Type::Unevaluated`]

use crate::{
    ast::{
        self, Ast, AstKind, BinOpKind, DeclList, DeclMarkers, UnaryOpKind, UpcastToAst, ast_new,
    },
    context::{CompilationContextInner, primitives},
    diagnostics::{HandledErr, cerror, cerror2, chint, cwarn},
    literals::{self, replace_escape_chars},
    ptr::{OPtr, Ptr},
    scope::{Scope, ScopeAndAggregateInfo, ScopeKind, ScopePos},
    scratch_pool::ScratchPool,
    util::{UnwrapDebug, concat_arr, then, unreachable_debug},
};
use core::str;
pub use error::*;
use lexer::{Keyword, Lexer, Span, Token, TokenKind, is_ascii_space_or_tab};
use parser_helper::ParserInterface;

pub mod error;
pub mod lexer;
pub mod parser_helper;

macro_rules! opt {
    ($self:expr, $method:ident($($arg:expr),* $(,)?), $prec:expr) => {{
        let _self: &mut Parser = $self;
        if _self.lex.peek_or_eof().kind.is_invalid_start($prec) {
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
        $p.lex.advance_while(|t| {
            match t.kind {
                $open_scope_pat => depth += 1,
                $until_pat if depth == 0 => return false,
                $close_scope_pat => depth -= 1,
                _ => {},
            };
            true
        });
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
            let res: Result<Ptr<ast::Ast>, ParseError> = try {
                $parse_item
            };
            let Ok(expr) = res else {
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

impl Ptr<Ast> {
    pub fn try_to_ident(self) -> Result<Ptr<ast::Ident>, HandledErr> {
        self.try_downcast::<ast::Ident>()
            .ok_or_else(|| cerror!(self.full_span(), "expected an identifier, got an expression"))
    }
}

pub fn parse_files_in_ctx<'a>(cctx: Ptr<CompilationContextInner>) -> &'a mut [Ptr<Ast>] {
    let mut stmts = &mut cctx.as_mut().stmts;
    // Note: this idx-based loop is needed because `ctx.files` might get mutated while the loop is
    // running.
    let mut idx = 0;
    while let Some(file) = cctx.files.get(idx).copied() {
        let start_idx = stmts.len();
        let mut p = Parser { lex: Lexer::new(file), cctx };
        p.parse_stmts_into(&mut stmts, false);
        while let Some(t) = p.lex.next() {
            debug_assert_eq!(t.kind, TokenKind::CloseBrace);
            cerror!(t.span, "unexpected token");
            p.parse_stmts_into(&mut stmts, false);
        }
        debug_assert!(p.lex.is_empty());
        let stmt_range = start_idx..stmts.len();
        file.as_mut().set_stmt_range(stmt_range.clone());

        let mut scope =
            Scope::from_stmts(&stmts[stmt_range], ScopeKind::File, &cctx.alloc).unwrap();
        scope.parent = Some(cctx.root_scope);
        scope.pos_in_parent = ScopePos(cctx.root_scope.decls.len() as u32);
        file.as_mut().scope = Some(scope);

        idx += 1;
    }
    stmts
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
            FollowingOperator::Call => self.call(lhs, ScratchPool::new(), None)?.upcast(),
            FollowingOperator::Index => {
                let idx = self.expr()?;
                let close = self.tok(TokenKind::CloseBracket)?;
                let mut_t = self.lex.next_if_kind(TokenKind::Keyword(Keyword::Mut));
                expr!(Index { mut_access: mut_t.is_some(), lhs, idx }, mut_t.unwrap_or(close).span)
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
            FollowingOperator::ArrayInitializer => self.parse_array_initializer(Some(lhs), span)?,
            FollowingOperator::SingleArgNoParenFn => {
                let Ok(lhs) = lhs.try_to_ident() else { panic!("SingleArgFn: unknown rhs") };
                let param = self.alloc(ast::Decl::from_ident(lhs))?;
                let params = self.alloc_one_val_slice(param)?;
                self.function_tail(params, span, min_precedence)?.upcast()
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
            FollowingOperator::Pipe => {
                let t = self.lex.peek_or_eof();
                match t.kind {
                    TokenKind::Keyword(Keyword::If | Keyword::Then) => {
                        self.advanced().if_after_cond(lhs, span, true)?.upcast()
                    },
                    TokenKind::Keyword(Keyword::Match) => {
                        todo!("|> match")
                    },
                    TokenKind::Keyword(Keyword::For) => {
                        let iter_var = self.advanced().ident()?;
                        self.opt_do();
                        let body = self.expr()?;

                        let iter_var = self.alloc(ast::Decl::from_ident(iter_var))?;
                        let scope = Scope::new(self.alloc_slice(&[iter_var])?, ScopeKind::ForLoop);
                        expr!(For { source: lhs, iter_var, body, scope, was_piped: true }, t.span)
                    },
                    TokenKind::Keyword(Keyword::While) => {
                        self.advanced().opt_do();
                        let body = self.expr()?;
                        expr!(While { condition: lhs, body, was_piped: true }, t.span)
                    },
                    _ => {
                        let func = self.expr_(PIPE_TARGET_PRECEDENCE)?;
                        let mut args = ScratchPool::new();
                        if let Some(dot) = func.try_downcast::<ast::Dot>()
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
                        } else {
                            args.push(lhs)?;
                        }
                        self.tok(TokenKind::OpenParenthesis)?;
                        self.call(func, args, Some(0))?.upcast()
                    },
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
            FollowingOperator::Decl(kind) => {
                self.decl_tail(self.alloc(ast::Decl::from_lhs(lhs)?)?, kind)?.upcast()
            },
        });
    }

    /// anything which has higher precedence than any operator
    fn value(&mut self, prec: u8) -> ParseResult<Ptr<Ast>> {
        let Token { kind, span } = self.lex.peek_or_eof();

        Ok(match kind {
            TokenKind::Ident => {
                let text = self.advanced().get_text_from_span(span);
                let sym = self.cctx.symbols.get_or_intern(text);
                expr!(Ident { sym, decl: None }, span)
            },
            TokenKind::Keyword(Keyword::Mut | Keyword::Rec | Keyword::Pub | Keyword::Static) => {
                self.var_decl(false)?.upcast()
            },
            TokenKind::Keyword(k @ Keyword::Struct) => {
                self.advanced().tok(TokenKind::OpenBrace)?;
                let ScopeAndAggregateInfo { scope, fields, consts } = self.struct_body(k)?;
                let close_b = self.tok(TokenKind::CloseBrace)?;
                expr!(
                    StructDef { scope, fields, consts, finished_members: 0 },
                    span.join(close_b.span)
                )
            },
            TokenKind::Keyword(k @ Keyword::Union) => {
                self.advanced().tok(TokenKind::OpenBrace)?;
                let ScopeAndAggregateInfo { scope, fields, consts } = self.struct_body(k)?;
                let close_b = self.tok(TokenKind::CloseBrace)?;
                expr!(
                    UnionDef { scope, fields, consts, finished_members: 0 },
                    span.join(close_b.span)
                )
            },
            TokenKind::Keyword(Keyword::Enum) => {
                self.advanced().tok(TokenKind::OpenBrace)?;
                let mut variants = Vec::new();
                //let consts = Vec::new(); // TODO: allow constants in enum block
                parse_in_block!(
                    self,
                    sep = [TokenKind::Semicolon, TokenKind::Comma],
                    in_block = true,
                    {
                        let Some(variant_ident) = opt!(self, ident(), MIN_PRECEDENCE)? else {
                            break;
                        };
                        let ty = then!(
                            self.lex.advance_if_kind(TokenKind::OpenParenthesis) => {
                            let ty_expr = self.expr()?;
                            self.tok(TokenKind::CloseParenthesis)?;
                            ty_expr
                        });
                        let variant_index = then!(
                            self.lex.advance_if_kind(TokenKind::Eq)
                            => self.expr()?
                        );
                        let mut decl = self.alloc(ast::Decl::new(variant_ident, None, span))?;
                        decl.var_ty_expr = ty;
                        decl.init = variant_index;
                        variants.push(decl);
                        decl.upcast() // Note: variants are not constant and thus always expect a trailing seperator
                    }
                );
                let close_b = self.tok(TokenKind::CloseBrace)?;
                let ScopeAndAggregateInfo { scope, fields, consts } =
                    Scope::for_aggregate(variants, vec![], &self.cctx.alloc, ScopeKind::Enum)?;
                expr!(
                    EnumDef {
                        scope,
                        variants: fields,
                        finished_members: 0,
                        variant_tags: None,
                        consts,
                        is_simple_enum: false,
                        tag_ty: None,
                    },
                    span.join(close_b.span)
                )
            },
            TokenKind::Keyword(Keyword::Unsafe) => todo!("unsafe"),
            TokenKind::Keyword(Keyword::If) => {
                let condition = self.advanced().expr()?;
                self.if_after_cond(condition, span, false)?.upcast()
            },
            TokenKind::Keyword(Keyword::Match) => {
                todo!("match body");
                /*
                let val = self.advanced().expr()?;
                let else_body = then!(self.lex.advance_if_kind(TokenKind::Keyword(Keyword::Else))
                    => self.expr()?);
                expr!(Match { val, else_body, was_piped: false }, span)
                */
            },
            TokenKind::Keyword(Keyword::For) => {
                let iter_var = self.advanced().ident()?;
                self.local_keyword("in")?;
                let source = self.expr()?;
                self.opt_do();
                let body = self.expr()?;

                let iter_var = self.alloc(ast::Decl::from_ident(iter_var))?;
                let scope = Scope::new(self.alloc_slice(&[iter_var])?, ScopeKind::ForLoop);
                expr!(For { source, iter_var, body, scope, was_piped: false }, span)
            },
            TokenKind::Keyword(Keyword::While) => {
                let condition = self.advanced().expr()?;
                self.opt_do();
                let body = self.expr()?;
                expr!(While { condition, body, was_piped: false }, span)
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
                use std::num::IntErrorKind;
                let val = match literals::parse_int_lit(&self.advanced().get_text_from_span(span)) {
                    Ok(val) => val,
                    Err(e) if *e.kind() == IntErrorKind::PosOverflow => {
                        cwarn!(
                            span,
                            "Currently, the compiler cannot store a literal this big internally. \
                             Using i64::MAX instead."
                        );
                        i64::MAX
                    },
                    Err(e) if *e.kind() == IntErrorKind::NegOverflow => {
                        cwarn!(
                            span,
                            "Currently, the compiler cannot store a literal this small \
                             internally. Using i64::MIN instead."
                        );
                        i64::MIN
                    },
                    Err(e) => return cerror2!(span, "invalid integer literal: {e}"),
                };
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
                let code = replace_escape_chars(&self.advanced().lex.get_code()[span]);
                let mut chars = code.chars();

                let start = chars.next();
                debug_assert_eq!(start, Some('\''));
                let end = chars.next_back();
                debug_assert_eq!(end, Some('\''));

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
                expr!(IntVal { val: byte as i64 }, span)
            },
            TokenKind::StrLit => {
                let lit = self.advanced().get_text_from_span(span);
                expr!(StrVal { text: Ptr::from(&lit[1..lit.len().saturating_sub(1)]) }, span)
            },
            TokenKind::MultilineStrLitLine => {
                // Note: Arena allocates in the wrong direction
                let mut scratch = Vec::with_capacity(1024);
                while let Some(t) = self.lex.next_if_kind(TokenKind::MultilineStrLitLine) {
                    let line_text = self.get_text_from_span(t.span);
                    debug_assert_eq!(&line_text[0..2], "\\\\");
                    // `\\ a` == "a" == `\\a`
                    let start_idx =
                        2 + line_text.as_bytes().get(2).copied().is_some_and(is_ascii_space_or_tab)
                            as usize;
                    scratch.extend_from_slice(line_text[start_idx..].as_bytes());
                }
                let bytes = self.alloc_slice(&scratch)?;
                let text = unsafe { std::str::from_utf8_unchecked(&bytes) };
                debug_assert!(text.ends_with('\n'));
                expr!(StrVal { text: Ptr::from(&text[0..text.len().saturating_sub(1)]) }, span)
            },
            TokenKind::OpenParenthesis => {
                // TODO: currently no tuples allowed!
                // () -> ...
                if self.advanced().lex.advance_if_kind(TokenKind::CloseParenthesis) {
                    self.tok(TokenKind::Arrow)?;
                    return Ok(self.function_tail(Ptr::empty_slice(), span, prec)?.upcast());
                }
                let mut first_expr = self.expr()?; // this assumes the parameter syntax is also a valid expression
                let t = self.lex.next_or_eof();
                let params = match t.kind {
                    // (expr)
                    TokenKind::CloseParenthesis if !self.lex.advance_if_kind(TokenKind::Arrow) => {
                        first_expr.parenthesis_count += 1;
                        return Ok(first_expr);
                    },
                    // (expr) -> ...
                    TokenKind::CloseParenthesis => {
                        let Some(decl) = first_expr.try_to_decl()? else {
                            return unexpected_expr(first_expr, "a parameter");
                        };
                        self.alloc_slice(&[decl])?
                    },
                    // (params...) -> ...
                    TokenKind::Comma => {
                        let Some(first_decl) = first_expr.try_to_decl()? else {
                            return unexpected_expr(first_expr, "a parameter");
                        };
                        let params = ScratchPool::new_with_first_val(first_decl)?;
                        let params = self.var_decl_list(TokenKind::Comma, params, true)?;
                        let expected_tok = if params.last().is_some_and(|d| d.is_lhs_only()) {
                            &EXPECTED_AFTER_IDENT_PARAM[..]
                        } else {
                            &EXPECTED_AFTER_PARAM[..]
                        };
                        self.tok_with_expected(TokenKind::CloseParenthesis, expected_tok)?;
                        self.tok(TokenKind::Arrow)?;
                        params
                    },
                    _ if first_expr.kind == AstKind::Ident => {
                        return unexpected_token(t, &EXPECTED_AFTER_IDENT_PARAM);
                    },
                    _ => return unexpected_token(t, &EXPECTED_AFTER_PARAM),
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
            TokenKind::Arrow => {
                self.lex.advance();
                self.function_tail(Ptr::empty_slice(), span, prec)?.upcast()
            },
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
                let (args, close_p_span) = self.advanced().parse_call(ScratchPool::new())?;
                let span = span.join(close_p_span);
                expr!(PositionalInitializer { lhs: None, args }, span)
            },
            TokenKind::DotOpenBrace => {
                let (fields, close_b_span) = self.advanced().parse_initializer_fields()?;
                let span = span.join(close_b_span);
                expr!(NamedInitializer { lhs: None, fields }, span)
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
                    arg.and_then(Ptr::<Ast>::try_downcast::<ast::StrVal>).ok_or_else(|| {
                        let span = arg.map(|a| a.full_span()).unwrap_or(self.lex.pos_span());
                        cerror!(span, "Expected {} after the #{directive_name} directive", usage)
                    })
                };

                let p = primitives();

                // function-like directives:
                if directive_name == "import" {
                    let path = parse_str_lit_arg("a path string literal")?;
                    let idx =
                        self.cctx.add_import(&path.text, Some(&self.lex.file.path), path.span)?;
                    expr!(ImportDirective { path, files_idx: idx }, span)
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
                    let type_ = self.expr()?;
                    expr!(SizeOfDirective { type_ }, span.join(directive_ident.span))
                } else if directive_name == "sizeof_val" {
                    let val = self.expr()?;
                    expr!(SizeOfValDirective { val }, span.join(directive_ident.span))
                } else if directive_name == "offsetof" {
                    self.tok(TokenKind::OpenParenthesis)?;
                    let type_ = self.expr()?;
                    self.tok(TokenKind::Comma)?;
                    let field = self.ident()?;
                    self.tok(TokenKind::CloseParenthesis)?;
                    expr!(OffsetOfDirective { type_, field }, span.join(directive_ident.span))
                }
                // annotation directives:
                else if directive_name == "no_mangle" {
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
                        debug_assert!(decl.ident.replacement.is_none());
                        decl.as_mut().ident.replacement = Some(main_ident);
                        func
                    }
                } else {
                    return cerror2!(span.join(directive_ident.span), "Unknown compiler directive");
                }
            },
            TokenKind::Dollar => todo!("TokenKind::Dollar"),
            TokenKind::At => todo!("TokenKind::At"),
            TokenKind::Tilde => todo!("TokenKind::Tilde"),
            TokenKind::Backslash => todo!("TokenKind::BackSlash"),
            TokenKind::Backtick => todo!("TokenKind::BackTick"),
            kind => return unexpected_token(Token { kind, span }, &[]),
        })
    }

    /// also parses the `}`
    fn parse_initializer_fields(
        &mut self,
    ) -> ParseResult<(Ptr<[(Ptr<ast::Ident>, OPtr<Ast>)]>, Span)> {
        let mut fields = ScratchPool::new();
        let close_b_span = loop {
            if let Some(t) = self.lex.next_if_kind(TokenKind::CloseBrace) {
                break t.span;
            }
            let ident = self.ident()?;
            let init = then!(self.lex.advance_if_kind(TokenKind::Eq) => self.expr()?);
            fields.push((ident, init))?;

            match self.lex.next_or_eof() {
                Token { kind: TokenKind::Comma, .. } => {},
                Token { kind: TokenKind::CloseBrace, span } => break span,
                t => {
                    return unexpected_token(
                        t,
                        if init.is_none() {
                            &[TokenKind::Eq, TokenKind::Comma, TokenKind::CloseBrace]
                        } else {
                            &[TokenKind::Comma, TokenKind::CloseBrace]
                        },
                    );
                },
            }
        };
        Ok((self.clone_slice_from_scratch_pool(fields)?, close_b_span))
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
                expr!($kind { lhs, $($field $(: $val )?),* }, open_b_span.join(close_b.span))
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
                let elements = self.scratch_pool_with_first_val(first_expr)?;
                let elements = self.clone_slice_from_scratch_pool(elements)?;
                new_arr_init!(ArrayInitializer { elements })
            },
            // `.[expr; count]`
            TokenKind::Semicolon => {
                let count = self.advanced().expr()?;
                new_arr_init!(ArrayInitializerShort { val: first_expr, count })
            },
            // `.[expr,]` or `.[expr, expr, ...]`
            TokenKind::Comma => {
                let elems = self.advanced().scratch_pool_with_first_val(first_expr)?;
                let elements = self.expr_list(TokenKind::Comma, elems)?.0;
                new_arr_init!(ArrayInitializer { elements })
            },
            _ => {
                return unexpected_token(t, &[
                    TokenKind::Comma,
                    TokenKind::Semicolon,
                    TokenKind::CloseBracket,
                ]);
            },
        })
    }

    /// parsing starts after the '->'
    fn function_tail(
        &mut self,
        params: DeclList,
        start_span: Span,
        min_precedence: u8,
    ) -> ParseResult<Ptr<ast::Fn>> {
        let expr = self.expr_(min_precedence)?;
        let between_expr_state = self.lex.get_state();
        let (ret_ty_expr, body) = if expr.kind != AstKind::Block
            && let Some(body) = opt!(self, expr(), min_precedence)?
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
        let body = Some(body);
        let params_scope = Scope::new(params, ScopeKind::Fn);
        Ok(ast_new!(
            Fn { params_scope, ret_ty_expr, ret_ty: None, body, has_known_ret_ty: false },
            start_span
        ))
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
            => self.expr_(IF_PRECEDENCE)?);
        Ok(ast_new!(If { condition, then_body, else_body, was_piped }, start_span))
    }

    /// `... ( ... )`
    /// `     ^` starts here
    /// TODO: `... ( <expr>, ..., param=<expr>, ... )`
    fn parse_call(&mut self, args: ScratchPool<Ptr<Ast>>) -> ParseResult<(Ptr<[Ptr<Ast>]>, Span)> {
        let res: ParseResult<_> = try {
            //let args = self.parse_call_args(args)?;
            let args = self.expr_list(TokenKind::Comma, args)?.0;
            let closing_paren_span =
                self.tok_with_expected(TokenKind::CloseParenthesis, &EXPECTED_AFTER_PARAM)?.span;
            (args, closing_paren_span)
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
        args: ScratchPool<Ptr<Ast>>,
        pipe_idx: Option<usize>,
    ) -> ParseResult<Ptr<ast::Call>> {
        let (args, closing_paren_span) = self.parse_call(args)?;
        Ok(ast_new!(Call { func, args, pipe_idx }, closing_paren_span))
    }

    /// expects next token to be '{' and parses until and including the '}'
    fn block(&mut self) -> ParseResult<Ptr<ast::Block>> {
        let open_b = self.lex.next().u();
        debug_assert_eq!(open_b.kind, TokenKind::OpenBrace);

        let mut stmts = Vec::new();
        let has_trailing_semicolon = self.parse_stmts_into(&mut stmts, true);

        let close_b = self.tok(TokenKind::CloseBrace)?;

        let span = open_b.span.join(close_b.span);
        let stmts = self.alloc_slice(&stmts)?;
        let scope = Scope::from_stmts(&stmts, ScopeKind::Block, &self.cctx.alloc)?;
        Ok(self.alloc(ast::Block::new(stmts, scope, has_trailing_semicolon, span))?)
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
    ) -> HasTrailingSemicolon {
        parse_in_block!(self, sep = [TokenKind::Semicolon], in_block = in_block, {
            let expr = self.expr()?;
            stmts.push(expr);
            expr
        })
    }

    /// Also returns a `has_trailing_sep` [`bool`].
    fn expr_list(
        &mut self,
        sep: TokenKind,
        mut list_pool: ScratchPool<Ptr<Ast>>,
    ) -> ParseResult<(Ptr<[Ptr<Ast>]>, bool)> {
        let mut has_trailing_sep = false;
        loop {
            let Some(expr) = opt!(self, expr(), MIN_PRECEDENCE)? else { break };
            list_pool.push(expr)?;
            has_trailing_sep = self.lex.advance_if_kind(sep);
            if !has_trailing_sep {
                break;
            }
        }
        debug_assert!(self.lex.peek().is_none_or(|t| t.kind != sep));
        Ok((self.clone_slice_from_scratch_pool(list_pool)?, has_trailing_sep))
    }

    /// Parses [`Parser::var_decl`] multiple times, seperated by `sep`. Also allows a trailing
    /// `sep`.
    fn var_decl_list(
        &mut self,
        sep: TokenKind,
        mut list: ScratchPool<Ptr<ast::Decl>>,
        allow_ident_only: bool,
    ) -> ParseResult<DeclList> {
        loop {
            let Some(decl) = opt!(self, var_decl(allow_ident_only), MIN_PRECEDENCE)? else {
                break;
            };
            list.push(decl)?;
            if !self.lex.advance_if_kind(sep) {
                break;
            }
        }
        debug_assert!(self.lex.peek().is_none_or(|t| t.kind != sep));
        self.clone_slice_from_scratch_pool(list)
    }

    /// also returns the field_count
    fn struct_body(&mut self, kind: Keyword) -> ParseResult<ScopeAndAggregateInfo> {
        let mut fields = Vec::new();
        let mut consts = Vec::new();
        parse_in_block!(self, sep = [TokenKind::Semicolon, TokenKind::Comma], in_block = true, {
            let expr = self.expr()?;
            if let Some(decl) = expr.try_downcast::<ast::Decl>() {
                if let Some(on_ty_expr) = decl.on_type {
                    cerror!(on_ty_expr.full_span(), "currently not supported"); // TODO
                } else if decl.is_const {
                    consts.push(decl);
                } else {
                    fields.push(decl);
                }
            } else {
                cerror!(expr.full_span(), "expected field or constant declaration");
            };
            expr
        });
        let kind = match kind {
            Keyword::Struct => ScopeKind::Struct,
            Keyword::Union => ScopeKind::Union,
            _ => unreachable_debug(),
        };
        Scope::for_aggregate(fields, consts, &self.cctx.alloc, kind)
    }

    fn var_decl(&mut self, allow_ident_only: bool) -> ParseResult<Ptr<ast::Decl>> {
        let markers = self.decl_markers()?;
        let lhs = self.expr_(ASSIGN_PRECEDENCE)?;
        let decl = self.alloc(ast::Decl::from_lhs(lhs)?)?;
        decl.as_mut().markers = markers;
        let t = self.lex.peek_or_eof();
        let kind = match t.kind {
            TokenKind::Colon => DeclTailKind::Typed,
            TokenKind::ColonEq => DeclTailKind::Var,
            TokenKind::ColonColon => DeclTailKind::Const,
            _ if allow_ident_only => return Ok(decl),
            _ => return unexpected_token(t, &DECL_TAIL_TOKENS),
        };
        self.advanced().decl_tail(decl, kind)
    }

    fn decl_markers(&mut self) -> ParseResult<DeclMarkers> {
        let mut markers = DeclMarkers::default();
        let mut t = self.lex.peek_or_eof();

        macro_rules! set_marker {
            ($variant:ident $mask:ident) => {
                if markers.get(DeclMarkers::$mask) {
                    let marker_text = Keyword::$variant.as_str();
                    return cerror2!(t.span, "duplicate marker '{marker_text}' on declaration");
                } else {
                    markers.set(DeclMarkers::$mask)
                }
            };
        }

        while t.kind != TokenKind::Ident {
            match t.kind {
                TokenKind::Keyword(Keyword::Mut) => set_marker!(Mut IS_MUT_MASK),
                TokenKind::Keyword(Keyword::Rec) => set_marker!(Rec IS_REC_MASK),
                TokenKind::Keyword(Keyword::Pub) => set_marker!(Pub IS_PUB_MASK),
                TokenKind::Keyword(Keyword::Static) => set_marker!(Static IS_STATIC_MASK),
                _ => {
                    return unexpected_token(t, &[
                        TokenKind::Ident,
                        TokenKind::Keyword(Keyword::Mut),
                        TokenKind::Keyword(Keyword::Rec),
                        TokenKind::Keyword(Keyword::Pub),
                        TokenKind::Keyword(Keyword::Static),
                    ])
                    .into();
                },
            }
            t = self.advanced().lex.peek_or_eof();
        }

        Ok(markers)
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
        Ok(decl)
    }

    fn ident(&mut self) -> ParseResult<Ptr<ast::Ident>> {
        let t = self.tok(TokenKind::Ident)?;
        self.ident_from_span(t.span)
    }

    /// this doesn't check if the text at span is valid
    fn ident_from_span(&self, span: Span) -> ParseResult<Ptr<ast::Ident>> {
        self.alloc(ast::Ident::new(self.get_text_from_span(span), span))
    }

    fn local_keyword(&mut self, local_keyword: &str) -> ParseResult<()> {
        let i = self.tok(TokenKind::Ident)?;
        if &*self.get_text_from_span(i.span) != local_keyword {
            return unexpected_token_expect1(i, format_args!("`{local_keyword}`"));
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
            return unexpected_token(t, expected);
        }
    }

    fn advanced(&mut self) -> &mut Self {
        self.lex.advance();
        self
    }

    // helpers:

    #[inline]
    fn alloc<T>(&self, val: T) -> ParseResult<Ptr<T>> {
        self.cctx.alloc.alloc(val)
    }

    #[inline]
    fn alloc_slice<T: Copy>(&self, slice: &[T]) -> ParseResult<Ptr<[T]>> {
        self.cctx.alloc.alloc_slice(slice)
    }

    fn scratch_pool_with_first_val<'bump, T>(
        &self,
        first: T,
    ) -> ParseResult<ScratchPool<'bump, T>> {
        ScratchPool::new_with_first_val(first)
    }

    /// Clones all values from a [`ScratchPool`] to `self.alloc`.
    #[inline]
    fn clone_slice_from_scratch_pool<T: Clone>(
        &self,
        scratch_pool: ScratchPool<T>,
    ) -> ParseResult<Ptr<[T]>> {
        scratch_pool.clone_to_slice_into_arena(&self.cctx.alloc)
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
            FollowingOperator::Call
            | FollowingOperator::Index
            | FollowingOperator::PositionalInitializer
            | FollowingOperator::NamedInitializer
            | FollowingOperator::ArrayInitializer
            | FollowingOperator::SingleArgNoParenFn
            | FollowingOperator::PostOp(_) => POSTOP_PRECEDENCE,
            FollowingOperator::BinOp(k) => k.precedence(),
            FollowingOperator::Range { .. } => RANGE_PRECEDENCE,
            FollowingOperator::Pipe => 4,
            FollowingOperator::Assign
            | FollowingOperator::BinOpAssign(_)
            | FollowingOperator::Decl(_) => ASSIGN_PRECEDENCE,
        }
    }
}

const MAX_PRECEDENCE: u8 = u8::MAX;
const DOT_PRECEDENCE: u8 = 24;
/// must be higher than [`POSTOP_PRECEDENCE`] and lower than [`DOT_PRECEDENCE`] to parse `... |> A.func(...)` correctly
const PIPE_TARGET_PRECEDENCE: u8 = 23;
/// for `*ty`, `[]ty`, `?ty`
const TY_PREFIX_PRECEDENCE: u8 = 22;
const POSTOP_PRECEDENCE: u8 = 21;
const PREOP_PRECEDENCE: u8 = 20;

const RANGE_PRECEDENCE: u8 = 10;
/// `a: ty = init`
/// `   ^^`
/// must be higher than [`FollowingOperator::Assign`]!
const DECL_TYPE_PRECEDENCE: u8 = 3;
/// `a = 1`
/// `a := init`
/// `a : ty = init`
/// `  ^`
const ASSIGN_PRECEDENCE: u8 = 2;
const IF_PRECEDENCE: u8 = 1;
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

type HasTrailingSemicolon = bool;

const EXPECTED_AFTER_IDENT_PARAM: [TokenKind; 4] =
    concat_arr!(VAR_DECL_TAIL_TOKENS, EXPECTED_AFTER_PARAM);
const EXPECTED_AFTER_PARAM: [TokenKind; 2] = [TokenKind::Comma, TokenKind::CloseParenthesis];
const VAR_DECL_TAIL_TOKENS: [TokenKind; 2] = [TokenKind::Colon, TokenKind::ColonEq];
const DECL_TAIL_TOKENS: [TokenKind; 3] = concat_arr!(VAR_DECL_TAIL_TOKENS, [TokenKind::ColonColon]);
