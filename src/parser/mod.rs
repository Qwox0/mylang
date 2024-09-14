use crate::{
    ast::{
        BinOpKind, DeclMarkerKind, DeclMarkers, Expr, ExprKind, Fn, Ident, PostOpKind, PreOpKind,
        Type, VarDecl,
    },
    ptr::Ptr,
    scratch_pool::ScratchPool,
};
use debug::TreeLines;
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
            _ => err!(NotAnIdent, self.span),
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

// methods may advance the parser even on error
// resetting is done by the callee
impl<'code, 'alloc> Parser<'code, 'alloc> {
    pub fn new(lex: Lexer<'code>, alloc: &'alloc bumpalo::Bump) -> Parser<'code, 'alloc> {
        Self { lex, alloc }
    }

    pub fn top_level_item(&mut self) -> ParseResult<Ptr<Expr>> {
        let res = self.expr().map_err(|err| {
            if err.kind == ParseErrorKind::NoInput {
                ParseError { kind: ParseErrorKind::Finished, ..err }
            } else {
                err
            }
        });
        self.lex
            .advance_while(|t| t.kind.is_whitespace() || t.kind == TokenKind::Semicolon);
        res
    }

    pub fn expr(&mut self) -> ParseResult<Ptr<Expr>> {
        self.expr_(MIN_PRECEDENCE)
    }

    pub fn expr_(&mut self, min_precedence: usize) -> ParseResult<Ptr<Expr>> {
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

    pub fn op_chain(&mut self, lhs: Ptr<Expr>, min_precedence: usize) -> ParseResult<Ptr<Expr>> {
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
                let close_b_span = loop {
                    self.ws0();
                    if let Some(t) = self.lex.next_if(|t| t.kind == TokenKind::CloseBrace) {
                        break t.span;
                    }
                    let ident = self.ident().context("initializer field ident")?;
                    self.ws0();
                    let init = self
                        .lex
                        .next_if(|t| t.kind == TokenKind::Eq)
                        .map(|_| self.expr().context("init expr"))
                        .transpose()?;
                    fields
                        .push((ident, init))
                        .map_err(|e| err!(x AllocErr(e), self.lex.pos_span()))?;

                    match self.next_tok() {
                        Ok(Token { kind: TokenKind::Comma, .. }) => {},
                        Ok(Token { kind: TokenKind::CloseBrace, span }) => break span,
                        t => t
                            .and_then(ParseError::unexpected_token)
                            .context("expected '=', ',' or '}'")?,
                    }
                };
                let fields = self.clone_slice_from_scratch_pool(fields)?;
                expr!(Initializer { lhs: Some(lhs), fields }, span.join(close_b_span))
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
                        let func = self.alloc(self.ident_from_span(t.span).into_expr())?;
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
    pub fn value(&mut self, min_precedence: usize) -> ParseResult<Ptr<Expr>> {
        let Token { kind, span } = self.next_tok().context("expected value")?;

        macro_rules! expr {
            ($kind:ident) => {
                expr_!($kind, span)
            };
            ($kind:ident ( $( $val:expr ),* $(,)? ) ) => {
                expr_!($kind($($val)*), span)
            };
            ($kind:ident { $( $field:ident $( : $val:expr )? ),* $(,)? } ) => {
                Expr::new(ExprKind::$kind{$($field $(:$val)?),*}, span)
            };
            ($($t:tt)*) => {
                expr_!($($t)*)
            }
        }

        let expr = match kind {
            TokenKind::Ident => expr!(Ident(self.get_text_from_span(span))),
            TokenKind::Keyword(k @ Keyword::Mut | k @ Keyword::Rec | k @ Keyword::Pub) => {
                self.ws0();
                let markers = match k {
                    Keyword::Mut => DeclMarkers { is_pub: false, is_mut: true, is_rec: false },
                    Keyword::Rec => DeclMarkers { is_pub: false, is_mut: false, is_rec: true },
                    Keyword::Pub => DeclMarkers { is_pub: true, is_mut: false, is_rec: false },
                    _ => unreachable!(),
                };
                let (decl, span_end) = self.var_decl_with_markers(markers)?;
                expr!(VarDecl(decl), span.join(span_end))
            },
            TokenKind::Keyword(Keyword::Struct) => {
                self.ws0();
                self.tok(TokenKind::OpenBrace).context("struct '{'")?;
                self.ws0();
                let fields = self.struct_fields()?;
                expr!(StructDef(fields), span)
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
            TokenKind::Literal(kind) => {
                expr!(Literal { kind, code: self.get_text_from_span(span) })
            },
            TokenKind::BoolLit(b) => expr!(BoolLit(b)),
            TokenKind::OpenParenthesis => {
                self.ws0();
                // TODO: currently no tuples allowed!
                // () -> ...
                if self.lex.advance_if(is_close_paren) {
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
                    if self.lex.advance_if(|t| t.kind == TokenKind::CloseParenthesis) {
                        break;
                    }
                    let (param, _end_span) = self.var_decl().context("function parameter")?;
                    // TODO: use end_span
                    params.push(param).map_err(|e| err!(x AllocErr(e), self.lex.pos_span()))?;
                    match self.ws0_and_next_tok() {
                        Ok(Token { kind: TokenKind::Comma, .. }) => self.ws0(),
                        Ok(Token { kind: TokenKind::CloseParenthesis, .. }) => break,
                        t => t
                            .and_then(ParseError::unexpected_token)
                            .context("expected ',' or ')'")?,
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
                let type_ = self.expr().expect("type after ?");
                expr!(OptionShort(ty(type_)), span.join(type_.span))
            },
            TokenKind::Pound => todo!("TokenKind::Pound"),
            TokenKind::Dollar => todo!("TokenKind::Dollar"),
            TokenKind::At => todo!("TokenKind::At"),
            TokenKind::Tilde => todo!("TokenKind::Tilde"),
            TokenKind::BackSlash => todo!("TokenKind::BackSlash"),
            TokenKind::BackTick => todo!("TokenKind::BackTick"),

            // TokenKind::CloseParenthesis | TokenKind::CloseBracket | TokenKind::CloseBrace => {
            //     return err!(NoInput, self.lex.pos_span());
            // },
            // TokenKind::Semicolon => return err!(NoInput, self.lex.pos_span()),
            // TokenKind::Semicolon => expr!(Semicolon(None), span),
            t => return err!(UnexpectedToken(t), span).context("expected valid value"),
        };
        self.alloc(expr)
    }

    /// parsing starts after the '->'
    pub fn function_tail(
        &mut self,
        params: Ptr<[VarDecl]>,
        start_span: Span,
    ) -> ParseResult<Ptr<Expr>> {
        let expr = self.expr().context("fn return type or body")?;
        let (ret_type, body) = match self.lex.next_if(|t| t.kind == TokenKind::OpenBrace) {
            Some(brace) => (ty(expr), self.block(brace.span).context("fn body")?),
            None => (Type::Unset, expr),
        };
        self.alloc(expr!(Fn(Fn { params, ret_type, body }), start_span))
    }

    pub fn if_after_cond(
        &mut self,
        condition: Ptr<Expr>,
        start_span: Span,
    ) -> ParseResult<Ptr<Expr>> {
        let then_body = self.expr_(IF_PRECEDENCE).context("then body")?;
        self.ws0();
        let else_body = self
            .lex
            .peek()
            .filter(|t| t.kind == TokenKind::Keyword(Keyword::Else))
            .inspect(|_| self.lex.advance())
            .map(|_| self.expr_(IF_PRECEDENCE).context("else body"))
            .transpose()?;
        self.alloc(expr!(If { condition, then_body, else_body }, start_span)) // TODO: correct span
    }

    /// `... ( ... )`
    /// `     ^` starts here
    pub fn call(
        &mut self,
        func: Ptr<Expr>,
        open_paren_span: Span,
        mut args: ScratchPool<Ptr<Expr>>,
    ) -> ParseResult<Ptr<Expr>> {
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
        self.alloc(expr!(Call { func, args }, closing_paren_span))
    }

    /// parses block context and '}', doesn't parse the '{'
    pub fn block(&mut self, open_brace_span: Span) -> ParseResult<Ptr<Expr>> {
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
        let span = open_brace_span.join(closing_brace_span);
        self.alloc(expr!(Block { stmts, has_trailing_semicolon }, span))
    }

    /// `struct { ... }`
    /// `         ^^^` Parses this
    pub fn struct_fields(&mut self) -> ParseResult<Ptr<[VarDecl]>> {
        let mut fields = ScratchPool::new();
        loop {
            if self.lex.advance_if(|t| t.kind == TokenKind::CloseBrace) {
                break;
            }
            let (field, _end_span) = self.var_decl().context("struct field")?;
            // TODO: use end_span
            fields.push(field).map_err(|e| err!(x AllocErr(e), self.lex.pos_span()))?;
            match self.ws0_and_next_tok() {
                Ok(Token { kind: TokenKind::Comma, .. }) => self.ws0(),
                Ok(Token { kind: TokenKind::CloseBrace, .. }) => break,
                t => t.and_then(ParseError::unexpected_token).context("expected ',' or '}'")?,
            }
        }
        self.clone_slice_from_scratch_pool(fields)
    }

    /*
    /// Parses [`ExprParser::var_decl`] multiple times, seperated by `sep`.
    ///
    /// `has_finished` receives two non-whitespace tokens behind each
    /// [`ExprParser::var_decl`]. Only the first token is consumed.
    pub fn var_decl_list(
        &mut self,
        mut has_finished: impl FnMut(ParseResult<Token>, Option<&Token>) -> ParseResult<bool>,
    ) -> ParseResult<Ptr<[VarDecl]>> {
        let mut list = ScratchPool::new();
        loop {
            let decl = self.var_decl().context("var_decl")?;
            list.push(decl).map_err(|e| err!(x AllocErr(e), self.lex.pos_span()))?;
            self.ws0();
            let sep = self.next_tok();
            self.ws0();
            if has_finished(sep, self.lex.peek().as_ref())? {
                break;
            }
        }
        self.clone_slice_from_scratch_pool(list)
    }
     */

    /// These options are allowed in a lot of places:
    /// * `[markers] ident`
    /// * `[markers] ident: ty`
    /// * `[markers] ident: ty = default`
    /// * `[markers] ident := default`
    ///
    /// That is the reason why this doesn't return a [`Ptr<Expr>`].
    pub fn var_decl(&mut self) -> ParseResult<(VarDecl, Span)> {
        self.var_decl_with_markers(DeclMarkers::default())
    }

    /// This will still parse more markers, if they exists.
    /// Also returns a [`Span`] which is described by `expr.span` in
    /// [`ExprKind::VarDecl`].
    pub fn var_decl_with_markers(
        &mut self,
        mut markers: DeclMarkers,
    ) -> ParseResult<(VarDecl, Span)> {
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
                TokenKind::Ident => break self.ident_from_span(t.span),
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
        let ty_expr = self.expr().context("decl type")?;
        self.ws0();
        let t = self.lex.peek();
        let init = t
            .filter(|t| matches!(t.kind, TokenKind::Eq | TokenKind::Colon))
            .map(|_| self.expr().context("variable initialization"))
            .transpose()?;
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
    fn alloc<T>(&self, val: T) -> ParseResult<Ptr<T>> {
        match self.alloc.try_alloc(val) {
            Result::Ok(ok) => Ok(Ptr::from(ok)),
            Result::Err(err) => err!(AllocErr(err), self.lex.pos_span()),
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
            .map_err(|err| err!(x AllocErr(err), self.lex.pos_span()))?
            .cast::<T>();

        Ok(Ptr::from(unsafe {
            core::ptr::copy_nonoverlapping(slice.as_ptr(), dst.as_ptr(), slice.len());
            core::slice::from_raw_parts_mut(dst.as_ptr(), slice.len())
        }))
    }

    #[inline]
    fn clone_slice_from_scratch_pool<T: Clone>(
        &self,
        scratch_pool: ScratchPool<T>,
    ) -> ParseResult<Ptr<[T]>> {
        scratch_pool
            .clone_to_slice_in_bump(&self.alloc)
            .map_err(|e| err!(x AllocErr(e), self.lex.pos_span()))
    }

    #[inline]
    fn alloc_empty_slice<T>(&self) -> Ptr<[T]> {
        Ptr::from(&[] as &[T])
    }

    fn get_text_from_span(&self, span: Span) -> Ptr<str> {
        self.lex.get_code()[span].into()
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
}

#[derive(Debug, Clone)]
pub struct StmtIter<'code, 'alloc> {
    parser: Parser<'code, 'alloc>,
}

impl<'code, 'alloc> Iterator for StmtIter<'code, 'alloc> {
    type Item = ParseResult<Ptr<Expr>>;

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
        let mut parser = Parser::new(Lexer::new(code), alloc);
        parser.ws0();
        Self { parser }
    }
}

impl Ident {
    pub fn try_from_tok(t: Token, lex: &Lexer<'_>) -> ParseResult<Ident> {
        let text = &lex.get_code()[t.span];
        if Keyword::from_str(text).is_ok() {
            return err!(NotAnIdent, lex.pos_span());
        }
        Ok(Ident { text: text.into(), span: t.span })
    }
}

#[inline]
fn is_close_paren(t: Token) -> bool {
    t.kind == TokenKind::CloseParenthesis
}

pub mod debug {
    use super::{DebugAst, Ident};
    use crate::{
        ast::{Type, VarDecl},
        ptr::Ptr,
    };

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

        pub fn write_tree<T: DebugAst>(&mut self, expr: &T) {
            self.scope_next_line(|l| expr.write_tree(l));
            self.write_minus(expr.to_text().len());
        }

        pub fn write_ident(&mut self, ident: &Ident) {
            self.scope_next_line(|l| l.write(&ident.text));
            self.write_minus(ident.text.len());
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

    pub fn var_decl_to_text(VarDecl { markers, ident, ty, default, is_const }: &VarDecl) -> String {
        format!(
            "{}{}{}{}{}{}{}",
            if markers.is_pub { "pub " } else { "" },
            mut_marker(markers.is_mut),
            if markers.is_rec { "rec " } else { "" },
            ident.text.as_ref(),
            if *ty != Type::Unset { ":" } else { "" },
            ty.to_text(),
            default
                .map(|default| format!(
                    "{}{}{}",
                    if *ty == Type::Unset { ":" } else { "" },
                    if *is_const { ":" } else { "=" },
                    default.to_text()
                ))
                .unwrap_or_default()
        )
    }

    pub fn var_decl_write_tree(
        VarDecl { markers, ident, ty, default, is_const }: &VarDecl,
        lines: &mut TreeLines,
    ) {
        lines.write(&format!(
            "{}{}{}",
            if markers.is_pub { "pub " } else { "" },
            mut_marker(markers.is_mut),
            if markers.is_rec { "rec " } else { "" }
        ));

        lines.write_ident(ident);

        if *ty != Type::Unset {
            lines.write(":");
            lines.write_tree(ty);
        }

        if let Some(default) = default {
            let default = default;
            lines.write(&format!(
                "{}{}",
                if *ty == Type::Unset { ":" } else { "" },
                if *is_const { ":" } else { "=" },
            ));
            lines.write_tree(default);
        }
    }

    pub fn many_to_text<T>(
        elements: &Ptr<[T]>,
        single_to_text: impl FnMut(&T) -> String,
        sep: &str,
    ) -> String {
        elements.iter().map(single_to_text).intersperse(sep.to_string()).collect()
    }

    pub fn many_expr_to_text<T: DebugAst>(elements: &Ptr<[T]>, sep: &str) -> String {
        many_to_text(elements, |t| t.to_text(), sep)
    }

    pub fn opt_to_text<T>(opt_expr: &Option<T>, inner: impl FnOnce(&T) -> String) -> String {
        opt_expr.as_ref().map(inner).unwrap_or_default()
    }

    pub fn opt_expr_to_text<T: DebugAst>(opt_expr: &Option<T>) -> String {
        opt_to_text(opt_expr, |t| t.to_text())
    }

    #[inline]
    pub fn mut_marker(is_mut: bool) -> &'static str {
        if is_mut { "mut " } else { "" }
    }
}

pub trait DebugAst {
    fn to_text(&self) -> String;

    fn write_tree(&self, lines: &mut TreeLines);

    fn print_tree(&self) {
        let mut lines = TreeLines::default();
        self.write_tree(&mut lines);
        println!("| {}", self.to_text());
        for l in lines.lines {
            println!("| {}", l.0);
        }
    }
}

impl<T: DebugAst> DebugAst for Ptr<T> {
    fn to_text(&self) -> String {
        self.as_ref().to_text()
    }

    fn write_tree(&self, lines: &mut TreeLines) {
        self.as_ref().write_tree(lines)
    }
}

impl DebugAst for Expr {
    fn to_text(&self) -> String {
        #[allow(unused)]
        match &self.kind {
            ExprKind::Ident(text) => text.to_string(),
            ExprKind::Literal { kind: _, code } => code.to_string(),
            ExprKind::BoolLit(b) => b.to_string(),
            ExprKind::ArraySemi { val, count } => {
                format!("[{};{}]", val.to_text(), count.to_text())
            },
            ExprKind::ArrayComma { elements } => {
                format!("[{}]", debug::many_expr_to_text(elements, ","))
            },
            ExprKind::Tuple { elements } => {
                format!("({})", debug::many_expr_to_text(elements, ","))
            },
            ExprKind::Fn(Fn { params, ret_type, body }) => format!(
                "({})->{}{}",
                debug::many_to_text(params, |decl| debug::var_decl_to_text(decl), ","),
                ret_type.to_text(),
                if matches!(body.kind, ExprKind::Block { .. }) {
                    body.to_text()
                } else {
                    format!("{{{}}}", body.to_text())
                }
            ),
            ExprKind::Parenthesis { expr } => format!("({})", expr.to_text()),
            ExprKind::Block { stmts, has_trailing_semicolon } => {
                format!(
                    "{{{}{}}}",
                    debug::many_to_text(stmts, |a| a.to_text(), ";"),
                    if *has_trailing_semicolon { ";" } else { "" }
                )
            },
            ExprKind::StructDef(fields) => {
                format!(
                    "struct {{ {} }}",
                    debug::many_to_text(fields, |f| debug::var_decl_to_text(f), ",")
                )
            },
            ExprKind::UnionDef(..) => panic!(),
            ExprKind::EnumDef {} => panic!(),
            ExprKind::OptionShort(ty) => {
                format!("?{}", ty.to_text())
            },
            ExprKind::Ptr { is_mut, ty } => {
                format!("*{}{}", debug::mut_marker(*is_mut), ty.to_text())
            },
            ExprKind::Initializer { lhs, fields } => {
                format!(
                    "{}.{{{}}}",
                    debug::opt_expr_to_text(lhs),
                    debug::many_to_text(
                        fields,
                        |(f, val)| format!(
                            "{}{}",
                            f.text.as_ref(),
                            debug::opt_to_text(val, |v| format!("={}", v.to_text())),
                        ),
                        ","
                    )
                )
            },
            ExprKind::Dot { lhs, rhs } => {
                format!("{}.{}", lhs.to_text(), rhs.text.as_ref())
            },
            //ExprKind::Colon { lhs, rhs } => { format!("{}:{}", lhs.to_text(),
            // rhs.to_text()) },
            ExprKind::PostOp { kind, expr } => panic!(),
            ExprKind::Index { lhs, idx } => panic!(),
            //ExprKind::CompCall { func, args } => panic!(),
            ExprKind::Call { func, args } => {
                format!("{}({})", func.to_text(), debug::many_to_text(args, |e| e.to_text(), ","))
            },
            ExprKind::PreOp { kind, expr, .. } => format!(
                "{}{}",
                match kind {
                    PreOpKind::AddrOf => "&",
                    PreOpKind::AddrMutOf => "&mut ",
                    PreOpKind::Deref => "*",
                    PreOpKind::Not => "!",
                    PreOpKind::Neg => "- ",
                },
                expr.to_text()
            ),
            ExprKind::BinOp { lhs, op, rhs, .. } => {
                format!("{} {} {}", lhs.to_text(), op.to_binop_text(), rhs.to_text())
            },
            ExprKind::Assign { lhs, rhs, .. } => {
                format!("{}={}", lhs.to_text(), rhs.to_text())
            },
            ExprKind::BinOpAssign { lhs, op, rhs, .. } => {
                format!("{}{}{}", lhs.to_text(), op.to_binop_assign_text(), rhs.to_text())
            },
            ExprKind::VarDecl(decl) => debug::var_decl_to_text(decl),
            ExprKind::If { condition, then_body, else_body } => {
                format!(
                    "if {} {}{}",
                    condition.to_text(),
                    then_body.to_text(),
                    debug::opt_to_text(else_body, |e| format!(" else {}", e.to_text()))
                )
            },
            ExprKind::Match { val, else_body } => todo!(),
            ExprKind::For { source, iter_var, body } => todo!(),
            ExprKind::While { condition, body } => todo!(),
            ExprKind::Catch { lhs } => todo!(),
            ExprKind::Pipe { lhs } => todo!(),
            ExprKind::Return { expr } => {
                format!("return{}", debug::opt_to_text(expr, |e| format!(" {}", e.to_text())))
            },
            ExprKind::Semicolon(expr) => format!("{};", debug::opt_expr_to_text(expr)),
        }
    }

    fn write_tree(&self, lines: &mut TreeLines) {
        match &self.kind {
            ExprKind::Ident(_) | ExprKind::Literal { .. } | ExprKind::BoolLit(_) => {
                lines.write(&self.to_text())
            },
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
                debug::many_expr_to_text(elements, ",");
                lines.write(")");
            },
            ExprKind::Fn(Fn { params, ret_type, body }) => {
                let body = body;
                lines.write("(");

                for (idx, decl) in params.into_iter().enumerate() {
                    if idx != 0 {
                        lines.write(",");
                    }

                    debug::var_decl_write_tree(decl, lines)
                }
                lines.write(")->");
                if *ret_type != Type::Unset {
                    lines.write_tree(ret_type);
                }
                if matches!(body.kind, ExprKind::Block { .. }) {
                    body.write_tree(lines);
                } else {
                    lines.write("{");
                    lines.write_tree(body);
                    lines.write("}");
                }
            },
            ExprKind::Parenthesis { expr } => {
                lines.write("(");
                lines.write_tree(expr);
                lines.write(")");
            },
            ExprKind::Block { stmts, has_trailing_semicolon } => {
                lines.write("{");
                let len = stmts.len();
                for (idx, s) in stmts.iter().enumerate() {
                    lines.write_tree(s);
                    if idx + 1 < len || *has_trailing_semicolon {
                        lines.write(";");
                    }
                }
                lines.write("}");
            },
            ExprKind::StructDef(fields) => {
                lines.write("struct { ");
                for (idx, field) in fields.into_iter().enumerate() {
                    if idx != 0 {
                        lines.write(",");
                    }

                    debug::var_decl_write_tree(field, lines)
                }
                lines.write(" }");
            },
            /*
            ExprKind::StructInit { fields, span } => panic!(),
            ExprKind::TupleStructDef { fields, span } => panic!(),
            ExprKind::Union { span } => panic!(),
            ExprKind::Enum { span } => panic!(),
            */
            ExprKind::OptionShort(ty) => {
                lines.write("?");
                lines.write_tree(ty);
            },
            ExprKind::Ptr { is_mut, ty } => {
                lines.write("*");
                if *is_mut {
                    lines.write("mut ");
                }
                lines.write_tree(ty);
            },
            ExprKind::Initializer { lhs, fields } => {
                if let Some(lhs) = lhs {
                    lines.write_tree(lhs);
                }
                lines.write(".{");
                for (idx, (field, val)) in fields.into_iter().enumerate() {
                    if idx != 0 {
                        lines.write(",");
                    }
                    lines.write_ident(field);
                    if let Some(val) = val {
                        lines.write("=");
                        lines.write_tree(val);
                    }
                }

                lines.write("}");
            },
            ExprKind::Dot { lhs, rhs } => {
                let lhs = lhs;
                lines.write_tree(lhs);
                lines.write(".");
                lines.write_ident(rhs);
            },
            /*
            ExprKind::PostOp { kind, expr, span } => panic!(),
            ExprKind::Index { lhs, idx, span } => panic!(),
            ExprKind::CompCall { func, args } => panic!(),
            */
            ExprKind::Call { func, args } => {
                let func = func;
                lines.write_tree(func);
                lines.write("(");
                let len = args.len();
                for (idx, arg) in args.into_iter().enumerate() {
                    let arg = arg;
                    lines.write_tree(arg);
                    if idx + 1 != len {
                        lines.write(",");
                    }
                }
                lines.write(")");
            },
            ExprKind::PreOp { kind, expr, .. } => {
                let expr = expr;
                lines.write(match kind {
                    PreOpKind::AddrOf => "&",
                    PreOpKind::AddrMutOf => "&mut ",
                    PreOpKind::Deref => "*",
                    PreOpKind::Not => "!",
                    PreOpKind::Neg => "- ",
                });
                lines.write_tree(expr);
            },
            ExprKind::BinOp { lhs, op, rhs, .. } => {
                let lhs = lhs;
                let rhs = rhs;
                lines.write_tree(lhs);
                lines.write(" ");
                lines.write(op.to_binop_text());
                lines.write(" ");
                lines.write_tree(rhs);
            },
            ExprKind::Assign { lhs, rhs, .. } => {
                let lhs = lhs;
                let rhs = rhs;
                lines.write_tree(lhs);
                lines.write("=");
                lines.write_tree(rhs);
            },
            ExprKind::BinOpAssign { lhs, op, rhs, .. } => {
                let lhs = lhs;
                let rhs = rhs;
                lines.write_tree(lhs);
                lines.write(op.to_binop_assign_text());
                lines.write_tree(rhs);
            },
            ExprKind::If { condition, then_body, else_body } => {
                let condition = condition;
                let then_body = then_body;
                lines.write("if ");
                lines.write_tree(condition);
                lines.write(" ");
                lines.write_tree(then_body);
                if let Some(else_body) = else_body {
                    let else_body = else_body;
                    lines.write(" else ");
                    lines.write_tree(else_body);
                }
            },
            ExprKind::VarDecl(decl) => debug::var_decl_write_tree(decl, lines),
            ExprKind::Return { expr } => {
                lines.write("return");
                if let Some(expr) = expr {
                    let expr = expr;
                    lines.write(" ");
                    lines.write_tree(expr);
                }
            },
            ExprKind::Semicolon(expr) => {
                if let Some(expr) = expr {
                    lines.write_tree(expr);
                }
                lines.write(";");
            },

            k => panic!("{:?}", k),
        }
    }
}

impl DebugAst for Type {
    fn to_text(&self) -> String {
        match self {
            Type::Void => "void".to_string(),
            Type::Never => "!".to_string(),
            Type::Float { bits } => format!("f{}", bits),
            Type::Function(_) => "fn".to_string(), // TODO: fn type as text
            Type::Unset => String::default(),
            Type::Unevaluated(expr) => expr.to_text(),
        }
    }

    fn write_tree(&self, lines: &mut TreeLines) {
        match self {
            Type::Void | Type::Never | Type::Float { .. } => lines.write(&self.to_text()),
            Type::Function(_) => lines.write(&self.to_text()), // TODO: fn type as text
            Type::Unset => {},
            Type::Unevaluated(expr) => expr.write_tree(lines),
        }
    }
}
