#![allow(unused_variables)]

use crate::{
    ast::{DeclMarkers, Expr, ExprKind, Ident, LitKind, PreOpKind, Type, VarDecl},
    parser::lexer::{Code, Span},
    ptr::Ptr,
    symbol_table::SymbolTable,
    util::display_span_in_code_with_label,
};

#[derive(Debug)]
pub enum SemaErrorKind {
    ConstDeclWithoutInit,
    /// TODO: maybe infer the type from the usage
    VarDeclNoType,

    MismatchedTypes {
        expected: Type,
        got: Type,
    },
    MismatchedTypesBinOp {
        lhs_ty: Type,
        rhs_ty: Type,
    },
    /// rust error:
    /// ```
    /// error[E0600]: cannot apply unary operator `!` to type `&'static str`
    ///   --> src/sema/mod.rs:70:17
    ///    |
    /// 70 |         let a = !"";
    ///    |                 ^^^ cannot apply unary operator `!`
    /// ```
    InvalidPreOp {
        ty: Type,
        kind: PreOpKind,
    },

    MissingArg,

    UnknownIdent(Ptr<str>),
    CannotInfer,

    // TODO: remove this
    Unimplemented,
}

#[derive(Debug)]
pub struct SemaError {
    pub kind: SemaErrorKind,
    pub span: Span,
}

impl SemaError {
    pub fn unknown_ident(i: Ident) -> SemaError {
        SemaError { kind: SemaErrorKind::UnknownIdent(i.text), span: i.span }
    }
}

type TypeResult<T = Type> = Result<T, SemaError>;

macro_rules! err_impl {
    ($kind:ident, $span:expr) => {
        SemaError { kind: SemaErrorKind::$kind, span: $span }
    };
    ($kind:ident ( $( $field:expr ),* $(,)? ), $span:expr) => {
        SemaError { kind: SemaErrorKind::$kind ( $($field),* ), span: $span }
    };
    ($kind:ident { $( $field:ident $( : $val:expr )? ),* $(,)? } , $span:expr) => {
        SemaError { kind: SemaErrorKind::$kind { $($field $(: $val)?),* }, span: $span }
    };
}
pub(crate) use err_impl;

macro_rules! err {
    (x $($t:tt)*) => {
        err_impl!($($t)*)
    };
    ($($t:tt)*) => {
        Err(err_impl!($($t)*))
    };
}

/// Semantic analyzer
pub struct Sema<'c> {
    code: &'c Code,

    pub symbols: SymbolTable<SemaSymbol>,
}

impl<'c> Sema<'c> {
    pub fn new(code: &'c Code) -> Sema<'c> {
        Sema { code, symbols: SymbolTable::with_one_scope() }
    }

    pub fn preload_top_level(&mut self, mut s: Ptr<Expr>) -> TypeResult<()> {
        match &mut s.kind {
            ExprKind::VarDecl(VarDecl {
                markers,
                ident,
                ty,
                default: Some(init),
                is_const: true,
            }) => match &mut init.kind {
                ExprKind::Fn { params, ret_type, body } => {
                    for decl in params.iter_mut() {
                        {
                            let this = &mut *self;
                            let VarDecl { markers, ident, ty, default, is_const } = decl;
                            this.eval_type(ty)?;
                            if let Some(init) = default {
                                let init_ty = this.analyze(*init)?;
                                debug_assert!(init_ty.is_valid());
                                if *ty == Type::Unset {
                                    *ty = init_ty;
                                } else if *ty != init_ty {
                                    let span = init.span;
                                    return err!(
                                        MismatchedTypes { expected: *ty, got: init_ty },
                                        span
                                    );
                                }
                            //} else if *is_const {
                            //    return err!(ConstDeclWithoutInit, expr.span);
                            } else if !ty.is_valid() {
                                return err!(VarDeclNoType, ident.span);
                            };
                            Ok(Type::Void)
                        }?;
                        // todo: type check params
                    }

                    //self.symbols.insert(&*ident.text, SemaSymbol::Function { expr: *init });
                    self.symbols.insert(&*ident.text, SemaSymbol::Variable {
                        markers: *markers,
                        ty: Type::Function(*init),
                    });
                    Ok(())
                },
                _ => todo!(),
            },
            ExprKind::Semicolon(Some(expr)) => self.preload_top_level(*expr),
            ExprKind::Semicolon(None) => Ok(()),
            _ => todo!(),
        }
    }

    pub fn analyze_top_level(&mut self, expr: Ptr<Expr>) -> TypeResult<()> {
        self.analyze(expr)?;
        Ok(())
    }

    /// this modifies the [`Expr`] behind `expr` to ensure that codegen is
    /// possible.
    pub fn analyze(&mut self, mut expr: Ptr<Expr>) -> TypeResult {
        let span = expr.span;

        /*
        macro_rules! todo {
            () => {
                err!(Unimplemented, span)
            };
        }
        */

        #[allow(unused_variables)]
        let ty = match &mut expr.kind {
            /*
            ExprKind::Ident(text) => match self.symbols.get(text) {
                Some(sym) => sym.get_type(),
                None => err!(UnknownIdent(*text), expr.span),
            },
            */
            ExprKind::Ident(text) => self
                .symbols
                .get(text)
                .map(SemaSymbol::get_type)
                .ok_or(err!(x UnknownIdent(*text), expr.span)),
            &mut ExprKind::Literal { kind, code } => match kind {
                // TODO: better literal handling
                LitKind::Char => todo!(),
                LitKind::BChar => todo!(),
                LitKind::Int => Ok(Type::Float { bits: 64 }),
                LitKind::Float => Ok(Type::Float { bits: 64 }),
                LitKind::Str => todo!(),
            },
            ExprKind::BoolLit(_) => todo!(),
            ExprKind::ArraySemi { val, count } => todo!(),
            ExprKind::ArrayComma { elements } => todo!(),
            ExprKind::Tuple { elements } => todo!(),
            ExprKind::Fn { mut params, ret_type, body } => {
                self.symbols.open_scope();
                for decl in params.iter_mut() {
                    self.analyze_var_decl(decl)?;

                    // todo: type check params
                    // let var = SemaSymbol::Variable { markers: decl.markers,
                    // ty: decl.ty }; self.symbols.insert(&*
                    // decl.ident.text, var);
                }
                let body_ty = self.analyze(*body)?;
                self.symbols.close_scope();
                Ok(Type::Function(expr.into()))
            },
            ExprKind::Parenthesis { expr } => todo!(),
            ExprKind::Block { stmts, has_trailing_semicolon } => {
                let mut ty = Type::Void;
                for s in stmts.iter() {
                    ty = self.analyze(*s)?;
                }
                if *has_trailing_semicolon { Ok(Type::Void) } else { Ok(ty) }
            },
            ExprKind::StructDef(_) => todo!(),
            ExprKind::UnionDef(_) => todo!(),
            ExprKind::EnumDef {} => todo!(),
            ExprKind::OptionShort(_) => todo!(),
            ExprKind::Ptr { is_mut, ty } => todo!(),
            ExprKind::Initializer { lhs, fields } => todo!(),
            ExprKind::Dot { lhs, rhs } => todo!(),
            ExprKind::PostOp { expr, kind } => todo!(),
            ExprKind::Index { lhs, idx } => todo!(),
            ExprKind::Call { func, args } => {
                let Ok(func) = func.try_to_ident() else { todo!("non ident function call") };
                let func = *self.symbols.get(&func.text).ok_or(SemaError::unknown_ident(func))?;
                match func {
                    SemaSymbol::Variable { markers, ty: Type::Function(mut func) } => {
                        let ExprKind::Fn { params, ret_type, body } = &mut func.kind else {
                            unreachable!()
                        };
                        self.validate_call(*params, *args)?;
                        if *ret_type == Type::Unset {
                            *ret_type = Type::Float { bits: 64 }; // FIXME: infer correct type
                        }
                        debug_assert!(ret_type.is_valid());
                        Ok(*ret_type)
                    },
                    _ => return Err(SemaError { kind: SemaErrorKind::Unimplemented, span }),
                }
            },
            &mut ExprKind::PreOp { kind, expr, .. } => {
                let ty = self.analyze(expr)?;
                let is_valid = match (kind, ty) {
                    (_, Type::Unset | Type::Unevaluated(_)) => return err!(CannotInfer, span),
                    (_, Type::Void) => false,
                    (_, Type::Never) => true,
                    (PreOpKind::AddrOf, Type::Float { .. }) => true,
                    (PreOpKind::AddrOf, Type::Function(_)) => todo!(),
                    (PreOpKind::AddrMutOf, Type::Float { .. }) => true,
                    (PreOpKind::AddrMutOf, Type::Function(_)) => todo!(),
                    (PreOpKind::Deref, Type::Float { .. }) => false,
                    (PreOpKind::Deref, Type::Function(_)) => false,
                    (PreOpKind::Not, Type::Float { .. }) => false,
                    (PreOpKind::Not, Type::Function(_)) => false,
                    (PreOpKind::Neg, Type::Float { .. }) => true,
                    (PreOpKind::Neg, Type::Function(_)) => false,
                };
                if is_valid { Ok(ty) } else { err!(InvalidPreOp { ty, kind }, span) }
            },
            ExprKind::BinOp { lhs, op, rhs, op_span } => {
                let lhs_ty = self.analyze(*lhs)?;
                debug_assert!(lhs_ty.is_valid());
                let rhs_ty = self.analyze(*rhs)?;
                debug_assert!(rhs_ty.is_valid());
                if lhs_ty == rhs_ty {
                    Ok(lhs_ty) // todo: check if binop can be applied to type
                } else {
                    err!(MismatchedTypesBinOp { lhs_ty, rhs_ty }, *op_span)
                }
            },
            ExprKind::Assign { lhs, rhs, .. } => {
                let lhs_ty = self.analyze(*lhs)?;
                debug_assert!(lhs_ty.is_valid());
                let rhs_ty = self.analyze(*rhs)?;
                debug_assert!(rhs_ty.is_valid());
                if lhs_ty == rhs_ty {
                    Ok(Type::Void) // todo: check if binop can be applied to type
                } else {
                    err!(MismatchedTypes { expected: lhs_ty, got: rhs_ty }, rhs.span)
                }
            },
            ExprKind::BinOpAssign { lhs, op, rhs, op_span } => {
                let lhs_ty = self.analyze(*lhs)?;
                debug_assert!(lhs_ty.is_valid());
                let rhs_ty = self.analyze(*rhs)?;
                debug_assert!(rhs_ty.is_valid());
                if lhs_ty == rhs_ty {
                    Ok(Type::Void) // todo: check if binop can be applied to type
                } else {
                    err!(MismatchedTypesBinOp { lhs_ty, rhs_ty }, *op_span)
                }
            },
            ExprKind::VarDecl(decl) => self.analyze_var_decl(decl),
            ExprKind::If { condition, then_body, else_body } => todo!(),
            ExprKind::Match { val, else_body } => todo!(),
            ExprKind::For { source, iter_var, body } => todo!(),
            ExprKind::While { condition, body } => todo!(),
            ExprKind::Catch { lhs } => todo!(),
            ExprKind::Pipe { lhs } => todo!(),
            ExprKind::Return { expr } => Ok(Type::Never),
            ExprKind::Semicolon(_) => todo!(),
        };
        display_span_in_code_with_label(expr.span, self.code, format!("type: {ty:?}")); // TODO: remove this
        ty
    }

    pub fn analyze_var_decl(&mut self, decl: &mut VarDecl) -> TypeResult {
        let VarDecl { markers, ident, ty, default, is_const } = decl;
        self.eval_type(ty)?;
        if let Some(init) = default {
            let init_ty = self.analyze(*init)?;
            debug_assert!(init_ty.is_valid());
            if *ty == Type::Unset {
                *ty = init_ty;
            } else if *ty != init_ty {
                let span = init.span;
                return err!(MismatchedTypes { expected: *ty, got: init_ty }, span);
            }
        //} else if *is_const {
        //    return err!(ConstDeclWithoutInit, expr.span);
        } else if !ty.is_valid() {
            return err!(VarDeclNoType, ident.span);
        };

        let var = SemaSymbol::Variable { markers: decl.markers, ty: decl.ty };
        self.symbols.insert(&*decl.ident.text, var);
        Ok(Type::Void)
    }

    /// works for function calls and call initializers
    /// ...(arg1, ..., argX, paramX1=argX1, ... paramN=argN)
    /// paramN1, ... -> default values
    pub fn validate_call(
        &mut self,
        params: Ptr<[VarDecl]>,
        args: Ptr<[Ptr<Expr>]>,
    ) -> TypeResult<()> {
        for (idx, p) in params.iter().enumerate() {
            let Some(arg) = args.get(idx) else {
                if p.default.is_some() {
                    continue;
                } else {
                    let span = todo!();
                    return err!(MissingArg, span);
                }
            };
            let arg_ty = self.analyze(*arg)?;
            debug_assert!(p.ty.is_valid()); // TODO: infer?
            if p.ty != arg_ty {
                return err!(MismatchedTypes { expected: p.ty, got: arg_ty }, arg.span);
            }
        }
        Ok(())
    }

    pub fn eval_type_expr(&mut self, ty_expr: Ptr<Expr>) -> TypeResult {
        let ty_expr = ty_expr;
        match ty_expr.kind {
            ExprKind::Ident(code) => match &*code {
                "f64" => Ok(Type::Float { bits: 64 }),
                _ => todo!(),
            },
            _ => todo!(),
        }
    }

    /// [`Type::Unevaluated`] -> [`Type`]
    pub fn eval_type(&mut self, ty: &mut Type) -> TypeResult {
        if let Type::Unevaluated(ty_expr) = *ty {
            *ty = self.eval_type_expr(ty_expr)?;
            debug_assert!(ty.is_valid());
        }
        Ok(*ty)
    }
}

#[derive(Debug, Clone, Copy)]
pub enum SemaSymbol {
    Variable { markers: DeclMarkers, ty: Type },
    // Function { expr: Ptr<Expr> },
}

impl SemaSymbol {
    pub fn get_type(&self) -> Type {
        match self {
            SemaSymbol::Variable { ty, .. } => *ty,
            // SemaSymbol::Function { expr } => Type::Function(*expr),
        }
    }
}
