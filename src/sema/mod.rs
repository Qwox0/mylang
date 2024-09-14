#![allow(unused_variables)]

use crate::{
    ast::{DeclMarkers, Expr, ExprKind, Fn, Ident, LitKind, PreOpKind, Type, VarDecl},
    parser::lexer::{Code, Span},
    ptr::Ptr,
    symbol_table::SymbolTable,
    util::display_span_in_code_with_label,
};
use std::ops::{FromResidual, Try};
use SemaResult::*;

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

    TopLevelDuplicate,
    UnexpectedTopLevelExpr(Ptr<Expr>),

    // TODO: remove this
    Unimplemented,
}

#[derive(Debug)]
pub struct SemaError {
    pub kind: SemaErrorKind,
    pub span: Span,
}

//type SemaResult<T = Type> = Result<T, SemaError>;
#[derive(Debug)]
pub enum SemaResult<T = Type> {
    Ok(T),
    NotFinished,
    Err(SemaError),
}

impl<T> SemaResult<T> {
    pub fn map_ok<U>(self, f: impl FnOnce(T) -> U) -> SemaResult<U> {
        match self {
            Ok(t) => Ok(f(t)),
            NotFinished => NotFinished,
            Err(err) => Err(err),
        }
    }
}

impl<T> Try for SemaResult<T> {
    type Output = T;
    type Residual = SemaResult<!>;

    fn from_output(output: Self::Output) -> Self {
        Ok(output)
    }

    fn branch(self) -> std::ops::ControlFlow<Self::Residual, Self::Output> {
        match self {
            Ok(ty) => std::ops::ControlFlow::Continue(ty),
            NotFinished => std::ops::ControlFlow::Break(SemaResult::NotFinished),
            Err(err) => std::ops::ControlFlow::Break(SemaResult::Err(err)),
        }
    }
}

impl<T> FromResidual<SemaResult<!>> for SemaResult<T> {
    fn from_residual(residual: SemaResult<!>) -> Self {
        match residual {
            Ok(never) => never,
            NotFinished => SemaResult::NotFinished,
            Err(err) => SemaResult::Err(err),
        }
    }
}

impl FromResidual<Result<!, SemaError>> for SemaResult {
    fn from_residual(residual: Result<!, SemaError>) -> Self {
        match residual {
            Result::Ok(never) => never,
            Result::Err(err) => Err(err),
        }
    }
}

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

    pub symbols: SymbolTable<Option<SemaSymbol>>,
    debug_types: bool,

    pub errors: Vec<SemaError>,
}

impl<'c> Sema<'c> {
    pub fn new(code: &'c Code, debug_types: bool) -> Sema<'c> {
        Sema { code, symbols: SymbolTable::with_one_scope(), debug_types, errors: Vec::new() }
    }

    pub fn preload_top_level(&mut self, mut s: Ptr<Expr>) {
        let res = try {
            let ExprKind::VarDecl(decl) = &mut s.kind else {
                err!(UnexpectedTopLevelExpr(s), s.span)?
            };
            debug_assert!(decl.is_const);
            debug_assert!(decl.default.is_some());

            if self.symbols.insert(&*decl.ident.text, None).is_some() {
                err!(TopLevelDuplicate, decl.ident.span)?
            }
        };
        if let Err(e) = res {
            self.errors.push(e);
        }
    }

    /// Returns if s was fully checked
    pub fn analyze_top_level(&mut self, s: Ptr<Expr>) -> bool {
        match self.analyze(s) {
            Ok(_) => true,
            NotFinished => false,
            Err(e) => {
                self.errors.push(e);
                true
            },
        }
    }

    /// this modifies the [`Expr`] behind `expr` to ensure that codegen is
    /// possible.
    pub fn analyze(&mut self, mut expr: Ptr<Expr>) -> SemaResult {
        let span = expr.span;

        #[allow(unused_variables)]
        let ty = match &mut expr.kind {
            ExprKind::Ident(text) => self.get_symbol(*text, expr.span).map_ok(SemaSymbol::get_type),
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
            ExprKind::Fn(func) => self.analyze_fn(func),
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
                let Result::Ok(func) = func.try_to_ident() else {
                    todo!("non ident function call")
                };
                let func = self.get_ident_symbol(func)?;
                match func {
                    SemaSymbol::Variable { markers, ty: Type::Function(mut func) } => {
                        let Fn { params, ret_type, body } = func.as_mut();
                        self.validate_call(*params, *args)?;
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
            ExprKind::BinOp { lhs, op, rhs } => {
                let lhs_ty = self.analyze(*lhs)?;
                debug_assert!(lhs_ty.is_valid());
                let rhs_ty = self.analyze(*rhs)?;
                debug_assert!(rhs_ty.is_valid());
                if lhs_ty == rhs_ty {
                    Ok(lhs_ty) // todo: check if binop can be applied to type
                } else {
                    err!(MismatchedTypesBinOp { lhs_ty, rhs_ty }, expr.span)
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
            ExprKind::BinOpAssign { lhs, op, rhs } => {
                let lhs_ty = self.analyze(*lhs)?;
                debug_assert!(lhs_ty.is_valid());
                let rhs_ty = self.analyze(*rhs)?;
                debug_assert!(rhs_ty.is_valid());
                if lhs_ty == rhs_ty {
                    Ok(Type::Void) // todo: check if binop can be applied to type
                } else {
                    err!(MismatchedTypesBinOp { lhs_ty, rhs_ty }, expr.span)
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
        if self.debug_types {
            display_span_in_code_with_label(expr.full_span(), self.code, format!("type: {ty:?}")); // TODO: remove this
        }
        ty
    }

    fn analyze_fn(&mut self, func: &mut Fn) -> SemaResult {
        let Fn { mut params, ret_type, body } = func;
        self.symbols.open_scope();
        let res = try {
            for decl in params.iter_mut() {
                self.analyze_var_decl(decl)?;
            }
            let body_ty = self.analyze(*body)?;
            Self::check_infered_type(ret_type, body_ty, body.span)?;
            Type::Function(func.into())
        };
        self.symbols.close_scope();
        res
    }

    fn analyze_var_decl(&mut self, decl: &mut VarDecl) -> SemaResult {
        let var = self.analyze_var_decl2(decl)?;
        if self
            .symbols
            .insert(&*decl.ident.text, Some(var))
            .is_some_and(|sym| sym.is_some())
        {
            println!("INFO: '{}' was shadowed in the same scope", &*decl.ident.text);
        }
        Ok(Type::Void)
    }

    fn analyze_var_decl2(&mut self, decl: &mut VarDecl) -> SemaResult<SemaSymbol> {
        let ty = &mut decl.ty;
        self.eval_type(ty)?;
        if let Some(init) = &mut decl.default {
            let init_ty = self.analyze(*init)?;
            Self::check_infered_type(ty, init_ty, init.span)?;
        } else if !ty.is_valid() {
            return err!(VarDeclNoType, decl.ident.span);
        };
        Ok(SemaSymbol::Variable { markers: decl.markers, ty: *ty })
    }

    /// works for function calls and call initializers
    /// ...(arg1, ..., argX, paramX1=argX1, ... paramN=argN)
    /// paramN1, ... -> default values
    fn validate_call(&mut self, params: Ptr<[VarDecl]>, args: Ptr<[Ptr<Expr>]>) -> SemaResult<()> {
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

    #[inline]
    #[must_use]
    fn check_infered_type(ty: &mut Type, infered_ty: Type, err_span: Span) -> SemaResult<()> {
        debug_assert!(infered_ty.is_valid());
        if *ty == Type::Unset {
            *ty = infered_ty;
        } else if *ty != infered_ty {
            return err!(MismatchedTypes { expected: *ty, got: infered_ty }, err_span);
        }
        Ok(())
    }

    fn eval_type_expr(&mut self, ty_expr: Ptr<Expr>) -> SemaResult {
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
    fn eval_type(&mut self, ty: &mut Type) -> SemaResult {
        if let Type::Unevaluated(ty_expr) = *ty {
            *ty = self.eval_type_expr(ty_expr)?;
            debug_assert!(ty.is_valid());
        }
        Ok(*ty)
    }

    #[inline]
    fn get_symbol(&self, name: Ptr<str>, err_span: Span) -> SemaResult<&SemaSymbol> {
        match self.symbols.get(&name) {
            Some(None) => NotFinished,
            Some(Some(sym)) => Ok(&sym),
            None => err!(UnknownIdent(name), err_span),
        }
    }

    #[inline]
    fn get_ident_symbol(&self, i: Ident) -> SemaResult<&SemaSymbol> {
        self.get_symbol(i.text, i.span)
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
