//! # Semantic analysis module
//!
//! Semantic analysis validates (and changes) all stored [`Type`]s in [`Expr`].

#![allow(unused_variables)]

use crate::{
    ast::{
        BinOpKind, DeclMarkers, Expr, ExprKind, ExprWithTy, Fn, Ident, LitKind, UnaryOpKind,
        VarDecl, VarDeclList,
    },
    defer_stack::DeferStack,
    parser::lexer::{Code, Span},
    ptr::Ptr,
    symbol_table::SymbolTable,
    type_::Type,
    util::{OkOrWithTry, display_span_in_code_with_label, forget_lifetime},
};
pub use err::{SemaError, SemaErrorKind, SemaResult};
use err::{SemaErrorKind::*, SemaResult::*};
use symbol::SemaSymbol;
use value::{EMPTY_PTR, SemaValue};

mod err;
mod symbol;
mod value;

/// This is a macro because `err_span` should only be evaluated when an error
/// happens.
/// ```ignore
/// fn check_or_set_type(target: &mut Type, new_ty: Type, err_span: Span) -> SemaResult<()>
/// ```
macro_rules! check_or_set_type {
    ($target:expr, $new_ty:expr, $err_span:expr $(,)?) => {{
        let target: &mut Type = $target;
        let new_ty: Type = $new_ty;
        debug_assert!(new_ty.is_valid());
        if *target == Type::Unset {
            *target = new_ty.finalize();
            Ok(())
        } else if target.matches(new_ty) {
            Ok(())
        } else {
            err(MismatchedTypes { expected: *target, got: new_ty }, $err_span)
        }
    }};
}

macro_rules! try_not_never {
    ($val:expr) => {{
        let v: SemaValue = $val;
        match v.ty {
            Type::Never => return Ok(v),
            _ => v,
        }
    }};
}

/// Semantic analyzer
pub struct Sema<'c, 'alloc, const DEBUG_TYPES: bool = false> {
    code: &'c Code,

    pub symbols: SymbolTable<SemaSymbol>,
    function_stack: Vec<Ptr<Fn>>,
    defer_stack: DeferStack,

    pub errors: Vec<SemaError>,

    alloc: &'alloc bumpalo::Bump,
}

impl<'c, 'alloc, const DEBUG_TYPES: bool> Sema<'c, 'alloc, DEBUG_TYPES> {
    pub fn new(code: &'c Code, alloc: &'alloc bumpalo::Bump) -> Sema<'c, 'alloc, DEBUG_TYPES> {
        Sema {
            code,
            symbols: SymbolTable::with_one_scope(),
            function_stack: vec![],
            defer_stack: DeferStack::default(),
            errors: vec![],
            alloc,
        }
    }

    pub fn preload_top_level(&mut self, mut s: Ptr<Expr>) {
        let res = try {
            let ExprKind::VarDecl(decl) = &mut s.kind else {
                err(UnexpectedTopLevelExpr(s), s.span)?
            };
            debug_assert!(decl.is_const);
            debug_assert!(decl.default.is_some());

            if self.symbols.insert(&*decl.ident.text, SemaSymbol::preload_symbol()).is_some() {
                err(TopLevelDuplicate, decl.ident.span)?
            }
        };
        if let Err(e) = res {
            self.errors.push(e);
        }
    }

    pub fn analyze_top_level(&mut self, s: Ptr<Expr>) -> SemaResult<(), ()> {
        match self.analyze(s, true) {
            Ok(_) => Ok(()),
            NotFinished => NotFinished,
            Err(e) => {
                self.errors.push(e);
                Err(())
            },
        }
    }

    /// This modifies the [`Expr`] behind `expr` to ensure that codegen is
    /// possible.
    pub fn analyze(&mut self, mut expr: Ptr<Expr>, is_const: bool) -> SemaResult<SemaValue> {
        //let span = expr.full_span();
        let span = expr.span;

        #[allow(unused_variables)]
        let res = match &mut expr.kind {
            ExprKind::Ident(text) => {
                if let Some(internal_ty) = self.try_internal_ty(*text) {
                    let t = self.alloc(internal_ty)?;
                    return Ok(SemaValue::new_const(Type::Type(t), EMPTY_PTR));
                }

                match self.get_symbol(*text, expr.span)? {
                    SemaSymbol::Finished(v) => v.into_const_checked(is_const, span),
                    SemaSymbol::NotFinished(Some(ty)) if !is_const => Ok(SemaValue::new(*ty)),
                    SemaSymbol::NotFinished(_) => NotFinished,
                }
            },
            &mut ExprKind::Literal { kind, code } => {
                let ty = match kind {
                    LitKind::Char => todo!(),
                    LitKind::BChar => todo!(),
                    LitKind::Int => Type::IntLiteral,
                    LitKind::Float => Type::FloatLiteral,
                    LitKind::Str => todo!(),
                };
                if is_const {
                    let val =
                        kind.parse(code, &self.alloc).map_err(|e| err_val(AllocErr(e), span))?;
                    Ok(SemaValue::new_const(ty, val))
                } else {
                    Ok(SemaValue::new(ty))
                }
            },
            &mut ExprKind::BoolLit(val) => Ok(SemaValue::const_bool(self.alloc(val)?)),
            ExprKind::ArrayTy { count, ty } => {
                let count_val = self.analyze(*count, true)?;
                let count = count_val.int().map_err(|kind| SemaError { kind, span: count.span })?;
                let count = count.try_into().expect("todo: array to long");
                let ty = self.analyze_type_expr(*ty)?;
                let arr_ty = Type::Array { len: count, elem_ty: ty };
                self.alloc(arr_ty).map_ok(SemaValue::type_)
            },
            ExprKind::ArrayTy2 { ty } => todo!("[]ty"),
            ExprKind::ArrayLit { elements } => {
                let mut val_ty: Option<Type> = None;
                for elem in elements.iter() {
                    let val = self.analyze(*elem, is_const)?;
                    let ty = if let Some(prev) = val_ty {
                        prev.common_type(val.ty).ok_or_else(|| {
                            err_val(
                                MismatchedTypes { expected: prev, got: val.ty },
                                elem.full_span(),
                            )
                        })?
                    } else {
                        val.ty
                    };
                    val_ty = Some(ty);
                }

                let count = elements.len();
                let val_ty = self.alloc(val_ty.expect("todo: empty array lit"))?;
                let ty = Type::Array { len: count, elem_ty: val_ty };
                if is_const {
                    let val = todo!();
                    Ok(SemaValue::new_const(ty, val))
                } else {
                    Ok(SemaValue::new(ty))
                }
            },
            ExprKind::ArrayLitShort { val, count } => {
                let val = self.analyze(*val, is_const)?;
                let count_val = self.analyze(*count, true)?;
                let count_val =
                    count_val.int().map_err(|kind| SemaError { kind, span: count.full_span() })?;
                if count_val.is_negative() {
                    return err(NegativeArrayLen, count.full_span());
                }

                let count = count_val.try_into().expect("todo: large arr len");
                let ty = Type::Array { len: count, elem_ty: self.alloc(val.ty)? };
                if is_const {
                    let val = todo!();
                    Ok(SemaValue::new_const(ty, val))
                } else {
                    Ok(SemaValue::new(ty))
                }
            },
            ExprKind::Tuple { elements } => todo!(),
            ExprKind::Fn(func) => {
                assert!(is_const, "todo: non-const function");
                self.function_stack.push(func.into());
                self.open_scope();
                let Fn { mut params, ret_type, body } = func;
                let res = try {
                    for param in params.iter_mut() {
                        assert!(!param.is_const, "todo: const param");
                        self.analyze_var_decl(param)?;
                    }
                    let body_ty = self.analyze(*body, false)?.ty;
                    check_or_set_type!(ret_type, body_ty, body.span)?; // TODO: better span if `body` is a block
                    SemaValue::new_const(Type::Function(func.into()), EMPTY_PTR)
                };
                self.close_scope()?;
                self.function_stack.pop();
                res
            },
            ExprKind::Parenthesis { expr } => self.analyze(*expr, is_const),
            ExprKind::Block { stmts, has_trailing_semicolon } => {
                self.open_scope();
                let res: SemaResult<SemaValue> = try {
                    let mut val = SemaValue::void();
                    for s in stmts.iter_mut() {
                        match self.analyze(s.expr, is_const) {
                            Ok(new_val) => {
                                s.ty = new_val.ty.finalize();
                                debug_assert!(s.ty.is_valid());
                                debug_assert!(new_val.check_constness(is_const));
                                val = new_val
                            },
                            NotFinished => NotFinished?,
                            Err(err) => {
                                self.errors.push(err);
                                val = SemaValue::never()
                            },
                        }
                    }
                    val
                };
                self.close_scope()?;
                let val = res?;
                Ok(if !*has_trailing_semicolon || val.ty == Type::Never {
                    val
                } else {
                    SemaValue::void()
                })
            },
            ExprKind::StructDef(fields) => {
                // if !is_const {
                //     todo!()
                // }
                for field in fields.iter_mut() {
                    if field.is_const {
                        todo!("const struct field")
                    }
                    let f = self.var_decl_to_value(field)?;
                }
                let ty = self.alloc(Type::Struct { fields: *fields })?;
                Ok(SemaValue::new_const(Type::Type(ty), EMPTY_PTR))
            },
            ExprKind::UnionDef(_) => todo!(),
            ExprKind::EnumDef {} => todo!(),
            ExprKind::OptionShort(_) => todo!(),
            ExprKind::Ptr { is_mut, ty } => todo!(),
            ExprKind::Initializer { lhs: Some(ty_expr), fields: values } => {
                assert!(!is_const, "todo: const initializer");
                let lhs_val = self.analyze(*ty_expr, false)?;
                match lhs_val.ty {
                    Type::Never => return Ok(SemaValue::never()),
                    Type::Type(t) => {
                        self.analyze_initializer(*t, *values, is_const, span)?;
                        Ok(SemaValue::new(*t))
                    },
                    Type::Ptr(t) => {
                        self.analyze_initializer(*t, *values, is_const, span)?;
                        Ok(lhs_val)
                    },
                    ty => err(CannotApplyInitializer { ty }, ty_expr.full_span()),
                }
            },
            ExprKind::Initializer { lhs: None, fields: values } => {
                // TODO: benchmark this
                let mut struct_ty_iter = self
                    .symbols
                    .iter()
                    .filter_map(|s| match s.get_type().ok() {
                        Some(Type::Type(struct_ty)) => Some(*struct_ty),
                        _ => None,
                    })
                    .filter(|struct_ty| match struct_ty {
                        Type::Struct { fields } => values
                            .iter()
                            .all(|(i, _)| fields.iter().any(|f| *f.ident.text == *i.text)),
                        _ => false,
                    });
                let Some(struct_ty) = struct_ty_iter.next() else {
                    return err(CannotInferInitializerTy, expr.span.start_pos());
                };
                if struct_ty_iter.next().is_some() {
                    return err(MultiplePossibleInitializerTy, expr.span.start_pos());
                }
                drop(struct_ty_iter);
                self.analyze_initializer(struct_ty, *values, is_const, span)?;
                Ok(SemaValue::new(struct_ty))
            },
            ExprKind::Dot { lhs, rhs } => {
                assert!(!is_const, "todo: const dot");
                let lhs_ty = try_not_never!(self.analyze_typed(lhs, is_const)?).ty;
                debug_assert!(lhs_ty.is_valid());
                if let Type::Struct { fields } | Type::Union { fields } = lhs_ty
                    && let Some(f) = fields.iter().find(|f| *f.ident.text == *rhs.text)
                {
                    Ok(SemaValue::new(f.ty))
                } else {
                    err(UnknownField { ty: lhs_ty, field: rhs.text }, rhs.span)
                }
            },
            ExprKind::Index { lhs, idx } => {
                assert!(!is_const, "todo: const index");
                let arr = try_not_never!(self.analyze_typed(lhs, is_const)?);
                let Type::Array { len, elem_ty } = arr.ty else {
                    return err(CanOnlyIndexArrays, lhs.full_span());
                };
                let idx_val = try_not_never!(self.analyze_typed(idx, is_const)?).finalize_ty();
                assert!(idx_val.is_int(), "todo: non-int index");
                idx.ty = idx_val.ty;
                Ok(SemaValue::new(*elem_ty))
            },
            ExprKind::Call { func: f, args, .. } => {
                assert!(!is_const, "todo: const call");
                // self.analyze(*fn_expr)?;
                // debug_assert!(fn_expr.ty.is_valid());
                let Result::Ok(func) = f.try_to_ident() else {
                    todo!("non ident function call")
                };
                let ty = match self.get_ident_symbol(func)? {
                    SemaSymbol::Finished(v @ SemaValue { ty, .. }) => {
                        // currently Type::Function doesn't store additional data in const_val.
                        // this might change in the future
                        assert!(v.is_const(), "todo: call of a non-const function");
                        ty
                    },
                    SemaSymbol::NotFinished(Some(ty)) => todo!(),
                    SemaSymbol::NotFinished(_) => return NotFinished,
                };
                match ty {
                    Type::Never => Ok(SemaValue::never()),
                    Type::Function(mut func) => {
                        let Fn { params, ret_type, body } = func.as_mut();
                        self.validate_call(*params, *args, is_const)?;
                        debug_assert!(ret_type.is_valid());
                        Ok(SemaValue::new(*ret_type))
                    },
                    _ => err(CallOfANotFunction, span),
                }
            },
            &mut ExprKind::UnaryOp { kind, expr, .. } => {
                assert!(!is_const, "todo: PreOp in const");
                let ty = self.analyze(expr, is_const)?.ty;
                let out_ty = match (kind, ty) {
                    (_, Type::Unset | Type::Unevaluated(_)) => return err(CannotInfer, span),
                    (_, Type::Void) => None,
                    (_, Type::Never) => Some(ty),
                    (
                        UnaryOpKind::AddrOf | UnaryOpKind::AddrMutOf,
                        Type::Ptr(..)
                        | Type::Int { .. }
                        | Type::IntLiteral
                        | Type::Bool
                        | Type::Float { .. }
                        | Type::FloatLiteral
                        | Type::Array { .. }
                        | Type::Struct { .. }
                        | Type::Union { .. }
                        | Type::Enum { .. },
                    ) => Some(Type::Ptr(self.alloc(ty)?)),
                    (UnaryOpKind::AddrOf | UnaryOpKind::AddrMutOf, Type::Function(_)) => todo!(),
                    (UnaryOpKind::Deref, Type::Ptr(pointee)) => Some(*pointee),
                    (
                        UnaryOpKind::Deref,
                        Type::Int { .. }
                        | Type::IntLiteral
                        | Type::Bool
                        | Type::Float { .. }
                        | Type::FloatLiteral
                        | Type::Function(_)
                        | Type::Array { .. },
                    ) => None,
                    (UnaryOpKind::Not, Type::Int { .. } | Type::IntLiteral | Type::Bool) => {
                        Some(ty)
                    },
                    (
                        UnaryOpKind::Not,
                        Type::Ptr(_)
                        | Type::Float { .. }
                        | Type::FloatLiteral
                        | Type::Function(_)
                        | Type::Array { .. },
                    ) => None,
                    (
                        UnaryOpKind::Neg,
                        Type::Int { .. }
                        | Type::IntLiteral
                        | Type::Float { .. }
                        | Type::FloatLiteral,
                    ) => Some(ty),
                    (
                        UnaryOpKind::Neg,
                        Type::Ptr(_) | Type::Bool | Type::Function(_) | Type::Array { .. },
                    ) => None,
                    (
                        _,
                        Type::Struct { .. }
                        | Type::Union { .. }
                        | Type::Enum { .. }
                        | Type::Type(_),
                    ) => todo!("UnaryOp {kind:?} for {ty:?}"),
                    (UnaryOpKind::Try, _) => todo!("try"),
                };
                match out_ty {
                    Some(t) => Ok(SemaValue::new(t)),
                    None => err(InvalidPreOp { ty, kind }, span),
                }
            },
            ExprKind::BinOp { lhs, op, rhs, val_ty: ty } => {
                assert!(!is_const, "todo: BinOp in const");
                let lhs_ty = self.analyze(*lhs, is_const)?.ty;
                debug_assert!(lhs_ty.is_valid());
                let rhs_ty = self.analyze(*rhs, is_const)?.ty;
                debug_assert!(rhs_ty.is_valid());
                if let Some(common_ty) = lhs_ty.common_type(rhs_ty) {
                    *ty = common_ty;
                    // todo: check if binop can be applied to type
                    Ok(match op {
                        BinOpKind::Mul
                        | BinOpKind::Div
                        | BinOpKind::Mod
                        | BinOpKind::Add
                        | BinOpKind::Sub => common_ty,
                        BinOpKind::ShiftL => todo!(),
                        BinOpKind::ShiftR => todo!(),
                        BinOpKind::BitAnd => todo!(),
                        BinOpKind::BitXor => todo!(),
                        BinOpKind::BitOr => todo!(),
                        BinOpKind::Eq
                        | BinOpKind::Ne
                        | BinOpKind::Lt
                        | BinOpKind::Le
                        | BinOpKind::Gt
                        | BinOpKind::Ge => {
                            *ty = common_ty.finalize();
                            Type::Bool
                        },
                        BinOpKind::And | BinOpKind::Or => {
                            if common_ty == Type::Bool {
                                Type::Bool
                            } else {
                                todo!()
                            }
                        },
                        BinOpKind::Range => todo!(),
                        BinOpKind::RangeInclusive => todo!(),
                    })
                } else {
                    err(MismatchedTypesBinOp { lhs_ty, rhs_ty }, expr.span)
                }
                .map_ok(SemaValue::new)
            },
            ExprKind::Assign { lhs, rhs, .. } => {
                assert!(!is_const, "todo: Assign in const");
                let lhs_ty = self.analyze_typed(lhs, is_const)?.ty;
                debug_assert!(lhs_ty.is_valid());
                let rhs_ty = self.analyze(*rhs, is_const)?.ty;
                debug_assert!(rhs_ty.is_valid());
                if lhs_ty.matches(rhs_ty) {
                    // todo: check if binop can be applied to type
                    Ok(SemaValue::void())
                } else {
                    err(MismatchedTypes { expected: lhs_ty, got: rhs_ty }, rhs.full_span())
                }
            },
            ExprKind::BinOpAssign { lhs, op, rhs } => {
                assert!(!is_const, "todo: BinOpAssign in const");
                let lhs_ty = self.analyze_typed(lhs, is_const)?.ty;
                debug_assert!(lhs_ty.is_valid());
                let rhs_ty = self.analyze(*rhs, is_const)?.ty;
                debug_assert!(rhs_ty.is_valid());
                if lhs_ty.matches(rhs_ty) {
                    // todo: check if binop can be applied to type
                    Ok(SemaValue::void())
                } else {
                    err(MismatchedTypesBinOp { lhs_ty, rhs_ty }, expr.span)
                }
            },
            ExprKind::VarDecl(decl) => {
                if is_const && !decl.is_const {
                    return err(NotAConstExpr, span);
                }
                self.analyze_var_decl(decl)?;
                Ok(SemaValue::void())
            },
            ExprKind::If { condition, then_body, else_body, .. } => {
                assert!(!is_const, "todo: if in const");
                let cond = self.analyze(*condition, is_const)?.ty;
                debug_assert!(cond.is_valid());
                if cond != Type::Bool && cond != Type::Never {
                    return err(
                        MismatchedTypes { expected: Type::Bool, got: cond },
                        condition.full_span(),
                    );
                }
                let then_ty = self.analyze(*then_body, is_const)?.ty;
                debug_assert!(then_ty.is_valid());
                if let Some(else_body) = else_body {
                    let else_ty = self.analyze(*else_body, is_const)?.ty;
                    debug_assert!(else_ty.is_valid());
                    then_ty.common_type(else_ty).map(SemaValue::new).ok_or_else2(|| {
                        err(
                            IncompatibleBranches { expected: then_ty, got: else_ty },
                            else_body.full_span(),
                        )
                    })
                } else if matches!(then_ty, Type::Void | Type::Never) {
                    Ok(SemaValue::void())
                } else {
                    err(MissingElseBranch, expr.full_span())
                }
            },
            ExprKind::Match { val, else_body, .. } => todo!(),
            ExprKind::For { source, iter_var, body, .. } => {
                let arr = try_not_never!(self.analyze_typed(source, is_const)?);
                let Type::Array { len: count, elem_ty } = arr.ty.finalize() else {
                    todo!("for over non-array")
                };

                self.open_scope();
                self.analyze_var_decl(&mut VarDecl {
                    markers: DeclMarkers::default(),
                    ident: *iter_var,
                    ty: *elem_ty,
                    default: None,
                    is_const: false,
                })?;
                let body_ty = self.analyze(*body, is_const)?;
                if !matches!(body_ty.ty, Type::Void | Type::Never) {
                    return err(CannotReturnFromLoop, body.full_span());
                }
                self.close_scope()?;
                //Ok(body_ty)
                Ok(SemaValue::void())
            },
            ExprKind::While { condition, body, .. } => {
                let cond_ty = try_not_never!(self.analyze(*condition, is_const)?).ty.finalize();
                if cond_ty != Type::Bool {
                    let span = condition.full_span();
                    return err(MismatchedTypes { expected: Type::Bool, got: cond_ty }, span);
                }
                self.open_scope();
                let body_ty = self.analyze(*body, is_const)?;
                if !matches!(body_ty.ty, Type::Void | Type::Never) {
                    return err(CannotReturnFromLoop, body.full_span());
                }
                self.close_scope()?;
                //Ok(body_ty)
                Ok(SemaValue::void())
            },
            ExprKind::Catch { lhs } => todo!(),
            ExprKind::Defer(expr) => {
                self.defer_stack.push_expr(*expr);
                Ok(SemaValue::void())
            },
            ExprKind::Return { expr: val } => {
                let ret_type = if let Some(val) = val {
                    self.analyze_typed(val, is_const)?.ty
                } else {
                    Type::Void
                };
                let Some(func) = self.function_stack.last_mut() else {
                    return err(ReturnNotInAFunction, expr.full_span());
                };
                check_or_set_type!(
                    &mut func.ret_type,
                    ret_type.finalize(),
                    val.map(|v| v.full_span()).unwrap_or(span),
                )?;
                if let Some(val) = val {
                    val.ty = func.ret_type;
                }
                Ok(SemaValue::never())
            },
            ExprKind::Break { expr } => {
                if expr.is_some() {
                    todo!("break with value")
                }
                // check if in loop
                Ok(SemaValue::never())
            },
            ExprKind::Continue => {
                // check if in loop
                Ok(SemaValue::never())
            },
            ExprKind::Semicolon(_) => todo!(),
        };
        // if let Ok(val) = res {
        //     expr.ty = val.ty;
        // }
        if DEBUG_TYPES {
            display_span_in_code_with_label(expr.full_span(), self.code, format!("type: {res:?}")); // TODO: remove this
        }
        res
    }

    pub fn analyze_typed(
        &mut self,
        expr: &mut ExprWithTy,
        is_const: bool,
    ) -> SemaResult<SemaValue> {
        let res = self.analyze(expr.expr, is_const);
        if let Ok(val) = res {
            expr.ty = val.ty;
        }
        res
    }

    pub fn analyze_initializer(
        &mut self,
        struct_ty: Type,
        initializer_values: Ptr<[(Ident, Option<Ptr<Expr>>)]>,
        is_const: bool,
        initializer_span: Span,
    ) -> SemaResult<()> {
        let Type::Struct { fields } = struct_ty else { panic!("todo: error") };

        let mut is_initialized_field = vec![false; fields.len()];
        for (f, init) in initializer_values.iter() {
            match try {
                let field = f.text;
                let Some((f_idx, f_decl)) = fields.find_field(&*field) else {
                    err(UnknownField { ty: struct_ty, field }, f.span)?
                };

                if is_initialized_field[f_idx] {
                    err(DuplicateInInitializer, f.span)?
                }
                is_initialized_field[f_idx] = true;

                let (init_ty, span) = match init {
                    Some(expr) => (self.analyze(*expr, is_const)?.ty, expr.full_span()),
                    None => (self.get_symbol(field, f.span)?.get_type()?, f.span),
                };
                debug_assert!(f_decl.ty.is_valid() && f_decl.ty != Type::Never);
                debug_assert!(init_ty.is_valid());
                if !f_decl.ty.matches(init_ty) {
                    err(MismatchedTypes { expected: f_decl.ty, got: init_ty }, span)?
                }
            } {
                Ok(()) => {},
                NotFinished => NotFinished?,
                Err(err) => self.errors.push(err),
            }
        }

        for missing_field in is_initialized_field
            .into_iter()
            .enumerate()
            .filter(|(_, is_init)| !is_init)
            .map(|(idx, _)| fields[idx])
            .filter(|f| f.default.is_none())
        {
            let field = missing_field.ident.text;
            self.errors.push(err_val(MissingFieldInInitializer { field }, initializer_span))
        }
        Ok(())
    }

    /// Returns the [`SemaValue`] repesenting the new variable, not the entire
    /// declaration. This also doesn't insert into `self.symbols`.
    fn var_decl_to_value(&mut self, decl: &mut VarDecl) -> SemaResult<SemaValue> {
        let ty = &mut decl.ty;
        self.eval_type(ty)?;
        if let Some(init) = &mut decl.default {
            let mut init_val = self.analyze(*init, decl.is_const)?;
            check_or_set_type!(ty, init_val.ty, init.span)?;
            init_val.ty = *ty;
            Ok(init_val)
        } else if *ty == Type::Unset {
            err(VarDeclNoType, decl.ident.span)
        } else {
            debug_assert!(ty.is_valid());
            debug_assert!(!decl.is_const);
            Ok(SemaValue::new(*ty))
        }
    }

    fn analyze_var_decl(&mut self, decl: &mut VarDecl) -> SemaResult<()> {
        let res = self.var_decl_to_value(decl);
        let name = &*decl.ident.text;
        match res {
            Ok(val) => {
                let _ = self.symbols.insert(name, SemaSymbol::Finished(val));
                Ok(())
            },
            NotFinished => {
                let _ = self.symbols.insert(name, SemaSymbol::NotFinished(None));
                NotFinished
            },
            Err(err) => {
                self.errors.push(err);
                let ty = decl.ty.into_valid().unwrap_or(Type::Never);
                let _ = self.symbols.insert(name, SemaSymbol::Finished(SemaValue::new(ty)));
                Ok(())
            },
        }

        // println!("INFO: '{}' was shadowed in the same scope",
        // &*decl.ident.text);
    }

    /// works for function calls and call initializers
    /// ...(arg1, ..., argX, paramX1=argX1, ... paramN=argN)
    /// paramN1, ... -> default values
    fn validate_call(
        &mut self,
        params: VarDeclList,
        args: Ptr<[Ptr<Expr>]>,
        is_const: bool,
    ) -> SemaResult<()> {
        // TODO: check for duplicate named arguments
        for (idx, p) in params.iter().enumerate() {
            let Some(arg) = args.get(idx) else {
                if p.default.is_some() {
                    continue;
                }
                return err(MissingArg, todo!());
            };
            let arg_ty = self.analyze(*arg, is_const)?.ty;
            debug_assert!(p.ty.is_valid()); // TODO: infer?
            if !p.ty.matches(arg_ty) {
                return err(MismatchedTypes { expected: p.ty, got: arg_ty }, arg.span);
            }
        }
        Ok(())
    }

    /// expects the [`Type`] of `ty_expr` to be [`Type::Type`]. This returns the
    /// inner type of [`Type::Type`].
    ///
    /// If you want the [`Type`] itself, use [`Sema::analyze`].
    fn analyze_type_expr(&mut self, ty_expr: Ptr<Expr>) -> SemaResult<Ptr<Type>> {
        match self.analyze(ty_expr, true)?.ty {
            Type::Never => todo!(),
            Type::Type(t) => Ok(t),
            _ => err(NotAType, ty_expr.full_span()),
        }
    }

    fn try_internal_ty(&mut self, name: Ptr<str>) -> Option<Type> {
        match name.bytes().next()? {
            b'i' => name[1..].parse().ok().map(|bits| Type::Int { bits, is_signed: true }),
            b'u' => name[1..].parse().ok().map(|bits| Type::Int { bits, is_signed: false }),
            b'f' => name[1..].parse().ok().map(|bits| Type::Float { bits }),
            _ => None,
        }
    }

    /// [`Type::Unevaluated`] -> a valid [`Type`]
    fn eval_type(&mut self, ty: &mut Type) -> SemaResult<Type> {
        if let Type::Unevaluated(ty_expr) = *ty {
            *ty = *self.analyze_type_expr(ty_expr)?;
            debug_assert!(ty.is_valid());
        }
        Ok(*ty)
    }

    #[inline]
    fn get_symbol(&self, name: Ptr<str>, err_span: Span) -> SemaResult<&SemaSymbol> {
        match self.symbols.get(&name) {
            Some(sym) => Ok(sym),
            None => err(UnknownIdent(name), err_span),
        }
    }

    #[inline]
    fn get_ident_symbol(&self, i: Ident) -> SemaResult<&SemaSymbol> {
        self.get_symbol(i.text, i.span)
    }

    fn open_scope(&mut self) {
        self.symbols.open_scope();
        self.defer_stack.open_scope();
    }

    fn close_scope(&mut self) -> SemaResult<()> {
        let res = self.analyze_defer_exprs();
        self.symbols.close_scope();
        self.defer_stack.close_scope();
        res
    }

    #[inline]
    fn analyze_defer_exprs(&mut self) -> SemaResult<()> {
        let exprs = unsafe { forget_lifetime(self.defer_stack.get_cur_scope()) };
        for expr in exprs.iter().rev() {
            self.analyze(*expr, false)?;
        }
        Ok(())
    }

    #[inline]
    fn alloc<T: core::fmt::Debug>(&self, val: T) -> SemaResult<Ptr<T>> {
        match self.alloc.try_alloc(val) {
            Result::Ok(ok) => Ok(Ptr::from(ok)),
            Result::Err(e) => err(AllocErr(e), todo!()),
        }
    }
}

#[inline]
pub fn err<T>(kind: SemaErrorKind, span: Span) -> SemaResult<T> {
    Err(SemaError { kind, span })
}

#[inline]
pub fn err_val(kind: SemaErrorKind, span: Span) -> SemaError {
    SemaError { kind, span }
}
