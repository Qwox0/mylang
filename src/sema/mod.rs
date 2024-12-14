//! # Semantic analysis module
//!
//! Semantic analysis validates (and changes) all stored [`Type`]s in [`Expr`].

#![allow(unused_variables)]

use crate::{
    ast::{
        BinOpKind, DeclMarkers, Expr, ExprKind, ExprWithTy, Fn, Ident, LitKind, UnaryOpKind,
        VarDecl, VarDeclListTrait,
    },
    defer_stack::DeferStack,
    parser::lexer::{Code, Span},
    ptr::Ptr,
    symbol_table::SymbolTable,
    type_::Type,
    util::{
        OkOrWithTry, UnwrapDebug, display_span_in_code_with_label, forget_lifetime,
        unreachable_debug,
    },
};
pub use err::{SemaError, SemaErrorKind, SemaResult};
use err::{SemaErrorKind::*, SemaResult::*};
use std::collections::HashSet;
pub use symbol::SemaSymbol;
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
pub struct Sema<'c, 'alloc> {
    code: &'c Code,

    pub symbols: SymbolTable<SemaSymbol>,
    struct_stack: Vec<Vec<Ptr<Type>>>,
    enum_stack: Vec<Vec<Ptr<Type>>>,
    function_stack: Vec<Ptr<Fn>>,
    defer_stack: DeferStack,

    pub errors: Vec<SemaError>,

    alloc: &'alloc bumpalo::Bump,

    #[cfg(debug_assertions)]
    debug_types: bool,
}

impl<'c, 'alloc> Sema<'c, 'alloc> {
    pub fn new(
        code: &'c Code,
        alloc: &'alloc bumpalo::Bump,
        debug_types: bool,
    ) -> Sema<'c, 'alloc> {
        Sema {
            code,
            symbols: SymbolTable::with_one_scope(),
            struct_stack: vec![vec![]],
            enum_stack: vec![vec![]],
            function_stack: vec![],
            defer_stack: DeferStack::default(),
            errors: vec![],
            alloc,
            #[cfg(debug_assertions)]
            debug_types,
        }
    }

    pub fn preload_top_level(&mut self, s: Ptr<Expr>) {
        let res = try {
            let item_ident = match s.kind {
                ExprKind::VarDecl(decl) => {
                    debug_assert!(decl.is_const);
                    debug_assert!(decl.default.is_some());
                    decl.ident
                },
                ExprKind::Extern { ident, .. } => ident,
                _ => err(UnexpectedTopLevelExpr(s), s.span)?,
            };

            if self.symbols.insert(&*item_ident.text, SemaSymbol::preload_symbol()).is_some() {
                err(TopLevelDuplicate, item_ident.span)?
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

        let res = match &mut expr.kind {
            ExprKind::Ident(text) => {
                if let Some(internal_ty) = Type::try_internal_ty(*text) {
                    let t = self.alloc(internal_ty)?;
                    return Ok(SemaValue::type_(t));
                }

                // TODO: remove this
                if &**text == "nil" {
                    return Ok(SemaValue::new(Type::Ptr { pointee_ty: Type::ptr_void() }));
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
                    LitKind::Str => Type::str_slice(),
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
            ExprKind::PtrTy { is_mut, ty } => {
                self.eval_type(ty)?;
                debug_assert!(ty.is_valid());
                let pointee_ty = self.alloc(*ty)?;
                self.alloc(Type::Ptr { pointee_ty }).map_ok(SemaValue::type_)
            },
            ExprKind::SliceTy { ty } => {
                self.eval_type(ty)?;
                debug_assert!(ty.is_valid());
                let elem_ty = self.alloc(*ty)?;
                self.alloc(Type::Slice { elem_ty }).map_ok(SemaValue::type_)
            },
            ExprKind::ArrayTy { count, ty } => {
                let count_val = self.analyze(*count, true)?;
                let count = count_val.int().map_err(|kind| SemaError { kind, span: count.span })?;
                let count = count.try_into().expect("todo: array to long");
                self.eval_type(ty)?;
                let elem_ty = self.alloc(*ty)?;
                let arr_ty = Type::Array { len: count, elem_ty };
                self.alloc(arr_ty).map_ok(SemaValue::type_)
            },
            ExprKind::Fn(func) => {
                assert!(is_const, "todo: non-const function");
                self.function_stack.push(func.into());
                self.open_scope();
                let Fn { mut params, ret_type, body } = func;
                self.eval_type(ret_type)?;
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
                            Ok(mut new_val) => {
                                new_val.ty = new_val.ty.finalize();
                                s.ty = new_val.ty;
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
                let mut field_names = HashSet::new();
                for field in fields.iter_mut() {
                    let is_duplicate = !field_names.insert(field.ident.text.as_ref());
                    if is_duplicate {
                        return err(DuplicateField, field.ident.span);
                    }
                    if field.is_const {
                        todo!("const struct field")
                    }
                    let f = self.var_decl_to_value(field)?;
                }
                let ty = self.alloc(Type::Struct { fields: *fields })?;
                self.struct_stack.last_mut().unwrap_debug().push(ty);
                Ok(SemaValue::type_(ty))
            },
            ExprKind::UnionDef(fields) => {
                let mut field_names = HashSet::new();
                for field in fields.iter_mut() {
                    let is_duplicate = !field_names.insert(field.ident.text.as_ref());
                    if is_duplicate {
                        return err(DuplicateField, field.ident.span);
                    }
                    if field.is_const {
                        todo!("const struct field")
                    }
                    if let Some(d) = field.default {
                        return err(UnionFieldWithDefaultValue, d.full_span());
                    }
                    let f = self.var_decl_to_value(field)?;
                }
                let ty = self.alloc(Type::Union { fields: *fields })?;
                Ok(SemaValue::type_(ty))
            },
            ExprKind::EnumDef(variants) => {
                let mut variant_names = HashSet::new();
                for variant in variants.iter_mut() {
                    let is_duplicate = !variant_names.insert(variant.ident.text.as_ref());
                    if is_duplicate {
                        return err(DuplicateEnumVariant, variant.ident.span);
                    }
                    let _ = self.var_decl_to_value(variant)?;
                }
                let ty = self.alloc(Type::Enum { variants: *variants })?;
                self.enum_stack.last_mut().unwrap_debug().push(ty);
                Ok(SemaValue::type_(ty))
            },
            ExprKind::OptionShort(ty) => {
                self.eval_type(ty)?;
                Ok(SemaValue::type_(self.alloc(Type::Option { ty: self.alloc(*ty)? })?))
            },
            ExprKind::PositionalInitializer { lhs: Some(ty_expr), lhs_ty, args } => {
                assert!(!is_const, "todo: const initializer");
                let lhs_val = self.analyze(*ty_expr, false)?;
                *lhs_ty = lhs_val.ty;
                match *lhs_ty {
                    Type::Never => return Ok(SemaValue::never()),
                    Type::Type(t) => {
                        let Type::Struct { fields } = *t else { todo!("error") };
                        self.validate_call(&*fields, args.as_ref(), is_const)?;
                        Ok(SemaValue::new(*t))
                    },
                    Type::Ptr { pointee_ty: t } => {
                        let Type::Struct { fields } = *t else { todo!("error") };
                        self.validate_call(&*fields, args.as_ref(), is_const)?;
                        Ok(lhs_val)
                    },
                    ty => err(CannotApplyInitializer { ty }, ty_expr.full_span()),
                }
            },
            ExprKind::PositionalInitializer { lhs: None, .. } => {
                err(CannotInferPositionalInitializerTy, span.start_pos())
            },
            ExprKind::NamedInitializer { lhs: Some(ty_expr), lhs_ty, fields: values } => {
                assert!(!is_const, "todo: const initializer");
                let lhs_val = self.analyze(*ty_expr, false)?;
                *lhs_ty = lhs_val.ty;
                match *lhs_ty {
                    Type::Never => return Ok(SemaValue::never()),
                    Type::Type(t) => {
                        self.validate_initializer(*t, *values, is_const, span)?;
                        Ok(SemaValue::new(*t))
                    },
                    Type::Ptr { pointee_ty: t } => {
                        self.validate_initializer(*t, *values, is_const, span)?;
                        Ok(lhs_val)
                    },
                    ty => err(CannotApplyInitializer { ty }, ty_expr.full_span()),
                }
            },
            ExprKind::NamedInitializer { lhs: None, lhs_ty, fields: values } => {
                // TODO: benchmark this
                let mut struct_ty_iter =
                    self.struct_stack.iter().flat_map(|scope| scope.iter()).copied().filter(
                        |struct_ty| match **struct_ty {
                            Type::Struct { fields } => values
                                .iter()
                                .all(|(i, _)| fields.iter().any(|f| *f.ident.text == *i.text)),
                            _ => false,
                        },
                    );
                let Some(struct_ty) = struct_ty_iter.next() else {
                    *lhs_ty = Type::Never;
                    return err(CannotInferNamedInitializerTy, span.start_pos());
                };
                if struct_ty_iter.next().is_some() {
                    *lhs_ty = Type::Never;
                    return err(MultiplePossibleInitializerTy, span.start_pos());
                }
                drop(struct_ty_iter);
                *lhs_ty = Type::Type(struct_ty);
                let struct_ty = *struct_ty;
                self.validate_initializer(struct_ty, *values, is_const, span)?;
                Ok(SemaValue::new(struct_ty))
            },
            ExprKind::ArrayInitializer { lhs: Some(_), lhs_ty, elements } => todo!(),
            ExprKind::ArrayInitializer { lhs: None, lhs_ty, elements } => {
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
            ExprKind::ArrayInitializerShort { lhs: Some(_), lhs_ty, val, count } => todo!(),
            ExprKind::ArrayInitializerShort { lhs: None, lhs_ty, val, count } => {
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
            ExprKind::Dot { lhs: Some(lhs), lhs_ty, rhs } => {
                assert!(!is_const, "todo: const dot");
                *lhs_ty = try_not_never!(self.analyze(*lhs, is_const)?).ty;
                debug_assert!(lhs_ty.is_valid());
                self.analyze_dot(*lhs_ty, rhs)
            },
            ExprKind::Dot { lhs: None, lhs_ty, rhs } => {
                // `.<ident>` must be an enum
                let mut enum_ty_iter =
                    self.enum_stack.iter().flat_map(|scope| scope.iter()).copied().filter(
                        |enum_ty| match **enum_ty {
                            Type::Enum { variants } => {
                                variants.iter().any(|v| *v.ident.text == *rhs.text)
                            },
                            _ => false,
                        },
                    );
                let Some(enum_ty) = enum_ty_iter.next() else {
                    return err(CannotInferNamedInitializerTy, expr.span.start_pos());
                };
                if enum_ty_iter.next().is_some() {
                    return err(MultiplePossibleInitializerTy, expr.span.start_pos());
                }
                drop(enum_ty_iter);
                *lhs_ty = Type::Type(enum_ty);
                self.analyze_dot(*lhs_ty, rhs)
            },
            ExprKind::Index { lhs, idx } => {
                assert!(!is_const, "todo: const index");
                let arr = try_not_never!(self.analyze_typed(lhs, is_const)?);
                let (Type::Array { elem_ty, .. } | Type::Slice { elem_ty }) = arr.ty else {
                    return err(CanOnlyIndexArrays, lhs.full_span());
                };
                let idx_val = try_not_never!(self.analyze_typed(idx, is_const)?).finalize_ty();
                idx.ty = idx_val.ty;
                Ok(SemaValue::new(match idx_val.ty {
                    Type::Int { .. } => *elem_ty,
                    Type::Range { elem_ty: i } | Type::RangeInclusive { elem_ty: i }
                        if matches!(*i, Type::Int { .. }) =>
                    {
                        Type::Slice { elem_ty }
                    },
                    t => todo!("other idx ({t:#?})"),
                }))
            },
            ExprKind::Call { func: f, args, .. } => {
                assert!(!is_const, "todo: const call");
                match self.analyze_typed(f, is_const)?.ty {
                    Type::Never => Ok(SemaValue::never()),
                    Type::Function(mut func) => {
                        let Fn { params, ret_type, body } = func.as_mut();
                        self.validate_call(&params, &args, is_const)?;
                        debug_assert!(ret_type.is_valid());
                        /*
                        if *ret_type == Type::Never {
                            self.function_stack.last_mut().unwrap_debug().ret_type = Type::Never;
                        }
                        */
                        Ok(SemaValue::new(*ret_type))
                    },
                    Type::EnumVariant { enum_ty, idx } => {
                        let Type::Enum { variants } = *enum_ty else { unreachable_debug() };
                        let variant = variants[idx];
                        self.validate_call(&[variant], &args, is_const)?;
                        Ok(SemaValue::new(*enum_ty))
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
                    (UnaryOpKind::AddrOf | UnaryOpKind::AddrMutOf, Type::Function(_)) => todo!(),
                    (UnaryOpKind::AddrOf | UnaryOpKind::AddrMutOf, ty) => {
                        Some(Type::Ptr { pointee_ty: self.alloc(ty)? })
                    },
                    (UnaryOpKind::Deref, Type::Ptr { pointee_ty }) => Some(*pointee_ty),
                    (UnaryOpKind::Deref, _) => None,
                    (UnaryOpKind::Not, Type::Int { .. } | Type::IntLiteral | Type::Bool) => {
                        Some(ty)
                    },
                    (UnaryOpKind::Not, _) => None,
                    (
                        UnaryOpKind::Neg,
                        Type::Int { .. }
                        | Type::IntLiteral
                        | Type::Float { .. }
                        | Type::FloatLiteral,
                    ) => Some(ty),
                    (UnaryOpKind::Neg, _) => None,
                    (UnaryOpKind::Try, _) => todo!("try"),
                };
                match out_ty {
                    Some(t) => Ok(SemaValue::new(t)),
                    None => err(InvalidPreOp { ty, kind }, span),
                }
            },
            ExprKind::BinOp { lhs, op, rhs, arg_ty } => {
                assert!(!is_const, "todo: BinOp in const");
                let lhs_ty = self.analyze(*lhs, is_const)?.ty;
                debug_assert!(lhs_ty.is_valid());
                let rhs_ty = self.analyze(*rhs, is_const)?.ty;
                debug_assert!(rhs_ty.is_valid());
                if let Some(common_ty) = lhs_ty.common_type(rhs_ty) {
                    *arg_ty = common_ty;
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
                            *arg_ty = common_ty.finalize();
                            Type::Bool
                        },
                        BinOpKind::And | BinOpKind::Or => {
                            if common_ty == Type::Bool {
                                Type::Bool
                            } else {
                                todo!()
                            }
                        },
                        BinOpKind::Range => Type::Range { elem_ty: self.alloc(common_ty)? },
                        BinOpKind::RangeInclusive => {
                            Type::RangeInclusive { elem_ty: self.alloc(common_ty)? }
                        },
                    })
                } else {
                    println!("MismatchedTypesBinOp {lhs_ty:?} {rhs_ty:?}",);
                    err(MismatchedTypesBinOp { lhs_ty, rhs_ty }, expr.span)
                }
                .map_ok(SemaValue::new)
            },
            ExprKind::Assign { lhs, rhs, .. } => {
                if is_const {
                    return err(AssignToConst, span);
                }
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
                let Some(common_ty) = lhs_ty.common_type(rhs_ty) else {
                    return err(MismatchedTypesBinOp { lhs_ty, rhs_ty }, expr.span);
                };
                lhs.ty = common_ty;
                // todo: check if binop can be applied to type
                Ok(SemaValue::void())
            },
            ExprKind::VarDecl(decl) => {
                if is_const && !decl.is_const {
                    return err(NotAConstExpr, span);
                }
                self.analyze_var_decl(decl)?;
                Ok(SemaValue::void())
            },
            ExprKind::Extern { ident, ty } => {
                self.eval_type(ty)?;
                if let Type::Function(f) = ty
                    && let Type::Type(ret_type) = f.ret_type
                {
                    f.ret_type = *ret_type;
                }
                let _ =
                    self.symbols.insert(&*ident.text, SemaSymbol::Finished(SemaValue::new(*ty)));
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
            // ExprKind::Semicolon(_) => todo!(),
        };
        #[cfg(debug_assertions)]
        if self.debug_types {
            display_span_in_code_with_label(expr.full_span(), self.code, format!("type: {res:?}"));
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

    pub fn validate_initializer(
        &mut self,
        struct_ty: Type,
        initializer_values: Ptr<[(Ident, Option<Ptr<Expr>>)]>,
        is_const: bool,
        initializer_span: Span,
    ) -> SemaResult<()> {
        let fields = match &struct_ty {
            Type::Struct { fields } => &**fields,
            Type::Slice { elem_ty } => &Type::slice_fields(*elem_ty),
            &ty => return err(CannotApplyInitializer { ty }, initializer_span),
        };

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

    fn analyze_dot(&mut self, lhs_ty: Type, rhs: &Ident) -> SemaResult<SemaValue> {
        match lhs_ty {
            Type::Struct { fields } | Type::Union { fields } => fields
                .iter()
                .find(|f| *f.ident.text == *rhs.text)
                .map(|f| SemaValue::new(f.ty))
                .ok_or_else2(|| err(UnknownField { ty: lhs_ty, field: rhs.text }, rhs.span)),
            Type::Type(p) => match *p {
                Type::Enum { variants } => variants
                    .iter()
                    .enumerate()
                    .find(|(_, f)| *f.ident.text == *rhs.text)
                    .map(|(idx, _)| SemaValue::new(Type::EnumVariant { enum_ty: p, idx }))
                    .ok_or_else2(|| err(UnknownField { ty: lhs_ty, field: rhs.text }, rhs.span)),
                _ => todo!("dot for type {:?}", *p),
            },
            Type::Slice { elem_ty } => Ok(SemaValue::new(match &*rhs.text {
                "ptr" => Type::Ptr { pointee_ty: elem_ty },
                "len" => Type::U64,
                _ => return err(UnknownField { ty: lhs_ty, field: rhs.text }, rhs.span),
            })),
            _ => todo!("err for invalid dot lhs"),
        }
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

        // println!("INFO: '{}' was shadowed in the same scope", &*decl.ident.text);
    }

    /// works for function calls and call initializers
    /// ...(arg1, ..., argX, paramX1=argX1, ... paramN=argN)
    /// paramN1, ... -> default values
    fn validate_call(
        &mut self,
        params: &[VarDecl],
        args: &[Ptr<Expr>],
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
                return err(MismatchedTypes { expected: p.ty, got: arg_ty }, arg.full_span());
            }
        }
        Ok(())
    }

    /// [`Type::Unevaluated`] -> a valid [`Type`]
    fn eval_type(&mut self, ty: &mut Type) -> SemaResult<()> {
        if let Type::Unevaluated(ty_expr) = *ty {
            *ty = *match self.analyze(ty_expr, true)?.ty {
                Type::Never => todo!(),
                Type::Type(t) => t,
                t @ Type::Function(_) => self.alloc(t)?, // TODO: don't alloc here
                _ => return Err(err_val(NotAType, ty_expr.full_span())),
            };
            debug_assert!(ty.is_valid());
        }
        Ok(())
    }

    #[inline]
    fn get_symbol(&self, name: Ptr<str>, err_span: Span) -> SemaResult<&SemaSymbol> {
        match self.symbols.get(&name) {
            Some(sym) => Ok(sym),
            None => err(UnknownIdent(name), err_span),
        }
    }

    fn open_scope(&mut self) {
        self.symbols.open_scope();
        self.struct_stack.push(vec![]);
        self.enum_stack.push(vec![]);
        self.defer_stack.open_scope();
    }

    fn close_scope(&mut self) -> SemaResult<()> {
        let res = self.analyze_defer_exprs();
        self.symbols.close_scope();
        self.struct_stack.pop();
        self.enum_stack.pop();
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
