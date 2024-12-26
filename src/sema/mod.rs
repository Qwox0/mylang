//! # Semantic analysis module
//!
//! Semantic analysis validates (and changes) all stored [`Type`]s in [`Expr`].

use crate::{
    ast::{
        BinOpKind, DeclMarkers, Expr, ExprKind, ExprWithTy, Fn, Ident, UnaryOpKind, VarDecl,
        VarDeclListTrait,
    },
    parser::lexer::{Code, Span},
    ptr::Ptr,
    scoped_stack::ScopedStack,
    symbol_table::SymbolTable,
    type_::{RangeKind, Type},
    util::{
        OkOrWithTry, UnwrapDebug, display_span_in_code_with_label, forget_lifetime,
        unreachable_debug,
    },
};
pub use err::{SemaError, SemaErrorKind, SemaResult};
use err::{SemaErrorKind::*, SemaResult::*};
use std::{assert_matches::debug_assert_matches, collections::HashSet};
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
        } else if let Some(common_ty) = target.common_type(new_ty) {
            *target = common_ty;
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
    defer_stack: ScopedStack<Ptr<Expr>>,

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
            defer_stack: ScopedStack::default(),
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
        match self.analyze(s, Type::Void, true) {
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
    pub fn analyze(
        &mut self,
        mut expr: Ptr<Expr>,
        ty_hint: Type,
        is_const: bool,
    ) -> SemaResult<SemaValue> {
        let span = expr.span;

        macro_rules! lit_to_val {
            ($ty:expr, $const_val:expr) => {
                if is_const {
                    let val = self.alloc($const_val)?.cast::<()>();
                    Ok(SemaValue::new_const($ty, val))
                } else {
                    Ok(SemaValue::new($ty))
                }
            };
        }

        let res = match &mut expr.kind {
            ExprKind::Ident(text) => {
                if let Some(internal_ty) = Type::try_internal_ty(*text) {
                    let t = self.alloc(internal_ty)?;
                    return Ok(SemaValue::type_(t));
                }

                // TODO: remove this
                if &**text == "nil" {
                    return Ok(SemaValue::new(Type::Ptr { pointee_ty: Type::ptr_never() }));
                }

                match self.get_symbol(*text, expr.span)? {
                    SemaSymbol::Finished(v) if v.check_constness(is_const) => Ok(*v),
                    SemaSymbol::Finished(_) => err(SemaErrorKind::NotAConstExpr, span),
                    SemaSymbol::NotFinished(Some(ty)) if !is_const => Ok(SemaValue::new(*ty)),
                    SemaSymbol::NotFinished(_) => NotFinished,
                }
            },
            ExprKind::IntLit(code) => {
                let ty = match ty_hint {
                    Type::Never
                    | Type::Int { .. }
                    | Type::Float { .. }
                    | Type::IntLiteral
                    | Type::FloatLiteral => ty_hint,
                    _ => Type::IntLiteral,
                };
                lit_to_val!(ty, code.parse::<i128>().unwrap_debug())
            },
            ExprKind::FloatLit(code) => {
                let ty = match ty_hint {
                    Type::Never | Type::Float { .. } | Type::FloatLiteral => ty_hint,
                    _ => Type::FloatLiteral,
                };
                lit_to_val!(ty, code.parse::<f64>().unwrap_debug())
            },
            ExprKind::BoolLit(val) => lit_to_val!(Type::Bool, *val),
            ExprKind::CharLit(char) => lit_to_val!(Type::U8, *char), // TODO: change this to a real char type
            ExprKind::BCharLit(byte) => lit_to_val!(Type::U8, *byte),
            #[allow(unused_variables)]
            ExprKind::StrLit(_) => lit_to_val!(Type::str_slice(), {
                todo!();
                ()
            }),
            #[allow(unused_variables)]
            ExprKind::PtrTy { ty, is_mut } => {
                self.eval_type(ty)?;
                debug_assert!(ty.is_valid());
                let pointee_ty = self.alloc(*ty)?;
                self.alloc(Type::Ptr { pointee_ty }).map_ok(SemaValue::type_)
            },
            #[allow(unused_variables)]
            ExprKind::SliceTy { ty, is_mut } => {
                self.eval_type(ty)?;
                debug_assert!(ty.is_valid());
                let elem_ty = self.alloc(*ty)?;
                self.alloc(Type::Slice { elem_ty }).map_ok(SemaValue::type_)
            },
            ExprKind::ArrayTy { count, ty } => {
                let count_val = self.analyze(*count, Type::U64, true)?;
                let count = count_val.int().map_err(|kind| SemaError { kind, span: count.span })?;
                let count = count.try_into().expect("todo: array to long");
                self.eval_type(ty)?;
                let elem_ty = self.alloc(*ty)?;
                let arr_ty = Type::Array { len: count, elem_ty };
                self.alloc(arr_ty).map_ok(SemaValue::type_)
            },
            ExprKind::Fn(func) => {
                assert!(is_const, "todo: non-const function");
                let fn_ptr = Ptr::from(&*func);
                let Fn { mut params, ret_type, body } = func;
                self.eval_type(ret_type)?;
                self.function_stack.push(fn_ptr);
                self.open_scope();
                let res: SemaResult<SemaValue> = try {
                    for param in params.iter_mut() {
                        assert!(!param.is_const, "todo: const param");
                        self.analyze_var_decl(param)?;
                    }
                    match body {
                        Some(body) => {
                            let body_ty = self.analyze(*body, *ret_type, false)?.ty;
                            check_or_set_type!(ret_type, body_ty, body.span)?; // TODO: better span if `body` is a block
                            SemaValue::new_const(Type::Function(func.into()), EMPTY_PTR)
                        },
                        None => SemaValue::new(Type::Type(self.alloc(Type::Function(fn_ptr))?)),
                    }
                };
                debug_assert!(!res.is_ok() || func.ret_type.is_valid());
                self.close_scope()?;
                self.function_stack.pop();
                res
            },
            ExprKind::Parenthesis { expr } => self.analyze(*expr, ty_hint, is_const),
            ExprKind::Block { stmts, has_trailing_semicolon } => {
                self.open_scope();
                let res: SemaResult<SemaValue> = try {
                    let mut val = SemaValue::void();
                    let max_idx = stmts.len().wrapping_sub(1);
                    for (idx, s) in stmts.iter_mut().enumerate() {
                        let expected_ty = if max_idx == idx { ty_hint } else { Type::Unset };
                        match self.analyze(s.expr, expected_ty, is_const) {
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
                    let _f = self.var_decl_to_value(field)?;
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
                    let _f = self.var_decl_to_value(field)?;
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
                let lhs_val = self.analyze(*ty_expr, Type::Unset, false)?;
                *lhs_ty = lhs_val.ty;
                match *lhs_ty {
                    Type::Never => return Ok(SemaValue::never()),
                    Type::Type(t) => {
                        let Type::Struct { fields } = *t else { todo!("error") };
                        self.validate_call(&*fields, args.iter().copied(), is_const, span.end())?;
                        Ok(SemaValue::new(*t))
                    },
                    Type::Ptr { pointee_ty: t } => {
                        let Type::Struct { fields } = *t else { todo!("error") };
                        self.validate_call(&*fields, args.iter().copied(), is_const, span.end())?;
                        Ok(lhs_val)
                    },
                    ty => err(CannotApplyInitializer { ty }, ty_expr.full_span()),
                }
            },
            ExprKind::PositionalInitializer { lhs: None, lhs_ty, args } => match ty_hint {
                t @ Type::Struct { fields } => {
                    *lhs_ty = Type::Type(self.alloc(t)?);
                    self.validate_call(&*fields, args.iter().copied(), is_const, span.end())?;
                    Ok(SemaValue::new(t))
                },
                _ => err(CannotApplyInitializer { ty: ty_hint }, span.start_pos()),
            },
            ExprKind::NamedInitializer { lhs: Some(ty_expr), lhs_ty, fields: values } => {
                assert!(!is_const, "todo: const initializer");
                let lhs_val = self.analyze(*ty_expr, Type::Unset, false)?;
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
                let struct_ty = if let Type::Struct { .. } | Type::Slice { .. } = ty_hint {
                    self.alloc(ty_hint)?
                } else {
                    debug_assert_matches!(ty_hint, Type::Unset);
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
                    #[cfg(debug_assertions)]
                    println!("INFO: lookup type {}", *struct_ty);
                    struct_ty
                };
                *lhs_ty = Type::Type(struct_ty);
                let struct_ty = *struct_ty;
                self.validate_initializer(struct_ty, *values, is_const, span)?;
                Ok(SemaValue::new(struct_ty))
            },
            #[allow(unused_variables)]
            ExprKind::ArrayInitializer { lhs: Some(_), elements, .. } => todo!(),
            ExprKind::ArrayInitializer { lhs: None, elements, .. } => {
                let mut val_ty: Option<Type> = None;
                for elem in elements.iter() {
                    let val = self.analyze(*elem, Type::Unset, is_const)?;
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
                    Ok(SemaValue::new_const(ty, todo!()))
                } else {
                    Ok(SemaValue::new(ty))
                }
            },
            #[allow(unused_variables)]
            ExprKind::ArrayInitializerShort { lhs: Some(_), lhs_ty, val, count } => todo!(),
            #[allow(unused_variables)]
            ExprKind::ArrayInitializerShort { lhs: None, lhs_ty, val, count } => {
                let val = self.analyze(*val, Type::U64, is_const)?;
                let count_val = self.analyze(*count, Type::Unset, true)?;
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
                *lhs_ty = try_not_never!(self.analyze(*lhs, Type::Unset, is_const)?).ty;
                debug_assert!(lhs_ty.is_valid());
                // field access or enum variant
                match *lhs_ty {
                    Type::Struct { fields } | Type::Union { fields } => {
                        if let Some(field) = fields.iter().find(|f| *f.ident.text == *rhs.text) {
                            return Ok(SemaValue::new(field.ty));
                        }
                    },
                    Type::Type(t) if let Type::Enum { variants } = *t => {
                        if let Some((variant_idx, _)) =
                            variants.iter().enumerate().find(|(_, f)| *f.ident.text == *rhs.text)
                        {
                            return Ok(SemaValue::enum_variant(t, variant_idx));
                        }
                    },
                    Type::Slice { elem_ty } => match &*rhs.text {
                        "ptr" => return Ok(SemaValue::new(Type::Ptr { pointee_ty: elem_ty })),
                        "len" => return Ok(SemaValue::new(Type::U64)),
                        _ => {},
                    },
                    _ => {},
                }

                let function =
                    match self.get_symbol(rhs.text, rhs.span).and_then(SemaSymbol::get_type) {
                        Ok(Type::Function(function)) => function,
                        NotFinished => return NotFinished,
                        _ => return err(UnknownField { ty: *lhs_ty, field: rhs.text }, rhs.span),
                    };

                Ok(SemaValue::new(Type::MethodStub { function, first_expr: *lhs }))
            },
            ExprKind::Dot { lhs: None, lhs_ty, rhs } => {
                // `.<ident>` must be an enum
                let (enum_ty, idx) = if let Type::Enum { variants } = ty_hint {
                    let Some(idx) = variants.iter().position(|v| *v.ident.text == *rhs.text) else {
                        return err(UnknownField { ty: ty_hint, field: rhs.text }, rhs.span);
                    };
                    (self.alloc(ty_hint)?, idx)
                } else {
                    //debug_assert_matches!(ty_hint, Type::Unset);
                    // TODO: benchmark this
                    let mut enum_ty_iter =
                        self.enum_stack.iter().flat_map(|scope| scope.iter()).copied().filter_map(
                            |enum_ty| match *enum_ty {
                                Type::Enum { variants } => {
                                    let idx =
                                        variants.iter().position(|v| *v.ident.text == *rhs.text)?;
                                    Some((enum_ty, idx))
                                },
                                _ => None,
                            },
                        );
                    let Some((enum_ty, idx)) = enum_ty_iter.next() else {
                        return err(CannotInferNamedInitializerTy, expr.span.start_pos());
                    };
                    if enum_ty_iter.next().is_some() {
                        return err(MultiplePossibleInitializerTy, expr.span.start_pos());
                    }
                    drop(enum_ty_iter);
                    #[cfg(debug_assertions)]
                    println!("INFO: lookup type {}", *enum_ty);
                    (enum_ty, idx)
                };
                *lhs_ty = Type::Type(enum_ty);
                Ok(SemaValue::enum_variant(enum_ty, idx))
            },
            ExprKind::Index { lhs, idx } => {
                assert!(!is_const, "todo: const index");
                let arr = try_not_never!(self.analyze_typed(lhs, Type::Unset, is_const)?);
                let (Type::Array { elem_ty, .. } | Type::Slice { elem_ty }) = arr.ty else {
                    return err(CanOnlyIndexArrays, lhs.full_span());
                };
                let idx_val =
                    try_not_never!(self.analyze_typed(idx, Type::Unset, is_const)?).finalize_ty();
                idx.ty = idx_val.ty;
                Ok(SemaValue::new(match idx_val.ty {
                    Type::Int { .. } => *elem_ty,
                    Type::Range { elem_ty: i, kind }
                        if i.matches_int() || kind == RangeKind::Full =>
                    {
                        Type::Slice { elem_ty }
                    },
                    t => todo!("other idx ({t:#?})"),
                }))
            },
            ExprKind::Cast { lhs, target_ty } => {
                self.eval_type(target_ty)?;
                let _v = self.analyze_typed(lhs, *target_ty, false)?;
                // TODO: check if cast is possible
                Ok(SemaValue::new(*target_ty))
            },
            ExprKind::Call { func: f, args, .. } => {
                assert!(!is_const, "todo: const call");
                match self.analyze_typed(f, Type::Unset, is_const)?.ty {
                    Type::Never => Ok(SemaValue::never()),
                    Type::Function(mut func) => {
                        let Fn { params, ret_type, .. } = func.as_mut();
                        self.validate_call(&params, args.iter().copied(), is_const, span.end())?;
                        debug_assert!(ret_type.is_valid());
                        Ok(SemaValue::new(*ret_type))
                    },
                    Type::MethodStub { mut function, first_expr } => {
                        let Fn { params, ret_type, .. } = function.as_mut();
                        let args = std::iter::once(first_expr).chain(args.iter().copied());
                        self.validate_call(&params, args, is_const, span.end())?;
                        debug_assert!(ret_type.is_valid());
                        Ok(SemaValue::new(*ret_type))
                    },
                    Type::EnumVariant { enum_ty, idx } => {
                        let Type::Enum { variants } = *enum_ty else { unreachable_debug() };
                        let variant = variants[idx];
                        self.validate_call(&[variant], args.iter().copied(), is_const, span.end())?;
                        Ok(SemaValue::new(*enum_ty))
                    },
                    _ => err(CallOfANonFunction, span),
                }
            },
            &mut ExprKind::UnaryOp { kind, expr, .. } => {
                //assert!(!is_const, "todo: PreOp in const");
                let ty = self.analyze(expr, Type::Unset, is_const)?.ty;
                let out_ty = match (kind, ty) {
                    (_, Type::Unset | Type::Unevaluated(_)) => return err(CannotInfer, span),
                    (_, Type::Void) => None,
                    (_, Type::Never) => Some(ty),
                    //(UnaryOpKind::AddrOf | UnaryOpKind::AddrMutOf, Type::Function(_)) => todo!(),
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
                let lhs_ty = self.analyze(*lhs, Type::Unset, is_const)?.ty;
                debug_assert!(lhs_ty.is_valid());
                let rhs_ty = self.analyze(*rhs, lhs_ty, is_const)?.ty;
                debug_assert!(rhs_ty.is_valid());
                let Some(common_ty) = lhs_ty.common_type(rhs_ty) else {
                    return err(MismatchedTypesBinOp { lhs_ty, rhs_ty }, expr.span);
                };
                *arg_ty = common_ty;
                // todo: check if binop can be applied to type
                let ty = match op {
                    BinOpKind::Mul
                    | BinOpKind::Div
                    | BinOpKind::Mod
                    | BinOpKind::Add
                    | BinOpKind::Sub
                    | BinOpKind::ShiftL
                    | BinOpKind::ShiftR
                    | BinOpKind::BitAnd
                    | BinOpKind::BitXor
                    | BinOpKind::BitOr => common_ty,
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
                };
                Ok(SemaValue::new(ty))
            },
            ExprKind::Range { start, end, is_inclusive } => {
                let (elem_ty, kind) = match (start, end) {
                    (None, None) => (Type::ptr_u0(), RangeKind::Full),
                    (None, Some(end)) => {
                        let end_ty = self.analyze(*end, Type::Unset, is_const)?.ty;
                        debug_assert!(end_ty.is_valid());
                        let kind =
                            if *is_inclusive { RangeKind::ToInclusive } else { RangeKind::To };
                        (self.alloc(end_ty)?, kind)
                    },
                    (Some(start), None) => {
                        let start_ty = self.analyze(*start, Type::Unset, is_const)?.ty;
                        debug_assert!(start_ty.is_valid());
                        (self.alloc(start_ty)?, RangeKind::From)
                    },
                    (Some(start), Some(end)) => {
                        let start_ty = self.analyze(*start, Type::Unset, is_const)?.ty;
                        debug_assert!(start_ty.is_valid());
                        let end_ty = self.analyze(*end, start_ty, is_const)?.ty;
                        debug_assert!(end_ty.is_valid());
                        let kind =
                            if *is_inclusive { RangeKind::BothInclusive } else { RangeKind::Both };
                        (self.alloc(end_ty)?, kind)
                    },
                };
                Ok(SemaValue::new(Type::Range { elem_ty, kind }))
            },
            ExprKind::Assign { lhs, rhs, .. } => {
                if is_const {
                    return err(AssignToConst, span);
                }
                assert!(!is_const, "todo: Assign in const");
                let lhs_ty = self.analyze_typed(lhs, Type::Unset, is_const)?.ty;
                debug_assert!(lhs_ty.is_valid());
                let rhs_ty = self.analyze(*rhs, lhs_ty, is_const)?.ty;
                debug_assert!(rhs_ty.is_valid());
                if lhs_ty.matches(rhs_ty) {
                    // todo: check if binop can be applied to type
                    Ok(SemaValue::void())
                } else {
                    err(MismatchedTypes { expected: lhs_ty, got: rhs_ty }, rhs.full_span())
                }
            },
            #[allow(unused_variables)]
            ExprKind::BinOpAssign { lhs, op, rhs } => {
                assert!(!is_const, "todo: BinOpAssign in const");
                let lhs_ty = self.analyze_typed(lhs, Type::Unset, is_const)?.ty;
                debug_assert!(lhs_ty.is_valid());
                let rhs_ty = self.analyze(*rhs, lhs_ty, is_const)?.ty;
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
                let cond = self.analyze(*condition, Type::Bool, is_const)?.ty;
                debug_assert!(cond.is_valid());
                if cond != Type::Bool && cond != Type::Never {
                    return err(
                        MismatchedTypes { expected: Type::Bool, got: cond },
                        condition.full_span(),
                    );
                }
                let then_ty = self.analyze(*then_body, Type::Unset, is_const)?.ty;
                debug_assert!(then_ty.is_valid());
                if let Some(else_body) = else_body {
                    let else_ty = self.analyze(*else_body, then_ty, is_const)?.ty;
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
            ExprKind::Match { .. } => todo!(),
            ExprKind::For { source, iter_var, body, .. } => {
                let source = try_not_never!(self.analyze_typed(source, Type::Unset, is_const)?);
                let elem_ty = match source.ty.finalize() {
                    Type::Array { elem_ty, .. } | Type::Slice { elem_ty } => elem_ty,
                    Type::Range { elem_ty, kind } if elem_ty.matches_int() && kind.has_start() => {
                        elem_ty
                    },
                    _ => todo!("for over non-array"),
                };

                self.open_scope();
                self.analyze_var_decl(&mut VarDecl {
                    markers: DeclMarkers::default(),
                    ident: *iter_var,
                    ty: *elem_ty,
                    default: None,
                    is_const: false,
                })?;
                let body_ty = self.analyze(*body, Type::Void, is_const)?;
                if !matches!(body_ty.ty, Type::Void | Type::Never) {
                    return err(CannotReturnFromLoop, body.full_span());
                }
                self.close_scope()?;
                Ok(SemaValue::void())
            },
            ExprKind::While { condition, body, .. } => {
                let cond_ty =
                    try_not_never!(self.analyze(*condition, Type::Bool, is_const)?).ty.finalize();
                if cond_ty != Type::Bool {
                    let span = condition.full_span();
                    return err(MismatchedTypes { expected: Type::Bool, got: cond_ty }, span);
                }
                self.open_scope();
                let body_ty = self.analyze(*body, Type::Void, is_const)?;
                if !matches!(body_ty.ty, Type::Void | Type::Never) {
                    return err(CannotReturnFromLoop, body.full_span());
                }
                self.close_scope()?;
                //Ok(body_ty)
                Ok(SemaValue::void())
            },
            #[allow(unused_variables)]
            ExprKind::Catch { lhs } => todo!(),
            ExprKind::Autocast { expr } => {
                let _val = self.analyze_typed(expr, ty_hint, is_const)?;
                // TODO: check if cast is possible
                if ty_hint.is_valid() {
                    Ok(SemaValue::new(ty_hint))
                } else {
                    err(SemaErrorKind::CannotInferAutocastTy, expr.full_span())
                }
            },
            ExprKind::Defer(expr) => {
                self.defer_stack.push_expr(*expr);
                Ok(SemaValue::void())
            },
            ExprKind::Return { expr: val } => {
                let Some(mut func) = self.function_stack.last().copied() else {
                    return err(ReturnNotInAFunction, expr.full_span());
                };
                let ret_type = if let Some(val) = val {
                    self.analyze_typed(val, func.ret_type, is_const)?.ty
                } else {
                    Type::Void
                };
                check_or_set_type!(
                    &mut func.ret_type,
                    ret_type,
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
            let text = match &res {
                Ok(v) => format!("{}", v.ty),
                res => format!("{:?}", res),
            };
            display_span_in_code_with_label(expr.full_span(), self.code, format!("type: {text}"));
        }
        res
    }

    pub fn analyze_typed(
        &mut self,
        expr: &mut ExprWithTy,
        ty_hint: Type,
        is_const: bool,
    ) -> SemaResult<SemaValue> {
        let res = self.analyze(expr.expr, ty_hint, is_const);
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
                    Some(expr) => (self.analyze(*expr, f_decl.ty, is_const)?.ty, expr.full_span()),
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
            let mut init_val = self.analyze(*init, *ty, decl.is_const)?;
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
        args: impl IntoIterator<Item = Ptr<Expr>>,
        is_const: bool,
        close_p_span: Span,
    ) -> SemaResult<()> {
        let mut args = args.into_iter();
        // TODO: check for duplicate named arguments
        for p in params.iter() {
            let Some(arg) = args.next() else {
                if p.default.is_some() {
                    continue;
                }
                return err(MissingArg, close_p_span);
            };
            let arg_ty = self.analyze(arg, p.ty, is_const)?.ty;
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
            *ty = *match self.analyze(ty_expr, Type::Unset, true)?.ty {
                Type::Never => Type::ptr_never(),
                Type::Type(t) => t,
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
            self.analyze(*expr, Type::Unset, false)?;
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
