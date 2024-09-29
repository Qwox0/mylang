#![allow(unused_variables)]

use crate::{
    ast::{
        BinOpKind, DeclMarkers, Expr, ExprKind, Fn, Ident, LitKind, PreOpKind, VarDecl, VarDeclList,
    },
    defer_stack::DeferStack,
    parser::lexer::{Code, Span},
    ptr::Ptr,
    symbol_table::SymbolTable,
    type_::Type,
    util::{display_span_in_code_with_label, forget_lifetime, OkOrWithTry},
};
pub use err::{SemaError, SemaErrorKind, SemaResult};
use err::{SemaErrorKind::*, SemaResult::*};
use symbol::SemaSymbol;
use value::{SemaValue, EMPTY_PTR};

mod err;
mod symbol;
mod value;

/// This is a macro because `err_span` should only be evaluated when an error
/// happens.
/// ```rust
/// fn check_or_set_type(target: &mut Type, new_ty: Type, err_span: Span) -> SemaResult<()>
/// ```
macro_rules! check_or_set_type {
    ($target:expr, $new_ty:expr, $err_span:expr $(,)?) => {{
        let target: &mut Type = $target;
        let new_ty: Type = $new_ty;
        debug_assert!(new_ty.is_valid());
        //if *target == Type::Unset || *target == Type::Never {
        if *target == Type::Unset {
            *target = new_ty;
            Ok(())
        } else if Type::assign_matches(*target, new_ty) {
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
    function_stack: Vec<Ptr<Fn>>,
    defer_stack: DeferStack,

    debug_types: bool,
    pub errors: Vec<SemaError>,

    alloc: &'alloc bumpalo::Bump,
}

impl<'c, 'alloc> Sema<'c, 'alloc> {
    pub fn new(
        code: &'c Code,
        debug_types: bool,
        alloc: &'alloc bumpalo::Bump,
    ) -> Sema<'c, 'alloc> {
        Sema {
            code,
            symbols: SymbolTable::with_one_scope(),
            function_stack: vec![],
            defer_stack: DeferStack::default(),
            debug_types,
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

    /// Returns if `s` was fully analyzed
    pub fn analyze_top_level(&mut self, s: Ptr<Expr>) -> bool {
        match self.analyze(s, true) {
            Ok(_) => true,
            NotFinished => false,
            Err(e) => {
                self.errors.push(e);
                true
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
            ExprKind::Ident(text) => match self.get_symbol(*text, expr.span)? {
                SemaSymbol::Finished(v) => v.into_const_checked(is_const, span),
                SemaSymbol::NotFinished(Some(ty)) if !is_const => Ok(SemaValue::new(*ty)),
                SemaSymbol::NotFinished(_) => NotFinished,
            },
            &mut ExprKind::Literal { kind, code } => {
                let ty = match kind {
                    // TODO: better literal handling
                    LitKind::Char => todo!(),
                    LitKind::BChar => todo!(),

                    // use this for the `benches::bench_frontend1` benchmark
                    // LitKind::Int => Type::Float { bits: 64 },
                    // LitKind::Float => Type::Float { bits: 64 },

                    // LitKind::Int => Type::Int { bits: 64, is_signed: true },
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
            ExprKind::ArrayTy { count, ty } => todo!(),
            ExprKind::ArrayTy2 { ty } => todo!(),
            ExprKind::ArrayLit { elements } => {
                let mut val_ty = None;
                for elem in elements.iter() {
                    let val = self.analyze(*elem, is_const)?;
                    match val_ty {
                        None => val_ty = Some(val.ty),
                        Some(ty) if !ty.matches(val.ty) => {
                            return err(
                                MismatchedTypes { expected: ty, got: val.ty },
                                elem.full_span(),
                            );
                        },
                        Some(_) => {},
                    }
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
                if !matches!(count_val.ty, Type::Int { .. } | Type::IntLiteral) {
                    return err(ExpectedNumber { got: count_val.ty }, count.full_span());
                }
                let Some(count_val) = count_val.const_val else {
                    return err(NotAConstExpr, count.full_span());
                };
                let count_val = *count_val.cast::<i128>();
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
                    for s in stmts.iter() {
                        match self.analyze(*s, is_const) {
                            Ok(new_val) => {
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
                if !is_const {
                    todo!()
                }
                for field in fields.iter_mut() {
                    let f = self.var_decl_to_value(field)?;
                    if f.is_const() {
                        todo!()
                    }
                }
                Ok(SemaValue::new_const(Type::Struct { fields: *fields }, EMPTY_PTR))
            },
            ExprKind::UnionDef(_) => todo!(),
            ExprKind::EnumDef {} => todo!(),
            ExprKind::OptionShort(_) => todo!(),
            ExprKind::Ptr { is_mut, ty } => todo!(),
            ExprKind::Initializer { lhs, fields: values } => {
                assert!(!is_const, "todo: const initializer");
                if let Some(ty_expr) = lhs {
                    let ty = self.eval_type_expr(*ty_expr)?;
                    let fields = match ty {
                        Type::Never => return Ok(SemaValue::never()),
                        Type::Struct { fields } => fields,
                        _ => return err(CannotApplyInitializer { ty }, ty_expr.full_span()),
                    };
                    self.validate_named_initializer(ty, fields, *values, is_const, expr.span)?;
                    Ok(SemaValue::new(ty))
                } else {
                    todo!("todo: initializer type")
                }
            },
            ExprKind::Dot { lhs, rhs } => {
                assert!(!is_const, "todo: const dot");
                let lhs_ty = self.analyze(*lhs, is_const)?.ty;
                debug_assert!(lhs_ty.is_valid());
                if let Type::Struct { fields } | Type::Union { fields } = lhs_ty
                    && let Some(f) = fields.iter().find(|f| *f.ident.text == *rhs.text)
                {
                    Ok(SemaValue::new(f.ty))
                } else {
                    err(UnknownField { ty: lhs_ty, field: rhs.text }, rhs.span)
                }
            },
            ExprKind::PostOp { expr, kind } => todo!(),
            ExprKind::Index { lhs, idx } => {
                assert!(!is_const, "todo: const index");
                let arr = try_not_never!(self.analyze(*lhs, is_const)?);
                let Type::Array { len, elem_ty } = arr.ty else {
                    return err(CanOnlyIndexArrays, lhs.full_span());
                };
                let idx = try_not_never!(self.analyze(*idx, is_const)?);
                assert!(
                    matches!(idx.ty, Type::Int { .. } | Type::IntLiteral),
                    "todo: non-int index"
                );
                Ok(SemaValue::new(*elem_ty))
            },
            ExprKind::Call { func: fn_expr, args } => {
                assert!(!is_const, "todo: const call");
                // self.analyze(*fn_expr)?;
                // debug_assert!(fn_expr.ty.is_valid());
                let Result::Ok(func) = fn_expr.try_to_ident() else {
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
            &mut ExprKind::PreOp { kind, expr, .. } => {
                assert!(!is_const, "todo: PreOp in const");
                let ty = self.analyze(expr, is_const)?.ty;
                let is_valid = match (kind, ty) {
                    (_, Type::Unset | Type::Unevaluated(_)) => return err(CannotInfer, span),
                    (_, Type::Void) => false,
                    (_, Type::Never) => true,
                    (
                        PreOpKind::AddrOf | PreOpKind::AddrMutOf,
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
                    ) => true,
                    (PreOpKind::AddrOf | PreOpKind::AddrMutOf, Type::Function(_)) => todo!(),
                    (PreOpKind::Deref, Type::Ptr(_)) => true,
                    (
                        PreOpKind::Deref,
                        Type::Int { .. }
                        | Type::IntLiteral
                        | Type::Bool
                        | Type::Float { .. }
                        | Type::FloatLiteral
                        | Type::Function(_)
                        | Type::Array { .. },
                    ) => false,
                    (PreOpKind::Not, Type::Int { .. } | Type::IntLiteral | Type::Bool) => true,
                    (
                        PreOpKind::Not,
                        Type::Ptr(_)
                        | Type::Float { .. }
                        | Type::FloatLiteral
                        | Type::Function(_)
                        | Type::Array { .. },
                    ) => false,
                    (
                        PreOpKind::Neg,
                        Type::Int { .. }
                        | Type::IntLiteral
                        | Type::Float { .. }
                        | Type::FloatLiteral,
                    ) => true,
                    (
                        PreOpKind::Neg,
                        Type::Ptr(_) | Type::Bool | Type::Function(_) | Type::Array { .. },
                    ) => false,
                    (
                        _,
                        Type::Struct { .. }
                        | Type::Union { .. }
                        | Type::Enum { .. }
                        | Type::Type(_),
                    ) => todo!(),
                };
                if is_valid {
                    Ok(SemaValue::new(ty))
                } else {
                    err(InvalidPreOp { ty, kind }, span)
                }
            },
            ExprKind::BinOp { lhs, op, rhs } => {
                assert!(!is_const, "todo: BinOp in const");
                let lhs_ty = self.analyze(*lhs, is_const)?.ty;
                debug_assert!(lhs_ty.is_valid());
                let rhs_ty = self.analyze(*rhs, is_const)?.ty;
                debug_assert!(rhs_ty.is_valid());
                if lhs_ty.matches(rhs_ty) {
                    // todo: check if binop can be applied to type
                    Ok(match op {
                        BinOpKind::Mul
                        | BinOpKind::Div
                        | BinOpKind::Mod
                        | BinOpKind::Add
                        | BinOpKind::Sub => lhs_ty,
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
                        | BinOpKind::Ge => Type::Bool,
                        BinOpKind::And | BinOpKind::Or => {
                            if lhs_ty == Type::Bool {
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
                let lhs_ty = self.analyze(*lhs, is_const)?.ty;
                debug_assert!(lhs_ty.is_valid());
                let rhs_ty = self.analyze(*rhs, is_const)?.ty;
                debug_assert!(rhs_ty.is_valid());
                if lhs_ty.matches(rhs_ty) {
                    // todo: check if binop can be applied to type
                    Ok(SemaValue::void())
                } else {
                    err(MismatchedTypes { expected: lhs_ty, got: rhs_ty }, rhs.span)
                }
            },
            ExprKind::BinOpAssign { lhs, op, rhs } => {
                assert!(!is_const, "todo: BinOpAssign in const");
                let lhs_ty = self.analyze(*lhs, is_const)?.ty;
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
            ExprKind::If { condition, then_body, else_body } => {
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
            ExprKind::Match { val, else_body } => todo!(),
            ExprKind::For { source, iter_var, body } => {
                let arr = try_not_never!(self.analyze(*source, is_const)?);
                let Type::Array { len: count, elem_ty } = arr.ty else {
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
                Ok(body_ty)
            },
            ExprKind::While { condition, body } => todo!(),
            ExprKind::Catch { lhs } => todo!(),
            ExprKind::Pipe { lhs } => todo!(),
            ExprKind::Defer(expr) => {
                self.defer_stack.push_expr(*expr);
                Ok(SemaValue::void())
            },
            ExprKind::Return { expr: val } => {
                let ret_type = if let Some(val) = val {
                    let t = self.analyze(*val, is_const)?.ty;
                    debug_assert!(t.is_valid());
                    t
                } else {
                    Type::Void
                };
                let Some(func) = self.function_stack.last_mut() else {
                    return err(ReturnNotInAFunction, expr.full_span());
                };
                check_or_set_type!(
                    &mut func.ret_type,
                    ret_type,
                    val.map(|v| v.full_span()).unwrap_or(expr.full_span()),
                )?;
                Ok(SemaValue::never())
            },
            ExprKind::Semicolon(_) => todo!(),
        };
        if let Ok(val) = res {
            expr.ty = val.ty;
        }
        if self.debug_types {
            display_span_in_code_with_label(expr.full_span(), self.code, format!("type: {res:?}")); // TODO: remove this
        }
        res
    }

    /// Returns the [`SemaValue`] repesenting the new variable, not the entire
    /// declaration. This also doesn't insert into `self.symbols`.
    fn var_decl_to_value(&mut self, decl: &mut VarDecl) -> SemaResult<SemaValue> {
        let ty = &mut decl.ty;
        self.eval_type(ty)?;
        if let Some(init) = &mut decl.default {
            let init_val = self.analyze(*init, decl.is_const)?;
            check_or_set_type!(ty, init_val.ty, init.span)?;
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
            if p.ty != arg_ty {
                return err(MismatchedTypes { expected: p.ty, got: arg_ty }, arg.span);
            }
        }
        Ok(())
    }

    /// ... { param1=arg1, param2, ... }
    /// `param2` = `params2=param2`
    /// all unset params -> default values
    ///
    /// this function gracefully stores all errors in `self.errors`.
    fn validate_named_initializer(
        &mut self,
        ty: Type,
        fields: VarDeclList,
        values: Ptr<[(Ident, Option<Ptr<Expr>>)]>,
        is_const: bool,
        initializer_span: Span,
    ) -> SemaResult<SemaValue> {
        assert!(!is_const, "todo: const named initializer");
        debug_assert!(matches!(ty, Type::Struct { fields: f } if f == fields));
        let mut is_initialized_field = vec![false; fields.len()];
        for (f, init) in values.iter() {
            match try {
                let field = f.text;
                let Some((f_idx, f_decl)) = fields.find_field(&*field) else {
                    err(UnknownField { ty, field }, f.span)?
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
                if !Type::assign_matches(f_decl.ty, init_ty) {
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
        Ok(SemaValue::new(ty))
    }

    fn eval_type_expr(&mut self, mut ty_expr: Ptr<Expr>) -> SemaResult<Type> {
        let res = match &mut ty_expr.kind {
            &mut ExprKind::Ident(code) => match &*code {
                "f64" => Ok(Type::Float { bits: 64 }),
                "i64" => Ok(Type::Int { bits: 64, is_signed: true }),
                "u64" => Ok(Type::Int { bits: 64, is_signed: false }),
                //_ => Ok(Type::Custom(code)),
                name => match self.get_symbol(code, ty_expr.full_span())? {
                    SemaSymbol::Finished(SemaValue { ty, const_val }) => {
                        if matches!(
                            ty,
                            Type::Never
                                | Type::Struct { .. }
                                | Type::Union { .. }
                                | Type::Enum { .. }
                                | Type::Type(_)
                        ) {
                            Ok(*ty)
                        } else {
                            err(NotAType, ty_expr.span)
                        }
                    },
                    SemaSymbol::NotFinished(_) => NotFinished,
                },
            },
            ExprKind::ArrayTy { count, ty } => {
                let count_val = self.analyze(*count, true)?;
                if !matches!(count_val.ty, Type::Int { .. } | Type::IntLiteral) {
                    return err(ExpectedNumber { got: count_val.ty }, count.span);
                }
                let Some(val) = count_val.const_val else {
                    panic!("`analyze` should have returned a const value")
                };
                let count = *val.cast();
                let ty = self.eval_type_expr(*ty)?;
                let ty = self.alloc(ty)?;
                Ok(Type::Array { len: count, elem_ty: ty })
            },
            ExprKind::ArrayTy2 { ty } => todo!("[]ty"),
            ExprKind::Ptr { is_mut, ty } => {
                let pointee = self.eval_type(ty)?;
                Ok(Type::Ptr(self.alloc(pointee)?)) // TODO: no alloc here
            },

            kind => todo!("{kind:?}"),
        };
        if let Ok(ty) = res {
            ty_expr.ty = Type::Type(self.alloc(ty)?);
        }
        res
    }

    /// [`Type::Unevaluated`] -> [`Type`]
    fn eval_type(&mut self, ty: &mut Type) -> SemaResult<Type> {
        if let Type::Unevaluated(ty_expr) = *ty {
            *ty = self.eval_type_expr(ty_expr)?;
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
