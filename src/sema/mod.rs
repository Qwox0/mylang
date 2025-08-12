//! # Semantic analysis module
//!
//! Semantic analysis validates (and changes) all stored [`ast::Type`]s in [`Expr`].

use crate::{
    ast::{
        self, Ast, AstEnum, AstKind, AstMatch, BinOpKind, DeclListExt, DeclMarkers, OptionTypeExt,
        TypeEnum, UnaryOpKind, UpcastToAst, ast_new, is_pos_arg, type_new,
    },
    context::{CompilationContextInner, ctx_mut, primitives as p},
    diagnostics::{
        DiagnosticReporter, HandledErr, InitializerKind, cerror, cerror2, chint, cinfo,
        cunimplemented, cwarn,
    },
    display_code::display,
    intern_pool::Symbol,
    parser::lexer::Span,
    ptr::{OPtr, Ptr},
    scope::{Scope, ScopePos},
    scoped_stack::ScopedStack,
    type_::{RangeKind, common_type, ty_match},
    util::{self, UnwrapDebug, then, unreachable_debug},
};
pub(crate) use err::{SemaError, SemaErrorKind, SemaResult};
use err::{SemaErrorKind::*, SemaResult::*};
use std::{fmt::Write, iter};

mod err;
pub mod primitives;

/// This is a macro because `err_span` should only be evaluated when an error happens.
/// ```ignore
/// fn check_or_set_type(ty: Ptr<ast::Type>, target_ty: &mut OPtr<ast::Type>, err_span: Span) -> SemaResult<()>
/// ```
macro_rules! check_and_set_target {
    ($ty:expr, $target_ty:expr, $err_span:expr $(,)?) => {{
        let ty: Ptr<ast::Type> = $ty;
        let target_ty: &mut OPtr<ast::Type> = $target_ty;
        let common_ty = if let Some(target_ty) = target_ty {
            match common_type(ty, *target_ty) {
                Some(t) => t,
                None => {
                    ctx_mut().error_mismatched_types($err_span, *target_ty, ty);
                    return SemaResult::HandledErr;
                },
            }
        } else {
            ty
        };
        *target_ty = Some(common_ty);
        target_ty.as_mut().u()
    }};
}

pub fn analyze(cctx: Ptr<CompilationContextInner>, stmts: &mut [Ptr<Ast>]) {
    let p = p();
    // validate top level stmts
    cctx.root_scope.as_mut().verify_no_duplicates();
    for file in cctx.files.iter().copied() {
        file.as_mut().scope.as_mut().u().verify_no_duplicates();
        for &s in stmts[file.stmt_range.u()].iter() {
            let Some(decl) = s.try_downcast::<ast::Decl>() else {
                if !s.kind.is_allowed_top_level() {
                    cerror!(s.full_span(), "unexpected top level expression");
                    s.set_replacement(p.never.upcast());
                    s.as_mut().ty = Some(p.never);
                }
                continue;
            };
            debug_assert!(!decl.is_const || decl.init.is_some());
            if !decl.is_const && !decl.markers.get(DeclMarkers::IS_STATIC_MASK) {
                cerror!(
                    decl.ident.span,
                    "Global variables must be marked as const (`{0} :: ...`) or static (`static \
                     {0} := ...`)",
                    decl.ident.sym
                );
                continue;
            }
            debug_assert!(decl.ty.is_none());
        }
    }

    let mut sema = Sema::new(cctx);
    let mut remaining_count = usize::MAX;
    while remaining_count > 0 {
        let old_remaining_count = remaining_count;
        remaining_count = 0;
        for file in cctx.files.iter().copied() {
            debug_assert!(file.scope.as_ref().u().parent.is_some());
            sema.open_scope(file.as_mut().scope.as_mut().u());
            let stmt_range = file.as_mut().stmt_range.as_mut().u();
            if let NotFinished { remaining } =
                analyze_scope(&mut stmts[..stmt_range.end], &mut stmt_range.start, |stmt| {
                    sema.analyze_top_level(stmt)
                })
            {
                //remaining_count += stmt_range.end - stmt_range.start;
                remaining_count += remaining
            }
            debug_assert!(
                sema.defer_stack.get_cur_scope().is_empty(),
                "file scope must not contain defer statements"
            );
            sema.close_scope();
        }
        // println!("finished statements: {:?}", finished);
        if remaining_count == old_remaining_count {
            cerror!(Span::ZERO, "cycle(s) detected:");
            for file in cctx.files.iter().copied() {
                for s in stmts[file.stmt_range.u()].iter().copied() {
                    let span = s
                        .try_downcast::<ast::Decl>()
                        .map(|d| d.ident.span)
                        .unwrap_or_else(|| s.full_span());
                    display(span).finish();
                }
            }
            break;
        }
    }
}

/// 0 1 2 3 4 5 6 7 8 9
/// . . x|.
/// 0 1 3 2 4 5 6 7 8 9
/// . . . x|.
/// 0 1 3 4 2 5 6 7 8 9
/// . . . . x|x x .
/// 0 1 3 4 7 5 6 2 8 9
/// . . . . . x x x|.
/// 0 1 3 4 7 8 6 2 5 9
/// . . . . . . x x x|
fn analyze_scope<T>(
    items: &mut [Ptr<T>],
    finished_count: &mut usize,
    mut analyze_item: impl FnMut(Ptr<T>) -> SemaResult<()>,
) -> SemaResult<(), HandledErr> {
    debug_assert!(*finished_count <= items.len());
    let mut ok = true;
    let mut remaining = 0;
    // index-based loop because `items` is mutated during the loop. In theory an iterator should
    // also work because only previous/already consumed items are swapped.
    let mut idx = *finished_count;
    while let Some(i) = items.get(idx).copied() {
        let res = analyze_item(i);
        if !matches!(res, SemaResult::NotFinished { .. }) {
            items.swap(idx, *finished_count);
            *finished_count += 1;
        }
        match res {
            Ok(()) => {},
            NotFinished { remaining: rem } => remaining += rem + 1,
            Err(e) => {
                ctx_mut().error2(&e);
                ok = false;
            },
        }
        idx += 1;
    }
    match ok {
        false => Err(HandledErr),
        true if remaining > 0 => NotFinished { remaining },
        true => Ok(()),
    }
}

/// Semantic analyzer
pub struct Sema {
    function_stack: Vec<Ptr<ast::Fn>>,
    decl_stack: Vec<Ptr<ast::Decl>>,
    defer_stack: ScopedStack<Ptr<Ast>>,

    cctx: Ptr<CompilationContextInner>,
    cur_scope: Ptr<Scope>,
    cur_scope_pos: ScopePos,
}

impl Sema {
    pub fn new(cctx: Ptr<CompilationContextInner>) -> Sema {
        Sema {
            function_stack: vec![],
            decl_stack: vec![],
            defer_stack: ScopedStack::default(),
            cctx,
            cur_scope: cctx.root_scope,
            cur_scope_pos: ScopePos(0),
        }
    }

    pub fn analyze_top_level(&mut self, s: Ptr<Ast>) -> SemaResult<()> {
        self.analyze(s, &Some(p().void_ty), true)
    }

    pub fn analyze(
        &mut self,
        expr: Ptr<Ast>,
        ty_hint: &OPtr<ast::Type>,
        is_const: bool,
    ) -> SemaResult<()> {
        let res = self._analyze_inner(expr, ty_hint, is_const);
        #[cfg(debug_assertions)]
        if self.cctx.args.debug_types {
            let label = match &res {
                Ok(()) => format!("type: {}", expr.ty.u()),
                NotFinished { remaining } => format!("not finished (remaining: {remaining})"),
                Err(e) => format!("err: {}", e.kind),
            };

            display(expr.full_span()).label(&label).finish();
        }
        #[cfg(debug_assertions)]
        if res.is_ok() {
            debug_assert!(expr.ty.is_some());
        }
        res
    }

    /// This modifies the [`Ast`] behind `expr` to ensure that codegen is possible.
    #[inline]
    fn _analyze_inner(
        &mut self,
        mut expr: Ptr<Ast>,
        ty_hint: &OPtr<ast::Type>, // reference to handle changing pointers (see crate::tests::function::specialize_return_type)
        is_const: bool,
    ) -> SemaResult<()> {
        // println!("analyze {:x?}: {:?} {:?}", expr, expr.kind, ast::debug::DebugAst::to_text(&expr));
        let span = expr.span;

        if expr.replacement.is_some() {
            expr.ty = Some(expr.rep().ty.u());
            // Does this work in general? Is this needed if NotFinished is removed?
            return Ok(());
        }

        let p = p();

        /// Like [`Sema::analyze`] but returns on error and never
        macro_rules! analyze {
            ($expr:expr, $ty_hint:expr) => {
                analyze!($expr, $ty_hint, is_const)
            };
            ($expr:expr, $ty_hint:expr, $is_const:expr) => {{
                let e: Ptr<Ast> = $expr;
                self.analyze(e, &$ty_hint, $is_const)?;
                if e.ty == p.never {
                    expr.as_mut().ty = Some(p.never);
                    return Ok(());
                }
                e.as_mut().ty.as_mut().u()
            }};
        }

        /// This exists because I want to use [`analyze!`]
        macro_rules! analyze_array_initializer_lhs {
            ($lhs:expr, $init_count:expr $(,)?) => {{
                let lhs: OPtr<Ast> = $lhs;
                if let Some(lhs) = lhs {
                    analyze!(lhs, None);
                }
                let lhs = lhs.or_else(|| {
                    // Here we can't set `expr.ty` to `ty_hint` because the there might be a length mismatch
                    ty_hint.try_downcast::<ast::ArrayTy>().map(|arr_ty| arr_ty.elem_ty)
                });

                if let Some(lhs) = lhs {
                    Some(if let Some(elem_ty) = lhs.try_downcast_type() {
                        elem_ty
                    } else if let Some(ptr_ty) = lhs.ty.try_downcast::<ast::PtrTy>()
                        && let Some(arr_ty) = ptr_ty.pointee.try_downcast::<ast::ArrayTy>()
                    {
                        if self.cctx.do_mut_checks && !ptr_ty.is_mut {
                            self.cctx.error_mutate_const_ptr(expr.full_span(), lhs);
                            return SemaResult::HandledErr;
                        }
                        expr.ty = Some(ptr_ty.upcast_to_type());
                        let count: usize = $init_count;
                        if count != arr_ty.len.int() {
                            return cerror2!(
                                expr.full_span(),
                                "Cannot initialize the array behind the pointer `{}` with {count} \
                                 items",
                                ptr_ty.upcast_to_type(),
                            );
                        }
                        arr_ty.elem_ty.downcast_type()
                    } else {
                        // TODO: also allow lhs slices?
                        self.cctx.error_cannot_apply_initializer(InitializerKind::Array, lhs);
                        return SemaResult::HandledErr;
                    })
                } else {
                    None
                }
            }};
        }

        match expr.matchable().as_mut() {
            AstEnum::Ident { sym, decl, span, .. } => match self.get_symbol(*sym) {
                None if let Some(mut i) = self.try_custom_bitwith_int_type(sym.text()) => {
                    i.span = expr.span;
                    i.ty = Some(p.type_ty);
                    const { debug_assert!(size_of::<ast::Ident>() >= size_of::<ast::IntTy>()) };
                    *expr.cast::<ast::IntTy>().as_mut() = i;
                    // decl is not set. Is this a problem?
                },
                None => return cerror2!(expr.span, "unknown identifier `{sym}`"),
                Some(sym) => {
                    let var_ty = self.get_symbol_var_ty(sym)?;
                    *decl = Some(sym);
                    expr.ty = Some(var_ty);
                    if sym.is_const {
                        if sym.init.is_some_and(|i| i.kind == AstKind::ExternDirective) {
                            return cerror2!(
                                *span,
                                "the use of extern symbols in constants is currently not \
                                 implemented"
                            );
                        };
                        expr.set_replacement(sym.const_val());
                    } else if is_const {
                        return cerror2!(
                            *span,
                            "cannot access a non-constant symbol at compile time"
                        );
                    }
                },
            },
            AstEnum::Block { stmts, has_trailing_semicolon, scope, .. } => {
                scope.verify_no_duplicates();
                self.open_scope(scope);
                let res: SemaResult<()> = try {
                    let max_idx = stmts.len().wrapping_sub(1);
                    for (idx, s) in stmts.iter().enumerate() {
                        let expected_ty = if max_idx == idx { ty_hint } else { &None };
                        match self.analyze(*s, expected_ty, false) {
                            Ok(()) => {
                                // s.ty = s.ty.finalize();
                                debug_assert!(s.ty.is_some());
                            },
                            NotFinished { remaining } => {
                                NotFinished::<!, !> { remaining: remaining + stmts.len() - idx }?
                            },
                            Err(err) => {
                                self.cctx.error2(&err);
                                s.as_mut().ty = Some(p.never);
                            },
                        }
                    }
                };
                self.close_scope();
                res?;
                let last_ty = stmts.last().map(|s| s.ty.u()).unwrap_or(p.void_ty);
                expr.ty = Some(if !*has_trailing_semicolon || last_ty == p.never {
                    last_ty
                } else {
                    p.void_ty
                })
            },
            AstEnum::PositionalInitializer { lhs, args, .. } => {
                let lhs = if let Some(lhs) = *lhs {
                    analyze!(lhs, None, false);
                    lhs
                } else if let Some(s_def) = ty_hint.try_downcast::<ast::StructDef>() {
                    s_def.upcast()
                } else {
                    return self.cctx.error_cannot_infer_initializer_lhs(expr).into();
                };
                if let Some(s) = lhs.try_downcast::<ast::StructDef>() {
                    // allow slices?
                    self.validate_call(&s.fields, args, span.end(), is_const)?;
                    expr.ty = Some(s.upcast_to_type());

                    if is_const {
                        let all_args = s
                            .fields
                            .iter()
                            .enumerate()
                            .map(|(idx, f)| args.get(idx).or(f.init).u());
                        let cv = self.create_aggregate_const_val(all_args)?;
                        expr.set_replacement(cv.upcast());
                    }
                } else if let Some(ptr_ty) = lhs.ty.try_downcast::<ast::PtrTy>()
                    && let Some(s) = ptr_ty.pointee.try_downcast::<ast::StructDef>()
                {
                    if self.cctx.do_mut_checks && !ptr_ty.is_mut {
                        self.cctx.error_mutate_const_ptr(expr.full_span(), lhs);
                        return SemaResult::HandledErr;
                    }
                    expr.ty = Some(ptr_ty.upcast_to_type());

                    if is_const {
                        return self.cctx.error_const_ptr_initializer(expr).into();
                    } else {
                        self.validate_call(&s.fields, args, span.end(), false)?;
                    }
                } else {
                    self.cctx.error_cannot_apply_initializer(InitializerKind::Positional, lhs);
                    return SemaResult::HandledErr;
                };
            },
            AstEnum::NamedInitializer { lhs, fields: values, .. } => {
                let lhs = if let Some(lhs) = *lhs {
                    analyze!(lhs, None, false);
                    lhs
                } else if let Some(s_def) = ty_hint.filter(|t| t.kind.is_struct_kind()) {
                    s_def.upcast()
                } else {
                    return self.cctx.error_cannot_infer_initializer_lhs(expr).into();
                };
                if let Some(struct_ty) = lhs.try_downcast_type()
                    && struct_ty.kind.is_struct_kind()
                {
                    let const_values =
                        self.validate_initializer(struct_ty, *values, is_const, span)?;
                    debug_assert_eq!(const_values.is_some(), is_const);
                    expr.ty = Some(struct_ty);
                    if let Some(elements) = const_values {
                        let val = ast_new!(AggregateVal { elements }, expr.span);
                        expr.set_replacement(val.upcast());
                    }
                } else if let Some(ptr_ty) = lhs.ty.try_downcast::<ast::PtrTy>()
                    && let Some(struct_ty) = ptr_ty.pointee.try_downcast_type()
                    && struct_ty.kind.is_struct_kind()
                {
                    if self.cctx.do_mut_checks && !ptr_ty.is_mut {
                        self.cctx.error_mutate_const_ptr(expr.full_span(), lhs);
                        return SemaResult::HandledErr;
                    }
                    if is_const {
                        return self.cctx.error_const_ptr_initializer(expr).into();
                    } else {
                        self.validate_initializer(struct_ty, *values, is_const, span)?;
                        expr.ty = Some(ptr_ty.upcast_to_type());
                    }
                } else {
                    self.cctx.error_cannot_apply_initializer(InitializerKind::Named, lhs);
                    return SemaResult::HandledErr;
                };
            },
            AstEnum::ArrayInitializer { lhs, elements, .. } => {
                let mut elem_iter = elements.iter().copied();

                let elem_ty = analyze_array_initializer_lhs!(*lhs, elements.len());
                let mut elem_ty = if let Some(elem_ty) = elem_ty {
                    elem_ty
                } else {
                    let Some(elem) = elem_iter.next() else {
                        expr.ty = Some(p.empty_array_ty.upcast_to_type()); // `.[]`
                        return Ok(());
                    };
                    *analyze!(elem, None)
                };
                #[cfg(debug_assertions)]
                let orig_elem_ty = elem_ty;

                for elem in elem_iter {
                    self.analyze(elem, &Some(elem_ty), is_const)?;
                    let ty = elem.ty.u();
                    let Some(common_ty) = common_type(ty, elem_ty) else {
                        self.cctx.error_mismatched_types(elem.full_span(), elem_ty, ty);
                        return SemaResult::HandledErr;
                    };
                    elem_ty = common_ty;
                }

                #[cfg(debug_assertions)]
                debug_assert!(lhs.is_none() || elem_ty == orig_elem_ty);

                if lhs.is_none() || expr.ty.is_none() {
                    let len = elements.len() as i64;
                    let len = ast_new!(IntVal { val: len }, Span::ZERO).upcast();
                    let arr_ty = type_new!(ArrayTy { len, elem_ty: elem_ty.upcast() });
                    //debug_assert!(expr.ty.is_none());
                    expr.ty = Some(arr_ty.upcast_to_type());
                }

                if is_const {
                    let cv = self.create_aggregate_const_val(*elements)?;
                    expr.set_replacement(cv.upcast());
                }
            },
            AstEnum::ArrayInitializerShort { lhs, val, count, .. } => {
                analyze!(*count, Some(p.u64), true);
                self.ty_match(*count, p.u64)?;
                let len = count
                    .try_downcast_const_val()
                    .ok_or_else(|| {
                        self.cctx
                            .error_non_const(*count, "Array length must be known at compile time")
                    })?
                    .upcast();

                let mut elem_ty = analyze_array_initializer_lhs!(*lhs, len.int());

                let val_ty = *analyze!(*val, elem_ty);
                let elem_ty =
                    check_and_set_target!(val_ty, &mut elem_ty, val.return_val_span()).upcast();
                let arr_ty = type_new!(ArrayTy { len, elem_ty });
                expr.ty = Some(arr_ty.upcast_to_type());

                if is_const {
                    let len = len.int();
                    let cv = self.create_aggregate_const_val(iter::repeat_n(*val, len))?;
                    expr.set_replacement(cv.upcast());
                }
            },
            AstEnum::Dot { has_lhs: true, lhs: Some(lhs), rhs, .. } => {
                let lhs_ty = *analyze!(*lhs, None);
                let t = if lhs_ty == p.module {
                    let m = lhs.downcast::<ast::ImportDirective>();
                    let Some(s) = self.cctx.files[m.files_idx]
                        .scope
                        .as_ref()
                        .u()
                        .find_decl_norec_unordered(rhs.sym, true)
                    else {
                        return cerror2!(
                            rhs.span,
                            "Cannot find symbol `{}` in module `{}`",
                            rhs.sym,
                            m.path.text.as_ref(),
                        );
                    };
                    rhs.decl = Some(s);
                    let ty = self.get_symbol_var_ty(s)?;
                    if let Some(cv) = s.try_const_val() {
                        expr.set_replacement(cv);
                    }
                    ty
                } else if lhs_ty == p.type_ty
                    && let Some(enum_ty) = lhs.try_downcast::<ast::EnumDef>()
                    && let Some((v_idx, _)) = enum_ty.variants.find_field(rhs.sym)
                {
                    self.analyze_enum_tag(expr, enum_ty, v_idx, is_const)?
                } else if lhs_ty == p.type_ty
                    && let TypeEnum::StructDef { consts, .. }
                    | TypeEnum::UnionDef { consts, .. }
                    | TypeEnum::EnumDef { consts, .. } = lhs.downcast_type().matchable().as_ref()
                {
                    // associated consts/methods
                    let Some((_, field)) = consts.find_field(rhs.sym) else {
                        cinfo!(rhs.span, "This associated const might be undefined");
                        // The definition of an associated const might be after it's first use (or
                        // even in another file).
                        //
                        // TODO: better handling of missing fields (currently only cycle detection
                        // is triggered)
                        return NotFinished { remaining: 2 };
                    };
                    let ty = self.get_symbol_var_ty(field)?;
                    debug_assert!(field.is_const);
                    expr.set_replacement(field.const_val());
                    ty
                } else if let TypeEnum::StructDef { fields, .. } | TypeEnum::UnionDef { fields, .. } =
                    lhs_ty.flatten_transparent().matchable().as_ref()
                    && let Some((f_idx, field)) = fields.find_field(rhs.sym)
                {
                    debug_assert!(!field.is_const);
                    // field access
                    if lhs_ty.kind == AstKind::PtrTy {
                        return cerror2!(
                            lhs.full_span(),
                            "automatic dereferencing of pointers is currently not allowed"
                        );
                    }
                    let ty = self.get_symbol_var_ty(field)?;
                    if is_const {
                        let Some(cv) = lhs.try_downcast_const_val() else {
                            return cerror2!(
                                lhs.full_span(),
                                "Cannot access a field of a non-constant value in a constant \
                                 context"
                            );
                        };
                        let const_field =
                            cv.downcast::<ast::AggregateVal>().elements.get(f_idx).u();
                        expr.set_replacement(const_field.upcast());
                    }
                    ty
                } else if let TypeEnum::StructDef { consts, .. }
                | TypeEnum::UnionDef { consts, .. }
                | TypeEnum::EnumDef { consts, .. } =
                    lhs_ty.flatten_transparent().matchable().as_ref()
                    && let Some((_, method)) = consts.find_field(rhs.sym)
                {
                    debug_assert!(method.is_const);
                    // method access
                    let method_ty = self.get_symbol_var_ty(method)?;
                    rhs.ty = Some(method_ty);
                    rhs.upcast().set_replacement(method.const_val());
                    if method_ty.try_downcast::<ast::Fn>().is_some() {
                        p.method_stub
                    } else if method_ty == p.never {
                        p.never
                    } else {
                        cerror!(
                            expr.full_span(),
                            "cannot access a static constant through a value"
                        );
                        chint!(
                            lhs.full_span(),
                            "consider replacing the value with its type '{}'",
                            lhs_ty // TODO: only show this hint iff lhs_ty has a name
                        );
                        return SemaResult::HandledErr;
                    }
                } else if let TypeEnum::SliceTy { elem_ty, is_mut, .. } = *lhs_ty.matchable()
                    && rhs.sym == p.ptr_sym
                {
                    // TODO: remove this allocation (test if cast SliceTy -> PointerTy is valid)
                    type_new!(PtrTy { pointee: elem_ty, is_mut }).upcast_to_type()
                } else if lhs_ty.kind == AstKind::SliceTy && rhs.sym == p.len_sym {
                    p.u64
                } else {
                    if rhs.replacement.is_some() {
                        debug_assert!(expr.ty.is_some());
                        return Ok(());
                    }
                    let mut ty = None;
                    // method-like call:
                    // TODO?: maybe change syntax to `arg~my_fn(...)`. using `.` both for method
                    //        calls and method-like calls might be confusing
                    if let Some(s) = self.get_symbol(rhs.sym) {
                        let var_ty = self.get_symbol_var_ty(s)?;
                        if let Some(f) = var_ty.try_downcast::<ast::Fn>()
                            && let Some(first_param) = f.params().get(0)
                            && ty_match(lhs.ty.u(), first_param.var_ty.u())
                        {
                            debug_assert!(s.is_const);
                            rhs.upcast().set_replacement(f.upcast());
                            ty = Some(p.method_stub);
                        } else if var_ty == p.never {
                            ty = Some(p.never);
                        }
                    }
                    ty.ok_or_else(|| self.cctx.error_unknown_field(*rhs, lhs_ty))?
                };
                expr.ty = Some(t);
            },
            AstEnum::Dot { has_lhs: true, lhs: None, .. } => unreachable_debug(),
            AstEnum::Dot { has_lhs: false, lhs, rhs, .. } => {
                if lhs.is_some() {
                    // TODO(without `NotFinished`): make this an assert
                    debug_assert!(expr.ty.is_some());
                    return Ok(());
                }
                // `.<ident>` must be an enum
                let Some(enum_ty) = ty_hint.try_downcast::<ast::EnumDef>() else {
                    if *ty_hint == p.never {
                        expr.ty = Some(p.never);
                        return Ok(());
                    }
                    return err(CannotInfer, expr.full_span());
                };
                let Some((v_idx, _)) = enum_ty.variants.find_field(rhs.sym) else {
                    return self.cctx.error_unknown_variant(*rhs, enum_ty.upcast_to_type()).into();
                };
                *lhs = Some(enum_ty.upcast());
                expr.ty = Some(self.analyze_enum_tag(expr, enum_ty, v_idx, is_const)?)
            },
            AstEnum::Index { mut_access, lhs, idx, .. } => {
                let lhs_ty = analyze!(*lhs, None);
                let (TypeEnum::SliceTy { elem_ty, .. } | TypeEnum::ArrayTy { elem_ty, .. }) =
                    lhs_ty.matchable().as_ref()
                else {
                    cerror!(lhs.full_span(), "cannot index into value of type {}", lhs_ty);
                    return SemaResult::HandledErr;
                };
                let elem_ty = elem_ty.downcast_type();
                let idx_ty = analyze!(*idx, None).finalize();

                match idx_ty.matchable().as_ref() {
                    TypeEnum::RangeTy { elem_ty: i, rkind, .. }
                        if i.matches_int() || *rkind == RangeKind::Full =>
                    {
                        self.validate_addr_of(*mut_access, *lhs, expr)?;
                        let slice_ty =
                            type_new!(SliceTy { elem_ty: elem_ty.upcast(), is_mut: *mut_access });
                        expr.ty = Some(slice_ty.upcast_to_type());

                        if is_const {
                            cunimplemented!(expr.full_span(), "const slicing");
                        }
                    },
                    _ if *mut_access => {
                        cerror!(
                            expr.span,
                            "The `mut` marker can only be used when slicing, not when indexing"
                        );
                        chint!(expr.span, "to reference the value mutably, use `.&mut` instead");
                        return SemaResult::HandledErr;
                    },
                    TypeEnum::IntTy { .. } => {
                        expr.ty = Some(elem_ty);

                        if is_const {
                            if lhs_ty.kind == AstKind::SliceTy {
                                cunimplemented!(
                                    expr.full_span(),
                                    "indexing into slice at compile time"
                                )
                            }
                            let arr = lhs.downcast::<ast::AggregateVal>();
                            let idx = idx.int::<usize>();
                            if idx >= arr.elements.len() {
                                return cerror2!(
                                    expr.full_span(),
                                    "index out of bounds: the length is {} but the index is {idx}",
                                    arr.elements.len(),
                                );
                            }
                            expr.set_replacement(arr.elements[idx].upcast());
                        }
                    },
                    _ => {
                        cerror!(idx.full_span(), "Cannot index into array with `{idx_ty}`");
                        return SemaResult::HandledErr;
                    },
                }
            },
            AstEnum::Cast { operand, target_ty, .. } => {
                let ty = self.analyze_type(*target_ty)?;
                analyze!(*operand, Some(ty));
                // TODO: check if cast is possible
                expr.ty = Some(ty);
            },
            AstEnum::Autocast { operand, .. } => {
                if ty_hint.is_some() {
                    analyze!(*operand, ty_hint);
                    // TODO: check if cast is possible
                    expr.ty = *ty_hint;
                } else {
                    return err(SemaErrorKind::CannotInferAutocastTy, expr.full_span());
                }
            },
            AstEnum::Call { func, args, .. } => {
                let call = expr.downcast::<ast::Call>();
                let ty_hint =
                    ty_hint.filter(|t| t.kind == AstKind::EnumDef && func.kind == AstKind::Dot); // I hope this doesn't conflict with `method_stub`
                let fn_ty = *analyze!(*func, ty_hint);
                expr.ty = Some(if let Some(f) = fn_ty.try_downcast::<ast::Fn>() {
                    if is_const {
                        return self.cctx.error_const_call(call).into();
                    }
                    self.validate_call(&f.params(), args, span.end(), false)?;

                    debug_assert!(is_finished_or_recursive(f, self));
                    f.ret_ty.unwrap_or(p.unknown_ty)
                } else if fn_ty == p.method_stub {
                    if is_const {
                        return self.cctx.error_const_call(call).into();
                    }
                    let dot = func.downcast::<ast::Dot>().as_mut();
                    let fn_ty = dot.rhs.upcast().downcast::<ast::Fn>();
                    let args = std::iter::once(dot.lhs.u())
                        .chain(args.iter().copied())
                        .collect::<Vec<_>>(); // TODO: bench no allocation
                    self.validate_call(&fn_ty.params(), &args, span.end(), false)?;

                    debug_assert!(is_finished_or_recursive(fn_ty, self));
                    fn_ty.ret_ty.unwrap_or(p.unknown_ty)
                } else if fn_ty == p.enum_variant {
                    let dot = func.flat_downcast::<ast::Dot>();
                    let enum_ty = dot.lhs.u().downcast::<ast::EnumDef>();
                    let variant = enum_ty.variants.find_field(dot.rhs.sym).u().1;
                    self.validate_call(&[variant], args, span.end(), is_const)?;

                    if is_const {
                        let tag = dot.upcast().downcast::<ast::IntVal>().upcast();
                        debug_assert_eq!(args.len(), 1);
                        let data = args.get(0).u();
                        let cv = self.create_aggregate_const_val([tag, data])?;
                        data.rep().as_mut().ty = Some(variant.var_ty.u());
                        expr.set_replacement(cv.upcast());
                    }
                    enum_ty.upcast_to_type()
                } else {
                    cerror!(
                        func.full_span(),
                        "Cannot call value of type '{}'; expected function",
                        fn_ty
                    );
                    return SemaResult::HandledErr;
                });
            },
            AstEnum::UnaryOp { op, operand, .. } => {
                let expr_ty;

                macro_rules! simple_unary_op {
                    ($op:tt $ast_node:ident) => {{
                        if is_const {
                            let const_val = operand.downcast_const_val();
                            let val = const_val.downcast::<ast::$ast_node>().val;
                            debug_assert!(expr.replacement.is_none());
                            // TODO: no allocation
                            expr.replacement = Some(self.alloc(ast_new!($ast_node {
                                val: $op val,
                                span: expr.full_span(),
                            }))?.upcast());
                        }
                        expr_ty
                    }};
                }

                let err = |operand_ty| {
                    cerror!(
                        expr.full_span(),
                        "Cannot apply unary operator `{op}` to type `{operand_ty}`"
                    );
                    SemaResult::HandledErr
                };

                expr.ty = Some(match *op {
                    UnaryOpKind::AddrOf | UnaryOpKind::AddrMutOf => {
                        expr_ty = *analyze!(*operand, None);
                        let is_mut = *op == UnaryOpKind::AddrMutOf;
                        self.validate_addr_of(is_mut, *operand, expr)?;
                        let pointee = expr_ty.upcast();
                        if is_const {
                            return cerror2!(
                                expr.full_span(),
                                "The \"address of\" operator is currently not supported at \
                                 compile time"
                            );
                        }
                        type_new!(PtrTy { pointee, is_mut }).upcast_to_type()
                    },
                    UnaryOpKind::Deref => {
                        expr_ty = *analyze!(*operand, None);
                        let Some(ptr_ty) = expr_ty.try_downcast::<ast::PtrTy>() else {
                            cerror!(
                                expr.full_span(),
                                "Cannot dereference value of type `{expr_ty}`",
                            );
                            return SemaResult::HandledErr;
                        };
                        if is_const {
                            return cerror2!(
                                expr.full_span(),
                                "The \"dereference\" operator is currently not supported at \
                                 compile time"
                            );
                        }
                        ptr_ty.pointee.downcast_type()
                    },
                    UnaryOpKind::Not => {
                        expr_ty = *analyze!(*operand, ty_hint);
                        if expr_ty == p.bool {
                            simple_unary_op!(!BoolVal)
                        } else if expr_ty.kind == AstKind::IntTy || expr_ty.is_int_lit() {
                            simple_unary_op!(!IntVal)
                        } else {
                            return err(expr_ty);
                        }
                    },
                    UnaryOpKind::Neg => {
                        expr_ty = *analyze!(*operand, ty_hint);
                        if expr_ty.is_sint() {
                            simple_unary_op!(-IntVal)
                        } else if expr_ty.is_int_lit() {
                            simple_unary_op!(-IntVal);
                            p.sint_lit
                        } else if expr_ty.kind == AstKind::FloatTy || expr_ty == p.float_lit {
                            simple_unary_op!(-FloatVal)
                        } else {
                            return err(expr_ty);
                        }
                    },
                    UnaryOpKind::Try => todo!("try"),
                });
            },
            AstEnum::BinOp { lhs, op, rhs, .. } => {
                let lhs_ty = *analyze!(*lhs, None);
                let rhs_ty = *analyze!(*rhs, Some(lhs_ty));
                let Some(mut common_ty) = common_type(rhs_ty, lhs_ty) else {
                    return err(MismatchedTypesBinOp { lhs_ty, rhs_ty }, expr.span);
                };
                // todo: check if binop can be applied to type
                expr.ty = Some(match op {
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
                        common_ty.finalize();
                        lhs.ty = Some(common_ty);
                        rhs.ty = Some(common_ty);
                        p.bool
                    },
                    BinOpKind::And | BinOpKind::Or => {
                        if common_ty == p.bool {
                            p.bool
                        } else {
                            todo!("err")
                        }
                    },
                });
                if is_const {
                    let lhs = lhs.try_downcast_const_val().ok_or_else(|| {
                        self.cctx.error_non_const(
                            *lhs,
                            "Left-hand side of compile time operation not known at compile time",
                        )
                    })?;
                    let rhs = rhs.try_downcast_const_val().ok_or_else(|| {
                        self.cctx.error_non_const(
                            *rhs,
                            "Right-hand side of compile time operation not known at compile time",
                        )
                    })?;

                    macro_rules! calc_num_binop {
                        ($op:tt $(, allow $float_val:ident)?) => {
                            if common_ty.kind == AstKind::IntTy || common_ty.is_int_lit() {
                                let val = lhs.downcast::<ast::IntVal>().val
                                    $op rhs.downcast::<ast::IntVal>().val;
                                Some(self.alloc(ast_new!(IntVal { val, span: expr.full_span() }))?
                                    .upcast())
                            } $( else if common_ty.kind == AstKind::FloatTy || common_ty == p.float_lit {
                                let val = lhs.float_val() $op rhs.float_val();
                                Some(self.alloc(ast_new!($float_val { val, span: expr.full_span() }))?
                                    .upcast())
                            })? else {
                                None
                            }
                        };
                    }

                    let Some(out_val) = (match op {
                        BinOpKind::Mul => calc_num_binop!(*, allow FloatVal),
                        BinOpKind::Div => calc_num_binop!(/, allow FloatVal),
                        BinOpKind::Mod => calc_num_binop!(%, allow FloatVal),
                        BinOpKind::Add => calc_num_binop!(+, allow FloatVal),
                        BinOpKind::Sub => calc_num_binop!(-, allow FloatVal),
                        BinOpKind::ShiftL => calc_num_binop!(<<),
                        BinOpKind::ShiftR => calc_num_binop!(>>),
                        BinOpKind::BitAnd => calc_num_binop!(&),
                        BinOpKind::BitXor => calc_num_binop!(^),
                        BinOpKind::BitOr => calc_num_binop!(|),
                        BinOpKind::Eq => todo!(),
                        BinOpKind::Ne => todo!(),
                        BinOpKind::Lt => todo!(),
                        BinOpKind::Le => todo!(),
                        BinOpKind::Gt => todo!(),
                        BinOpKind::Ge => todo!(),
                        BinOpKind::And => todo!(),
                        BinOpKind::Or => todo!(),
                    }) else {
                        return cerror2!(
                            expr.full_span(),
                            "unimplemented compiletime binary operation {op:?}"
                        );
                    };
                    out_val.as_mut().ty = expr.ty;
                    expr.replacement = Some(out_val);
                }
            },
            AstEnum::Range { start, end, is_inclusive, .. } => {
                let (elem_ty, rkind): (Ptr<ast::Type>, _) = match (start, end) {
                    (None, None) => {
                        expr.ty = Some(p.full_range);
                        return Ok(());
                    },
                    (None, Some(end)) => (
                        *analyze!(*end, None),
                        if *is_inclusive { RangeKind::ToInclusive } else { RangeKind::To },
                    ),
                    (Some(start), None) => (*analyze!(*start, None), RangeKind::From),
                    (Some(start), Some(end)) => {
                        let kind =
                            if *is_inclusive { RangeKind::BothInclusive } else { RangeKind::Both };
                        let start_ty = *analyze!(*start, None);
                        let end_ty = *analyze!(*end, Some(start_ty));
                        let Some(common_ty) = common_type(start_ty, end_ty) else {
                            return err(
                                MismatchedTypesBinOp { lhs_ty: start_ty, rhs_ty: end_ty },
                                expr.span,
                            );
                        };
                        (common_ty, kind)
                    },
                };
                expr.ty = Some(type_new!(RangeTy { elem_ty, rkind }).upcast_to_type());
            },
            AstEnum::Assign { lhs, rhs, .. } => {
                assert!(!is_const, "todo: Assign in const");
                let lhs_ty = *analyze!(*lhs, None);
                analyze!(*rhs, Some(lhs_ty));
                // todo: check if binop can be applied to type
                self.ty_match(*rhs, lhs_ty)?;
                self.validate_lvalue(*lhs, expr)?;
                expr.ty = Some(p.void_ty);
            },
            AstEnum::BinOpAssign { lhs, rhs, op, .. } => {
                assert!(!is_const, "todo: BinOpAssign in const");
                let lhs_ty = *analyze!(*lhs, None);
                let rhs_ty = *analyze!(*rhs, Some(lhs_ty));
                if let Some(ptr_ty) = lhs_ty.try_downcast::<ast::PtrTy>() {
                    cerror!(
                        lhs.full_span(),
                        "Cannot apply binary operatator `{}` to pointer type `{lhs_ty}`",
                        op.to_binop_assign_text()
                    );
                    if ty_match(rhs_ty, ptr_ty.pointee.downcast_type()) {
                        chint!(
                            lhs.full_span(),
                            "Consider dereferencing the pointer first", // TODO: code example
                        )
                    }
                    return SemaResult::HandledErr;
                }
                // TODO: check if binop can be applied to type
                self.ty_match(*rhs, lhs_ty)?;
                self.validate_lvalue(*lhs, expr)?;
                expr.ty = Some(p.void_ty);
            },
            AstEnum::Decl { on_type, .. } => {
                self.analyze_decl(expr.downcast::<ast::Decl>(), is_const, None)?;
                if on_type.is_none() {
                    self.cur_scope_pos.inc();
                }
            },
            AstEnum::If { condition, then_body, else_body, .. } => {
                assert!(!is_const, "todo: if in const");
                let bool_ty = p.bool;
                analyze!(*condition, Some(bool_ty));
                self.ty_match(*condition, bool_ty)?;

                self.analyze(*then_body, ty_hint, is_const)?;
                let then_ty = then_body.ty.u();
                expr.ty = if let Some(else_body) = *else_body {
                    self.analyze(else_body, &Some(then_ty), is_const)?;
                    let else_ty = else_body.ty.u();
                    let Some(common_ty) = common_type(then_ty, else_ty) else {
                        return cerror2!(
                            else_body.full_span(),
                            "'then' and 'else' branches have incompatible types"
                        );
                    };
                    Some(common_ty)
                } else if then_body.can_ignore_yielded_value() {
                    Some(p.void_ty)
                } else {
                    return cerror2!(
                        then_body.return_val_span(),
                        "Cannot yield a value from this `if` because it doesn't have an `else` \
                         branch."
                    );
                }
            },
            AstEnum::Match { .. } => todo!(),
            AstEnum::For { source, iter_var, body, scope, .. } => {
                scope.verify_no_duplicates();

                analyze!(*source, None).finalize();
                let elem_ty = match source.ty.matchable().as_ref() {
                    TypeEnum::ArrayTy { elem_ty, .. } | TypeEnum::SliceTy { elem_ty, .. } => {
                        elem_ty.downcast_const_val().downcast_type()
                    },
                    TypeEnum::RangeTy { elem_ty, rkind, .. }
                        if elem_ty.matches_int() && rkind.has_start() =>
                    {
                        *elem_ty
                    },
                    _ => {
                        return cerror2!(
                            source.full_span(),
                            "cannot iterate over value of type `{}`",
                            source.ty.u()
                        );
                    },
                };

                self.open_scope(scope);
                let res = (|| {
                    iter_var.var_ty = Some(elem_ty);
                    self.analyze_decl(*iter_var, false, None)?;

                    self.analyze(*body, &Some(p.void_ty), is_const)?;
                    if !body.can_ignore_yielded_value() {
                        self.cctx.error_cannot_yield_from_loop_block(body.return_val_span());
                        return SemaResult::HandledErr;
                    }
                    Ok(())
                })();
                self.close_scope();
                res?;
                expr.ty = Some(p.void_ty);
            },
            AstEnum::While { condition, body, .. } => {
                let bool_ty = p.bool;
                analyze!(*condition, Some(bool_ty));
                self.ty_match(*condition, bool_ty)?;

                //self.open_scope(); // currently not needed
                let res: SemaResult<()> = try {
                    self.analyze(*body, &Some(p.void_ty), is_const)?; // TODO: check if scope is closed on `NotFinished?`
                    if !body.can_ignore_yielded_value() {
                        self.cctx.error_cannot_yield_from_loop_block(body.return_val_span());
                        return SemaResult::HandledErr;
                    }
                };
                //self.close_scope();
                res?;
                expr.ty = Some(p.void_ty);
            },
            // AstEnum::Catch { .. } => todo!(),
            AstEnum::Defer { stmt, .. } => {
                self.analyze(*stmt, &None, false)?;
                self.defer_stack.push(*stmt);
                expr.ty = Some(p.void_ty);
            },
            AstEnum::Return { val, parent_fn, .. } => {
                let Some(mut func) = self.function_stack.last().copied() else {
                    return err(ReturnNotInAFunction, expr.full_span());
                };
                *parent_fn = Some(func);
                expr.ty = Some(p.never);
                let val_ty = if let Some(val) = *val {
                    self.analyze(val, &func.ret_ty, is_const)?;
                    val.ty.u()
                } else {
                    p.void_ty
                };
                check_and_set_target!(
                    val_ty,
                    &mut func.ret_ty,
                    val.map(|v| v.full_span()).unwrap_or(span)
                );
            },
            AstEnum::Break { val, .. } => {
                if val.is_some() {
                    todo!("break with value")
                }
                // TODO: check if in loop
                expr.ty = Some(p.never)
            },
            AstEnum::Continue { .. } => {
                // TODO: check if in loop
                expr.ty = Some(p.never)
            },
            AstEnum::Empty { .. } => {
                expr.ty = Some(p.void_ty);
            },
            AstEnum::ImportDirective { .. } => {
                expr.ty = Some(p.module);
            },
            AstEnum::ExternDirective { decl, .. } => {
                let ty = self.analyze_extern_directive(expr, decl, *ty_hint)?;
                if ty.kind != AstKind::Fn && decl.u().is_const {
                    // TODO: is this correct?
                    return cerror2!(
                        decl.u().lhs_span(),
                        "`#extern` declaration must be a function or marked as `static`"
                    );
                }
                expr.ty = Some(ty);
            },
            AstEnum::IntrinsicDirective { intrinsic_name, decl, .. } => {
                let ty = self.analyze_extern_directive(expr, decl, *ty_hint)?;
                if !intrinsic_name.text.starts_with("llvm.") {
                    return cerror2!(
                        intrinsic_name.span,
                        "Currently only llvm directives are supported. This directive name does \
                         not start with \"llvm.\""
                    );
                }
                let Some(f) = ty.try_downcast::<ast::Fn>() else {
                    // Do non-function intrinsics exist?
                    self.cctx.error_mismatched_types(
                        ty.upcast().full_span(),
                        "function type of intrinsic",
                        ty,
                    );
                    return SemaResult::HandledErr;
                };
                expr.ty = Some(f.upcast_to_type())
            },
            AstEnum::ProgramMainDirective { .. } => {
                if self.cctx.args.is_lib {
                    return cerror2!(
                        expr.full_span(),
                        "The program entry point is not available when compiling with `--lib`."
                    );
                }
                debug_assert!(!self.cctx.args.is_lib);
                let entry_point_sym = self.cctx.entry_point;
                let root_file = self.cctx.files[self.cctx.root_file_idx.u()];
                let Some(main) =
                    root_file.scope.as_ref().u().find_decl_norec_unordered(entry_point_sym, true)
                else {
                    return cerror2!(
                        root_file.full_span().start(),
                        "Couldn't find the entry point '{entry_point_sym}' in '{}'",
                        self.cctx.path_in_proj(&root_file.path).display()
                    );
                };
                let main_ty = self.get_symbol_var_ty(main)?;
                if main_ty != p.never {
                    let Some(main_fn) = main_ty.try_downcast::<ast::Fn>() else {
                        return cerror2!(
                            main.ident.span,
                            "Expected the entry point to be a function"
                        );
                    };
                    let main_ret_ty = main_fn.ret_ty.u();
                    if main_ret_ty != p.void_ty
                        && main_ret_ty != p.never
                        // not handled in `runtime.mylang`:
                        && main_ret_ty.kind != AstKind::IntTy
                    {
                        return cerror2!(
                            main.ident.span,
                            "Entry point '{}' has invalid return type `{}`",
                            entry_point_sym,
                            main_ret_ty,
                        );
                    }
                    expr.set_replacement(main.const_val());
                }
                expr.ty = Some(main_ty);
            },
            AstEnum::SimpleDirective { ret_ty, .. } => {
                expr.ty = Some(*ret_ty);
            },

            AstEnum::IntVal { val, .. } => {
                expr.ty = Some(match ty_hint {
                    Some(t) if matches!(t.kind, AstKind::IntTy | AstKind::FloatTy) => *t,
                    _ if val.is_negative() => p.sint_lit,
                    _ => p.int_lit,
                });
            },
            AstEnum::FloatVal { .. } => {
                let ty = ty_hint.filter(|t| t.kind == AstKind::FloatTy).unwrap_or(p.float_lit);
                expr.ty = Some(ty);
            },
            AstEnum::BoolVal { .. } => expr.ty = Some(p.bool),
            AstEnum::CharVal { .. } => expr.ty = Some(p.u8), // TODO: use `p.char`
            /*
            AstEnum::BCharLit { .. } => {
                expr.downcast::<ast::BCharLit>().reinterpret_as_const();
                finish_ret!(ast::Type::U8)
            },
            */
            AstEnum::StrVal { .. } => expr.ty = Some(p.str_slice_ty),
            AstEnum::PtrVal { .. } => todo!(),
            AstEnum::AggregateVal { .. } => todo!(),
            AstEnum::Fn { params_scope, ret_ty_expr, ret_ty, body, .. } => {
                params_scope.verify_no_duplicates();

                let fn_ptr = expr.downcast::<ast::Fn>();
                let fn_hint = ty_hint.and_then(|t| t.try_downcast::<ast::Fn>());

                let mut is_fn_ty = *ty_hint == p.type_ty;

                self.function_stack.push(fn_ptr);
                self.open_scope(params_scope);
                let res: SemaResult<()> = try {
                    for (p_idx, param) in params_scope.decls.iter().enumerate() {
                        if param.is_const {
                            return cerror2!(
                                param.lhs_span(),
                                "constant function parameters are currently not implemented"
                            );
                        }
                        if param.var_ty_expr.is_none()
                            && let Some(fn_hint) = fn_hint
                            && let Some(p_hint) = fn_hint.params().get(p_idx)
                        {
                            //debug_assert!(param.var_ty.is_none()); // TODO: without NotFinished
                            param.as_mut().var_ty = Some(p_hint.var_ty.u());
                        }
                        self.analyze_decl(
                            *param,
                            false,
                            then!(is_fn_ty => VarDeclSpecialCase::FnTyParam),
                        )?;
                    }

                    let mut has_known_ret = false; // needed because of `NotFinished`. `ret_ty` might get set in a previous try
                    if let Some(ret) = *ret_ty_expr {
                        *ret_ty = Some(self.analyze_type(ret)?);
                        has_known_ret = true;
                    } else if let Some(fn_hint) = fn_hint {
                        *ret_ty = Some(fn_hint.ret_ty.u());
                        has_known_ret = true;
                    }

                    if !is_fn_ty && has_known_ret {
                        // Needed for indirect recursion. For direct recursion without an explicit
                        // return type, see `get_symbol_var_ty`
                        self.allow_unfinished_use(expr, fn_ptr.upcast_to_type());
                    }

                    if let Some(body) = *body {
                        self.analyze(body, ret_ty, false)?;
                        check_and_set_target!(body.ty.u(), ret_ty, body.return_val_span())
                            .finalize();
                    } else {
                        debug_assert!(is_fn_ty);
                        //panic_debug!("this function has already been analyzed as a function type")
                    }
                };
                self.close_scope();
                self.function_stack.pop();
                res?;
                debug_assert!(ret_ty.is_some());

                if *ret_ty == p.unknown_ty {
                    let rec_fn_decl = self
                        .decl_stack
                        .last()
                        .filter(|d| d.init.is_some_and(|i| i.rep().p_eq(fn_ptr)));
                    return cerror2!(
                        rec_fn_decl.map(|d| d.ident.span).unwrap_or_else(|| expr.full_span()),
                        "cannot infer the return type of this recursive function"
                    );
                }

                is_fn_ty = is_fn_ty || (*ret_ty == p.type_ty && body.u().kind != AstKind::Block);
                expr.ty = Some(if is_fn_ty {
                    let Some(b) = *body else { return Ok(()) }; // only needed for `NotFinished`
                    *ret_ty_expr = Some(b);
                    *ret_ty = Some(b.downcast_type());
                    *body = None;
                    p.type_ty
                } else {
                    fn_ptr.upcast_to_type()
                });
            },

            AstEnum::SimpleTy { .. } | AstEnum::IntTy { .. } | AstEnum::FloatTy { .. } => {
                expr.ty = Some(p.type_ty)
            },
            AstEnum::PtrTy { pointee, .. } => {
                self.analyze_type(*pointee)?;
                expr.ty = Some(p.type_ty);
            },
            AstEnum::SliceTy { elem_ty, .. } => {
                self.analyze_type(*elem_ty)?;
                expr.ty = Some(p.type_ty);
            },
            AstEnum::ArrayTy { len, elem_ty, .. } => {
                self.analyze_type(*elem_ty)?;
                let u64_ty = p.u64;
                analyze!(*len, Some(u64_ty), true);
                self.ty_match(*len, u64_ty)?;
                if !len.rep().is_const_val() {
                    return cerror2!(
                        len.full_span(),
                        "cannot evaluate the array length at compile time"
                    );
                }
                expr.ty = Some(p.type_ty);
            },
            AstEnum::StructDef { scope, finished_members, consts, .. } => {
                scope.verify_no_duplicates();
                self.allow_unfinished_use(expr, p.type_ty);
                self.open_scope(scope);
                let res = analyze_scope(scope.decls.as_mut(), finished_members, |member| {
                    debug_assert!(member.is_const == consts.contains(&member));
                    self.var_decl_to_value(member, Some(VarDeclSpecialCase::Field))
                });
                self.close_scope();
                res?;
                debug_assert_eq!(expr.ty, p.type_ty);
            },
            AstEnum::UnionDef { scope, finished_members, consts, .. } => {
                scope.verify_no_duplicates();
                self.allow_unfinished_use(expr, p.type_ty);
                self.open_scope(scope);
                let res = analyze_scope(scope.decls.as_mut(), finished_members, |member| {
                    debug_assert!(member.is_const == consts.contains(&member));
                    if !member.is_const
                        && let Some(d) = member.init
                    {
                        return cerror2!(d.full_span(), "union fields cannot have default values");
                    }
                    self.var_decl_to_value(member, Some(VarDeclSpecialCase::Field))
                });
                self.close_scope();
                res?;
                debug_assert_eq!(expr.ty, p.type_ty);
            },
            AstEnum::EnumDef {
                scope,
                variants,
                finished_members,
                variant_tags,
                is_simple_enum,
                tag_ty,
                ..
            } => {
                let mut repr_ty = Some(p.int_lit);
                let mut used_tags = self.cctx.alloc.alloc_uninit_slice(variants.len())?;
                let mut tag = 0;
                *is_simple_enum = true;

                scope.verify_no_duplicates();
                self.allow_unfinished_use(expr, p.type_ty);
                self.open_scope(scope);
                let mut idx = 0;
                let res = analyze_scope(scope.decls.as_mut(), finished_members, |member| {
                    if member.is_const {
                        return Ok(());
                    }
                    if member.var_ty_expr.is_none() {
                        member.as_mut().var_ty = Some(p.void_ty);
                    }

                    let variant_idx = member.as_mut().init.take();
                    let _ = self.var_decl_to_value(member, Some(VarDeclSpecialCase::Field))?;
                    member.as_mut().init = variant_idx;
                    if member.var_ty != p.void_ty {
                        *is_simple_enum = false;
                    }
                    if let Some(variant_idx) = variant_idx {
                        self.analyze(variant_idx, &repr_ty, true)?;
                        check_and_set_target!(
                            variant_idx.ty.u(),
                            &mut repr_ty,
                            variant_idx.full_span()
                        );
                        tag = variant_idx.int();
                    }

                    const ALLOW_DUPLICATE_TAG: bool = true;

                    // TODO: replace linear search?
                    if !ALLOW_DUPLICATE_TAG
                        && unsafe { used_tags[..idx].assume_init_ref() }.contains(&tag)
                    {
                        return cerror2!(member.ident.span, "Duplicate enum variant tag");
                    }
                    used_tags[idx].write(tag);
                    tag += 1;
                    idx += 1;
                    Ok(())
                });
                self.close_scope();
                res?;
                debug_assert_eq!(variants.len(), used_tags.len());
                *variant_tags = Some(Ptr::from_ref(unsafe { used_tags.assume_init_ref() }));

                let repr_ty = repr_ty.u();
                let min_size_bits = util::variant_count_to_tag_size_bits(variants.len());
                *tag_ty = Some(if repr_ty.is_int_lit() {
                    let is_signed = repr_ty == p.sint_lit;
                    let Some(int_ty) = self.int_primitive(min_size_bits, is_signed) else {
                        return cerror2!(
                            expr.span,
                            "enums which can't be represented by an `u128` are currently not \
                             supported. This enum would require {min_size_bits} bits."
                        );
                    };
                    int_ty
                } else {
                    repr_ty.downcast::<ast::IntTy>()
                });
                debug_assert_eq!(expr.ty, p.type_ty);
            },
            AstEnum::RangeTy { .. } => todo!(),
            AstEnum::OptionTy { inner_ty: ty, .. } => {
                self.analyze_type(*ty)?;
                expr.ty = Some(p.type_ty);
            },
        }
        #[cfg(debug_assertions)]
        if expr.ty.is_none() {
            display(expr.full_span()).label("missing type").finish();
            debug_assert!(expr.ty.is_some());
        }
        Ok(())
    }

    fn analyze_and_finalize_with_known_type(
        &mut self,
        expr: Ptr<Ast>,
        expected_ty: Ptr<ast::Type>,
        is_const: bool,
    ) -> SemaResult<()> {
        self.analyze(expr, &Some(expected_ty), is_const)?;
        let expr_ty = expr.ty.u();
        if !ty_match(expr_ty, expected_ty) {
            self.cctx.error_mismatched_types(expr.full_span(), expected_ty, expr_ty);
            return SemaResult::HandledErr;
        }
        debug_assert!(expected_ty.is_finalized()); // => common_ty() is not needed.
        expr.as_mut().ty = Some(expected_ty);
        Ok(())
    }

    pub fn analyze_enum_tag(
        &mut self,
        expr: Ptr<Ast>,
        enum_ty: Ptr<ast::EnumDef>,
        v_idx: usize,
        is_const: bool,
    ) -> SemaResult<Ptr<ast::Type>> {
        let p = p();
        if is_const {
            let tag = enum_ty.variant_tags.u()[v_idx] as i64;
            let tag = ast_new!(IntVal { val: tag }, Span::ZERO).upcast();
            tag.as_mut().ty = Some(enum_ty.tag_ty.or_not_finished()?.upcast_to_type());
            expr.set_replacement(tag);
        }
        // enum variant
        Ok(if enum_ty.variants[v_idx].var_ty.u() == p.void_ty {
            enum_ty.upcast_to_type()
        } else {
            p.enum_variant
        })
    }

    pub fn try_custom_bitwith_int_type(&self, name: &str) -> Option<ast::IntTy> {
        let is_signed = match name.bytes().next() {
            Some(b'i') => true,
            Some(b'u') => false,
            _ => return None,
        };
        let bits = name[1..].parse().ok()?;
        debug_assert!(![8, 16, 32, 64, 128].contains(&bits));
        Some(type_new!(local IntTy { bits, is_signed }))
    }

    /// To allow recursive (or indirectly recursive) functions and types, we need to set the
    /// `var_ty` of these symbols
    /// may be recursive directly or indirectly. To allow this
    fn allow_unfinished_use(&self, expr: Ptr<Ast>, ty: Ptr<ast::Type>) {
        expr.as_mut().ty = Some(ty);
        if let Some(decl) = self.decl_stack.last()
            && decl.init == expr
            && decl.var_ty.is_none()
        {
            decl.as_mut().var_ty = Some(ty);
        }
    }

    pub fn validate_initializer(
        &mut self,
        struct_ty: Ptr<ast::Type>,
        initializer_values: Ptr<[(Ptr<ast::Ident>, Option<Ptr<Ast>>)]>,
        is_const: bool,
        initializer_span: Span,
    ) -> SemaResult<OPtr<[Ptr<ast::ConstVal>]>> {
        let fields = match struct_ty.matchable().as_ref() {
            TypeEnum::StructDef { fields, .. } => &**fields,
            TypeEnum::SliceTy { elem_ty, is_mut, .. } => {
                &self.slice_fields(elem_ty.downcast_type(), *is_mut)?
            },
            _ => unreachable_debug(),
        };

        let mut ok = true;
        macro_rules! on_err {
            () => {
                ok = false;
                continue
            };
        }
        let mut const_values =
            then!(is_const => self.cctx.alloc.alloc_slice_default(fields.len())?);
        macro_rules! handle_const_val {
            ($f_idx:expr, $val_expr:expr) => {
                if let Some(const_values) = const_values.as_mut() {
                    if let Some(cv) = $val_expr.try_downcast_const_val() {
                        const_values[$f_idx] = Some(cv);
                    } else {
                        self.cctx.error_non_const_initializer_field($val_expr);
                        on_err!();
                    }
                }
            };
        }
        let mut is_initialized_field = vec![false; fields.len()];
        for (idx, (f, init)) in initializer_values.as_mut().iter_mut().enumerate() {
            let Some((f_idx, f_decl)) = fields.find_field(f.sym) else {
                self.cctx.error_unknown_field(*f, struct_ty);
                on_err!();
            };

            if is_initialized_field[f_idx] {
                cerror!(f.span, "Duplicate field in named initializer");
                let (prev, prev_init) = initializer_values.iter().find(|v| v.0.sym == f.sym).u();
                let prev_span = prev.span.maybe_join(prev_init.map(|init| init.full_span()));
                chint!(prev_span, "first initialization here");
                on_err!();
            }
            is_initialized_field[f_idx] = true;

            let init = *init.get_or_insert(f.upcast());
            match self.analyze_and_finalize_with_known_type(init, f_decl.var_ty.u(), is_const) {
                Ok(()) => {},
                NotFinished { remaining } => {
                    return NotFinished { remaining: remaining + initializer_values.len() - idx };
                },
                Err(err) => {
                    self.cctx.error2(&err);
                    on_err!();
                },
            }
            handle_const_val!(f_idx, init);
        }

        for (f_idx, _) in
            is_initialized_field.into_iter().enumerate().filter(|(_, is_init)| !is_init)
        {
            let field = fields[f_idx];
            let Some(init) = field.init else {
                cerror!(initializer_span, "missing field `{}` in initializer", field.ident.sym);
                on_err!();
            };
            handle_const_val!(f_idx, init);
        }
        if ok {
            Ok(const_values.map(|cvalues| cvalues.u()))
        } else {
            SemaResult::HandledErr
        }
    }

    fn create_aggregate_const_val(
        &mut self,
        all_element_exprs: impl IntoIterator<IntoIter = impl ExactSizeIterator<Item = Ptr<ast::Ast>>>,
    ) -> SemaResult<Ptr<ast::AggregateVal>> {
        let mut all_element_exprs = all_element_exprs.into_iter();
        let mut elements = self.cctx.alloc.alloc_slice_default(all_element_exprs.len())?;
        let mut ok = true;
        for (elem, arg) in elements.iter_mut().zip(all_element_exprs.by_ref()) {
            if arg.ty.is_none() {
                return NotFinished { remaining: all_element_exprs.count() + 1 };
            }
            if let Some(cv) = arg.try_downcast_const_val() {
                *elem = Some(cv);
            } else {
                self.cctx.error_non_const_initializer_field(arg);
                ok = false;
            }
        }
        if !ok {
            return SemaResult::HandledErr;
        }
        Ok(ast_new!(AggregateVal { elements: elements.u() }, Span::ZERO))
    }

    /// Returns the [`SemaValue`] repesenting the new variable, not the entire
    /// declaration. This also doesn't insert into `self.symbols`.
    fn var_decl_to_value(
        &mut self,
        decl_ptr: Ptr<ast::Decl>,
        kind: Option<VarDeclSpecialCase>,
    ) -> SemaResult<()> {
        self.decl_stack.push(decl_ptr);
        let res = self.var_decl_to_value_inner(decl_ptr, kind);
        self.decl_stack.pop();
        res
    }

    #[inline]
    fn var_decl_to_value_inner(
        &mut self,
        decl_ptr: Ptr<ast::Decl>,
        kind: Option<VarDeclSpecialCase>,
    ) -> SemaResult<()> {
        let p = p();
        let decl = decl_ptr.as_mut();
        let is_first_pass = decl.ty.is_none(); // TODO(without `NotFinished`): remove this
        if is_first_pass && let Some(ty_expr) = decl.on_type {
            let ty = self.analyze_type(ty_expr)?;
            match ty.matchable().as_mut() {
                TypeEnum::StructDef { fields, consts, .. }
                | TypeEnum::UnionDef { fields, consts, .. }
                | TypeEnum::EnumDef { variants: fields, consts, .. } => {
                    let name = decl.ident.sym;
                    if let Some((_, prev)) =
                        fields.find_field(name).or_else(|| consts.find_field(name))
                    {
                        // TODO: remove this duplicate logic
                        cerror!(
                            decl_ptr.lhs_span(),
                            "duplicate definition of `{}`",
                            decl_ptr.display_lhs()
                        );
                        chint!(prev.lhs_span(), "previous definition here");
                        return SemaResult::HandledErr;
                    }
                    consts.push(decl_ptr);
                },
                _ if ty == p.never => return SemaResult::HandledErr,
                _ => {
                    return cerror2!(
                        ty_expr.span,
                        "cannot define an associated variable on a primitive type"
                    );
                },
            }
        }
        decl.ty = Some(p.void_ty);
        if let Some(t) = decl.var_ty_expr {
            let ty = self.analyze_type(t)?;
            debug_assert!(decl.var_ty.is_none_or(|t| ty_match(t, ty))); // TODO(without `NotFinished`): remove this
            decl.var_ty = Some(ty);
            if ty == p.never {
                decl.ty = Some(p.never);
            }
        }
        if let Some(init) = decl.init {
            let is_field = kind == Some(VarDeclSpecialCase::Field);
            let is_static = decl.markers.get(DeclMarkers::IS_STATIC_MASK);
            let is_init_const = decl.is_const || is_static || is_field;
            self.analyze(init, &decl.var_ty, is_init_const)?;
            let var_ty =
                check_and_set_target!(init.ty.u(), &mut decl.var_ty, init.full_span()).finalize();
            init.as_mut().ty = Some(var_ty);
            if is_init_const && var_ty != p.never && !init.rep().is_const_val() {
                // Ideally all branches in `_analyze_inner` should handle the `is_const` parameter.
                if is_static {
                    return cerror2!(
                        init.full_span(),
                        "Currently the initial value of a static must be known at compile time"
                    );
                }
                return cerror2!(init.full_span(), "Cannot evaluate value at compile time");
            }
            Ok(())
        } else if decl.var_ty.is_some() {
            debug_assert!(!decl.is_const);
            Ok(())
        } else if kind == Some(VarDeclSpecialCase::FnTyParam) {
            debug_assert!(decl.var_ty_expr.is_none());
            debug_assert!(decl.init.is_none());

            let ty_name = decl.ident.upcast();

            // The parameter name is changed to prevent `analyze_type` and further analysis from
            // finding the parameter itself as a type definition.
            decl.ident = p.ignored_name;
            decl.var_ty_expr = Some(ty_name);
            let ty = self.analyze_type(ty_name)?;
            decl.var_ty = Some(ty);
            Ok(())
        } else {
            cerror!(decl_ptr.upcast().full_span(), "cannot infer type of `{}`", decl_ptr.ident.sym);
            chint!(decl_ptr.upcast().full_span(), "consider explicitly specifying the type");
            // TODO: `my_var: /* type */`
            // TODO: `      ++++++++++++`
            return SemaResult::HandledErr;
        }
    }

    fn analyze_decl(
        &mut self,
        mut decl: Ptr<ast::Decl>,
        is_const: bool,
        kind: Option<VarDeclSpecialCase>,
    ) -> SemaResult<()> {
        let p = p();
        let _ = is_const; // TODO: non-toplevel constant contexts?
        let res = self.var_decl_to_value(decl, kind);
        #[cfg(debug_assertions)]
        if self.cctx.args.debug_types && decl.ident.span != Span::ZERO {
            let label = match &res {
                Ok(()) => format!("type: {}", decl.var_ty.u()),
                NotFinished { remaining } => format!("not finished (remaining: {remaining})"),
                Err(e) => format!("err: {}", e.kind),
            };
            display(decl.ident.span).label(&label).finish();
        }
        match res {
            Err(err) => {
                self.cctx.error2(&err);
                decl.var_ty = Some(p.never);
                decl.ty = Some(p.never);
                if let Some(init) = decl.init.as_mut() {
                    // see `tests::ordering::correctly_handle_error_in_later_cycles`
                    init.ty = Some(p.never);
                    init.rep().ty = Some(p.never);
                }
                Ok(())
            },
            NotFinished { remaining } => NotFinished { remaining },
            Ok(()) => {
                debug_assert!(decl.var_ty.is_some());
                decl.ident.decl = Some(decl);
                Ok(())
            },
        }
    }

    /// Also used for `#intrinsic` directive.
    fn analyze_extern_directive(
        &self,
        directive: Ptr<ast::Ast>,
        decl_out: &mut OPtr<ast::Decl>,
        ty_hint: OPtr<ast::Type>,
    ) -> SemaResult<Ptr<ast::Type>> {
        let Some(&decl) = self.decl_stack.last().filter(|decl| decl.init == directive) else {
            return cerror2!(
                directive.full_span(),
                "The #{0} directive must be preceeded by a declaration, like `f : (int) -> int : \
                 #{0}`.",
                extern_directive_name(directive)
            );
        };
        *decl_out = Some(decl);
        debug_assert_eq!(decl.var_ty, ty_hint);
        let Some(ty) = ty_hint else {
            return cerror2!(
                decl.lhs_span(),
                "An #{0} declaration requires an explicit type annotation. like `f : (int) -> int \
                 : #{0}`",
                extern_directive_name(directive)
            );
        };
        if decl.on_type.is_some() || !self.function_stack.is_empty() {
            return cerror2!(
                decl.lhs_span(),
                "An #{0} declaration must be in global/file scope.",
                extern_directive_name(directive)
            );
        }
        Ok(ty)
    }

    /// works for function calls and call initializers
    /// ...(arg1, ..., argX, paramX1=argX1, ... paramN=argN)
    /// paramN1, ... -> default values
    fn validate_call(
        &mut self,
        params: &[Ptr<ast::Decl>],
        args: &[Ptr<ast::Ast>],
        close_p_span: Span,
        is_const: bool,
    ) -> SemaResult<()> {
        // positional args
        let mut pos_idx = 0;
        while let Some(pos_arg) = args.get(pos_idx).copied().filter(is_pos_arg) {
            let Some(&param) = params.get(pos_idx) else {
                let pos_arg_count = args.iter().copied().filter(is_pos_arg).count();
                return cerror2!(
                    pos_arg.full_span(),
                    "Got {pos_arg_count} positional arguments, but expected at most {} arguments",
                    params.len(),
                );
            };
            let param_ty = param.var_ty.or_not_finished()?;
            self.analyze_and_finalize_with_known_type(pos_arg, param_ty, is_const)?;

            pos_idx += 1;
        }

        // named args
        let remaining_args = &args[pos_idx..];
        let remaining_params = &params[pos_idx..];
        let mut was_set = vec![false; remaining_params.len()];
        for &arg in remaining_args {
            if is_pos_arg(&arg) {
                return cerror2!(
                    arg.full_span(),
                    "Cannot specify a positional argument after named arguments"
                );
            }
            let named_arg = arg.downcast::<ast::Assign>();
            let Some(arg_name) = named_arg.lhs.try_downcast::<ast::Ident>() else {
                return cerror2!(named_arg.lhs.full_span(), "Expected a parameter name");
            };
            let Some((param_idx, param)) = remaining_params.find_field(arg_name.sym) else {
                if let Some((idx, _)) = params[..pos_idx].find_field(arg_name.sym) {
                    self.cctx.error_duplicate_named_arg(arg_name);
                    chint!(
                        args[idx].full_span(),
                        "The parameter has already been set by this positional argument"
                    )
                } else {
                    cerror!(arg_name.span, "Unknown parameter");
                }
                return SemaResult::HandledErr;
            };
            if was_set[param_idx] {
                self.cctx.error_duplicate_named_arg(arg_name);
                chint!(remaining_args[param_idx].full_span(), "set here already");
                return SemaResult::HandledErr;
            } else {
                was_set[param_idx] = true;
            }
            let param_ty = param.var_ty.or_not_finished()?;
            self.analyze_and_finalize_with_known_type(named_arg.rhs, param_ty, is_const)?;
        }

        // missing args
        let mut missing_params = remaining_params
            .iter()
            .copied()
            .zip(was_set)
            .filter(|(p, was_set)| !was_set && p.init.is_none())
            .map(|(p, _)| p);
        if let Some(first) = missing_params.next() {
            let mut missing_params_list = String::new();
            let mut idx = 0;
            for p in iter::once(first).chain(missing_params) {
                if idx != 0 {
                    write!(&mut missing_params_list, ", ").unwrap();
                }
                let param_ty = p.var_ty.or_not_finished()?;
                write!(&mut missing_params_list, "`{}: {}`", p.ident.sym, param_ty).unwrap();
                idx += 1;
            }
            cerror!(
                close_p_span,
                "Missing argument{0} for parameter{0} {1}",
                if idx > 1 { "s" } else { "" },
                missing_params_list
            );
            chint!(first.upcast().full_span(), "parameter defined here");
            return SemaResult::HandledErr;
        }
        Ok(())
    }

    fn resolve_mutated_value(&self, mut expr: Ptr<Ast>) -> MutatedValue {
        use MutatedValue::*;
        loop {
            match expr.matchable2() {
                AstMatch::Ident(ident) => return Var(ident),
                AstMatch::Dot(dot) => {
                    debug_assert_ne!(dot.lhs.u().ty.u().kind, AstKind::PtrTy); // autodereferencing is not implemented
                    expr = dot.lhs.u();
                },
                AstMatch::Index(index) => {
                    debug_assert_ne!(index.lhs.ty.u().kind, AstKind::PtrTy); // autodereferencing is not implemented
                    if index.lhs.ty == p().never {
                        return None;
                    } else if index.lhs.ty.u().kind == AstKind::SliceTy {
                        return Slice(index.lhs);
                    } else {
                        expr = index.lhs;
                    }
                },
                AstMatch::UnaryOp(op) if op.op == UnaryOpKind::Deref => {
                    if op.operand.ty == p().never {
                        return None;
                    }
                    return Ptr(op.operand);
                },
                _ => return None,
            }
        }
    }

    fn validate_lvalue(&self, lvalue: Ptr<Ast>, full_expr: Ptr<Ast>) -> SemaResult<()> {
        match lvalue.matchable2() {
            AstMatch::Ident(_) | AstMatch::Dot(_) | AstMatch::Index(_) => {},
            AstMatch::UnaryOp(op) if op.op == UnaryOpKind::Deref => {},
            _ => {
                return cerror2!(
                    lvalue.full_span(),
                    "Cannot assign a value to an expression of kind '{:?}'",
                    lvalue.kind
                );
            },
        }
        if !self.cctx.do_mut_checks {
            return Ok(());
        }
        match self.resolve_mutated_value(lvalue) {
            MutatedValue::Var(ident) => {
                let Some(decl) = ident.decl else {
                    cwarn!(ident.span, "INTERNAL: This ident doesn't have a declaration");
                    return Ok(());
                };
                if decl.markers.get(DeclMarkers::IS_MUT_MASK) {
                    return Ok(());
                }
                let var_name = decl.ident.sym;
                // let is_direct_assignment = lvalue == ident; // TODO: change error messages?
                if decl.is_const {
                    cerror!(full_expr.full_span(), "Cannot assign to constant '{var_name}'");
                } else {
                    cerror!(
                        full_expr.full_span(),
                        "Cannot assign to immutable variable '{var_name}'",
                    );
                    chint!(decl.ident.span, "consider making '{var_name}' mutable");
                }
                SemaResult::HandledErr
            },
            MutatedValue::Ptr(ptr) => {
                let ptr_ty = ptr.ty.downcast::<ast::PtrTy>();
                if ptr_ty.is_mut {
                    return Ok(());
                }
                cerror!(
                    full_expr.full_span(),
                    "Cannot mutate the value behind an immutable pointer"
                );
                chint!(
                    ptr.full_span(),
                    "The pointer type `{}` is not `mut`",
                    ptr_ty.upcast_to_type()
                );
                SemaResult::HandledErr
            },
            MutatedValue::Slice(slice) => {
                let slice_ty = slice.ty.downcast::<ast::SliceTy>();
                if slice_ty.is_mut {
                    return Ok(());
                }
                cerror!(full_expr.full_span(), "Cannot mutate the elements of an immutable slice");
                chint!(
                    slice.full_span(),
                    "The slice type `{}` is not `mut`",
                    slice_ty.upcast_to_type()
                );
                SemaResult::HandledErr
            },
            MutatedValue::None => Ok(()),
        }
    }

    // also used for slicing
    fn validate_addr_of(
        &self,
        is_mut_addr_of: bool,
        operand: Ptr<Ast>,
        full_expr: Ptr<Ast>,
    ) -> SemaResult<()> {
        if !self.cctx.do_mut_checks || !is_mut_addr_of {
            return Ok(());
        }
        match self.resolve_mutated_value(operand) {
            MutatedValue::Var(ident) => {
                let Some(decl) = ident.decl else {
                    cwarn!(ident.span, "INTERNAL: This ident doesn't have a declaration");
                    return Ok(());
                };
                if decl.markers.get(DeclMarkers::IS_MUT_MASK) {
                    return Ok(());
                }
                if decl.is_const {
                    cwarn!(
                        full_expr.full_span(),
                        "The mutable pointer will reference a local copy of `{}`, not the \
                         constant itself",
                        ident.sym
                    );
                    Ok(())
                } else {
                    cerror!(ident.span, "Cannot mutably reference `{}`", ident.sym);
                    chint!(decl.ident.span, "because `{}` is not marked as `mut`", ident.sym);
                    SemaResult::HandledErr
                }
            },
            MutatedValue::Ptr(ptr) => {
                let ptr_ty = ptr.ty.downcast::<ast::PtrTy>();
                if !ptr_ty.is_mut {
                    self.cctx.error_mutate_const_ptr(full_expr.full_span(), ptr);
                    return SemaResult::HandledErr;
                }
                Ok(())
            },
            MutatedValue::Slice(slice) => {
                let slice_ty = slice.ty.downcast::<ast::SliceTy>();
                if !slice_ty.is_mut {
                    self.cctx.error_mutate_const_slice(full_expr.full_span(), slice);
                    return SemaResult::HandledErr;
                }
                Ok(())
            },
            MutatedValue::None => Ok(()),
        }
    }

    fn analyze_type(&mut self, ty_expr: Ptr<Ast>) -> SemaResult<Ptr<ast::Type>> {
        let p = p();
        self.analyze(ty_expr, &Some(p.type_ty), true)?;
        let ty = ty_expr.ty.u();
        if ty != p.type_ty {
            if ty == p.never {
                return Ok(p.never);
            }
            self.cctx.error_mismatched_types(ty_expr.full_span(), p.type_ty, ty);
            return SemaResult::HandledErr;
        }
        Ok(ty_expr.downcast_type())
    }

    fn ty_match(&mut self, expr: Ptr<Ast>, expected_ty: Ptr<ast::Type>) -> SemaResult<()> {
        let got_ty = expr.ty.u();
        if ty_match(got_ty, expected_ty) {
            Ok(())
        } else {
            self.cctx.error_mismatched_types(expr.full_span(), expected_ty, got_ty);
            SemaResult::HandledErr
        }
    }

    pub fn int_primitive(&self, bits: u32, is_signed: bool) -> OPtr<ast::IntTy> {
        let p = p();

        macro_rules! i {
            ($signed:ident, $unsigned:ident) => {
                if is_signed { p.$signed } else { p.$unsigned }
            };
        }

        let int_ty = match bits {
            0 => p.u0,
            ..=8 => i!(i8, u8),
            ..=16 => i!(i16, u16),
            ..=32 => i!(i32, u32),
            ..=64 => i!(i64, u64),
            ..=128 => i!(i128, u128),
            _ => return None,
        };
        Some(int_ty.downcast::<ast::IntTy>())
    }

    pub fn slice_fields(
        &self,
        elem_ty: Ptr<ast::Type>,
        is_mut: bool,
    ) -> SemaResult<[Ptr<ast::Decl>; 2]> {
        let p = p();
        let elem_ptr_ty = type_new!(PtrTy { pointee: elem_ty.upcast(), is_mut });
        let mut ptr = self.alloc(ast::Decl::new(p.slice_ptr_field_ident, None, Span::ZERO))?;
        ptr.var_ty = Some(elem_ptr_ty.upcast_to_type());
        Ok([ptr, p.slice_len_field])
    }

    /// Note: the returned [`ast::Decl`] might not be fully analyzed.
    #[inline]
    fn get_symbol(&self, sym: Symbol) -> OPtr<ast::Decl> {
        self.cur_scope.find_decl(sym, self.cur_scope_pos, true)
    }

    fn get_symbol_var_ty(&self, sym: Ptr<ast::Decl>) -> SemaResult<Ptr<ast::Type>> {
        Ok(if let Some(var_ty) = sym.var_ty {
            var_ty
        } else if sym.is_const
            && let Some(f) = sym.init.u().try_downcast::<ast::Fn>()
            && self.decl_stack.iter().rev().any(|d| *d == sym)
        {
            // This case is needed for recursive functions without an explicit return type. Those
            // functions cannot use `allow_unfinished_use` because the return type might change
            // multiple times during inference and call sites must not use a partially inferred
            // return type.
            // TODO: add hint message to recommend adding at least one explicit return type to
            //       indirectly recursive functions.
            debug_assert!(sym.init.u().replacement.is_none());
            f.upcast_to_type()
        } else {
            return NotFinished { remaining: 1 };
        })
    }

    fn open_scope(&mut self, new_scope: &mut Scope) {
        #[cfg(debug_assertions)]
        debug_assert!(new_scope.was_checked_for_duplicates);
        debug_assert!(new_scope.parent.is_none_or(|p| p == self.cur_scope));
        //debug_assert!(new_scope.parent.is_none()); // For when NotFinished is removed
        if new_scope.parent.is_none() {
            new_scope.parent = Some(self.cur_scope);
            new_scope.pos_in_parent = self.cur_scope_pos;
        }

        //println!("#### open pos: {:?} {:?}", new_scope.pos_in_parent, new_scope.kind);

        self.cur_scope = Ptr::from_ref(new_scope);
        self.cur_scope_pos = ScopePos(0);
        self.defer_stack.open_scope();
    }

    fn close_scope(&mut self) {
        //self.symbols.close_scope();
        self.cur_scope_pos = self.cur_scope.pos_in_parent;
        //println!("#### close pos: {:?} {:?}", self.cur_decl_pos, self.cur_scope.kind);
        self.cur_scope = self.cur_scope.parent.u();
        self.defer_stack.close_scope();
    }

    #[inline]
    fn alloc<T>(&self, val: T) -> SemaResult<Ptr<T>> {
        Ok(self.cctx.alloc.alloc(val)?)
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

#[derive(Debug, Clone, Copy)]
pub enum MutatedValue {
    Var(Ptr<ast::Ident>),
    Ptr(Ptr<ast::Ast>),
    Slice(Ptr<ast::Ast>),
    None,
}

impl MutatedValue {
    pub fn try_get_var(self) -> OPtr<ast::Ident> {
        match self {
            MutatedValue::Var(ptr) => Some(ptr),
            MutatedValue::Ptr(ptr) | MutatedValue::Slice(ptr) => ptr.try_downcast::<ast::Ident>(),
            MutatedValue::None => None,
        }
    }
}

fn is_finished_or_recursive(f: Ptr<ast::Fn>, _self: &Sema) -> bool {
    f.ret_ty.is_some() || f == *_self.function_stack.last().u() // TODO: check all previous fns
}

pub trait OptionSemaExt<T> {
    fn or_not_finished(self) -> SemaResult<T, !>;
}

impl<T> OptionSemaExt<T> for Option<T> {
    fn or_not_finished(self) -> SemaResult<T, !> {
        match self {
            Some(t) => Ok(t),
            None => NotFinished { remaining: 1 },
        }
    }
}

fn extern_directive_name(directive: Ptr<ast::Ast>) -> &'static str {
    match directive.kind {
        AstKind::ExternDirective => "extern",
        AstKind::IntrinsicDirective => "intrinsic",
        _ => unreachable_debug(),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum VarDeclSpecialCase {
    FnTyParam,
    Field,
}

const _: () = assert!(size_of::<Option<VarDeclSpecialCase>>() == 1);
