//! # Semantic analysis module
//!
//! Semantic analysis validates (and changes) all stored [`ast::Type`]s in [`Expr`].

use crate::{
    ast::{
        self, Ast, AstEnum, AstKind, BinOpKind, Decl, DeclListExt, OptionAstExt, OptionTypeExt,
        TypeEnum, UnaryOpKind, UpcastToAst, ast_new, type_new,
    },
    context::{CompilationContextInner, primitives as p},
    diagnostic_reporter::{DiagnosticReporter, cerror, chint},
    display_code::display,
    parser::lexer::Span,
    ptr::{OPtr, Ptr},
    scoped_stack::ScopedStack,
    source_file::SourceFile,
    symbol_table::{SymbolTable2, linear_search_symbol},
    type_::{RangeKind, common_type, ty_match},
    util::{UnwrapDebug, forget_lifetime_mut, panic_debug, then, unreachable_debug},
};
pub(crate) use err::{SemaError, SemaErrorKind, SemaResult};
use err::{SemaErrorKind::*, SemaResult::*};
use std::collections::HashSet;

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
                None => return err(MismatchedTypes { expected: *target_ty, got: ty }, $err_span),
            }
        } else {
            ty
        };
        *target_ty = Some(common_ty);
    }};
}

pub fn analyze(cctx: Ptr<CompilationContextInner>, stmts: &[Ptr<Ast>]) -> Vec<usize> {
    // validate top level stmts
    for file in cctx.files.iter().copied() {
        let file_stmts = &stmts[file.stmt_range.u()];
        for (idx, s) in file_stmts.iter().copied().enumerate() {
            let Some(decl) = s.try_downcast::<ast::Decl>() else {
                cerror!(s.full_span(), "unexpected top level expression");
                continue;
            };
            debug_assert!((decl.is_const && decl.init.is_some()) || decl.is_extern);
            debug_assert!(decl.ty.is_none());
            if let Some(dup) = linear_search_symbol(&file_stmts[..idx], &decl.ident.text) {
                cerror!(decl.ident.span, "Duplicate definition in module scope.");
                chint!(dup.ident.span, "First definition here")
            }
        }
    }

    let mut sema = Sema::new(cctx, Ptr::from_ref(stmts));
    let mut finished = vec![false; stmts.len()];
    let mut remaining_count = stmts.len();
    let mut order = Vec::with_capacity(stmts.len());
    while finished.iter().any(std::ops::Not::not) {
        let old_remaining_count = remaining_count;
        debug_assert!(stmts.len() == finished.len());
        remaining_count = 0;
        for file in cctx.files.iter().copied() {
            sema.cur_file = Some(file);
            debug_assert!(sema.symbols.is_empty());
            sema.open_scope();
            let stmt_range = file.stmt_range.u();
            for (idx, (s, finished)) in
                stmts[file.stmt_range.u()].iter().zip(&mut finished[stmt_range]).enumerate()
            {
                if *finished {
                    continue;
                }
                let res = sema.analyze_top_level(*s);
                *finished = res != SemaResult::NotFinished;
                match res {
                    SemaResult::Ok(_) => order.push(idx + stmt_range.start),
                    SemaResult::NotFinished => remaining_count += 1,
                    SemaResult::Err(_) => {},
                }
            }
            debug_assert!(
                sema.defer_stack.get_cur_scope().is_empty(),
                "file scope must not contain defer statements"
            );
            sema.close_scope().ok().u();
        }
        // println!("finished statements: {:?}", finished);
        if remaining_count == old_remaining_count {
            eprintln!("cycle(s) detected:");
            for (s, finished) in stmts.iter().zip(finished) {
                if finished {
                    continue;
                }
                let span = s
                    .try_downcast::<ast::Decl>()
                    .map(|d| d.ident.span)
                    .unwrap_or_else(|| s.full_span());
                display(span).finish();
            }
            panic!("cycle detected") // TODO: find location of cycle
        }
    }

    order
}

/// Semantic analyzer
pub struct Sema {
    file_level_symbols: Ptr<[Ptr<Ast>]>,
    /// doesn't contain file-level symbols
    pub symbols: SymbolTable2,
    function_stack: Vec<Ptr<ast::Fn>>,
    defer_stack: ScopedStack<Ptr<Ast>>,

    cctx: Ptr<CompilationContextInner>,
    cur_file: OPtr<SourceFile>,
}

impl Sema {
    pub fn new(cctx: Ptr<CompilationContextInner>, file_level_symbols: Ptr<[Ptr<Ast>]>) -> Sema {
        Sema {
            file_level_symbols,
            symbols: ScopedStack::default(),
            function_stack: vec![],
            defer_stack: ScopedStack::default(),
            cctx,
            cur_file: None,
        }
    }

    pub fn analyze_top_level(&mut self, s: Ptr<Ast>) -> SemaResult<(), ()> {
        match self.analyze(s, &Some(p().void_ty), true) {
            Ok(_) => Ok(()),
            NotFinished => NotFinished,
            Err(e) => {
                self.cctx.error2(&e);
                Err(())
            },
        }
    }

    pub fn analyze(
        &mut self,
        expr: Ptr<Ast>,
        ty_hint: &OPtr<ast::Type>,
        is_const: bool,
    ) -> SemaResult<()> {
        let res = self._analyze_inner(expr, ty_hint, is_const);
        #[cfg(debug_assertions)]
        if self.cctx.debug_types {
            let label = match &res {
                Ok(()) => format!("type: {}", expr.ty.u()),
                NotFinished => "not finished".to_string(),
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
            debug_assert!(expr.ty.is_some());
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

        match expr.matchable().as_mut() {
            AstEnum::Ident { text, .. } => {
                let sym = self.get_symbol(*text, expr.span);
                if sym.is_err()
                    && let Some(mut i) = self.try_custom_bitwith_int_type(*text)
                {
                    i.span = expr.span;
                    i.ty = Some(p.type_ty);
                    debug_assert!(size_of::<ast::Ident>() >= size_of::<ast::IntTy>());
                    *expr.cast::<ast::IntTy>().as_mut() = i;
                } else {
                    let sym = sym?;
                    if sym.is_const {
                        debug_assert!(expr.replacement.is_none());
                        expr.replacement = Some(sym.const_val());
                    } else if is_const {
                        if sym.is_extern {
                            todo!("use of extern symbol in const expr")
                        };
                        return err(NotAConstExpr, expr.full_span());
                    }
                    expr.ty = Some(sym.var_ty.u());
                }
            },
            AstEnum::Block { stmts, has_trailing_semicolon, .. } => {
                self.open_scope();
                let res: SemaResult<()> = try {
                    let max_idx = stmts.len().wrapping_sub(1);
                    for (idx, s) in stmts.iter().enumerate() {
                        let expected_ty = if max_idx == idx { ty_hint } else { &None };
                        match self.analyze(*s, expected_ty, is_const) {
                            Ok(()) => {
                                // s.ty = s.ty.finalize();
                                debug_assert!(s.ty.is_some());
                            },
                            NotFinished => NotFinished?,
                            Err(err) => {
                                self.cctx.error2(&err);
                                s.as_mut().ty = Some(p.never);
                            },
                        }
                    }
                };
                self.close_scope()?;
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
                    return err(CannotInferInitializerTy, expr.full_span());
                };
                if let Some(mut s) = lhs.try_downcast::<ast::StructDef>() {
                    // allow slices?
                    self.validate_call(&mut s.fields, args.iter_mut(), span.end())?;
                    expr.ty = Some(s.upcast_to_type())
                } else if let Some(ptr_ty) = lhs.ty.try_downcast::<ast::PtrTy>()
                    && let Some(mut s) = ptr_ty.pointee.try_downcast::<ast::StructDef>()
                {
                    self.validate_call(&mut s.fields, args.iter_mut(), span.end())?;
                    expr.ty = Some(ptr_ty.upcast_to_type())
                } else {
                    return err(CannotApplyInitializer { ty: lhs.ty.u() }, lhs.full_span());
                };
            },
            AstEnum::NamedInitializer { lhs, fields: values, .. } => {
                let lhs = if let Some(lhs) = *lhs {
                    analyze!(lhs, None, false);
                    lhs
                } else if let Some(s_def) = ty_hint.filter(|t| t.kind.is_struct_kind()) {
                    s_def.upcast()
                } else {
                    return err(CannotInferInitializerTy, expr.full_span());
                };
                if let Some(struct_ty) = lhs.try_downcast_type()
                    && struct_ty.kind.is_struct_kind()
                {
                    self.validate_initializer(struct_ty, *values, is_const, span)?;
                    expr.ty = Some(struct_ty)
                } else if let Some(ptr_ty) = lhs.ty.try_downcast::<ast::PtrTy>()
                    && let Some(pointee) = ptr_ty.pointee.try_downcast_const_val()
                    && pointee.kind.is_struct_kind()
                {
                    let struct_ty = pointee.downcast_type();
                    self.validate_initializer(struct_ty, *values, is_const, span)?;
                    expr.ty = Some(ptr_ty.upcast_to_type())
                } else {
                    return err(CannotApplyInitializer { ty: lhs.ty.u() }, lhs.full_span());
                };
            },
            AstEnum::ArrayInitializer { lhs, elements, .. } => {
                if let Some(lhs) = *lhs {
                    analyze!(lhs, None);
                }
                let lhs = lhs.or_else(|| ty_hint.upcast().filter(|t| t.kind == AstKind::ArrayTy));

                let mut elem_iter = elements.iter();

                let mut elem_ty = if let Some(lhs) = lhs {
                    let arr_ty = if let Some(arr_ty) = lhs.try_downcast::<ast::ArrayTy>() {
                        expr.ty = Some(arr_ty.upcast_to_type());
                        arr_ty
                    } else if let Some(ptr_ty) = lhs.ty.try_downcast::<ast::PtrTy>()
                        && let Some(arr_ty) = ptr_ty.pointee.try_downcast::<ast::ArrayTy>()
                    {
                        expr.ty = Some(ptr_ty.upcast_to_type());
                        arr_ty
                    } else {
                        return err(CannotApplyInitializer { ty: lhs.ty.u() }, lhs.full_span());
                    };

                    if elements.len() != arr_ty.len.int() {
                        return err(
                            MismatchedArrayLen { expected: arr_ty.len.int(), got: elements.len() },
                            expr.full_span(),
                        );
                    }

                    arr_ty.elem_ty.downcast_type()
                } else {
                    let Some(elem) = elem_iter.next() else {
                        expr.ty = Some(p.empty_array_ty.upcast_to_type()); // `.[]`
                        return Ok(());
                    };
                    *analyze!(*elem, None)
                };
                #[cfg(debug_assertions)]
                let orig_elem_ty = elem_ty;

                for elem in elem_iter {
                    let ty = *analyze!(*elem, Some(elem_ty));
                    let Some(common_ty) = common_type(ty, elem_ty) else {
                        return err(
                            MismatchedTypes { expected: elem_ty, got: ty },
                            elem.full_span(),
                        );
                    };
                    elem_ty = common_ty;
                }

                #[cfg(debug_assertions)]
                debug_assert!(lhs.is_none() || elem_ty == orig_elem_ty);

                if lhs.is_none() {
                    debug_assert!(expr.ty.is_none());
                    let len = elements.len() as i64;
                    let len = self.alloc(ast_new!(IntVal { span: Span::ZERO, val: len }))?.upcast();
                    let elem_ty = elem_ty.upcast();
                    let arr_ty = self.alloc(type_new!(ArrayTy { len, elem_ty }))?;
                    expr.ty = Some(arr_ty.upcast_to_type());
                }
            },
            AstEnum::ArrayInitializerShort { lhs, val, count, .. } => {
                assert!(!is_const, "todo: const array");
                assert!(lhs.is_none(), "todo: array initializer with lhs");
                let u64_ty = p.u64;
                let count_ty = *analyze!(*count, Some(u64_ty), true);
                debug_assert!(count.is_const_val());
                if count_ty != u64_ty {
                    return err(
                        MismatchedTypes { expected: u64_ty, got: count_ty },
                        count.full_span(),
                    );
                }
                let elem_ty = analyze!(*val, None).upcast();
                let arr_ty =
                    self.alloc(type_new!(ArrayTy { len: *count, elem_ty }))?.upcast_to_type();
                expr.ty = Some(arr_ty);
            },
            AstEnum::Dot { lhs: Some(lhs), rhs, .. } => {
                let lhs_ty = *analyze!(*lhs, None);
                let lhs = lhs.rep();
                let t = if let Some(m) = lhs.try_downcast::<ast::ImportDirective>()
                    && let Some(s) =
                        self.cctx.files[m.files_idx].find_symbol(&rhs.text, self.file_level_symbols)
                {
                    debug_assert_eq!(lhs_ty, p.module);
                    let Some(ty) = s.var_ty else { return NotFinished };
                    debug_assert!(expr.replacement.is_none());
                    expr.replacement = Some(s.const_val());
                    ty
                } else if lhs_ty == p.type_ty
                    && let Some(enum_ty) = lhs.try_downcast::<ast::EnumDef>()
                    && let Some((_, variant)) = enum_ty.variants.find_field(&rhs.text)
                {
                    if variant.var_ty.u() == p.void_ty {
                        enum_ty.upcast_to_type()
                    } else {
                        p.enum_variant
                    }
                } else if let TypeEnum::StructDef { fields, .. } | TypeEnum::UnionDef { fields, .. } =
                    lhs_ty.matchable().as_ref()
                    && let Some((_, field)) = fields.find_field(&rhs.text)
                {
                    field.var_ty.u()
                } else if let TypeEnum::SliceTy { elem_ty, is_mut, .. } = *lhs_ty.matchable()
                    && &*rhs.text == "ptr"
                {
                    // TODO: remove this allocation (test if cast SliceTy -> PointerTy is valid)
                    self.alloc(type_new!(PtrTy { pointee: elem_ty, is_mut }))?.upcast_to_type()
                } else if lhs_ty.kind == AstKind::SliceTy && &*rhs.text == "len" {
                    p.u64
                } else {
                    if rhs.replacement.is_some() {
                        debug_assert!(expr.ty.is_some());
                        return Ok(());
                    }
                    // method-like call:
                    match self.get_symbol(rhs.text, rhs.span) {
                        Ok(s)
                            if s.var_ty == p.fn_val
                                && let f = s.init.u().downcast::<ast::Fn>()
                                && let Some(first_param) = f.params.get(0)
                                && ty_match(lhs.ty.u(), first_param.var_ty.u()) =>
                        {
                            debug_assert!(rhs.replacement.is_none());
                            rhs.replacement = Some(f.upcast());
                        },
                        NotFinished => return NotFinished,
                        _ => return err(UnknownField { ty: lhs_ty, field: rhs.text }, rhs.span),
                    };
                    p.method_stub
                };
                expr.ty = Some(t);
            },
            AstEnum::Dot { lhs: lhs @ None, rhs, .. } => {
                // `.<ident>` must be an enum
                let Some(enum_ty) = ty_hint.try_downcast::<ast::EnumDef>() else {
                    if *ty_hint == p.never {
                        expr.ty = Some(p.never);
                        return Ok(());
                    }
                    return err(CannotInfer, expr.full_span());
                };
                let Some((_, variant)) = enum_ty.variants.find_field(&rhs.text) else {
                    let ty = enum_ty.upcast_to_type();
                    return err(UnknownField { ty, field: rhs.text }, rhs.span);
                };
                *lhs = Some(enum_ty.upcast());
                expr.ty = Some(if variant.var_ty.u() == p.void_ty {
                    enum_ty.upcast_to_type()
                } else {
                    p.enum_variant
                });
            },
            AstEnum::Index { lhs, idx, .. } => {
                assert!(!is_const, "todo: const index");
                let lhs_ty = analyze!(*lhs, None);
                let (elem_ty, is_mut) = match lhs_ty.matchable().as_ref() {
                    TypeEnum::SliceTy { elem_ty, is_mut, .. } => (*elem_ty, *is_mut),
                    TypeEnum::ArrayTy { elem_ty, .. } => (*elem_ty, true),
                    // _ => cerror!(lhs.full_span(), "cannot index into value of type {}", lhs_ty),
                    _ => return err(CanOnlyIndexArrays, lhs.full_span()),
                };
                let elem_ty = elem_ty.downcast_const_val().downcast_type();
                analyze!(*idx, None).finalize();
                expr.ty = Some(match idx.ty.matchable().as_ref() {
                    TypeEnum::IntTy { .. } => elem_ty,
                    TypeEnum::RangeTy { elem_ty: i, rkind, .. }
                        if i.matches_int() || *rkind == RangeKind::Full =>
                    {
                        let elem_ty = elem_ty.upcast();
                        self.alloc(type_new!(SliceTy { elem_ty, is_mut }))?.upcast_to_type()
                    },
                    _ => return err(InvalidArrayIndex { ty: idx.ty.u() }, idx.full_span()),
                });
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
                if is_const {
                    return err(NotAConstExpr, expr.full_span());
                }
                let ty_hint =
                    ty_hint.filter(|t| t.kind == AstKind::EnumDef && func.kind == AstKind::Dot); // I hope this doesn't conflict with `method_stub`
                let fn_ty = *analyze!(*func, ty_hint);
                expr.ty = Some(if let Some(fn_ty) = func.try_downcast::<ast::Fn>().as_mut() {
                    let params = &mut fn_ty.params;
                    self.validate_call(params, args.iter_mut(), span.end())?;
                    fn_ty.ret_ty.u()
                } else if fn_ty == p.method_stub {
                    let dot = func.downcast::<ast::Dot>().as_mut();
                    debug_assert_eq!(dot.rhs.upcast().rep().kind, AstKind::Fn);
                    let fn_ty = dot.rhs.upcast().downcast::<ast::Fn>();
                    let args = std::iter::once(dot.lhs.as_mut().u()).chain(args.iter_mut());
                    self.validate_call(&mut fn_ty.as_mut().params, args, span.end())?;
                    fn_ty.ret_ty.u()
                } else if fn_ty == p.enum_variant {
                    let Some(dot) = func.try_downcast::<ast::Dot>() else { unreachable_debug() };
                    let enum_ty = dot.lhs.u().downcast::<ast::EnumDef>();
                    let variant = enum_ty.variants.find_field(&dot.rhs.text).u().1;
                    self.validate_call(&mut [variant], args.iter_mut(), span.end())?;
                    enum_ty.upcast_to_type()
                } else {
                    display(expr.full_span()).finish();
                    panic!("{:#?}", func);
                    return err(CallOfANonFunction, expr.full_span());
                });
            },
            AstEnum::UnaryOp { op, operand, .. } => {
                //assert!(!is_const, "todo: PreOp in const");
                let expr_ty = *analyze!(*operand, None);
                let const_val = then!(is_const => operand.downcast_const_val());
                expr.ty = Some(match *op {
                    UnaryOpKind::AddrOf | UnaryOpKind::AddrMutOf => {
                        let is_mut = *op == UnaryOpKind::AddrMutOf;
                        let pointee = expr_ty.upcast();
                        debug_assert!(const_val.is_none(), "todo: const addr of");
                        self.alloc(type_new!(PtrTy { pointee, is_mut }))?.upcast_to_type()
                    },
                    UnaryOpKind::Deref
                        if let Some(ptr_ty) = expr_ty.try_downcast::<ast::PtrTy>() =>
                    {
                        debug_assert!(const_val.is_none(), "todo: const deref");
                        ptr_ty.pointee.downcast_type()
                    },
                    UnaryOpKind::Not if expr_ty == p.bool => {
                        if let Some(const_val) = const_val {
                            let b = const_val.downcast::<ast::BoolVal>().val;
                            debug_assert!(expr.replacement.is_none());
                            // TODO: no allocation
                            expr.replacement = Some(
                                self.alloc(ast_new!(BoolVal { val: !b, span: Span::ZERO }))?
                                    .upcast(),
                            )
                        }
                        p.bool
                    },
                    UnaryOpKind::Neg if expr_ty.kind == AstKind::IntTy || expr_ty == p.int_lit => {
                        if let Some(const_val) = const_val {
                            let i = const_val.downcast::<ast::IntVal>().val;
                            debug_assert!(expr.replacement.is_none());
                            // TODO: no allocation
                            expr.replacement = Some(
                                self.alloc(ast_new!(IntVal { val: -i, span: Span::ZERO }))?
                                    .upcast(),
                            )
                        }
                        expr_ty
                    },
                    UnaryOpKind::Neg
                        if expr_ty.kind == AstKind::FloatTy || expr_ty == p.float_lit =>
                    {
                        if let Some(const_val) = const_val {
                            let f = const_val.downcast::<ast::FloatVal>().val;
                            debug_assert!(expr.replacement.is_none());
                            // TODO: no allocation
                            expr.replacement = Some(
                                self.alloc(ast_new!(FloatVal { val: -f, span: Span::ZERO }))?
                                    .upcast(),
                            )
                        }
                        expr_ty
                    },
                    UnaryOpKind::Deref | UnaryOpKind::Not | UnaryOpKind::Neg => {
                        return err(InvalidPreOp { ty: expr_ty, op: *op }, span);
                    },
                    UnaryOpKind::Try => todo!("try"),
                });
            },
            AstEnum::BinOp { lhs, op, rhs, .. } => {
                assert!(!is_const, "todo: BinOp in const");
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
            },
            AstEnum::Range { start, end, is_inclusive, .. } => {
                let (elem_ty, rkind): (Ptr<ast::Type>, _) = match (start, end) {
                    (None, None) => (p.u0, RangeKind::Full),
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
                let range_ty = self.alloc(type_new!(RangeTy { elem_ty, rkind }))?;
                expr.ty = Some(range_ty.upcast_to_type());
            },
            AstEnum::Assign { lhs, rhs, .. } => {
                assert!(!is_const, "todo: Assign in const");
                let lhs_ty = *analyze!(*lhs, None);
                let rhs_ty = *analyze!(*rhs, Some(lhs_ty));
                if !ty_match(lhs_ty, rhs_ty) {
                    return err(MismatchedTypes { expected: lhs_ty, got: rhs_ty }, rhs.full_span());
                }
                // todo: check if binop can be applied to type
                expr.ty = Some(p.void_ty);
            },
            AstEnum::BinOpAssign { lhs, rhs, .. } => {
                assert!(!is_const, "todo: BinOpAssign in const");
                let lhs_ty = *analyze!(*lhs, None);
                let rhs_ty = *analyze!(*rhs, Some(lhs_ty));
                if !ty_match(lhs_ty, rhs_ty) {
                    return err(MismatchedTypesBinOp { lhs_ty, rhs_ty }, rhs.full_span());
                }
                // todo: check if binop can be applied to type
                expr.ty = Some(p.void_ty);
            },
            AstEnum::Decl { .. } => self.analyze_decl(expr.downcast::<ast::Decl>(), is_const)?,
            //AstEnum::Extern { .. } => self.analyze_extern(expr.downcast::<ast::Extern>())?,
            AstEnum::If { condition, then_body, else_body, .. } => {
                assert!(!is_const, "todo: if in const");
                let bool_ty = p.bool;
                let cond_ty = *analyze!(*condition, Some(bool_ty));
                if cond_ty != bool_ty {
                    let span = condition.full_span();
                    return err(MismatchedTypes { expected: bool_ty, got: cond_ty }, span);
                }
                self.analyze(*then_body, &None, is_const)?;
                let then_ty = then_body.ty.u();
                expr.ty = if let Some(else_body) = *else_body {
                    self.analyze(else_body, &Some(then_ty), is_const)?;
                    let else_ty = else_body.ty.u();
                    let Some(common_ty) = common_type(then_ty, else_ty) else {
                        return err(
                            IncompatibleBranches { expected: then_ty, got: else_ty },
                            else_body.full_span(),
                        );
                    };
                    Some(common_ty)
                } else if then_body.ty == Some(p.void_ty) || then_body.ty == Some(p.never) {
                    Some(p.void_ty)
                } else {
                    return err(MissingElseBranch, expr.full_span());
                }
            },
            AstEnum::Match { .. } => todo!(),
            AstEnum::For { source, iter_var, body, .. } => {
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
                    _ => todo!("for over non-array"),
                };

                self.open_scope();
                let res: SemaResult<()> = try {
                    let mut iter_var_decl = ast::Decl::from_ident(*iter_var);
                    iter_var_decl.var_ty = Some(elem_ty);
                    // SAFETY: `iter_var_decl` lives until `close_scope` is called so this is fine
                    self.analyze_decl(Ptr::from_ref(&iter_var_decl), false)?;

                    self.analyze(*body, &Some(p.void_ty), is_const)?;
                    if body.ty.u() != p.void_ty && body.ty.u() != p.never {
                        return err(CannotReturnFromLoop, body.full_span());
                    }
                };
                self.close_scope()?;
                res?;
                expr.ty = Some(p.void_ty);
            },
            AstEnum::While { condition, body, .. } => {
                let bool_ty = p.bool;
                analyze!(*condition, Some(bool_ty));
                if condition.ty.u() != bool_ty {
                    let got = condition.ty.u();
                    return err(MismatchedTypes { expected: bool_ty, got }, condition.full_span());
                }
                self.open_scope();
                self.analyze(*body, &Some(p.void_ty), is_const)?; // TODO: check if scope is closed on `NotFinished?`
                if body.ty.u() != p.void_ty && body.ty.u() != p.never {
                    return err(CannotReturnFromLoop, body.full_span());
                }
                self.close_scope()?;
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
            AstEnum::ImportDirective { .. } => {
                expr.ty = Some(p.module);
            },

            AstEnum::IntVal { .. } => {
                let ty = ty_hint
                    .filter(|t| matches!(t.kind, AstKind::IntTy | AstKind::FloatTy))
                    .unwrap_or(p.int_lit);
                expr.ty = Some(ty);
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
            AstEnum::Fn { params, ret_ty_expr, ret_ty, body, .. } => {
                assert!(is_const, "todo: non-const function");
                let fn_ptr = expr.downcast::<ast::Fn>();
                if let Some(ret) = *ret_ty_expr {
                    *ret_ty = Some(self.analyze_type(ret)?);
                }
                self.function_stack.push(fn_ptr);
                self.open_scope();
                let res: SemaResult<()> = try {
                    for param in params.iter_mut() {
                        assert!(!param.is_const, "todo: const param");
                        self.analyze_decl(*param, false)?;
                    }

                    if let Some(body) = *body {
                        self.analyze(body, ret_ty, false)?;
                        check_and_set_target!(
                            body.ty.u(),
                            ret_ty,
                            body.try_downcast::<ast::Block>()
                                .and_then(|b| b.stmts.last().copied())
                                .unwrap_or(body)
                                .full_span()
                        );
                        ret_ty.as_mut().u().finalize();
                    } else {
                        panic_debug("this function has already been analyzed as a function type")
                    }
                };
                self.close_scope()?;
                self.function_stack.pop();
                res?;
                debug_assert!(ret_ty.is_some());

                let is_fn_ty = *ret_ty == p.type_ty && body.u().kind != AstKind::Block;
                if is_fn_ty {
                    *ret_ty_expr = Some(body.u());
                    *ret_ty = Some(body.cv().downcast_type());
                    *body = None;
                    expr.ty = Some(p.type_ty);
                } else {
                    // expr.ty = Some(fn_ptr.upcast_to_type());
                    //expr.ty = Some(p.type_ty);
                    expr.ty = Some(p.fn_val);
                }
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
                let len_ty = *analyze!(*len, Some(u64_ty), true);
                if !ty_match(len_ty, u64_ty) {
                    return err(MismatchedTypes { expected: u64_ty, got: len_ty }, len.span);
                }
                if !len.rep().is_const_val() {
                    return err(NotAConstExpr, len.full_span());
                }
                expr.ty = Some(p.type_ty);
            },
            AstEnum::StructDef { fields, .. } => {
                let mut field_names = HashSet::new();
                for field in fields.iter().copied() {
                    let is_duplicate = !field_names.insert(field.ident.text.as_ref());
                    if is_duplicate {
                        return err(DuplicateField, field.ident.span);
                    }
                    if field.is_const {
                        todo!("const struct field")
                    }
                    let _f = self.var_decl_to_value(field)?;
                }
                //self.struct_stack.last_mut().unwrap_debug().push(ty);
                expr.ty = Some(p.type_ty);
            },
            AstEnum::UnionDef { fields, .. } => {
                let mut field_names = HashSet::new();
                for field in fields.iter().copied() {
                    let is_duplicate = !field_names.insert(field.ident.text.as_ref());
                    if is_duplicate {
                        return err(DuplicateField, field.ident.span);
                    }
                    if field.is_const {
                        todo!("const struct field")
                    }
                    if let Some(d) = field.init {
                        return err(UnionFieldWithDefaultValue, d.full_span());
                    }
                    let _f = self.var_decl_to_value(field)?;
                }
                expr.ty = Some(p.type_ty);
            },
            AstEnum::EnumDef { variants, .. } => {
                let mut variant_names = HashSet::new();
                for variant in variants.iter_mut() {
                    let is_duplicate = !variant_names.insert(variant.ident.text.as_ref());
                    if is_duplicate {
                        return err(DuplicateEnumVariant, variant.ident.span);
                    }
                    if variant.var_ty.is_none() {
                        variant.var_ty = Some(p.void_ty);
                    }
                    let _ = self.var_decl_to_value(*variant)?;
                }
                //self.enum_stack.last_mut().unwrap_debug().push(ty);
                expr.ty = Some(p.type_ty);
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
        let Some(common_ty) = common_type(expr_ty, expected_ty) else {
            return err(MismatchedTypes { expected: expected_ty, got: expr_ty }, expr.full_span());
        };
        expr.as_mut().ty = Some(common_ty);
        Ok(())
    }

    pub fn try_custom_bitwith_int_type(&self, name: Ptr<str>) -> Option<ast::IntTy> {
        let is_signed = match name.bytes().next() {
            Some(b'i') => true,
            Some(b'u') => false,
            _ => return None,
        };
        let bits = name[1..].parse().ok()?;
        debug_assert!(![8, 16, 32, 64, 128].contains(&bits));
        Some(type_new!(IntTy { bits, is_signed }))
    }

    pub fn validate_initializer(
        &mut self,
        struct_ty: Ptr<ast::Type>,
        mut initializer_values: Ptr<[(Ptr<ast::Ident>, Option<Ptr<Ast>>)]>,
        is_const: bool,
        initializer_span: Span,
    ) -> SemaResult<()> {
        let fields = match struct_ty.matchable().as_ref() {
            TypeEnum::StructDef { fields, .. } => &**fields,
            TypeEnum::SliceTy { elem_ty, is_mut, .. } => {
                &self.slice_fields(elem_ty.downcast_const_val().downcast_type(), *is_mut)?
            },
            _ => unreachable_debug(),
        };

        let mut is_initialized_field = vec![false; fields.len()];
        for (f, init) in initializer_values.iter_mut() {
            match try {
                let field = f.text;
                let Some((f_idx, f_decl)) = fields.find_field(&field) else {
                    err(UnknownField { ty: struct_ty, field }, f.span)?
                };

                if is_initialized_field[f_idx] {
                    err(DuplicateInInitializer, f.span)?
                }
                is_initialized_field[f_idx] = true;

                if init.is_none() {
                    *init = Some(f.upcast())
                }
                self.analyze_and_finalize_with_known_type(init.u(), f_decl.var_ty.u(), is_const)?;
            } {
                Ok(()) => {},
                NotFinished => NotFinished?,
                Err(err) => self.cctx.error2(&err),
            }
        }

        for missing_field in is_initialized_field
            .into_iter()
            .enumerate()
            .filter(|(_, is_init)| !is_init)
            .map(|(idx, _)| fields[idx])
            .filter(|f| f.init.is_none())
        {
            let field = missing_field.ident.text;
            self.cctx
                .error2(&err_val(MissingFieldInInitializer { field }, initializer_span));
        }
        Ok(())
    }

    /// Returns the [`SemaValue`] repesenting the new variable, not the entire
    /// declaration. This also doesn't insert into `self.symbols`.
    fn var_decl_to_value(&mut self, decl_ptr: Ptr<ast::Decl>) -> SemaResult<()> {
        let p = p();
        let decl = decl_ptr.as_mut();
        decl.ty = Some(p.void_ty);
        if let Some(t) = decl.var_ty_expr {
            let ty = self.analyze_type(t)?;
            decl.var_ty = Some(ty);
            if decl.is_extern
                && let Some(f) = ty.try_downcast::<ast::Fn>()
            {
                // TODO: remove this special case
                debug_assert!(f.body.is_none());
                debug_assert!(decl.init.is_none());
                decl.init = Some(f.upcast());
                decl.var_ty = Some(p.fn_val);
                decl.is_const = true;
                return Ok(());
            }
            if ty == p.never {
                decl.ty = Some(p.never);
            }
        }
        if let Some(init) = decl.init {
            self.analyze(init, &decl.var_ty, decl.is_const)?;
            check_and_set_target!(init.ty.u(), &mut decl.var_ty, init.full_span());
            decl.var_ty.as_mut().u().finalize();
            Ok(())
        } else if decl.var_ty.is_some() {
            debug_assert!(!decl.is_const);
            Ok(())
        } else {
            err(VarDeclNoType, decl_ptr.upcast().full_span())
        }
    }

    fn analyze_decl(&mut self, mut decl: Ptr<ast::Decl>, is_const: bool) -> SemaResult<()> {
        let p = p();
        let res = try {
            if is_const && !(decl.is_const || decl.is_extern) {
                err(NotAConstExpr, decl.upcast().full_span())?;
            }
            self.var_decl_to_value(decl)?
        };
        #[cfg(debug_assertions)]
        if self.cctx.debug_types {
            let label = match &res {
                Ok(()) => format!("type: {}", decl.var_ty.u()),
                NotFinished => "not finished".to_string(),
                Err(e) => format!("err: {}", e.kind),
            };
            display(decl.ident.span).label(&label).finish();
        }
        let res = match res {
            Err(err) => {
                self.cctx.error2(&err);
                decl.var_ty = Some(p.never);
                decl.ty = Some(p.never);
                /* TODO: make sure the const_val of this decl is also `never`
                if let Some(init) = decl.init.as_mut() {
                    init.replacement = Some(p.never.upcast());
                }
                */
                Ok(())
            },
            NotFinished => NotFinished,
            Ok(()) => {
                decl.ty = Some(p.void_ty);
                Ok(())
            },
        };
        /*
        let is_top_level = self.function_stack.len() == 0;
        if !is_top_level {
            debug_assert!(!self.symbols.get_cur_scope().contains(&decl));
            self.symbols.push(decl);
        }
        */
        self.symbols.push(decl);
        res
    }

    /// works for function calls and call initializers
    /// ...(arg1, ..., argX, paramX1=argX1, ... paramN=argN)
    /// paramN1, ... -> default values
    fn validate_call<'i>(
        &mut self,
        params: &mut [Ptr<ast::Decl>],
        args: impl IntoIterator<Item = &'i mut Ptr<Ast>>,
        close_p_span: Span,
    ) -> SemaResult<()> {
        let mut args = args.into_iter();
        // TODO: check for duplicate named arguments
        for p in params.iter_mut() {
            let Some(arg) = args.next() else {
                if p.init.is_some() {
                    continue;
                }
                return err(MissingArg, close_p_span);
            };
            self.analyze_and_finalize_with_known_type(*arg, p.var_ty.u(), false)?;
        }
        Ok(())
    }

    fn analyze_type(&mut self, ty_expr: Ptr<Ast>) -> SemaResult<Ptr<ast::Type>> {
        let p = p();
        self.analyze(ty_expr, &Some(p.type_ty), true)?;
        let ty = ty_expr.ty.u();
        if ty != p.type_ty {
            if ty == p.never {
                return Ok(p.never);
            }
            return err(NotAType, ty_expr.full_span());
        }
        Ok(ty_expr.downcast_type())
    }

    pub fn slice_fields(
        &self,
        elem_ty: Ptr<ast::Type>,
        is_mut: bool,
    ) -> SemaResult<[Ptr<ast::Decl>; 2]> {
        let p = p();
        let elem_ptr_ty = self.alloc(type_new!(PtrTy { pointee: elem_ty.upcast(), is_mut }))?;
        let mut ptr = self.alloc(Decl::new(p.slice_ptr_field_ident, Span::ZERO))?;
        ptr.var_ty = Some(elem_ptr_ty.upcast_to_type());
        Ok([ptr, p.slice_len_field])
    }

    #[inline]
    fn get_symbol(&self, name: Ptr<str>, err_span: Span) -> SemaResult<Ptr<ast::Decl>> {
        match self
            .symbols
            .get(&name)
            .or_else(|| self.cur_file.u().find_symbol(&name, self.file_level_symbols))
            .or_else(|| self.cctx.primitives_scope.find_symbol(&name))
        {
            Some(sym) if sym.var_ty.is_some() => Ok(sym),
            Some(_) => NotFinished,
            None => err(UnknownIdent(name), err_span),
        }
    }

    fn open_scope(&mut self) {
        self.symbols.open_scope();
        //self.struct_stack.push(vec![]);
        //self.enum_stack.push(vec![]);
        self.defer_stack.open_scope();
    }

    fn close_scope(&mut self) -> SemaResult<()> {
        let res = self.analyze_defer_exprs();
        self.symbols.close_scope();
        //self.struct_stack.pop();
        //self.enum_stack.pop();
        self.defer_stack.close_scope();
        res
    }

    #[inline]
    fn analyze_defer_exprs(&mut self) -> SemaResult<()> {
        let exprs = unsafe { forget_lifetime_mut(self.defer_stack.get_cur_scope_mut()) };
        for expr in exprs.iter().rev() {
            self.analyze(*expr, &None, false)?;
        }
        Ok(())
    }

    #[inline]
    fn alloc<T: core::fmt::Debug>(&self, val: T) -> SemaResult<Ptr<T>> {
        match self.cctx.alloc.alloc(val) {
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
