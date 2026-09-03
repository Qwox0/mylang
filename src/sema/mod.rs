//! # Semantic analysis module
//!
//! Semantic analysis validates (and changes) all stored [`ast::Type`]s in [`Expr`].

use crate::{
    ast::{
        self, Ast, AstEnum, AstKind, AstMatch, BinOpKind, DeclFlags, DeclListExt, FnFlags,
        GenericSlotFlags, InitializerFlags, OPtrExt, OPtrTypeExt, RangeKind, StructFlags, TypeEnum,
        TypeMatch, UnaryOpKind, UpcastToAst, ast_new, debug::DebugAst, is_pos_arg, type_new,
    },
    context::{CompilationContextInner, primitives as p, tmp_alloc},
    diagnostics::{HandledErr, cerror, cerror2, chint, common::*, cunimplemented, cwarn},
    display_code::{debug_expr, display},
    intern_pool::Symbol,
    parser::lexer::Span,
    ptr::{OPtr, Ptr},
    scope::{Scope, ScopeFlags, ScopeKind, error_duplicate_in_unordered_scope, setup_scopes},
    scoped_stack::ScopedStack,
    scratch_allocator::TmpPtr,
    sema::generics::{
        Polymorphable, PolymorphableMatch, PolymorphableType, accumulate_generic, generic_match,
    },
    source_file::SourceFile,
    type_::{
        AllowOptionalCoercion, common_type, common_type_restrict_optional_coerction, finalize_ty,
        struct_offset, ty_match, ty_match_quiet,
    },
    util::{
        self, BigIntExt, BitFlags, IteratorExt, OptionExt, UnwrapDebug, VecExt, debug_only_assert,
        then, ui, unreachable_debug, wrap_display,
    },
};
use err::{SemaResult, SemaResult::*};
use num::BigInt;
use std::{assert_matches::debug_assert_matches, fmt::Write, iter, ops::Not};

mod err;
pub mod generics;
pub mod primitives;

/// [`None`] return value means [`CompilationContextInner::error_mismatched_types`]
fn check_or_infer_type(
    ty: Ptr<ast::Type>,
    target_ty: &mut OPtr<ast::Type>,
    strict_target: bool,
) -> Option<()> {
    if let Some(target_ty) = target_ty {
        if strict_target {
            if !ty_match(ty, *target_ty) {
                return None;
            }
        } else {
            *target_ty = common_type(*target_ty, ty)?;
        }
    } else {
        *target_ty = Some(ty);
    }
    Some(())
}

macro_rules! check_or_infer_target {
    ($ty:expr, $target_ty:expr, $strict_target:expr, $err_span:expr $(,)?) => {{
        let strict_target: bool = $strict_target;
        let ty: Ptr<ast::Type> = $ty;
        let target_ty: &mut OPtr<ast::Type> = $target_ty;
        let Some(()) = check_or_infer_type(ty, target_ty, strict_target) else {
            return error_mismatched_types($err_span, target_ty.u(), ty).into();
        };
        target_ty.as_mut().u()
    }};
}

pub fn analyze(cctx: Ptr<CompilationContextInner>, stmts: &mut [Ptr<Ast>]) {
    // validate top level stmts
    let _ignore = cctx.primitives_scope.as_mut().verify_no_duplicates();
    for file in cctx.files().iter().copied() {
        let _ignore = file.as_mut().scope.as_mut().u().verify_no_duplicates();
        for &s in stmts[file.stmt_range.u()].iter() {
            let Some(decl) = s.try_downcast::<ast::Decl>() else {
                if !s.kind.is_allowed_top_level() {
                    cerror!(s.full_span(), "unexpected top level expression");
                    s.as_mut().ty = Some(p().err_ty);
                    s.set_replacement(p().err_ty.upcast());
                }
                continue;
            };
            debug_assert!(!decl.is_const || decl.init.is_some());
            if !decl.is_const && !decl.flags.get(DeclFlags::IS_STATIC) {
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

    let mut units = tmp_alloc()
        .alloc_slice_fill_iter(stmts.iter().map(|_s| SemaUnit {
            #[cfg(debug_assertions)]
            stmt: *_s,
            waiting_for: None,
        }))
        .unwrap();

    fn iter_unfinished_files<'a>(
        cctx: Ptr<CompilationContextInner>,
        stmts: &'a mut [Ptr<Ast>],
        units: &'a mut [SemaUnit],
    ) -> impl Iterator<Item = (Ptr<SourceFile>, UnfinishedMembers<'a, Ast>)> {
        let stmts = Ptr::from_ref(stmts);
        let units = Ptr::from_ref(units);
        cctx.as_ref().files().iter().filter_map(move |&file| {
            let stmt_range = file.as_mut().stmt_range.as_mut().u();
            then!(!stmt_range.is_empty() => (file, UnfinishedMembers {
                items: &mut stmts.as_mut()[..stmt_range.end],
                units: &mut units.as_mut()[..stmt_range.end],
                finished_count: &mut stmt_range.start,
            }))
        })
    }

    let mut finished = 0;
    while finished < units.len() + sema.unfinished_instantiation_units.len() {
        let mut continued = false;

        for (file, UnfinishedMembers { items: stmts, units, finished_count }) in
            iter_unfinished_files(cctx, stmts, &mut units)
        {
            debug_assert!(file.scope.as_ref().u().parent.is_some());
            let osh = sema.open_scope(file.as_mut().scope.as_mut().u());
            let prev_count = *finished_count;
            let res = analyze_scope(stmts, units, finished_count, |stmt, _unit| {
                sema.analyze_top_level(stmt)
            });
            continued |= res.continued;
            finished += *finished_count - prev_count;
            debug_assert!(
                sema.defer_stack.get_cur_scope().is_empty(),
                "file scope must not contain defer statements"
            );
            sema.close_scope(osh);
        }

        traverse_unfinished_insts!(sema; unit, inst => {
            // open_scope/close_scope are not needed
            debug_assert!(inst.main_scope().parent.is_some());
            debug_assert!(inst.generics_scope().u().parent.is_some());
            if !unit.waiting_for.as_ref().is_none_or(UnitDependency::resolved) {
                TraverseResult::Skip
            } else {
                continued = true;
                sema.analyze(inst.upcast(), &None, false).into()
            }
        });

        if !continued {
            struct CycleDetectionState {
                has_non_cycle_error: bool,
                has_cycle: bool,
            }
            let mut state = CycleDetectionState { has_non_cycle_error: false, has_cycle: false };

            fn unknown_symbol_error<'u>(
                unit: &'u SemaUnit,
                expr: Ptr<Ast>,
                state: &mut CycleDetectionState,
            ) -> TraverseResult {
                let dep = unit.waiting_for.as_ref().u();
                let was_handled = dep.emit_missing_dep_error(expr);
                if was_handled {
                    state.has_non_cycle_error = true;
                    TraverseResult::CompleteItem
                } else {
                    state.has_cycle = true;
                    TraverseResult::Skip
                }
            }
            for (_, mut stmts) in iter_unfinished_files(cctx, stmts, &mut units) {
                stmts.traverse_unfinished(|s, unit| unknown_symbol_error(unit, s, &mut state));
            }
            traverse_unfinished_insts!(sema; unit, inst => unknown_symbol_error(unit, inst.upcast(), &mut state));

            // see `no_cycle_error_after_other_dependency_errors`
            if state.has_non_cycle_error {
                continue;
            } else if !state.has_cycle {
                break;
            }

            cerror!(Span::ZERO, "cycle(s) detected:"); // TODO: detect individual cycles
            fn print_cycle_element(stmt: Ptr<Ast>, unit: &SemaUnit) -> TraverseResult {
                let dep = unit.waiting_for.as_ref().u();
                if let UnitDependency::Scope(ty) = dep {
                    ty.member_sema_state().u().traverse_unfinished(|member, unit| {
                        print_cycle_element(member.upcast(), unit)
                    });
                }

                let span = stmt
                    .try_downcast::<ast::Decl>()
                    .map(|d| d.ident.span)
                    .unwrap_or_else(|| stmt.full_span());

                let mut label = format!("waiting for ");
                match dep {
                    UnitDependency::ExprType(_) => write!(&mut label, ": {dep:?}"),
                    UnitDependency::VarType(d) => write!(&mut label, "type of {}", d.ident.sym),
                    UnitDependency::ConstVal(d) => {
                        write!(&mut label, "constant value of {}", d.ident.sym)
                    },
                    UnitDependency::RetTy(f) => write!(
                        &mut label,
                        "return type of {}",
                        f.decl.map(|d| d.ident.sym.text()).unwrap_or("lamda (TODO: more info)")
                    ),
                    UnitDependency::TypeLayout(ty) => {
                        write!(&mut label, "memory layout of type `{ty}`")
                    },
                    UnitDependency::EnumVariantTag(variant) => {
                        write!(&mut label, "tag value of {}", variant.ident.sym) // TODO: print enum ty
                    },
                    UnitDependency::_AssociatedConst(_) | UnitDependency::_Dot(_) => {
                        unreachable_debug()
                    },
                    UnitDependency::Scope(_) => write!(&mut label, "some members"),
                }
                .unwrap();

                display(span).label(&label).finish();
                TraverseResult::Skip
            }
            for (_, mut stmts) in iter_unfinished_files(cctx, stmts, &mut units) {
                stmts.traverse_unfinished(print_cycle_element);
            }
            traverse_unfinished_insts!(sema; unit, inst => print_cycle_element(inst.upcast(), unit));

            break;
        }
    }
}

#[derive(Debug)]
struct AnalyzeScopeResult {
    ok: bool,
    finished: bool,
    continued: bool,
}

impl AnalyzeScopeResult {
    fn as_sema_result(self, ty_with_scope: Ptr<ast::Type>) -> SemaResult<()> {
        debug_assert!(ty_with_scope.get_scope().is_some());
        match self.ok {
            false => Err(HandledErr),
            true if self.finished => Ok(()),
            _ => NotFinished(UnitDependency::Scope(ty_with_scope)),
        }
    }
}

fn analyze_scope2(
    decls: &mut [Ptr<ast::Decl>],
    units: &mut Option<TmpPtr<[SemaUnit]>>,
    finished_count: &mut usize,
    analyze_item: impl FnMut(Ptr<ast::Decl>, &SemaUnit) -> SemaResult<()>,
) -> AnalyzeScopeResult {
    if units.is_none() {
        *units = Some(
            tmp_alloc()
                .alloc_slice_fill_iter(decls.iter().map(|_d| SemaUnit {
                    #[cfg(debug_assertions)]
                    stmt: _d.upcast(),
                    waiting_for: None,
                }))
                .unwrap(),
        );
    }
    analyze_scope(decls, units.as_mut().u(), finished_count, analyze_item)
}

fn analyze_scope<T>(
    items: &mut [Ptr<T>],
    units: &mut [SemaUnit],
    finished_count: &mut usize,
    mut analyze_item: impl FnMut(Ptr<T>, &SemaUnit) -> SemaResult<()>,
) -> AnalyzeScopeResult {
    let mut ok = true;
    let mut continued = false;

    traverse_unfinished_scope_members(items, units, finished_count, |item, unit| {
        if !unit.waiting_for.as_ref().is_none_or(UnitDependency::resolved) {
            return TraverseResult::Skip;
        }
        continued = true;

        let res = analyze_item(item, unit);
        if matches!(res, Err(_)) {
            ok = false;
        }
        res.into()
    });

    debug_assert!(*finished_count <= items.len());
    AnalyzeScopeResult { ok, finished: *finished_count == items.len(), continued }
}

pub enum TraverseResult {
    /// keep member unfinished and keep current dependency
    Skip,
    /// keep member unfinished
    UpdateDep(UnitDependency),
    CompleteItem,
}

impl<T> From<SemaResult<T>> for TraverseResult {
    fn from(res: SemaResult<T>) -> Self {
        match res {
            NotFinished(dep) => TraverseResult::UpdateDep(dep),
            Ok(_) | Err(HandledErr) => TraverseResult::CompleteItem,
        }
    }
}

impl TraverseResult {
    fn from_finished(finished: bool) -> Self {
        if finished { TraverseResult::CompleteItem } else { TraverseResult::Skip }
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
///
/// . = sema finished; x = sema waiting
fn traverse_unfinished_scope_members<T>(
    items: &mut [Ptr<T>],
    units: &mut [SemaUnit],
    finished_members: &mut usize,
    mut analyze_item: impl FnMut(Ptr<T>, &SemaUnit) -> TraverseResult,
) {
    debug_assert!(*finished_members <= items.len());
    debug_assert_eq!(items.len(), units.len());

    let mut idx = *finished_members;
    while let Some(unit) = units.get_mut(idx) {
        let item = *items.get(idx).u();
        match analyze_item(item, unit) {
            TraverseResult::Skip => {},
            TraverseResult::UpdateDep(dep) => unit.update_dep(dep),
            TraverseResult::CompleteItem => {
                finish_item_in_scope(idx, items, units, finished_members);
            },
        }
        idx += 1;
    }
}

// Has to be a macro because `sema` reference cannot be used while `unit` reference is active.
// Normal predicate signatures cannot express this kind of information.
macro_rules! traverse_unfinished_insts {
    ($sema:ident; $unit:ident, $inst:ident => $body:expr) => {{
        debug_assert_eq!(
            $sema.unfinished_instantiations.len(),
            $sema.unfinished_instantiation_units.len()
        );
        let mut idx = 0;
        while let Some($unit) = $sema.unfinished_instantiation_units.get(idx) {
            let $inst = *$sema.unfinished_instantiations.get(idx).u();
            let res: TraverseResult = $body;
            match res {
                TraverseResult::Skip => idx += 1,
                TraverseResult::UpdateDep(dep) => {
                    // Note: body might mutate unfinished_instantiation_units which
                    // invalidates `unit`. Rust catches this.
                    $sema.unfinished_instantiation_units.get_mut(idx).u().update_dep(dep);
                    idx += 1;
                },
                TraverseResult::CompleteItem => {
                    $sema.unfinished_instantiations.swap_remove(idx);
                    $sema.unfinished_instantiation_units.swap_remove(idx);
                },
            }
        }
    }};
}
use traverse_unfinished_insts;

#[derive(Debug)]
pub struct UnfinishedMembers<'a, T> {
    pub items: &'a mut [Ptr<T>],
    pub units: &'a mut [SemaUnit],
    pub finished_count: &'a mut usize,
}

impl<'a, T> UnfinishedMembers<'a, T> {
    #[inline]
    pub fn unfinished_units(&self) -> &[SemaUnit] {
        &self.units[*self.finished_count..]
    }

    pub fn iter(self) -> impl ExactSizeIterator<Item = (&'a mut SemaUnit, Ptr<T>)> {
        self.units.iter_mut().zip_exact(self.items.iter().copied())
    }

    #[inline]
    fn traverse_unfinished(&mut self, f: impl FnMut(Ptr<T>, &SemaUnit) -> TraverseResult) {
        traverse_unfinished_scope_members(self.items, self.units, self.finished_count, f);
    }
}

pub fn finish_item_in_scope<T>(
    item_idx: usize,
    items: &mut [Ptr<T>],
    units: &mut [SemaUnit],
    finished_count: &mut usize,
) {
    debug_assert!(*finished_count <= item_idx);
    items.swap(item_idx, *finished_count);
    units.swap(item_idx, *finished_count);
    *finished_count += 1;
}

/// Semantic analyzer
pub struct Sema {
    decl_stack: Vec<Ptr<ast::Decl>>,
    defer_stack: ScopedStack<Ptr<Ast>>,
    loop_stack: Vec<Ptr<ast::Ast>>,

    cctx: Ptr<CompilationContextInner>,
    cur_scope: Ptr<Scope>,

    unfinished_instantiations: Vec<Ptr<Polymorphable>>,
    unfinished_instantiation_units: Vec<SemaUnit>,

    #[cfg(debug_assertions)]
    debug_scope_level: usize,
}

#[derive(Debug, PartialEq, Eq)]
pub enum UnitDependency {
    ExprType(Ptr<ast::Ast>),
    VarType(Ptr<ast::Decl>),
    ConstVal(Ptr<ast::Decl>),
    RetTy(Ptr<ast::Fn>),
    TypeLayout(Ptr<ast::Type>),
    EnumVariantTag(Ptr<ast::Decl>),

    _AssociatedConst(Ptr<ast::Dot>),
    _Dot(Ptr<ast::Dot>),
    Scope(Ptr<ast::Type>),
}

impl UnitDependency {
    #[allow(non_snake_case)]
    pub fn AssociatedConst(dot: Ptr<ast::Dot>) -> Result<UnitDependency, HandledErr> {
        let ok: Option<()> = try {
            dot.lhs?.downcast_type().get_associated_external_consts()?;
        };

        if ok.is_some() {
            Result::Ok(UnitDependency::_AssociatedConst(dot))
        } else {
            Result::Err(error_missing_associated_const(dot))
        }
    }

    #[allow(non_snake_case)]
    pub fn Dot(dot: Ptr<ast::Dot>) -> Result<UnitDependency, HandledErr> {
        let ok: Option<()> = try {
            dot.lhs?.ty?.flatten_transparent().get_associated_external_consts()?;
        };

        if ok.is_some() {
            Result::Ok(UnitDependency::_Dot(dot))
        } else {
            Result::Err(error_missing_field(dot))
        }
    }

    pub fn resolved(&self) -> bool {
        match self {
            UnitDependency::ExprType(expr) => expr.ty.is_some(),
            UnitDependency::VarType(d) => d.var_ty.is_some(),
            UnitDependency::ConstVal(d) => d.const_val().is_ok(),
            UnitDependency::RetTy(f) => f.ret_ty.is_some(),
            UnitDependency::TypeLayout(ty) => type_layout_finished(*ty),
            UnitDependency::EnumVariantTag(variant) => try_get_enum_variant_tag(*variant).is_some(),
            UnitDependency::_AssociatedConst(dot) => {
                debug_assert_eq!(dot.lhs.u().ty, p().type_ty);
                let ty = dot.lhs.u().downcast_type();
                ty.get_associated_external_consts().u().find_field(dot.rhs.sym).is_some()
            },
            UnitDependency::_Dot(dot) => {
                debug_assert_ne!(dot.lhs.u().ty, p().type_ty);
                let ty = dot.lhs.u().ty.u().flatten_transparent();
                debug_assert!(ty.get_fields().is_none_or(|f| f.find_field(dot.rhs.sym).is_none()));
                ty.get_associated_external_consts().u().find_field(dot.rhs.sym).is_some()
            },
            UnitDependency::Scope(ty_with_scope) => ty_with_scope
                .member_sema_state()
                .u()
                .unfinished_units()
                .iter()
                .any(|u| u.waiting_for.as_ref().is_none_or(UnitDependency::resolved)),
        }
    }

    pub fn emit_missing_dep_error(&self, stmt: Ptr<Ast>) -> bool {
        match self {
            UnitDependency::_AssociatedConst(dot) => {
                error_missing_associated_const(*dot);
            },
            UnitDependency::_Dot(dot) => {
                error_missing_field(*dot);
            },
            UnitDependency::Scope(s) => {
                let mut member_state = s.member_sema_state().u();
                debug_assert!(member_state.unfinished_units().len() >= 1);
                member_state.traverse_unfinished(|member, unit| {
                    TraverseResult::from_finished(
                        unit.waiting_for.as_ref().u().emit_missing_dep_error(member.upcast()),
                    )
                });
                // skips marking type definition as error, because it seams unnecessary
                return member_state.unfinished_units().is_empty();
            },
            UnitDependency::ExprType(_)
            | UnitDependency::VarType(_)
            | UnitDependency::ConstVal(_)
            | UnitDependency::RetTy(_)
            | UnitDependency::TypeLayout(_)
            | UnitDependency::EnumVariantTag(_) => return false,
        }

        // mark expression as error to resolve other dependencies
        let p = p();
        stmt.as_mut().ty = Some(p.err_ty);
        match stmt.matchable2() {
            AstMatch::Decl(d) => d.as_mut().var_ty = Some(p.err_ty),
            AstMatch::Fn(f) if f.ret_ty.is_none() => f.as_mut().ret_ty = Some(p.err_ty),
            _ => {},
        }
        true
    }

    /// checks that [`UnitDependency::resolved`] doesn't panic
    pub fn validate(&self) -> bool {
        #[cfg(debug_assertions)]
        let _ = self.resolved();
        true
    }
}

#[derive(Debug)]
pub struct SemaUnit {
    #[allow(unused)]
    #[cfg(debug_assertions)]
    stmt: Ptr<Ast>,

    pub waiting_for: Option<UnitDependency>,
}

impl SemaUnit {
    fn update_dep(&mut self, new_dep: UnitDependency) {
        debug_only_assert!(new_dep.validate());

        #[cfg(debug_assertions)]
        if let Some(prev_dep) = &self.waiting_for
            && !matches!(prev_dep, UnitDependency::Scope(_))
        {
            debug_assert_ne!(new_dep, *prev_dep);
        }
        self.waiting_for = Some(new_dep);
    }
}

impl Sema {
    pub fn new(cctx: Ptr<CompilationContextInner>) -> Sema {
        Sema {
            decl_stack: vec![],
            defer_stack: ScopedStack::default(),
            loop_stack: vec![],
            cctx,
            cur_scope: cctx.primitives_scope,
            unfinished_instantiations: vec![],
            unfinished_instantiation_units: vec![],
            #[cfg(debug_assertions)]
            debug_scope_level: 0,
        }
    }

    pub fn analyze_top_level(&mut self, s: Ptr<Ast>) -> SemaResult<()> {
        self.analyze(s, &Some(p().void_ty), true).map(|_| ())
    }

    pub fn analyze<'a>(
        &mut self,
        expr: Ptr<Ast>,
        ty_hint: &OPtr<ast::Type>,
        is_const: bool,
    ) -> SemaResult<&'a mut Ptr<ast::Type>> {
        let res = self._analyze_inner(expr, ty_hint, is_const);
        #[cfg(debug_assertions)]
        if self.cctx.args.debug_types {
            let label = match &res {
                Ok(()) => format!("type: {}", expr.ty.u()),
                NotFinished(dep) => format!("not finished ({dep:?})"),
                Err(e) => format!("err: {:?}", e),
            };

            display(expr.full_span()).label(&label).finish();
        }

        #[cfg(debug_assertions)]
        if !matches!(res, NotFinished(_)) {
            self.check_post_sema_invariance(expr);
        }

        res.map(|()| {
            let ty = expr.as_mut().ty.as_mut().u();
            debug_assert_ne!(ty.kind, AstKind::ArrayLikeContainer);
            ty
        })
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

        if let Some(rep) = expr.try_rep() {
            debug_assert!(expr.ty.is_some());
            if rep.ty.u().propagates_out() {
                expr.ty = Some(rep.ty.u()); // see `tests::sema::correctly_handle_error_in_later_cycles`
            }
            return Ok(());
        }

        let p = p();

        macro_rules! not_never {
            ($ty:expr) => {{
                let ty = $ty;
                if ty.propagates_out() {
                    let t: &Ptr<ast::Type> = &ty;
                    expr.as_mut().ty = Some(*t);
                    return Ok(());
                }
                ty
            }};
        }

        /// Like [`Sema::analyze`] but returns on error and never
        macro_rules! analyze {
            ($expr:expr, $ty_hint:expr) => {
                analyze!($expr, $ty_hint, is_const)
            };
            ($expr:expr, $ty_hint:expr, $is_const:expr) => {{
                let e: Ptr<Ast> = $expr;
                not_never!(self.analyze(e, &$ty_hint, $is_const)?)
            }};
        }

        match expr.matchable().as_mut() {
            AstEnum::Ident { sym, decl, span, .. } => match self.cur_scope.find_decl(*sym) {
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
                    let is_extern = sym.init.is_some_and(|i| i.kind == AstKind::ExternDirective);
                    debug_assert!(!is_extern || sym.is_const);
                    if is_const && is_extern {
                        return cerror2!(
                            *span,
                            "the use of extern symbols in constants is currently not implemented"
                        );
                    } else if sym.is_const {
                        expr.set_replacement(sym.const_val()?.upcast());
                    } else if is_const {
                        // no const-check here. see `prefer_type_error_over_non_const_error`
                    };
                },
            },
            AstEnum::Block { stmts, finished, has_trailing_semicolon, decl_scope, .. } => {
                debug_assert!(decl_scope.flags.get(ScopeFlags::WAS_CHECKED_FOR_DUPLICATES));
                let osh = self.open_scope(decl_scope);
                let res: SemaResult<()> = try {
                    let max_idx = stmts.len().wrapping_sub(1);
                    while let Some(s) = stmts.get(*finished) {
                        let expected_ty = if max_idx == *finished { ty_hint } else { &None };
                        match self.analyze(s, expected_ty, false) {
                            Ok(_ty) => {
                                // s.ty = s.ty.finalize();
                                debug_assert!(s.ty.is_some());
                            },
                            NotFinished(dep) => NotFinished(dep)?,
                            Err(HandledErr) => s.as_mut().ty = Some(p.err_ty),
                        }

                        if let Some(decl) = s.try_downcast::<ast::Decl>()
                            && decl.on_type.is_none()
                        {
                            decl_scope.add_decl_to_block(decl);
                        }

                        *finished += 1;
                    }
                };
                self.close_scope(osh);
                res?;
                let last_ty = stmts.last().map(|s| s.ty.u()).unwrap_or(p.void_ty);
                expr.ty = Some(if !*has_trailing_semicolon || last_ty.propagates_out() {
                    last_ty
                } else {
                    p.void_ty
                })
            },
            AstEnum::PositionalInitializer { args, resolved_struct_inst, .. } => {
                let lhs = self.analyze_initializer_lhs(expr, *ty_hint)?;
                if let Some(s) = lhs.try_downcast_type2()
                    && s.kind.is_struct_kind()
                {
                    let inst = self.validate_pos_initializer(
                        s,
                        args,
                        span.end(),
                        is_const,
                        lhs, // only used for span
                    )?;
                    *resolved_struct_inst = Some(inst);
                    expr.ty = Some(inst);

                    if is_const {
                        let fields = s.downcast_struct_def().fields;
                        let all_args =
                            fields.iter().enumerate().map(|(idx, f)| args.get(idx).or(f.init).u());
                        let cv = self.create_aggregate_const_val(all_args)?;
                        expr.set_replacement(cv.upcast());
                    }
                } else if let Some(ptr_ty) = lhs.ty.try_downcast::<ast::PtrTy>()
                    && let Some(s) = ptr_ty.pointee.try_downcast_type()
                    && s.kind.is_struct_kind()
                {
                    self.validate_mutation(MutationKind::Initialize, lhs, expr)?;

                    if is_const {
                        return error_const_ptr_initializer(expr).into();
                    } else {
                        let inst = self.validate_pos_initializer(
                            s,
                            args,
                            span.end(),
                            false,
                            expr, // only used for span
                        )?;
                        *resolved_struct_inst = Some(inst);

                        expr.ty = Some(
                            if let Some(struct_def) = s.try_downcast::<ast::StructDef>()
                                && struct_def.flags.get(StructFlags::IS_GENERIC)
                            {
                                debug_assert_eq!(inst.kind, AstKind::StructDef);
                                type_new!(PtrTy { is_mut: ptr_ty.is_mut, pointee: inst.upcast() })
                                    .upcast_to_type()
                            } else {
                                ptr_ty.upcast_to_type()
                            },
                        );
                    }
                } else if lhs.ty.u().propagates_out() {
                    expr.ty = Some(lhs.ty.u());
                    return Ok(());
                } else if let Some(lhs_ty) = lhs.try_downcast_type()
                    && lhs_ty.propagates_out()
                {
                    expr.ty = Some(lhs_ty);
                    return Ok(());
                } else {
                    return error_cannot_apply_initializer(lhs, expr).into();
                };
            },
            AstEnum::NamedInitializer { fields, resolved_struct_inst, .. } => {
                let initializer = expr.downcast::<ast::NamedInitializer>();
                let lhs = self.analyze_initializer_lhs(expr, *ty_hint)?;
                if let Some(struct_ty) = lhs.try_downcast_type2()
                    && struct_ty.kind.is_struct_kind()
                {
                    initializer.as_mut().flags.set(InitializerFlags::IS_TYPE_INIT);
                    let inst =
                        self.validate_named_initializer(struct_ty, *fields, is_const, initializer)?;
                    *resolved_struct_inst = Some(inst);
                    expr.ty = Some(inst);
                } else if let Some(ptr_ty) = lhs.ty.try_downcast::<ast::PtrTy>()
                    && let Some(struct_ty) = ptr_ty.pointee.try_downcast_type()
                    && struct_ty.kind.is_struct_kind()
                {
                    initializer.as_mut().flags.set(InitializerFlags::IS_PTR_INIT);
                    self.validate_mutation(MutationKind::Initialize, lhs, expr)?;
                    if is_const {
                        return error_const_ptr_initializer(expr).into();
                    } else {
                        let inst = self.validate_named_initializer(
                            struct_ty,
                            *fields,
                            is_const,
                            initializer,
                        )?;
                        *resolved_struct_inst = Some(inst);
                        expr.ty = Some(
                            if let Some(struct_def) = struct_ty.try_downcast::<ast::StructDef>()
                                && struct_def.flags.get(StructFlags::IS_GENERIC)
                            {
                                debug_assert_eq!(inst.kind, AstKind::StructDef);
                                type_new!(PtrTy { is_mut: ptr_ty.is_mut, pointee: inst.upcast() })
                                    .upcast_to_type()
                            } else {
                                ptr_ty.upcast_to_type()
                            },
                        );
                        expr.ty = Some(if inst != struct_ty {
                            debug_assert!(
                                struct_ty
                                    .downcast::<ast::StructDef>()
                                    .flags
                                    .get(StructFlags::IS_GENERIC)
                            );
                            debug_assert!(
                                inst.downcast::<ast::StructDef>()
                                    .flags
                                    .get(StructFlags::IS_INSTANTIATION)
                            );
                            type_new!(PtrTy { is_mut: ptr_ty.is_mut, pointee: inst.upcast() })
                                .upcast_to_type()
                        } else {
                            ptr_ty.upcast_to_type()
                        });
                        expr.ty = Some(ptr_ty.upcast_to_type());
                    }
                } else if lhs.ty.u().propagates_out() {
                    expr.ty = Some(lhs.ty.u());
                    return Ok(());
                } else if let Some(lhs_ty) = lhs.try_downcast_type()
                    && lhs_ty.propagates_out()
                {
                    expr.ty = Some(lhs_ty);
                    return Ok(());
                } else {
                    return error_cannot_apply_initializer(lhs, expr).into();
                };
            },
            AstEnum::ArrayInitializer { elements, .. } => {
                let elem_ty = if let Some(elem_ty) =
                    self.analyze_array_initializer_lhs(expr, elements.len(), *ty_hint)?
                {
                    for elem in elements.into_iter() {
                        self.analyze_and_check_type(elem, elem_ty, is_const)?;
                    }
                    elem_ty
                } else {
                    let mut elem_ty = None;
                    for elem in elements.into_iter() {
                        self.analyze_and_accumulate_type(elem, &mut elem_ty, &None, is_const)?;
                    }
                    match elem_ty {
                        Some(t) => t,
                        None => {
                            expr.ty = Some(p.empty_array_ty.upcast_to_type()); // `.[]`
                            return Ok(());
                        },
                    }
                };

                //debug_assert!(expr.ty.is_none());
                if expr.ty.is_none() {
                    let len = elements.len();
                    let mut len = ast_new!(IntVal { val: BigInt::from(len) }, Span::ZERO).upcast();
                    len.ty = Some(p.usize());
                    let arr_ty = type_new!(ArrayTy { len, elem_ty: elem_ty.upcast() });
                    expr.ty = Some(arr_ty.upcast_to_type());
                }

                if is_const {
                    let cv = self.create_aggregate_const_val(*elements)?;
                    expr.set_replacement(cv.upcast());
                }
            },
            AstEnum::ArrayInitializerShort { val, count, .. } => {
                self.analyze_and_check_type(*count, p.u64, true)?;
                let len = *count;
                let len_val = len
                    .try_downcast_const_val()
                    .ok_or_else(|| error_non_const(*count, "array length"))?
                    .upcast()
                    .int();

                let mut elem_ty = self.analyze_array_initializer_lhs(expr, len_val, *ty_hint)?;

                let val_ty = *self.analyze(*val, &elem_ty, is_const)?;
                let elem_ty =
                    check_or_infer_target!(val_ty, &mut elem_ty, true, val.return_val_span())
                        .upcast();
                let arr_ty = type_new!(ArrayTy { len, elem_ty });
                expr.ty = Some(arr_ty.upcast_to_type());

                if is_const {
                    let cv = self.create_aggregate_const_val(iter::repeat_n(*val, len_val))?;
                    expr.set_replacement(cv.upcast());
                }
            },
            AstEnum::Dot { has_lhs, lhs, rhs, .. } => {
                let dot = expr.downcast::<ast::Dot>();
                let lhs_ty = if let Some(lhs) = lhs.filter(|_| *has_lhs) {
                    *analyze!(lhs, None)
                } else if let Some(ty_hint) = *ty_hint {
                    *lhs = Some(ty_hint.upcast());
                    p.type_ty
                } else {
                    return cerror2!(
                        expr.full_span(),
                        "Cannot infer enum variant or type of associated constant"
                    );
                };
                let lhs = lhs.u();
                let decl;
                let t = if lhs_ty == p.module {
                    debug_assert!(*has_lhs);
                    let m = lhs.downcast::<ast::ImportDirective>();
                    let Some(s) = self.cctx.files()[m.files_idx.u()]
                        .scope
                        .as_ref()
                        .u()
                        .find_decl_norec(rhs.sym, false)
                    else {
                        return cerror2!(
                            rhs.span,
                            "Cannot find symbol `{}` in module `{}`",
                            rhs.sym,
                            m.path.text.as_ref(),
                        );
                    };
                    decl = Some(s);
                    let ty = self.get_symbol_var_ty(s)?;
                    if let Some(cv) = s.try_const_val() {
                        expr.set_replacement(cv?.upcast());
                    }
                    ty
                } else if lhs_ty == p.type_ty {
                    let lhs = lhs.try_downcast_type_inst()?;
                    let Some(member) = find_in_namespace(lhs, rhs.sym) else {
                        return NotFinished(UnitDependency::AssociatedConst(dot)?);
                    };
                    decl = Some(member);
                    if member.is_const {
                        // associated consts/methods: `MyType.MY_CONST`, `MyType.my_method`
                        let ty = self.get_symbol_var_ty(member)?;
                        expr.set_replacement(member.const_val()?.upcast());
                        ty
                    } else if let Some(enum_ty) = lhs.try_downcast::<ast::EnumDef>() {
                        // enum variant: `MyEnum.Variant`
                        let is_simple_variant = get_var_ty(member)? == p.void_ty;
                        if is_const {
                            let variant_idx =
                                enum_ty.variants.iter().position(|v| *v == member).u();
                            let tag = ast_new!(
                                EnumVal {
                                    is_valid: is_simple_variant,
                                    enum_ty,
                                    variant_idx,
                                    data: None
                                },
                                Span::ZERO
                            )
                            .upcast();
                            tag.as_mut().ty = Some(p.enum_variant);
                            expr.set_replacement(tag);
                        }
                        if is_simple_variant { enum_ty.upcast_to_type() } else { p.enum_variant }
                    } else {
                        return cerror2!(
                            expr.full_span(),
                            "Cannot access field on type. Consider creating a value of type \
                             `{lhs}` first"
                        );
                    }
                } else if let Some(member) =
                    find_in_namespace(lhs_ty.flatten_transparent(), rhs.sym)
                {
                    if !member.is_const {
                        // field
                        if lhs_ty.kind == AstKind::PtrTy {
                            return cerror2!(
                                lhs.full_span(),
                                "automatic dereferencing of pointers is currently not allowed"
                            );
                        }
                        decl = Some(member);
                        let ty = self.get_symbol_var_ty(member)?;
                        if is_const {
                            let Some(cv) = lhs.try_downcast_const_val() else {
                                return cerror2!(
                                    lhs.full_span(),
                                    "Cannot access a field of a non-constant value in a constant \
                                     context"
                                );
                            };
                            let fields = lhs_ty.flatten_transparent().get_fields().u();
                            let f_idx = fields.iter().position(|f| *f == member).u();
                            let const_field =
                                cv.downcast::<ast::AggregateVal>().elements.get(f_idx).u();
                            expr.set_replacement(const_field.upcast());
                        }
                        ty
                    } else {
                        // method
                        let method_ty = self.get_symbol_var_ty(member)?;
                        decl = Some(member);
                        rhs.ty = Some(method_ty);
                        rhs.upcast().set_replacement(member.const_val()?.upcast());
                        if method_ty.try_downcast::<ast::Fn>().is_some() {
                            p.method_stub
                        } else if method_ty.propagates_out() {
                            method_ty
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
                    }
                } else if let TypeEnum::SliceTy { elem_ty, is_mut, .. } = *lhs_ty.matchable()
                    && rhs.sym == p.ptr_sym
                {
                    decl = None; // TODO
                    // TODO: remove this allocation (test if cast SliceTy -> PointerTy is valid)
                    type_new!(PtrTy { pointee: elem_ty, is_mut }).upcast_to_type()
                } else if lhs_ty.kind == AstKind::SliceTy && rhs.sym == p.len_sym {
                    decl = None; // TODO
                    p.u64
                } else if let ty = lhs_ty.flatten_transparent()
                    // `lhs_ty.propagates_out()` implies `lhs_ty.flatten_transparent() == lhs_ty`
                    && ty.propagates_out()
                {
                    decl = None;
                    ty
                } else {
                    if rhs.replacement.is_some() {
                        debug_assert!(expr.ty.is_some());
                        return Ok(());
                    }
                    let mut ty = None;
                    // method-like call:
                    // TODO?: maybe change syntax to `arg~my_fn(...)`. using `.` both for method
                    //        calls and method-like calls might be confusing
                    if let Some(s) = self.cur_scope.find_decl(rhs.sym) {
                        let var_ty = self.get_symbol_var_ty(s)?;
                        if let Some(f) = var_ty.try_downcast::<ast::Fn>()
                            && let Some(first_param) = f.params().get(0)
                            && ty_match_quiet(lhs.ty.u(), first_param.var_ty.u(), true)
                        {
                            debug_assert!(s.is_const);
                            rhs.ty = Some(var_ty);
                            rhs.upcast().set_replacement(f.upcast());
                            ty = Some(p.method_stub);
                        } else if var_ty.propagates_out() {
                            ty = Some(var_ty);
                        }
                        decl = Some(s);
                    } else {
                        decl = None;
                    }
                    match ty {
                        Some(ty) => ty,
                        None => return NotFinished(UnitDependency::Dot(dot)?),
                    }
                };
                rhs.decl = decl;
                expr.ty = Some(t);
            },
            AstEnum::Index { mut_access, lhs, idx, .. } => {
                let idx_ty = self.analyze(*idx, &None, is_const)?.finalize();

                let tmp_ty_hint = if idx_ty.kind == AstKind::RangeTy {
                    ty_hint.and_then(|t| t.inner_ty())
                } else {
                    *ty_hint
                }
                .map(|elem_ty| {
                    type_new!(ArrayLikeContainer { elem_ty }, tmp_alloc()).upcast_to_type()
                });

                let lhs_ty = analyze!(*lhs, tmp_ty_hint);
                debug_assert!(lhs.ty != tmp_ty_hint, "temporary allocation won't be valid");
                let (TypeEnum::SliceTy { elem_ty, .. } | TypeEnum::ArrayTy { elem_ty, .. }) =
                    lhs_ty.matchable().as_ref()
                else {
                    return cerror2!(lhs.full_span(), "cannot index into value of type {}", lhs_ty);
                };
                let elem_ty = elem_ty.downcast_type();

                match idx_ty.matchable().as_ref() {
                    TypeEnum::RangeTy { elem_ty: i, rkind, .. }
                        if i.matches_int() || *rkind == RangeKind::Full =>
                    {
                        if *mut_access {
                            self.validate_mutation(MutationKind::Slice, *lhs, expr)?;
                        }
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
                            let idx = &idx.downcast::<ast::IntVal>().val;
                            let Some(idx) =
                                usize::try_from(idx).ok().filter(|idx| *idx < arr.elements.len())
                            else {
                                return cerror2!(
                                    expr.full_span(),
                                    "index out of bounds: the length is {} but the index is {idx}",
                                    arr.elements.len(),
                                );
                            };
                            expr.set_replacement(arr.elements[idx].upcast());
                        }
                    },
                    // TODO: allow `never` index?
                    //_ if idx_ty.propagates_out() => expr.ty = Some(idx_ty),
                    _ => {
                        return cerror2!(
                            idx.full_span(),
                            "Cannot index into array with `{idx_ty}`"
                        );
                    },
                }
            },
            AstEnum::Cast { operand, target_ty, .. } => {
                let target_ty = self.analyze_type_inst(*target_ty)?;
                let () = self.analyze_cast(*operand, target_ty, expr, is_const)?;
            },
            AstEnum::Autocast { operand, .. } => {
                let Some(target_ty) = *ty_hint else {
                    return cerror2!(expr.full_span(), "Cannot infer target type of autocast");
                };
                let () = self.analyze_cast(*operand, target_ty, expr, is_const)?;
            },
            AstEnum::Call { func, resolved_fn_inst, args, .. } => {
                let call = expr.downcast::<ast::Call>();
                let fn_ty = *analyze!(*func, ty_hint);
                if let Some(fn_ty) = fn_ty.try_downcast::<ast::Fn>() {
                    if is_const {
                        return error_const_call(call).into();
                    }

                    expr.ty = Some(self.validate_fn_call(
                        fn_ty,
                        args,
                        *ty_hint,
                        resolved_fn_inst,
                        call,
                    )?);
                } else if fn_ty == p.method_stub {
                    if is_const {
                        return error_const_call(call).into();
                    }
                    let dot = func.downcast::<ast::Dot>().as_mut();
                    let fn_ty = dot.rhs.ty.u().upcast().downcast::<ast::Fn>();
                    let args = std::iter::once(dot.lhs.u())
                        .chain(args.iter().copied())
                        .collect::<Vec<_>>(); // TODO: bench no allocation
                    expr.ty = Some(self.validate_fn_call(
                        fn_ty,
                        &args,
                        *ty_hint,
                        resolved_fn_inst,
                        call,
                    )?);
                } else if fn_ty == p.type_ty
                    && let ty = func.downcast_type2()
                    && let Some(polymorphable) = func.try_downcast_polymorphable()
                {
                    if !polymorphable.is_generic() {
                        return cerror2!(expr.full_span(), "Cannot call non-generic type `{ty}`");
                    }

                    macro_rules! call {
                        ($ty:expr) => {
                            self.validate_call(
                                CallKind::Type,
                                $ty,
                                args,
                                expr.span.end(),
                                true,
                                expr,
                            )?
                            .upcast_to_type()
                        };
                    }

                    let inst = match polymorphable.matchable2() {
                        PolymorphableMatch::Fn(f) => call!(f), // TODO: this is not correct
                        PolymorphableMatch::StructDef(s) => call!(s),
                        PolymorphableMatch::EnumDef(e) => call!(e),
                    };
                    *resolved_fn_inst = Some(inst);
                    expr.set_replacement(inst.upcast());
                    expr.ty = Some(inst.ty.u());
                } else if fn_ty == p.enum_variant
                    && let Some(i) = func.try_flat_downcast::<ast::Ident>()
                {
                    debug_assert_eq!(i.decl, p.some_variant);
                    if args.len() != 1 {
                        return cerror2!(span, "expected 1 argument, got {}", args.len());
                    }
                    let val = args.get(0).u();
                    let opt_ty =
                        if let Some(ty_hint) = ty_hint.try_downcast_ty_hint::<ast::OptionTy>() {
                            let inner_ty = ty_hint.inner_ty.downcast_type();
                            let val_ty = analyze!(val, Some(inner_ty));
                            /* // see `error_optional_orelse_optional` > "decide if these errors are better"
                            let _ = self.ty_match(val, inner_ty);
                            ty_hint
                            */
                            type_new!(OptionTy { inner_ty: val_ty.upcast() })
                        } else {
                            // Use ty_hint anyway to improve type inference in some error cases.
                            // see `error_optional_orelse_optional` > "inference still works"
                            type_new!(OptionTy { inner_ty: analyze!(val, ty_hint).upcast() })
                        };
                    expr.ty = Some(opt_ty.upcast_to_type());

                    if is_const {
                        let Some(cv) = val.try_downcast_const_val() else {
                            return error_non_const(val, "expected constant value").into();
                        };
                        let optional_cv = ast_new!(
                            OptionalVal { is_some: true, val: Some(cv) },
                            expr.full_span()
                        );
                        expr.set_replacement(optional_cv.upcast());
                    }
                } else if fn_ty == p.enum_variant {
                    let dot = func.flat_downcast::<ast::Dot>();
                    let enum_ty = dot.lhs.u().downcast::<ast::EnumDef>();
                    let variant = enum_ty.variants.find_field(dot.rhs.sym).u().1;
                    self.validate_call(
                        CallKind::EnumInit { variant },
                        enum_ty,
                        args,
                        span.end(),
                        is_const,
                        expr,
                    )?;
                    expr.ty = Some(enum_ty.upcast_to_type());

                    if is_const {
                        let cv = dot.upcast().downcast::<ast::EnumVal>();
                        debug_assert_eq!(args.len(), 1);
                        let data = args.get(0).u().downcast_const_val();
                        data.rep().as_mut().ty = Some(variant.var_ty.u());
                        cv.as_mut().data = Some(data);
                        expr.set_replacement(cv.upcast());
                    }
                } else {
                    return cerror2!(
                        func.full_span(),
                        "Cannot call value of type '{}'; expected function",
                        fn_ty
                    );
                };
            },
            AstEnum::UnaryOp { op, operand, .. } => {
                macro_rules! simple_unary_op_const {
                    ($op:tt $ast_node:ident) => {{
                        if is_const {
                            let const_val = operand.downcast_const_val();
                            let val = const_val.downcast::<ast::$ast_node>().val.clone();
                            // TODO: no allocation
                            expr.set_replacement(self.alloc(ast_new!($ast_node {
                                val: $op val,
                                span: expr.full_span(),
                            }))?.upcast());
                        }
                    }};
                }

                let err = |operand_ty| {
                    cerror!(
                        expr.full_span(),
                        "Cannot apply unary operator `{op}` to type `{operand_ty}`"
                    );
                    SemaResult::HandledErr
                };

                match *op {
                    UnaryOpKind::AddrOf | UnaryOpKind::AddrMutOf => {
                        let ty_hint = ty_hint
                            .try_downcast_ty_hint::<ast::PtrTy>()
                            .map(|ptr_ty| ptr_ty.pointee.downcast_type_inst());
                        let pointee = *analyze!(*operand, ty_hint);
                        let is_mut = *op == UnaryOpKind::AddrMutOf;
                        if is_mut {
                            self.validate_mutation(MutationKind::AddrOf, *operand, expr)?;
                        }
                        expr.ty = Some(
                            type_new!(PtrTy { pointee: pointee.upcast(), is_mut }).upcast_to_type(),
                        );
                        if pointee.kind == AstKind::Fn {
                            if is_mut {
                                return cerror2!(
                                    expr.full_span(),
                                    "Cannot mutably reference functions"
                                );
                            }
                            let const_val = operand.downcast_const_val();
                            debug_assert!(const_val.kind == AstKind::Fn);
                            expr.set_replacement(const_val.upcast());
                        } else if is_const {
                            let Some(sym) =
                                operand.try_get_symbol_decl().filter(|d| d.is_allowed_in_const())
                            else {
                                return cerror2!(
                                    expr.full_span(),
                                    "Can only take the address of a static or constant value at \
                                     compile time"
                                );
                            };
                            expr.set_replacement(
                                ast_new!(StaticPtrVal { sym }, Span::ZERO).upcast(),
                            );
                        }
                    },
                    UnaryOpKind::Deref => {
                        let operand_ty = *analyze!(*operand, None);
                        let Some(ptr_ty) = operand_ty.try_downcast::<ast::PtrTy>() else {
                            return cerror2!(
                                expr.full_span(),
                                "Cannot dereference value of type `{operand_ty}`",
                            );
                        };
                        if is_const {
                            return cerror2!(
                                expr.full_span(),
                                "The \"dereference\" operator is currently not supported at \
                                 compile time"
                            );
                        }
                        expr.ty = Some(ptr_ty.pointee.downcast_type());
                    },
                    UnaryOpKind::Not => {
                        let operand_ty = *analyze!(*operand, ty_hint);
                        if operand_ty == p.bool {
                            expr.ty = Some(operand_ty);
                            simple_unary_op_const!(!BoolVal);
                        } else if operand_ty.kind == AstKind::IntTy {
                            expr.ty = Some(operand_ty);
                            simple_unary_op_const!(!IntVal);
                        } else {
                            return err(operand_ty);
                        }
                    },
                    UnaryOpKind::Neg => {
                        let operand_ty = *analyze!(*operand, ty_hint);
                        if operand_ty.is_sint() {
                            expr.ty = Some(operand_ty);
                            simple_unary_op_const!(-IntVal);
                        } else if operand_ty == p.int_lit.upcast_to_type() {
                            expr.ty = Some(p.sint_lit.upcast_to_type());
                            simple_unary_op_const!(-IntVal);
                        } else if operand_ty.kind == AstKind::FloatTy {
                            expr.ty = Some(operand_ty);
                            simple_unary_op_const!(-FloatVal);
                        } else {
                            return err(operand_ty);
                        }
                    },
                    UnaryOpKind::Try => todo!("try"),
                }
            },
            AstEnum::BinOp { lhs, op, rhs, .. } => {
                let lhs_ty = *self.analyze(*lhs, &None, is_const)?;
                let rhs_ty = *self.analyze(*rhs, &Some(lhs_ty), is_const)?;
                let Some(mut common_ty) = common_type(lhs_ty, rhs_ty) else {
                    return error_mismatched_types_binop(expr.span, lhs_ty, rhs_ty).into();
                };
                not_never!(common_ty);
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
                        error_non_const(
                            *lhs,
                            format_args!("left-hand side of `{}`", op.as_binop_text()),
                        )
                    })?;
                    let rhs = rhs.try_downcast_const_val().ok_or_else(|| {
                        error_non_const(
                            *rhs,
                            format_args!("right-hand side of `{}`", op.as_binop_text()),
                        )
                    })?;

                    macro_rules! calc_num_binop {
                        ($op:tt $(, allow $float_val:ident)? $(, convert_rhs $convert_rhs:ident)?) => {
                            if common_ty.kind == AstKind::IntTy {
                                let rhs = &rhs.downcast::<ast::IntVal>().val;
                                $( let rhs = $convert_rhs(rhs); )?
                                let val = &lhs.downcast::<ast::IntVal>().val $op rhs;
                                Some(self.alloc(ast_new!(IntVal { val, span: expr.full_span() }))?
                                    .upcast())
                            } $( else if common_ty.kind == AstKind::FloatTy {
                                let val = lhs.float_val() $op rhs.float_val();
                                Some(self.alloc(ast_new!($float_val { val, span: expr.full_span() }))?
                                    .upcast())
                            })? else {
                                None
                            }
                        };
                    }

                    fn to_isize(big_int: &BigInt) -> isize {
                        isize::try_from(big_int).expect("value too big")
                    }

                    let Some(out_val) = (match op {
                        BinOpKind::Mul => calc_num_binop!(*, allow FloatVal),
                        BinOpKind::Div => calc_num_binop!(/, allow FloatVal),
                        BinOpKind::Mod => calc_num_binop!(%, allow FloatVal),
                        BinOpKind::Add => calc_num_binop!(+, allow FloatVal),
                        BinOpKind::Sub => calc_num_binop!(-, allow FloatVal),
                        BinOpKind::ShiftL => calc_num_binop!(<<, convert_rhs to_isize),
                        BinOpKind::ShiftR => calc_num_binop!(>>, convert_rhs to_isize),
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
                            return error_mismatched_types_binop(expr.span, start_ty, end_ty)
                                .into();
                        };
                        (common_ty, kind)
                    },
                };
                expr.ty = Some(type_new!(RangeTy { elem_ty, rkind }).upcast_to_type());
            },
            AstEnum::OrElse { lhs, rhs, .. } => {
                let alloc_opt_ty = |t: Ptr<ast::Type>| type_new!(OptionTy { inner_ty: t.upcast() });
                let lhs_ty_alloc = ty_hint.map(alloc_opt_ty);
                let lhs_ty = analyze!(*lhs, lhs_ty_alloc.upcast_to_type());
                let Some(lhs_opt_ty) = lhs_ty.try_downcast::<ast::OptionTy>() else {
                    return error_mismatched_types_custom(
                        lhs.full_span(),
                        "an optional value",
                        *lhs_ty,
                    )
                    .into();
                };
                let lhs_inner = lhs_opt_ty.inner_ty.downcast_type();

                debug_assert_eq!(
                    p.null_lit.var_ty.unwrap().downcast::<ast::OptionTy>().inner_ty.downcast_type(),
                    p.never
                );
                let rhs_hint = if lhs_inner != p.never { &Some(lhs_inner) } else { ty_hint };
                let rhs_ty = *self.analyze(*rhs, rhs_hint, is_const)?;
                let common_ty = common_type_restrict_optional_coerction(
                    lhs_inner,
                    rhs_ty,
                    AllowOptionalCoercion::RHS,
                )
                .unwrap_or_else(|| {
                    error_mismatched_types(rhs.full_span(), lhs_inner, rhs_ty);
                    if let Some(rhs_optional_depth) = rhs.ty.map(ast::Type::count_optional_nesting)
                        && let lhs_optional_depth = lhs_ty.count_optional_nesting()
                        && lhs_optional_depth <= rhs_optional_depth
                    {
                        chint!(expr.span, "Consider using `or` operator instead");
                    }
                    lhs_inner
                });

                if common_ty != lhs_inner {
                    let lhs_opt_ty = lhs_ty_alloc.unwrap_or_else(|| alloc_opt_ty(common_ty));
                    lhs_opt_ty.as_mut().inner_ty = common_ty.upcast();
                    *lhs_ty = lhs_opt_ty.upcast_to_type();
                }
                expr.ty = Some(common_ty);
            },
            AstEnum::Assign { lhs, rhs, .. } => {
                assert!(!is_const, "todo: Assign in const");
                let lhs_ty = *self.analyze(*lhs, &None, is_const)?;
                let rhs_ty = *self.analyze(*rhs, &Some(lhs_ty), is_const)?;
                // todo: check if binop can be applied to type
                self.ty_match(*rhs, lhs_ty)?;
                self.validate_lvalue(*lhs, expr)?;
                not_never!(rhs_ty);
                expr.ty = Some(p.void_ty);
            },
            AstEnum::BinOpAssign { lhs, rhs, op, .. } => {
                assert!(!is_const, "todo: BinOpAssign in const");
                let lhs_ty = *self.analyze(*lhs, &None, is_const)?;
                let rhs_ty = *self.analyze(*rhs, &Some(lhs_ty), is_const)?;
                if let Some(ptr_ty) = lhs_ty.try_downcast::<ast::PtrTy>() {
                    cerror!(
                        lhs.full_span(),
                        "Cannot apply binary operatator `{}` to pointer type `{lhs_ty}`",
                        op.as_binop_assign_text()
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
                not_never!(rhs_ty);
                expr.ty = Some(p.void_ty);
            },
            AstEnum::Decl { .. } => {
                self.analyze_decl(expr.downcast::<ast::Decl>(), false).ignore_error()?
            },
            AstEnum::If { condition, then_body, else_body, .. } => {
                assert!(!is_const, "todo: if in const");
                let bool_ty = p.bool;
                analyze!(*condition, Some(bool_ty));
                self.ty_match(*condition, bool_ty)?;

                let then_ty = *self.analyze(*then_body, ty_hint, is_const)?;
                expr.ty = if let Some(else_body) = *else_body {
                    let else_ty = *self.analyze(else_body, &Some(then_ty), is_const)?;
                    let Some(common_ty) = common_type(then_ty, else_ty) else {
                        return cerror2!(
                            else_body.full_span(),
                            "'then' and 'else' branches have incompatible types (`{then_ty}`; \
                             `{else_ty}`)"
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
            AstEnum::Switch { val, cases, else_body, .. } => {
                let val_ty = analyze!(*val, None).finalize();

                let source = PatternSource::new(val_ty);
                match source.ty.matchable2() {
                    TypeMatch::EnumDef(_) | TypeMatch::OptionTy(_) => {},
                    _ => return cerror2!(val.full_span(), "Cannot switch on type `{}`", val_ty),
                }

                let narrowed_symbol = val.try_downcast::<ast::Ident>();

                let mut out_ty = None;
                for case in cases.iter_mut() {
                    let case_ty = *self.analyze(case.case, &Some(source.ty), true)?;
                    if !(ty_match(case_ty, source.ty) || ty_match(case_ty, p.enum_variant)) {
                        return error_mismatched_types_custom(
                            case.case.full_span(),
                            "enum variant",
                            case_ty,
                        )
                        .into();
                    }
                    let narrowed_ty = match source.ty.matchable2() {
                        TypeMatch::EnumDef(_) => {
                            let case_val = case.case.downcast::<ast::EnumVal>();
                            if case_val.data.is_some() {
                                cerror!(
                                    case.case.full_span(),
                                    "Enum cases with associated data are currently not implemented"
                                );
                                continue;
                            }
                            let inner_ty = case_val.variant().var_ty.u();
                            if inner_ty == p.void_ty { source.ty } else { inner_ty }
                        },
                        TypeMatch::OptionTy(o) => {
                            let v = case.case.downcast::<ast::OptionalVal>();
                            if v.val.is_some() {
                                cerror!(
                                    case.case.full_span(),
                                    "Cases with associated data are currently not implemented"
                                );
                                continue;
                            }
                            if v.is_some { o.inner_ty.downcast_type() } else { p.void_ty }
                        },
                        _ => unreachable_debug(),
                    };
                    if let Some(narrowed_symbol) = narrowed_symbol {
                        if case.scope.decls.is_empty() {
                            let decl_in_case =
                                self.alloc(ast::Decl::from_ident(narrowed_symbol))?;
                            decl_in_case.as_mut().var_ty = Some(source.ty_in_body(narrowed_ty));
                            case.scope.decls.push(decl_in_case);
                        } else {
                            debug_assert_eq!(case.scope.decls.len(), 1);
                            debug_assert_eq!(case.scope.decls[0].ident, narrowed_symbol);
                        }
                    }

                    if !case.scope.flags.get(ScopeFlags::WAS_CHECKED_FOR_DUPLICATES) {
                        let _ignore = case.scope.verify_no_duplicates();
                    }
                    let osh = self.open_scope(&mut case.scope);
                    let res =
                        self.analyze_and_accumulate_type(case.body, &mut out_ty, ty_hint, is_const);
                    self.close_scope(osh);
                    res?;
                }
                if let Some(else_body) = *else_body {
                    self.analyze_and_accumulate_type(else_body, &mut out_ty, ty_hint, is_const)?;
                } else if let TypeMatch::EnumDef(e) = source.ty.matchable2() {
                    let mut variant_used = tmp_alloc().alloc_slice_with(e.variants.len(), false)?;
                    for case in cases.iter() {
                        let v = case.case.downcast::<ast::EnumVal>();
                        variant_used[v.variant_idx] = true;
                    }
                    if variant_used.iter().any(Not::not) {
                        let missing = e
                            .variants
                            .into_iter()
                            .zip(variant_used.as_ref())
                            .filter(|(_, used)| !*used)
                            .map(|(v, _)| wrap_display!("`.{}`", v.ident.sym.text()))
                            .join_fancy_list("and");

                        let err = cerror!(
                            expr.span.start().join(val.full_span()),
                            "missing cases {missing} in exhaustive switch on enum `{val_ty}`",
                        );
                        chint!(expr.span.end(), "Consider adding an `else` case");
                        return err.into();
                    }
                } else if let TypeMatch::OptionTy(_) = source.ty.matchable2() {
                    let mut variant_used = [false; 2];
                    for case in cases.iter() {
                        let v = case.case.downcast::<ast::OptionalVal>();
                        variant_used[v.is_some as usize] = true;
                    }
                    if variant_used.iter().any(Not::not) {
                        let missing = ["`null`", "`Some`"]
                            .into_iter()
                            .zip(variant_used)
                            .filter(|(_, used)| !*used)
                            .map(|(v, _)| v)
                            .join_fancy_list("and");

                        let err = cerror!(
                            expr.span.start().join(val.full_span()),
                            "missing cases {missing} in exhaustive switch on enum `{val_ty}`",
                        );
                        chint!(expr.span.end(), "Consider adding an `else` case");
                        return err.into();
                    }
                } else {
                    unreachable_debug()
                }
                expr.ty = Some(out_ty.unwrap_or(p.never));
            },
            AstEnum::For { source_expr, iter_var, body, scope, .. } => {
                if !scope.flags.get(ScopeFlags::WAS_CHECKED_FOR_DUPLICATES) {
                    let _ignore = scope.verify_no_duplicates();
                }

                let source_ty = analyze!(*source_expr, None).finalize();
                let source = PatternSource::new(source_ty);

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
                            source_expr.full_span(),
                            "cannot iterate over value of type `{source_ty}`",
                        );
                    },
                };

                let osh = self.open_scope(scope);
                self.loop_stack.push(expr);
                let res = (|| {
                    iter_var.var_ty = Some(source.ty_in_body(elem_ty));
                    self.analyze_decl(*iter_var, false).ignore_error()?;

                    self.analyze(*body, &Some(p.void_ty), is_const)?;
                    if !body.can_ignore_yielded_value() {
                        return error_cannot_yield_from_loop_block(body.return_val_span()).into();
                    }
                    Ok(())
                })();
                self.loop_stack.pop_expect(expr);
                self.close_scope(osh);
                res?;
                expr.ty = Some(p.void_ty);
            },
            AstEnum::While { condition, body, .. } => {
                let bool_ty = p.bool;
                analyze!(*condition, Some(bool_ty));
                self.ty_match(*condition, bool_ty)?;

                //self.open_scope(); // currently not needed
                self.loop_stack.push(expr);
                let res: SemaResult<()> = try {
                    self.analyze(*body, &Some(p.void_ty), is_const)?; // TODO: check if scope is closed on `NotFinished?`
                    if !body.can_ignore_yielded_value() {
                        Err(error_cannot_yield_from_loop_block(body.return_val_span()))?
                    }
                };
                self.loop_stack.pop_expect(expr);
                //self.close_scope();
                res?;
                expr.ty = Some(p.void_ty);
            },
            AstEnum::Loop { body, break_ty, .. } => {
                self.loop_stack.push(expr);
                let res = self.analyze(*body, &Some(p.void_ty), is_const); // TODO: check if scope is closed on `NotFinished?`
                self.loop_stack.pop_expect(expr);
                res?;
                if !body.can_ignore_yielded_value() {
                    return Err(error_cannot_yield_from_loop_block(body.return_val_span()));
                }
                expr.ty = Some(break_ty.unwrap_or(p.never));
            },
            // AstEnum::Catch { .. } => todo!(),
            AstEnum::Defer { stmt, .. } => {
                self.analyze(*stmt, &None, false)?;
                self.defer_stack.push(*stmt);
                expr.ty = Some(p.void_ty);
            },
            AstEnum::Return { val, parent_fn, .. } => {
                let Some(mut func) = self.get_cur_fn() else {
                    return cerror2!(expr.full_span(), "Cannot use `return` outside of a function");
                };
                *parent_fn = Some(func);
                expr.ty = Some(p.never);
                let val_ty = if let Some(val) = *val {
                    // TODO: set func.ret_ty to err_ty on error?
                    *self.analyze(val, &func.ret_ty, is_const)?
                } else {
                    p.void_ty
                };
                check_or_infer_target!(
                    val_ty,
                    &mut func.ret_ty,
                    func.flags.get(FnFlags::HAS_KNOWN_RET_TY),
                    val.map(|v| v.full_span()).unwrap_or(span)
                );
            },
            AstEnum::Break { val, .. } => {
                if val.is_some() {
                    todo!("break with value")
                }
                let Some(loop_) = self.loop_stack.last() else {
                    return cerror2!(expr.span, "`break` can only be used inside a loop");
                };
                match loop_.matchable2() {
                    AstMatch::For(_f) => {},
                    AstMatch::While(_w) => {},
                    AstMatch::Loop(l) => {
                        l.as_mut().break_ty = Some(p.void_ty);
                    },
                    _ => unreachable_debug(),
                }
                // TODO: check if in loop
                expr.ty = Some(p.never)
            },
            AstEnum::Continue { .. } => {
                let Some(_) = self.loop_stack.last() else {
                    return cerror2!(expr.span, "`continue` can only be used inside a loop");
                };
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
                    return error_mismatched_types_custom(
                        ty.upcast().full_span(),
                        "function type of intrinsic",
                        ty,
                    )
                    .into();
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
                let start_file = self.cctx.start_file();
                let Some(main) =
                    start_file.scope.as_ref().u().find_decl_norec(entry_point_sym, true)
                else {
                    return cerror2!(
                        start_file.full_span().start(),
                        "Couldn't find the entry point '{entry_point_sym}' in '{}'",
                        self.cctx.path_in_proj(&start_file.path).display()
                    );
                };
                let main_ty = self.get_symbol_var_ty(main)?;
                expr.ty = Some(main_ty);
                let Some(main_fn) = main_ty.try_downcast::<ast::Fn>() else {
                    if let Some(ty_hint) = *ty_hint
                        && ty_hint.propagates_out()
                    {
                        expr.ty = Some(ty_hint);
                        return Ok(());
                    }
                    return cerror2!(main.ident.span, "Expected the entry point to be a function");
                };
                let main_ret_ty = main_fn.ret_ty.u();
                if main_ret_ty != p.void_ty
                    && main_ret_ty.kind != AstKind::IntTy // not handled in `runtime.mylang`
                    && !main_ret_ty.propagates_out()
                {
                    return cerror2!(
                        main.ident.span,
                        "Entry point '{}' has invalid return type `{}`",
                        entry_point_sym,
                        main_ret_ty,
                    );
                }
                expr.set_replacement(main.const_val()?.upcast());
            },
            AstEnum::SizeOfDirective { type_, .. } => {
                let ty = self.analyze_type_inst(*type_)?;
                if !type_layout_finished(ty) {
                    return NotFinished(UnitDependency::TypeLayout(ty));
                }
                expr.ty = Some(p.int_lit.upcast_to_type());
                expr.set_replacement(ast::IntVal::new(ty.size())?.upcast());
            },
            AstEnum::SizeOfValDirective { val, .. } => {
                let ty = *analyze!(*val, None);
                if !type_layout_finished(ty) {
                    return NotFinished(UnitDependency::TypeLayout(ty));
                }
                expr.ty = Some(p.int_lit.upcast_to_type());
                expr.set_replacement(ast::IntVal::new(ty.size())?.upcast());
            },
            AstEnum::AlignOfDirective { type_, .. } => {
                let ty = self.analyze_type_inst(*type_)?;
                if !type_layout_finished(ty) {
                    return NotFinished(UnitDependency::TypeLayout(ty));
                }
                expr.ty = Some(p.int_lit.upcast_to_type());
                expr.set_replacement(ast::IntVal::new(ty.alignment())?.upcast());
            },
            AstEnum::OffsetOfDirective { type_, field, .. } => {
                let ty = self.analyze_type_inst(*type_)?;
                let Some(s_def) = ty.try_downcast_struct_def() else {
                    return cerror2!(type_.full_span(), "expected struct type");
                };
                let Some((f_idx, _)) = s_def.fields.find_field(field.sym) else {
                    return error_unknown_field(*field, s_def.upcast_to_type()).into();
                };
                let offset = struct_offset(&s_def.fields, f_idx);
                expr.ty = Some(p.int_lit.upcast_to_type());
                expr.set_replacement(ast::IntVal::new(offset)?.upcast());
            },
            AstEnum::SimpleDirective { ret_ty, .. } => {
                expr.ty = Some(*ret_ty);
            },

            AstEnum::IntVal { val, .. } => {
                expr.ty = Some(match ty_hint {
                    Some(t) if matches!(t.kind, AstKind::IntTy | AstKind::FloatTy) => *t,
                    _ if val.is_negative() => p.sint_lit.upcast_to_type(),
                    _ => p.int_lit.upcast_to_type(),
                });
            },
            AstEnum::FloatVal { .. } => {
                let ty = ty_hint
                    .filter(|t| t.kind == AstKind::FloatTy)
                    .unwrap_or(p.float_lit.upcast_to_type());
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
            AstEnum::RawPtrVal { .. } | AstEnum::StaticPtrVal { .. } => todo!(),
            AstEnum::EnumVal { .. } => todo!(),
            AstEnum::OptionalVal { .. } => unreachable_debug(),
            AstEnum::AggregateVal { .. } => todo!(),
            AstEnum::Fn { ret_ty_expr, ret_ty, body, flags, .. } => {
                let mut f = expr.downcast::<ast::Fn>();

                if let Some(decl) = self.decl_stack.last()
                    && decl.init == f.upcast()
                {
                    f.as_mut().decl.set_or_expect(*decl);
                }

                if !f.flags.get(FnFlags::IS_INSTANTIATION) {
                    setup_scopes(&mut f.as_mut().params_scope, f.generics_scope, self.cur_scope);
                } else {
                    debug_assert!(f.params_scope.flags.get(ScopeFlags::WAS_SETUP));
                    debug_assert!(f.generics_scope.u().flags.get(ScopeFlags::WAS_SETUP));
                }

                let fn_hint = ty_hint.try_downcast_ty_hint::<ast::Fn>();
                let might_be_fn_ty = *ty_hint == p.type_ty;

                let osh = self.jump_open_scope(&f.params_scope);
                let res = (|| {
                    for (p_idx, param) in f.as_ref().params_scope.decls.iter().enumerate() {
                        if param.is_const && !param.flags.get(DeclFlags::IS_GENERIC) {
                            let err = cerror2!(
                                param.lhs_span(),
                                "constant function parameters are currently not implemented"
                            );
                            chint!(param.lhs_span(), "Consider using a generic parameter instead");
                            return err;
                        }
                        if might_be_fn_ty {
                            param.as_mut().flags.set(DeclFlags::IS_FN_TY_PARAM);
                        } else if param.var_ty_expr.is_none()
                            && let Some(fn_hint) = fn_hint
                            && let Some(p_hint) = fn_hint.params().get(p_idx)
                        {
                            //debug_assert!(param.var_ty.is_none()); // TODO: without NotFinished
                            param.as_mut().var_ty = Some(p_hint.var_ty.u());
                        }

                        if param.flags.get(DeclFlags::IS_GENERIC) {
                            if !f.flags.get(FnFlags::IS_INSTANTIATION) {
                                self.analyze_fn_generic(*param, f)?;
                                param.as_mut().scope = None;
                                param.as_mut().is_const = true;
                                self.analyze_explicit_generic_decl(*param)
                            } else {
                                debug_assert!(param.is_const);
                                debug_assert!(param.var_ty.is_some());
                                debug_assert!(param.init.is_some());
                                Ok(())
                            }
                        } else {
                            self.analyze_decl(*param, false)
                        }
                        .ignore_error()?
                    }

                    if f.flags.get(FnFlags::HAS_KNOWN_RET_TY) {
                        // Happens iff sema of analyze_fn_body has to yield
                        debug_assert!(f.ret_ty.is_some())
                    } else if let Some(ret) = f.ret_ty_expr {
                        f.ret_ty = Some(self.analyze_type_inst(ret)?);
                        f.flags.set(FnFlags::HAS_KNOWN_RET_TY);
                    } else if let Some(fn_hint) = fn_hint {
                        f.ret_ty = Some(fn_hint.ret_ty.u());
                        f.flags.set(FnFlags::HAS_KNOWN_RET_TY);
                    }

                    if !might_be_fn_ty
                        && f.flags.get(FnFlags::HAS_KNOWN_RET_TY)
                        // decl_stack whould be wrong:
                        && !f.flags.get(FnFlags::IS_INSTANTIATION)
                    {
                        // Needed for indirect recursion. For direct recursion without an explicit
                        // return type, see `get_symbol_var_ty`
                        self.allow_unfinished_use(f.upcast(), f.upcast_to_type());
                    }

                    let contains_generic = f.flags.get(FnFlags::IS_GENERIC)
                        || f.flags.get(FnFlags::IS_INSTANTIATION_WITH_GENERICS);
                    if contains_generic {
                        if might_be_fn_ty {
                            return cerror2!(
                                f.generics().first().u().full_span(),
                                "Currently generic function types are not implemented"
                            );
                        }
                    } else if let Some(body) = f.body {
                        let body_ty = self
                            .analyze(body, &f.ret_ty, false)
                            .handle_err()? // prevents some misleading cycle errors
                            .copied()
                            .unwrap_or(p.err_ty);
                        let ret_ty = check_or_infer_target!(
                            body_ty,
                            &mut f.as_mut().ret_ty,
                            f.flags.get(FnFlags::HAS_KNOWN_RET_TY),
                            body.return_val_span()
                        );
                        if *ret_ty == p.rec_ret_ty {
                            let rec_fn_decl = self
                                .decl_stack
                                .last()
                                .filter(|d| d.init.is_some_and(|i| i.rep().p_eq(f)));
                            return cerror2!(
                                rec_fn_decl.map(|d| d.ident.span).unwrap_or_else(|| f.full_span()),
                                "cannot infer the return type of this recursive function"
                            );
                        }
                        ret_ty.finalize2(Some(f.ret_ty_expr.unwrap_or(body)), false)?;
                    } else {
                        debug_assert!(f.flags.get(FnFlags::IS_TYPE));
                        //panic_debug!("this function has already been analyzed as a function type")
                    }
                    Ok(())
                })();
                self.close_scope(osh);
                res?;
                debug_assert!(ret_ty.is_some() || flags.get(FnFlags::IS_GENERIC));

                let is_fn_ty = *ty_hint == p.type_ty
                    || (*ret_ty == p.type_ty && body.u().kind != AstKind::Block);
                expr.ty = Some(if is_fn_ty {
                    flags.set(FnFlags::IS_TYPE);
                    let Some(b) = *body else { return Ok(()) }; // only needed for `NotFinished`
                    *ret_ty_expr = Some(b);
                    *ret_ty = Some(b.downcast_type());
                    *body = None;
                    p.type_ty
                } else {
                    f.upcast_to_type()
                });
            },

            AstEnum::SimpleTy { .. } | AstEnum::IntTy { .. } | AstEnum::FloatTy { .. } => {
                expr.ty = Some(p.type_ty)
            },
            AstEnum::PtrTy { pointee, .. } => {
                self.analyze_type_inst(*pointee)?;
                expr.ty = Some(p.type_ty);
            },
            AstEnum::SliceTy { elem_ty, .. } => {
                self.analyze_type_inst(*elem_ty)?;
                expr.ty = Some(p.type_ty);
            },
            AstEnum::ArrayTy { len, elem_ty, .. } => {
                self.analyze_type_inst(*elem_ty)?;
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
            AstEnum::StructDef {
                flags,
                scope,
                fields,
                generics_scope,
                sema_units,
                finished_members,
                ..
            } => {
                let expr = expr.downcast::<ast::StructDef>();

                if !flags.get(StructFlags::IS_INSTANTIATION) {
                    setup_scopes(scope, *generics_scope, self.cur_scope);
                } else {
                    debug_assert!(scope.flags.get(ScopeFlags::WAS_SETUP));
                    debug_assert!(generics_scope.u().flags.get(ScopeFlags::WAS_SETUP));
                }
                self.allow_unfinished_use(expr.upcast(), p.type_ty);

                if !flags.get(StructFlags::GENERICS_ANALYZED) && flags.get(StructFlags::IS_GENERIC)
                {
                    let generics_scope = generics_scope.as_ref().u();
                    let osh = self.jump_open_scope(generics_scope);
                    let res = analyze_scope2(
                        generics_scope.as_mut().decls.as_mut(),
                        sema_units,
                        finished_members,
                        |generic, _unit| self.analyze_explicit_generic_decl(generic),
                    );
                    self.close_scope(osh);
                    let res = res.as_sema_result(expr.upcast_to_type()); // TODO: implementation of UnitDependency::Scope is not correct for this case
                    debug_assert_matches!(res, Ok(()));

                    *sema_units = None;
                    flags.set(StructFlags::GENERICS_ANALYZED);
                }

                let osh = self.jump_open_scope(scope);
                let contains_generic = flags.get(StructFlags::IS_GENERIC)
                    || flags.get(StructFlags::IS_INSTANTIATION_WITH_GENERICS);
                let res = if !contains_generic {
                    analyze_scope2(
                        scope.decls.as_mut(),
                        sema_units,
                        finished_members,
                        |member, _unit| self.analyze_decl(member, false),
                    )
                    .as_sema_result(expr.upcast_to_type())
                } else {
                    //debug_assert!(!flags.get(StructFlags::IS_INSTANTIATION));
                    try {
                        for field in fields.into_iter() {
                            debug_assert!(field.flags.get(DeclFlags::IS_DATA_MEMBER));
                            self.analyze_decl(field, true)?
                        }
                    }
                };
                self.close_scope(osh);
                res?;
                *sema_units = None;
                debug_assert_eq!(expr.ty, p.type_ty);
            },
            AstEnum::UnionDef { scope, sema_units, finished_members, .. } => {
                let expr = expr.downcast::<ast::UnionDef>();

                if !scope.flags.get(ScopeFlags::WAS_CHECKED_FOR_DUPLICATES) {
                    let _ignore = scope.verify_no_duplicates();
                }
                self.allow_unfinished_use(expr.upcast(), p.type_ty);
                let osh = self.open_scope(scope);
                let res = analyze_scope2(
                    scope.decls.as_mut(),
                    sema_units,
                    finished_members,
                    |member, _unit| {
                        if !member.is_const
                            && let Some(d) = member.init
                        {
                            return cerror2!(
                                d.full_span(),
                                "union fields cannot have default values"
                            );
                        }
                        self.analyze_decl(member, false)
                    },
                );
                self.close_scope(osh);
                res.as_sema_result(expr.upcast_to_type())?;
                *sema_units = None;
                debug_assert_eq!(expr.ty, p.type_ty);
            },
            AstEnum::EnumDef {
                scope,
                variants,
                sema_units,
                finished_members,
                is_simple_enum,
                tag_ty,
                ..
            } => {
                let expr = expr.downcast::<ast::EnumDef>();

                let mut repr_ty = Some(tag_ty.get_or_insert(p.int_lit).upcast_to_type());

                if !scope.flags.get(ScopeFlags::WAS_CHECKED_FOR_DUPLICATES) {
                    let _ignore = scope.verify_no_duplicates();
                }
                self.allow_unfinished_use(expr.upcast(), p.type_ty);
                let osh = self.open_scope(scope);
                let res = analyze_scope2(
                    scope.decls.as_mut(),
                    sema_units,
                    finished_members,
                    |member, _unit| {
                        if member.is_const {
                            return self.analyze_decl(member, false);
                        }
                        if member.var_ty_expr.is_none() {
                            member.as_mut().var_ty = Some(p.void_ty);
                        }

                        debug_assert!(member.flags.get(DeclFlags::IS_DATA_MEMBER));
                        self.analyze_decl(member, true)?;
                        if member.var_ty != p.void_ty {
                            *is_simple_enum = false;
                        }
                        let tag = if let Some(variant_tag) = member.init {
                            check_or_infer_target!(
                                *self.analyze(variant_tag, &repr_ty, true).no_err_ty()?,
                                &mut repr_ty,
                                false,
                                variant_tag.full_span()
                            );
                            variant_tag.downcast::<ast::IntVal>().val.clone()
                        } else {
                            // PERF: terrible implementation but works for now
                            let v_idx = variants.into_iter().position(|v| v == member).u();
                            match v_idx.checked_sub(1).map(|i| variants[i]) {
                                Some(prev_variant) => {
                                    get_enum_variant_tag(prev_variant)?.val.clone() + 1
                                },
                                None => num::BigInt::ZERO,
                            }
                        };

                        /*
                        const ALLOW_DUPLICATE_TAG: bool = true;

                        // TODO: replace linear search?
                        if !ALLOW_DUPLICATE_TAG
                            && unsafe { used_tags[..idx].assume_init_ref() }.contains(&tag)
                        {
                            return cerror2!(member.ident.span, "Duplicate enum variant tag");
                        }
                        */

                        debug_assert_eq!(
                            member.init.is_some(),
                            member.flags.get(DeclFlags::HAS_INIT_EXPR)
                        );
                        match &mut member.as_mut().init {
                            Some(init) => debug_assert_eq!(init.rep().kind, AstKind::IntVal),
                            i @ None => {
                                *i = Some(ast_new!(IntVal { val: tag }, Span::ZERO).upcast());
                            },
                        }

                        Ok(())
                    },
                );
                self.close_scope(osh);

                *tag_ty = Some(repr_ty.u().downcast::<ast::IntTy>());

                res.as_sema_result(expr.upcast_to_type())?;
                *sema_units = None;

                let repr_ty = repr_ty.u().downcast::<ast::IntTy>();
                *tag_ty = Some(if repr_ty.bits.is_none() {
                    let min_size_bits = util::variant_count_to_tag_size_bits(variants.len());
                    let Some(int_ty) = self.int_primitive(min_size_bits, repr_ty.is_signed) else {
                        return cerror2!(
                            expr.span,
                            "enums which can't be represented by an `u128` are currently not \
                             supported. This enum would require {min_size_bits} bits."
                        );
                    };
                    int_ty
                } else {
                    repr_ty
                });
                debug_assert_eq!(expr.ty, p.type_ty);
            },
            AstEnum::RangeTy { .. } => todo!(),
            AstEnum::OptionTy { inner_ty, .. } => {
                self.analyze_type_inst(*inner_ty)?;
                expr.ty = Some(p.type_ty);
            },
            AstEnum::ArrayLikeContainer { .. } => unreachable_debug(),
            AstEnum::GenericSlot { name, .. } => {
                let g = expr.downcast::<ast::GenericSlot>();

                match self.cur_scope.kind {
                    ScopeKind::FnParams => {
                        let f = self.cur_scope.get_expr().u().downcast::<ast::Fn>();

                        if f.flags.get(FnFlags::IS_INSTANTIATION) {
                            expr.ty = Some(*self.analyze(g.name.upcast(), ty_hint, is_const)?);
                            expr.set_replacement(g.name.upcast());
                            return Ok(());
                        }

                        let generic_decl = match g.name.decl {
                            Some(d) => d,
                            None => g.generate_decl(ty_hint.u(), &self.cctx.alloc)?,
                        };

                        self.analyze_fn_generic(generic_decl, f)?
                    },
                    ScopeKind::Struct => {
                        let s = self.cur_scope.get_expr().u().downcast::<ast::StructDef>();
                        let err = cerror!(
                            expr.full_span(),
                            "Cannot define generics inside a struct body"
                        );
                        debug_assert!(&s.span.file.u().code[s.span].starts_with("struct"));
                        if let Some(last_generic) = s.generics().last() {
                            chint!(
                                last_generic.full_span().after(),
                                "Consider adding the struct generic here. ${}",
                                name.sym,
                            );
                        } else {
                            chint!(
                                Span::pos(s.span.start + "struct".len(), s.span.file),
                                "Consider adding the struct generic here. (${})",
                                name.sym,
                            );
                        }
                        return err.into();
                    },
                    ScopeKind::Union => todo!(),
                    ScopeKind::Enum => todo!(),
                    ScopeKind::Root | ScopeKind::File => {
                        return cerror2!(
                            expr.full_span(),
                            "Generics cannot be defined at top-level",
                        );
                    },
                    ScopeKind::Block | ScopeKind::ForLoop | ScopeKind::SwitchCase => {
                        return cerror2!(
                            expr.full_span(),
                            "Generics cannot be defined inside a block",
                        );
                    },
                    ScopeKind::FnGenerics | ScopeKind::StructGenerics | ScopeKind::EnumGenerics => {
                        todo!()
                    },
                }

                expr.ty = Some(g.upcast_to_type());
            },
        }
        #[cfg(debug_assertions)]
        if expr.ty.is_none() {
            display(expr.full_span()).label("missing type").finish();
            debug_assert!(expr.ty.is_some());
        }
        Ok(())
    }

    #[cfg(debug_assertions)]
    fn check_post_sema_invariance(&mut self, expr: Ptr<Ast>) {
        //debug_assert!(expr.ty.is_some());

        match expr.matchable2() {
            AstMatch::Fn(f) => {
                if !f.flags.get(FnFlags::IS_GENERIC) {
                    debug_assert!(f.ret_ty.is_some());
                    //debug_assert!(f.ret_ty.u().is_finalized2(true));
                }
            },
            AstMatch::ArrayLikeContainer(_) => unreachable_debug(),
            _ => {}, // TODO
        }
    }

    fn ty_match(&mut self, expr: Ptr<Ast>, expected_ty: Ptr<ast::Type>) -> SemaResult<()> {
        let got_ty = expr.ty.u();
        if !ty_match(got_ty, expected_ty) {
            return error_mismatched_types(expr.full_span(), expected_ty, got_ty).into();
        }
        Ok(())
    }

    /// TODO: check [`ast::Type::propagates_out`], like in `analyze!`
    fn analyze_and_check_type(
        &mut self,
        expr: Ptr<Ast>,
        expected_ty: Ptr<ast::Type>,
        is_const: bool,
    ) -> SemaResult<&mut Ptr<ast::Type>> {
        let ty = self.analyze(expr, &Some(expected_ty), is_const)?;
        let () = self.ty_match(expr, expected_ty)?;
        Ok(ty)
    }

    fn analyze_and_accumulate_type(
        &mut self,
        expr: Ptr<Ast>,
        ty_acc: &mut OPtr<ast::Type>,
        ty_hint: &Option<Ptr<ast::Type>>,
        is_const: bool,
    ) -> SemaResult<Ptr<ast::Type>> {
        let ty = *self.analyze(expr, &ty_acc.or(*ty_hint), is_const)?;
        accumulate_type(ty_acc, ty, Some(expr))?;
        Ok(ty)
    }

    fn analyze_type_inst(&mut self, ty_expr: Ptr<Ast>) -> SemaResult<Ptr<ast::Type>> {
        let p = p();
        let ty = *self.analyze(ty_expr, &Some(p.type_ty), true)?;
        if ty_match(ty, p.type_ty) {
            Ok(ty_expr.try_downcast_type_inst()?)
        } else {
            error_mismatched_types(ty_expr.full_span(), p.type_ty, ty).into()
        }
    }

    pub fn try_custom_bitwith_int_type(&self, name: &str) -> Option<ast::IntTy> {
        let is_signed = match name.bytes().next() {
            Some(b'i') => true,
            Some(b'u') => false,
            _ => return None,
        };
        let bits = name[1..].parse().ok()?;
        debug_assert!(![8, 16, 32, 64, 128].contains(&bits));
        Some(type_new!(local IntTy { bits: Some(bits), is_signed }))
    }

    /// To allow recursive (or indirectly recursive) functions and types, we need to set the
    /// `var_ty` of these symbols
    fn allow_unfinished_use(&self, expr: Ptr<Ast>, ty: Ptr<ast::Type>) {
        expr.as_mut().ty = Some(ty);
        if let Some(decl) = self.decl_stack.last()
            && decl.init == expr
            && decl.var_ty.is_none()
        {
            decl.as_mut().var_ty = Some(ty);
        }
    }

    fn analyze_initializer_lhs(
        &mut self,
        initializer_expr: Ptr<ast::Ast>,
        ty_hint: OPtr<ast::Type>,
    ) -> SemaResult<Ptr<ast::Ast>> {
        let lhs = match initializer_expr.matchable().as_mut() {
            AstEnum::PositionalInitializer { lhs, parsed_with_lhs, .. }
            | AstEnum::NamedInitializer { lhs, parsed_with_lhs, .. } => {
                if !*parsed_with_lhs && let Some(lhs) = *lhs {
                    return Ok(lhs);
                }
                lhs
            },
            _ => unreachable_debug(),
        };

        Ok(if let Some(lhs) = *lhs {
            self.analyze(lhs, &None, false)?;
            lhs
        } else if let Some(ty_hint) = ty_hint {
            *lhs = Some(ty_hint.upcast());
            ty_hint.upcast()
        } else {
            cerror!(initializer_expr.full_span(), "cannot infer struct type");
            chint!(initializer_expr.span.start(), "consider specifying the type explicitly");
            return Err(HandledErr);
        })
    }

    fn analyze_array_initializer_lhs(
        &mut self,
        initializer_expr: Ptr<ast::Ast>,
        count: usize,
        ty_hint: OPtr<ast::Type>,
    ) -> SemaResult<OPtr<ast::Type>> {
        let lhs = match initializer_expr.matchable().as_mut() {
            AstEnum::ArrayInitializer { lhs, parsed_with_lhs, .. }
            | AstEnum::ArrayInitializerShort { lhs, parsed_with_lhs, .. } => {
                if !*parsed_with_lhs && let Some(lhs) = *lhs {
                    return Ok(Some(lhs.downcast_type()));
                }
                lhs
            },
            _ => unreachable_debug(),
        };

        let lhs = if let Some(lhs) = *lhs {
            self.analyze(lhs, &None, false)?;
            lhs
        } else if let Some(elem_ty) = ty_hint.and_then(|t| t.inner_ty()) {
            *lhs = Some(elem_ty.upcast());
            return Ok(Some(elem_ty));
        } else {
            return Ok(None);
        };

        Ok(Some(if let Some(elem_ty) = lhs.try_downcast_type() {
            elem_ty
        } else if let Some(ptr_ty) = lhs.ty.try_downcast::<ast::PtrTy>()
            && let Some(arr_ty) = ptr_ty.pointee.try_downcast::<ast::ArrayTy>()
        {
            self.validate_mutation(MutationKind::Initialize, lhs, initializer_expr)?;
            initializer_expr.as_mut().ty = Some(ptr_ty.upcast_to_type());
            if count != arr_ty.len.int() {
                return cerror2!(
                    initializer_expr.full_span(),
                    "Cannot initialize the array behind the pointer `{}` with {count} items",
                    ptr_ty.upcast_to_type(),
                );
            }
            arr_ty.elem_ty.downcast_type()
        } else {
            // TODO: also allow lhs slices?
            return error_cannot_apply_initializer(lhs, initializer_expr).into();
        }))
    }

    fn analyze_cast(
        &mut self,
        operand: Ptr<ast::Ast>,
        target_ty: Ptr<ast::Type>,
        expr: Ptr<Ast>,
        is_const: bool,
    ) -> SemaResult<()> {
        let op_ty = *self.analyze(operand, &Some(target_ty), is_const)?;
        if !self.validate_cast(op_ty, target_ty) {
            return cerror2!(expr.full_span(), "cannot cast `{op_ty}` to `{target_ty}`");
        }
        expr.as_mut().ty = Some(target_ty);

        if is_const {
            let cv = match (op_ty.matchable2(), target_ty.matchable2()) {
                // TODO: remove this int -> ptr cast?
                (TypeMatch::IntTy(_), TypeMatch::PtrTy(_)) => {
                    let i_val = operand.downcast::<ast::IntVal>();
                    ast_new!(RawPtrVal { val: ui(&i_val.val) }, i_val.span).upcast()
                },
                (TypeMatch::IntTy(_), TypeMatch::OptionTy(o))
                    if o.inner_ty.downcast_type().kind == AstKind::PtrTy =>
                {
                    let i_val = operand.downcast::<ast::IntVal>();
                    ast_new!(RawPtrVal { val: ui(&i_val.val) }, i_val.span).upcast()
                },
                (TypeMatch::EnumDef(_), TypeMatch::IntTy(_)) => {
                    get_enum_variant_tag(operand.downcast::<ast::EnumVal>().variant())?.upcast()
                },
                (TypeMatch::GenericSlot(_), _) => todo!("Generic"),
                (_, TypeMatch::GenericSlot(_)) => todo!("Generic"),
                // TODO: correctly handle other cases
                _ => operand.rep(),
            };
            expr.set_replacement(cv);
        }

        Ok(())
    }

    #[must_use]
    fn validate_cast(&self, ty: Ptr<ast::Type>, target_ty: Ptr<ast::Type>) -> bool {
        let p = p();
        match (ty.matchable2(), target_ty.matchable2()) {
            (..) if ty == p.any => false,
            (TypeMatch::OptionTy(o1), TypeMatch::OptionTy(o2)) => {
                // TODO: inlined optionals
                self.validate_cast(o1.inner_ty.downcast_type(), o2.inner_ty.downcast_type())
            },
            (TypeMatch::OptionTy(p), TypeMatch::IntTy(i))
                if p.inner_ty.downcast_type().kind == AstKind::PtrTy =>
            {
                i.bits.u() == 64
            },
            (TypeMatch::PtrTy(_), TypeMatch::IntTy(i)) => i.bits.u() == 64,
            (TypeMatch::OptionTy(_), _) => false,
            // TODO: more checks
            _ => true,
        }
    }

    fn validate_named_initializer(
        &mut self,
        struct_ty: Ptr<ast::Type>,
        initializer_values: Ptr<[(Ptr<ast::Ident>, Option<Ptr<Ast>>)]>,
        is_const: bool,
        initializer_expr: Ptr<ast::NamedInitializer>,
    ) -> SemaResult<Ptr<ast::Type>> {
        let fields = match struct_ty.matchable().as_ref() {
            TypeEnum::StructDef { fields, .. } => &**fields,
            TypeEnum::SliceTy { elem_ty, is_mut, .. } => {
                &self.slice_fields(elem_ty.downcast_type(), *is_mut)?
            },
            _ => unreachable_debug(),
        };

        let mut ok = true;
        macro_rules! on_err {
            () => {{
                ok = false;
                continue;
            }};
        }
        let mut const_values =
            then!(is_const => self.cctx.alloc.alloc_slice_default(fields.len())?);
        macro_rules! handle_const_val {
            ($f_idx:expr, $val_expr:expr) => {
                if let Some(const_values) = const_values.as_mut() {
                    if let Some(cv) = $val_expr.try_downcast_const_val() {
                        const_values[$f_idx] = Some(cv);
                    } else {
                        error_non_const_initializer_field($val_expr);
                        on_err!();
                    }
                }
            };
        }
        let mut is_initialized_field = vec![false; fields.len()];
        for (f, init) in initializer_values.as_mut().iter_mut() {
            let Some((f_idx, f_decl)) = fields.find_field(f.sym) else {
                error_unknown_field(*f, struct_ty);
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
            let field_ty = get_var_ty(f_decl)?;
            match self.analyze_and_check_type(init, field_ty, is_const) {
                Ok(_ty) => {},
                NotFinished(dep) => return NotFinished(dep),
                Err(HandledErr) => on_err!(),
            }
            handle_const_val!(f_idx, init);
        }

        for (f_idx, _) in
            is_initialized_field.into_iter().enumerate().filter(|(_, is_init)| !is_init)
        {
            let field = fields[f_idx];
            let Some(init) = field.init else {
                cerror!(
                    initializer_expr.span,
                    "missing field `{}` in initializer of `{struct_ty}`",
                    field.ident.sym
                );
                on_err!();
            };
            get_ty(init)?;
            handle_const_val!(f_idx, init);
        }

        if !ok {
            return SemaResult::HandledErr;
        }

        let inst = match struct_ty.try_downcast::<ast::StructDef>() {
            Some(struct_ty) => self
                .finalize_instantiation(struct_ty, initializer_expr.upcast())?
                .upcast_to_type(),
            None => struct_ty,
        };

        if is_const {
            let elements = const_values.u().u();
            debug_assert_eq!(initializer_expr.lhs.u().downcast_type2(), struct_ty);
            let val = ast_new!(AggregateVal { elements }, initializer_expr.span);
            val.as_mut().ty = Some(inst);
            initializer_expr.upcast().set_replacement(val.upcast());
        }

        // TODO: check if this is a good idea for SliceTy
        Ok(inst)
    }

    fn validate_pos_initializer(
        &mut self,
        struct_ty: Ptr<ast::Type>,
        args: &[Ptr<ast::Ast>],
        close_p_span: Span,
        is_const: bool,
        expr: Ptr<Ast>,
    ) -> SemaResult<Ptr<ast::Type>> {
        let inst = self.validate_call(
            CallKind::PositionalInitializer,
            struct_ty.downcast_struct_def(),
            args,
            close_p_span,
            is_const,
            expr,
        )?;

        Ok(if let Some(_) = struct_ty.try_downcast::<ast::SliceTy>() {
            debug_assert!(!inst.flags.get(StructFlags::IS_INSTANTIATION));
            struct_ty
        } else {
            inst.upcast_to_type()
        })
    }

    fn create_aggregate_const_val(
        &mut self,
        all_element_exprs: impl IntoIterator<IntoIter = impl ExactSizeIterator<Item = Ptr<ast::Ast>>>,
    ) -> SemaResult<Ptr<ast::AggregateVal>> {
        let mut all_element_exprs = all_element_exprs.into_iter();
        let mut elements = self.cctx.alloc.alloc_slice_default(all_element_exprs.len())?;
        let mut ok = true;
        for (elem, arg) in elements.iter_mut().zip(all_element_exprs.by_ref()) {
            get_ty(arg)?;
            if let Some(cv) = arg.try_downcast_const_val() {
                *elem = Some(cv);
            } else {
                error_non_const_initializer_field(arg);
                ok = false;
            }
        }
        if !ok {
            return SemaResult::HandledErr;
        }
        Ok(ast_new!(AggregateVal { elements: elements.u() }, Span::ZERO))
    }

    #[inline]
    fn analyze_decl_inner(&mut self, decl_ptr: Ptr<ast::Decl>, skip_init: bool) -> SemaResult<()> {
        let p = p();
        let decl = decl_ptr.as_mut();

        let is_static = decl.flags.get(DeclFlags::IS_STATIC);
        if is_static && decl.is_const {
            // TODO: I'm not happy that this is possible syntactically
            return cerror2!(
                decl.lhs_span(),
                "cannot mark a declaration as `static` and as const (`::`)"
            );
        }

        let is_first_pass = decl.ty.is_none(); // TODO(without `NotFinished`): remove this
        if is_first_pass && let Some(ty_expr) = decl.on_type {
            let ty = self.analyze_type_inst(ty_expr)?; // TODO: Don't require a generic instantiation
            match ty.matchable().as_mut() {
                TypeEnum::StructDef { fields, external_consts, .. }
                | TypeEnum::UnionDef { fields, external_consts, .. }
                | TypeEnum::EnumDef { variants: fields, external_consts, .. } => {
                    let name = decl.ident.sym;
                    if let Some((_, prev)) =
                        fields.find_field(name).or_else(|| external_consts.find_field(name))
                    {
                        // TODO: remove this duplicate logic
                        cerror!(
                            decl.lhs_span(),
                            "duplicate definition of `{}`",
                            decl.display_lhs()
                        );
                        chint!(prev.lhs_span(), "previous definition here");
                        return SemaResult::HandledErr;
                    }
                    external_consts.push(decl_ptr);
                },
                _ if ty == p.err_ty => return Err(HandledErr), // TODO: ty.propagates_out()?
                _ => {
                    return cerror2!(
                        ty_expr.span,
                        "cannot define an associated variable on a primitive type"
                    );
                },
            }
        }
        decl.ty = Some(p.void_ty);
        if let Some(t) = decl.var_ty_expr
            && decl.var_ty.is_none()
        {
            let ty = self.analyze_type_inst(t)?;
            debug_assert!(decl.var_ty.is_none_or(|t| ty_match(t, ty))); // TODO(without `NotFinished`): remove this
            decl.var_ty = Some(ty);
            if ty == p.err_ty {
                return Err(HandledErr);
            }
        }
        if let Some(init) = decl.init.filter(|_| !skip_init) {
            if init.is_custom_type() {
                let ty = init.downcast_type2();
                // TODO: bench vs decl field on individual ast nodes
                let _old = crate::context::ctx_mut().ty_names.insert(ty, decl.ident.sym);
                //debug_assert!(old.is_none());
            }

            let is_init_const = decl.is_const
                || is_static
                || decl.flags.get(DeclFlags::IS_DATA_MEMBER)
                || decl.flags.get(DeclFlags::IS_GENERIC);
            let init_ty = *self.analyze(init, &decl.var_ty, is_init_const)?;
            let var_ty = check_or_infer_target!(init_ty, &mut decl.var_ty, true, init.full_span());
            let var_ty = if !decl.is_const { var_ty.finalize_allow_generic() } else { *var_ty };
            // init.ty is finalized later
            if is_init_const && var_ty != p.err_ty && !init.rep().is_const_val() {
                // Ideally all branches in `_analyze_inner` should handle the `is_const` parameter.
                let span = init.full_span();
                return if is_static {
                    // TODO: add static initialization?
                    cerror!(span, "The initial value of a static must be known at compile time")
                } else {
                    error_non_const_custom(
                        init,
                        "Cannot access a non-constant symbol at compile time",
                        "Cannot evaluate value at compile time",
                    )
                }
                .into();
            }
            Ok(())
        } else if decl.var_ty.is_some() {
            debug_assert!(!decl.is_const || decl.flags.get(DeclFlags::IS_GENERIC));
            Ok(())
        } else if decl.flags.get(DeclFlags::IS_FN_TY_PARAM) {
            debug_assert!(decl.var_ty_expr.is_none());
            debug_assert!(decl.init.is_none());

            let ty_name = decl.ident.upcast();

            // The parameter name is changed to prevent `analyze_type` and further analysis from
            // finding the parameter itself as a type definition.
            decl.ident = p.ignored_name;
            decl.var_ty_expr = Some(ty_name);
            let ty = self.analyze_type_inst(ty_name)?;
            decl.var_ty = Some(ty);
            Ok(())
        } else {
            cerror!(decl_ptr.upcast().full_span(), "cannot infer type of `{}`", decl.ident.sym);
            chint!(decl_ptr.upcast().full_span(), "consider explicitly specifying the type");
            // TODO: `my_var: /* type */`
            // TODO: `      ++++++++++++`
            return SemaResult::HandledErr;
        }
    }

    fn analyze_decl(&mut self, mut decl: Ptr<ast::Decl>, skip_init: bool) -> SemaResult<()> {
        let p = p();
        decl.scope.set_or_expect(self.cur_scope);
        self.decl_stack.push(decl);
        let res = self.analyze_decl_inner(decl, skip_init);
        self.decl_stack.pop();
        #[cfg(debug_assertions)]
        if self.cctx.args.debug_types && decl.ident.span != Span::ZERO {
            let label = match &res {
                Ok(()) => format!("type: {}", decl.var_ty.u()),
                NotFinished(dep) => {
                    use crate::util::OptionExt;
                    format!("not finished ({dep:?}; type: {})", decl.var_ty.display())
                },
                Err(e) => format!("err: {:?}", e),
            };
            display(decl.ident.span).label(&label).finish();
        }
        match res {
            Err(HandledErr) => {
                decl.var_ty = Some(p.err_ty);
                decl.ty = Some(p.err_ty);
                if let Some(init) = decl.init.as_mut() {
                    // see `tests::ordering::correctly_handle_error_in_later_cycles`
                    init.ty = Some(p.err_ty);
                    init.rep().ty = Some(p.err_ty);
                }
            },
            NotFinished(_) => {},
            Ok(()) => {
                let var_ty = decl.var_ty.u();
                if var_ty.propagates_out() {
                    decl.ty = Some(var_ty);
                } else if let Some(f) = var_ty.try_downcast::<ast::Fn>() {
                    f.as_mut().flags.set(FnFlags::IS_NAMED);
                }
                decl.ident.decl = Some(decl);
            },
        }
        res
    }

    fn analyze_explicit_generic_decl(&mut self, decl: Ptr<ast::Decl>) -> SemaResult<()> {
        let p = p();

        debug_assert!(decl.flags.get(DeclFlags::IS_GENERIC));
        debug_assert!(decl.is_const);
        if decl.var_ty_expr.is_none() && decl.init.is_none() {
            decl.as_mut().var_ty = Some(p.type_ty);
        }

        let () = self.analyze_decl(decl, false)?;

        // The GenericSlot is put as the const_val of `decl`, which causes it to be distributed to
        // uses of `decl`
        if let Some(default_expr) = decl.as_mut().init.replace(decl.generic.u().upcast())
            && decl.flags.get(DeclFlags::HAS_INIT_EXPR)
        {
            debug_assert!(default_expr.kind != AstKind::GenericSlot);
            decl.generic.u().default.set_or_expect(default_expr.downcast_const_val());
        }

        Ok(())
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
        if decl.on_type.is_some() || self.get_cur_fn().is_some() {
            return cerror2!(
                decl.lhs_span(),
                "An #{0} declaration must be in global/file scope.",
                extern_directive_name(directive)
            );
        }
        Ok(ty)
    }

    fn analyze_fn_generic(
        &mut self,
        generic_decl: Ptr<ast::Decl>,
        f: Ptr<ast::Fn>,
    ) -> SemaResult<()> {
        let generic = generic_decl.generic.u();
        debug_assert_eq!(generic.name.decl, generic_decl);

        if generic.as_mut().flags.get(GenericSlotFlags::WAS_ADDED_TO_SCOPE) {
            debug_assert!(f.flags.get(FnFlags::IS_GENERIC));
            debug_assert!(f.generics_scope.u().decls.contains(&generic.name.decl.u()));
            return Ok(());
        }

        let generics_scope = f.as_mut().generics_scope.get_or_insert_with(|| {
            let s = self.cctx.alloc.alloc(Scope::new(vec![], ScopeKind::FnGenerics)).unwrap();
            s.as_mut().expr.set_once(f.upcast());
            debug_assert!(f.params_scope.flags.get(ScopeFlags::WAS_SETUP));
            s.as_mut().setup(f.params_scope.parent.u());
            f.as_mut().params_scope.parent = Some(s);
            s
        });
        if let Result::Err(dup) = generics_scope.add_decl(generic_decl) {
            return error_duplicate_in_unordered_scope(ScopeKind::FnParams, generic_decl, dup)
                .into();
        }
        f.as_mut().flags.set(FnFlags::IS_GENERIC);
        generic.as_mut().flags.set(GenericSlotFlags::WAS_ADDED_TO_SCOPE);

        Ok(())
    }

    fn validate_fn_call(
        &mut self,
        fn_ty: Ptr<ast::Fn>,
        args: &[Ptr<ast::Ast>],
        ty_hint: OPtr<ast::Type>,
        resolved_inst: &mut OPtr<ast::Type>,
        expr: Ptr<ast::Call>,
    ) -> SemaResult<Ptr<ast::Type>> {
        if resolved_inst.is_none() {
            let inst = self.validate_call(
                CallKind::Function { ty_hint },
                fn_ty,
                args,
                expr.span.end(),
                false,
                expr.upcast(),
            )?;
            *resolved_inst = Some(inst.upcast_to_type());
        }
        let inst = resolved_inst.u().downcast::<ast::Fn>();

        match inst.ret_ty {
            Some(ty) => Ok(ty),
            None if inst == self.get_cur_fn().u() => Ok(p().rec_ret_ty), // TODO: check all previous fns
            None => NotFinished(UnitDependency::RetTy(inst)),
        }
    }

    fn validate_call<T: PolymorphableType>(
        &mut self,
        kind: CallKind,
        ty: Ptr<T>,
        args: &[Ptr<ast::Ast>],
        close_p_span: Span,
        is_const: bool,
        expr: Ptr<Ast>,
    ) -> SemaResult<Ptr<T>> {
        let res = self._validate_call(kind, ty, args, close_p_span, is_const, expr);

        if ty.flags().get(T::FLAG_IS_GENERIC) {
            for g in ty.generics() {
                let g = g.init.u().downcast::<ast::GenericSlot>();
                g.as_mut().cur_inst = None;
            }
        }

        res
    }

    fn _validate_call<T: PolymorphableType>(
        &mut self,
        kind: CallKind,
        ty: Ptr<T>,
        args: &[Ptr<Ast>],
        close_p_span: Span,
        is_const: bool,
        expr: Ptr<Ast>,
    ) -> SemaResult<Ptr<T>> {
        debug_assert!(
            ty.flags().get(T::FLAG_IS_INSTANTIATION)
                || ty.generics().iter().all(|g| g.generic.u().cur_inst.is_none())
        );

        let is_enum_init = matches!(kind, CallKind::EnumInit { .. });
        let has_varargs = matches!(kind, CallKind::Function { .. })
            && ty.upcast().downcast::<ast::Fn>().flags.get(FnFlags::HAS_VARARGS);

        fn get_params<T: PolymorphableType>(kind: &CallKind, ty: Ptr<T>) -> &[Ptr<ast::Decl>] {
            match kind {
                CallKind::Function { .. } => {
                    let fn_ty = ty.upcast().downcast::<ast::Fn>();
                    fn_ty.as_ref().params()
                },
                CallKind::EnumInit { variant } => {
                    debug_assert!(
                        ty.upcast().downcast::<ast::EnumDef>().variants.contains(&variant)
                    );
                    std::slice::from_ref(variant)
                },
                CallKind::PositionalInitializer => {
                    // also used for SliceTy
                    let ty = ty.upcast().downcast::<ast::StructDef>();
                    ty.fields.as_ref()
                },
                CallKind::Type { .. } => ty.as_ref().generics(),
            }
        }

        let params = get_params(&kind, ty);

        let pos_arg_count = args.iter().copied().take_while(is_pos_arg).count();

        let mut normal_pos_arg_count = pos_arg_count;
        if pos_arg_count > params.len() {
            if has_varargs {
                normal_pos_arg_count = params.len();
            } else {
                return cerror2!(
                    args.get(params.len()).u().full_span(),
                    "Got {pos_arg_count} positional arguments, but expected at most {} arguments",
                    params.len(),
                );
            }
        }

        let normal_pos_args = &args[..normal_pos_arg_count];
        let var_args = &args[normal_pos_arg_count..pos_arg_count];
        let named_args = &args[pos_arg_count..];

        let params_for_normal_pos_args = &params[..normal_pos_arg_count];
        let params_for_named_args = params.get(pos_arg_count..).unwrap_or(&[]);

        fn analyze_if_generic_arg(
            sema: &mut Sema,
            param: Ptr<ast::Decl>,
            arg_val: Ptr<ast::Ast>,
        ) -> SemaResult<()> {
            if param.flags.get(DeclFlags::IS_GENERIC) {
                sema.analyze_and_check_type(arg_val, get_var_ty(param)?, true)?;
                param
                    .generic
                    .u()
                    .as_mut()
                    .cur_inst
                    .set_once(arg_val.downcast_const_val().finalize_allow_generic());
            }
            Ok(())
        }

        let mut ok = true;

        // handle explicit positional generics
        for (idx, pos_arg) in normal_pos_args.iter().enumerate() {
            let res = analyze_if_generic_arg(self, *params.get(idx).u(), *pos_arg);
            ok = res.is_ok()? && ok
        }

        // resolve params of named args; handle duplicate (named) args;
        // handle named explicit generic annotations
        let mut was_set_by_named = vec![false; params_for_named_args.len()];
        for arg in named_args {
            if is_pos_arg(arg) {
                return cerror2!(
                    arg.full_span(),
                    "Cannot specify a positional argument after named arguments"
                );
            }
            let named_arg = arg.downcast::<ast::Assign>();
            let Some(arg_name) = named_arg.lhs.try_downcast::<ast::Ident>() else {
                return cerror2!(named_arg.lhs.full_span(), "Expected a parameter name");
            };
            let param = if let Some((param_idx, param)) =
                params_for_named_args.find_field(arg_name.sym)
            {
                if was_set_by_named[param_idx] {
                    error_duplicate_named_arg(arg_name);
                    chint!(named_args[param_idx].full_span(), "set here already");
                    return SemaResult::HandledErr;
                }
                was_set_by_named[param_idx] = true;
                param
            } else if let Some(g_decl) = ty.generics_scope().and_then(|g| g.find_decl(arg_name.sym))
            {
                g_decl
            } else {
                if let Some((idx, _)) = params_for_normal_pos_args.find_field(arg_name.sym) {
                    error_duplicate_named_arg(arg_name);
                    chint!(
                        args[idx].full_span(),
                        "The parameter has already been set by this positional argument"
                    )
                } else {
                    cerror!(arg_name.span, "Unknown parameter");
                }
                return SemaResult::HandledErr;
            };
            arg_name.as_mut().decl.set_once(param);
            let res = analyze_if_generic_arg(self, param, named_arg.rhs);
            ok = res.is_ok()? && ok
        }

        if !ok {
            return Err(HandledErr);
        }

        // infer generic based on return value
        if let CallKind::Function { ty_hint: Some(ty_hint) } = kind
            && let Some(g_def) =
                ty.upcast().downcast::<ast::Fn>().ret_ty.try_downcast::<ast::GenericSlot>()
        {
            // TODO: this is not called when type return type is inferred (see infer_generic_based_on_inferred_return_type)
            let _ignore = accumulate_generic(g_def, ty_hint.upcast_to_const_val(), false);
        }

        // analyze non-generic parameters (might cause generics to be inferred)
        {
            let pos_args = normal_pos_args
                .iter()
                .copied()
                .zip_exact(params_for_normal_pos_args.iter().copied());
            let named_args = named_args.iter().map(|a| {
                let assign = a.downcast::<ast::Assign>();
                (assign.rhs, assign.lhs.downcast::<ast::Ident>().decl.u())
            });
            for (arg_val, param) in pos_args.chain(named_args) {
                let res = self.analyze_and_check_type(arg_val, get_var_ty(param)?, is_const);
                ok = res.is_ok()? && ok;
            }

            for var_arg in var_args {
                match self.analyze(*var_arg, &None, is_const) {
                    Ok(ty) => {
                        ty.finalize();
                    },
                    NotFinished(dep) => return NotFinished(dep),
                    Err(HandledErr) => ok = false,
                }
            }

            if !ok {
                return Err(HandledErr);
            }
        }

        // missing args
        struct MissingParam(Ptr<ast::Decl>);
        impl std::fmt::Display for MissingParam {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(f, "`{}: {}`", self.0.ident.sym, self.0.var_ty.u())
            }
        }
        let mut missing_params = params_for_named_args
            .iter()
            .copied()
            .zip_exact(was_set_by_named)
            .filter(|(p, was_set)| !was_set && !p.has_default(is_enum_init))
            .map(|(p, _)| MissingParam(p));
        if let Some(first) = missing_params.next() {
            let mut missing_params_list = first.to_string();
            let mut plural = false;
            for p in missing_params {
                missing_params_list.push_str(", ");
                let _ = missing_params_list.write_fmt(format_args!("{p}"));
                plural = true;
            }
            cerror!(
                close_p_span,
                "Missing argument{0} for parameter{0} {1}",
                if plural { "s" } else { "" },
                missing_params_list
            );
            chint!(first.0.upcast().full_span(), "parameter defined here");
            return SemaResult::HandledErr;
        }

        // generic instantiation
        let inst = self.finalize_instantiation(ty, expr)?;

        // finalize args to prevent leaking GenericSlot type into arg expressions
        if inst.flags().get(T::FLAG_IS_INSTANTIATION) && ty.get_kind() == AstKind::Fn {
            let inst_params = get_params(&kind, inst);
            debug_assert!(normal_pos_args.len() <= inst_params.len());

            let pos_args = normal_pos_args.iter().zip(inst_params).map(|d| (*d.0, *d.1));
            let named_args = named_args.iter().map(|assign| {
                let named_arg = assign.downcast::<ast::Assign>();
                let lhs = named_arg.lhs.downcast::<ast::Ident>();
                if ty.flags().get(T::FLAG_IS_GENERIC) {
                    let ty_param_decl = lhs.decl.u();
                    debug_assert!(!inst.main_scope().decls.contains(&ty_param_decl)); // ty_param_decl cannot be used
                    let inst_param = if ty_param_decl.flags.get(DeclFlags::IS_GENERIC) {
                        inst.generics_scope().u().as_ref()
                    } else {
                        inst.main_scope()
                    }
                    .find_decl_norec(lhs.sym, false)
                    .u();
                    lhs.as_mut().decl = Some(inst_param);
                } else {
                    debug_assert!(ty == inst);
                }
                (named_arg.rhs, lhs.decl.u())
            });

            for (arg, inst_param) in pos_args.chain(named_args) {
                debug_assert!(
                    inst.main_scope().decls.contains(&inst_param)
                        || inst.generics_scope().is_some_and(|s| s.decls.contains(&inst_param))
                );
                self.finalize_expr(arg, inst_param.var_ty.u())?;
                debug_assert!(ty.generics().iter().all(|g| g.generic.u().cur_inst.is_none()));
            }
        }

        debug_assert!(ty.generics().iter().all(|g| g.generic.u().cur_inst.is_none()));
        Ok(inst)
    }

    fn finalize_instantiation<T: PolymorphableType>(
        &mut self,
        ty: Ptr<T>,
        expr: Ptr<Ast>,
    ) -> SemaResult<Ptr<T>> {
        if !ty.flags().get(T::FLAG_IS_GENERIC) {
            return Ok(ty);
        }
        debug_assert!(!ty.flags().get(T::FLAG_IS_INSTANTIATION));

        // finalize generics & handle missing generics
        let mut err = false;
        let mut generic_inst = Vec::with_capacity(ty.generics_scope().u().decls.len());
        let mut is_instantiation_with_generics = false;
        for g_decl in ty.generics_scope().u().decls.iter() {
            let g = g_decl.generic.u();
            let Some(mut inst_val) = g.as_mut().cur_inst.take() else {
                cerror!(
                    expr.full_span(),
                    "Cannot infer value of generic argument `{}`",
                    g.name.sym
                );
                err = true;
                continue;
            };
            inst_val.finalize_allow_generic();
            debug_assert!(inst_val.replacement.is_none());
            if inst_val.kind == AstKind::GenericSlot {
                is_instantiation_with_generics = true;
            }
            generic_inst.push(inst_val);
        }
        debug_assert!(ty.generics().iter().all(|g| g.generic.u().cur_inst.is_none()));
        if err {
            return Err(HandledErr);
        }

        // Look for already compiled instantiation
        for inst in ty.polymorphs().iter().copied() {
            debug_assert!(inst.flags().get(T::FLAG_IS_INSTANTIATION));
            let i_generics = &inst.generics_scope().u().decls;
            debug_assert_eq!(generic_inst.len(), i_generics.len());
            if generic_inst
                .iter()
                .zip(i_generics)
                .all(|(inst, c)| generic_match(*inst, c.const_val().u(), true))
            {
                return Ok(inst);
            }
        }

        // Compile and Add new instantiation
        let inst = ty.clone_ast(&self.cctx.alloc);
        inst.flags().set(T::FLAG_IS_INSTANTIATION);
        inst.flags().unset(T::FLAG_IS_GENERIC);
        if is_instantiation_with_generics {
            // TODO: only create real instantiations
            inst.flags().set(T::FLAG_IS_INSTANTIATION_WITH_GENERICS);
        }

        debug_assert!(ty.main_scope().flags.get(ScopeFlags::WAS_SETUP));
        debug_assert!(ty.generics_scope().u().flags.get(ScopeFlags::WAS_SETUP));
        setup_scopes(
            &mut inst.main_scope(),
            inst.generics_scope(),
            ty.generics_scope().u().parent.u(),
        );

        for (idx, c) in inst.generics_scope().u().decls.iter().enumerate() {
            let inst_val = generic_inst.get(idx).u().upcast();
            c.as_mut().flags.set(DeclFlags::IS_INSTANTIATION_VALUE);
            c.as_mut().var_ty = Some(inst_val.ty.u());
            c.as_mut().init = Some(inst_val);
        }

        // Add instantiation immediately to prevent recursive instantiation. (see `tests::generics::recursive_generic_instantiation`)
        ty.polymorphs().push(inst);

        let res = self.analyze(inst.upcast(), &None, false);
        if let NotFinished(dep) = res {
            self.unfinished_instantiations.push(inst.upcast().downcast_polymorphable());
            self.unfinished_instantiation_units
                .push(SemaUnit { stmt: inst.upcast(), waiting_for: Some(dep) });
        }
        let _ = res; // ignored. see `dont_duplicate_dependency_errors_in_generic_items`
        Ok(inst)
    }

    fn finalize_expr(&mut self, expr: Ptr<Ast>, out_ty: Ptr<ast::Type>) -> Result<(), HandledErr> {
        use Result::*;

        let ty = expr.as_mut().ty.as_mut().u();

        if ty.propagates_out() {
            // never is important for trimming unreachable code
            return Ok(());
        }

        // This seams very fragile, but maybe it's fine.
        // TODO: recursively finalize children aswell
        match expr.matchable2() {
            AstMatch::PositionalInitializer(_) => todo!(),
            AstMatch::NamedInitializer(i) => {
                let i = i.as_mut();
                debug_assert!(
                    i.flags.get(InitializerFlags::IS_TYPE_INIT)
                        ^ i.flags.get(InitializerFlags::IS_PTR_INIT)
                );

                let struct_ty = if i.flags.get(InitializerFlags::IS_TYPE_INIT) {
                    out_ty
                } else {
                    out_ty.downcast::<ast::PtrTy>().pointee.downcast_type()
                };
                debug_assert_matches!(struct_ty.kind, AstKind::StructDef | AstKind::SliceTy);
                i.resolved_struct_inst = Some(struct_ty);
            },
            _ => {},
        }

        finalize_ty(ty, out_ty, true);
        Ok(())
    }

    fn validate_mutation(
        &self,
        mutation_kind: MutationKind,
        mutated: Ptr<Ast>,
        expr: Ptr<Ast>,
    ) -> SemaResult<()> {
        if !self.cctx.do_mut_checks {
            // TODO: assigning to const is never valid
            return Ok(());
        }

        let p = p();

        // TODO?: allow `ptr: **mut u8; ptr.*.* = 1;`
        const DEEP_MUT_CHECK: bool = true;

        enum InvalidMutation {
            Var(Ptr<ast::Ident>),
            Ptr(Ptr<ast::Ast>),
            Slice(Ptr<ast::Ast>),
            /// `slice[..]mut`
            Reslice(Ptr<ast::Ast>),
        }

        let err = 'err: {
            #[derive(PartialEq, Eq)]
            enum K {
                /// Mutation of the value itself, like in `a = 1;`
                Assign,
                /// Mutation of the elements behind the reference
                Deref,
            }

            let mut kind = K::Assign;

            macro_rules! handle_deref {
                ($operand:expr) => {{
                    if $operand.ty.u().propagates_out() {
                        return Ok(());
                    }
                    let ptr_ty = $operand.ty.downcast::<ast::PtrTy>();
                    if !ptr_ty.is_mut {
                        break 'err InvalidMutation::Ptr($operand);
                    }
                    if !DEEP_MUT_CHECK {
                        return Ok(());
                    }
                    // mutated = $operand.rep();
                    kind = K::Deref;
                }};
            }

            /// array[0]     -> field access
            /// array[..]mut -> &mut
            /// slice[0]     -> .*
            /// slice[..]mut -> .* + &mut
            macro_rules! handle_index {
                ($lhs:expr, $idx:expr) => {{
                    debug_assert_ne!($lhs.ty.u().kind, AstKind::PtrTy); // autodereferencing is not implemented

                    if $lhs.ty.u().propagates_out() {
                        return Ok(());
                    }

                    let do_slice = $idx.ty.u().kind == AstKind::RangeTy;
                    debug_assert!(!do_slice || kind != K::Deref);

                    match $lhs.ty.u().matchable().as_ref() {
                        TypeEnum::ArrayTy { .. } => {
                            // mutated = $lhs.rep();
                        },
                        TypeEnum::SliceTy { is_mut, .. } => {
                            if *is_mut {
                                return Ok(());
                            } else if do_slice {
                                break 'err InvalidMutation::Reslice($lhs);
                            } else {
                                break 'err InvalidMutation::Slice($lhs);
                            }
                        },
                        _ => unreachable_debug(),
                    }
                }};
            }

            match mutation_kind {
                MutationKind::Initialize => handle_deref!(mutated),
                MutationKind::Slice => {
                    let index = expr.downcast::<ast::Index>();
                    debug_assert_eq!(index.idx.ty.u().kind, AstKind::RangeTy);
                    handle_index!(index.lhs, index.idx);
                },
                _ => {},
            }

            let mut mutated = mutated;
            loop {
                match mutated.matchable2() {
                    AstMatch::Ident(ident) => {
                        let decl = ident.decl.u();
                        if kind == K::Assign && !decl.flags.get(DeclFlags::IS_MUT) {
                            break 'err InvalidMutation::Var(ident);
                        } else {
                            return Ok(());
                        }
                    },
                    AstMatch::Dot(dot) => {
                        let lhs = dot.lhs.u();
                        debug_assert_ne!(lhs.ty.u().kind, AstKind::PtrTy); // autodereferencing is not implemented
                        mutated =
                            if lhs.ty == p.module { dot.rhs.upcast() } else { dot.lhs.u().rep() };
                    },
                    AstMatch::Index(index) => {
                        handle_index!(index.lhs, index.idx);
                        mutated = index.lhs.rep();
                    },
                    AstMatch::UnaryOp(op) if op.op == UnaryOpKind::Deref => {
                        handle_deref!(op.operand);
                        mutated = op.operand.rep();
                    },
                    _ => return Ok(()),
                }
            }
        };

        // Errors:

        let op = match mutation_kind {
            MutationKind::Assign => "assign to",
            MutationKind::Initialize => "initialize",
            MutationKind::AddrOf | MutationKind::Slice => "mutably reference",
        };
        match (err, mutation_kind) {
            (InvalidMutation::Var(var), MutationKind::Assign) if var.decl.u().is_const => {
                cerror!(expr.full_span(), "Cannot {op} constant `{}`", var.sym);
            },
            (InvalidMutation::Var(var), MutationKind::AddrOf) if var.decl.u().is_const => {
                cwarn!(
                    expr.full_span(),
                    "The mutable pointer will reference a local copy of `{}`, not the constant \
                     itself",
                    var.sym
                );
                return Ok(());
            },
            (InvalidMutation::Var(var), k) => {
                debug_assert_ne!(k, MutationKind::Initialize);
                // Cannot be inlined (see <https://github.com/rust-lang/rust/pull/145838>)
                let v = if mutated == var.upcast() { "it" } else { &format!("`{}`", var.sym) };
                cerror!(
                    expr.full_span(),
                    "Cannot {op} `{}`, as {v} is not declared as mutable",
                    mutated.to_text(false),
                );
                chint!(var.decl.u().ident.span, "consider changing `{}` to be mutable", var.sym);
            },
            (InvalidMutation::Ptr(ptr), MutationKind::Initialize) => {
                let p = if mutated == ptr { "it" } else { &format!("`{}`", ptr.to_text(false)) };
                cerror!(
                    expr.full_span(),
                    "Cannot {op} the value behind `{}`, because {p} is an immutable pointer",
                    mutated.to_text(false),
                );
                chint!(ptr.full_span(), "The pointer type `{}` is not `mut`", ptr.ty.u());
            },
            (InvalidMutation::Ptr(ptr), _) => {
                cerror!(
                    expr.full_span(),
                    "Cannot {op} `{}`, which is behind the immutable pointer `{}`",
                    mutated.to_text(false),
                    ptr.to_text(false)
                );
                chint!(ptr.full_span(), "The pointer type `{}` is not `mut`", ptr.ty.u());
            },
            (InvalidMutation::Slice(slice), k) => {
                debug_assert_ne!(k, MutationKind::Initialize);
                cerror!(
                    expr.full_span(),
                    "Cannot {op} `{}`, which is behind the immutable slice `{}`",
                    mutated.to_text(false),
                    slice.to_text(false)
                );
                chint!(slice.full_span(), "The slice type `{}` is not `mut`", slice.ty.u());
            },
            (InvalidMutation::Reslice(slice), k) => {
                debug_assert_eq!(k, MutationKind::Slice);
                let s =
                    if mutated == slice { "it" } else { &format!("`{}`", slice.to_text(false)) };
                cerror!(
                    expr.full_span(),
                    "Cannot {op} the elements of `{}`, because {s} is an immutable slice",
                    mutated.to_text(false),
                );
                chint!(slice.full_span(), "The slice type `{}` is not `mut`", slice.ty.u());
            },
        }
        SemaResult::HandledErr
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
        self.validate_mutation(MutationKind::Assign, lvalue, full_expr)
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

    fn slice_fields(
        &self,
        elem_ty: Ptr<ast::Type>,
        is_mut: bool,
    ) -> SemaResult<[Ptr<ast::Decl>; 2]> {
        let p = p();
        let elem_ptr_ty = type_new!(PtrTy { pointee: elem_ty.upcast(), is_mut });
        let mut ptr = self.alloc(ast::Decl::from_ident(p.slice_ptr_field_ident))?;
        ptr.flags.set(DeclFlags::IS_DATA_MEMBER);
        ptr.var_ty = Some(elem_ptr_ty.upcast_to_type());
        Ok([ptr, p.slice_len_field])
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
            #[cfg(debug_assertions)]
            if sym.flags.get(DeclFlags::IS_GENERIC) {
                cwarn!(
                    sym.full_span(),
                    "Trying to access generic decl var_ty before it's analyzed. Maybe the Generic \
                     was not inferred"
                );
            }
            return NotFinished(UnitDependency::VarType(sym));
        })
    }

    fn get_cur_fn(&self) -> OPtr<ast::Fn> {
        let mut cur_scope = self.cur_scope;
        loop {
            if cur_scope.kind == ScopeKind::FnParams {
                return Some(cur_scope.get_expr().u().downcast::<ast::Fn>());
            }
            cur_scope = cur_scope.parent?;
        }
    }

    fn open_scope(&mut self, new_scope: &mut Scope) -> OpenScopeHandle {
        debug_only_assert!(new_scope.flags.get(ScopeFlags::WAS_CHECKED_FOR_DUPLICATES));
        if new_scope.kind == ScopeKind::File {
            debug_assert!(new_scope.parent.is_some());
        } else {
            debug_assert!(new_scope.parent.is_none_or(|p| p == self.cur_scope));
            //debug_assert!(new_scope.parent.is_none()); // For when NotFinished is removed
        }
        if new_scope.parent.is_none() {
            new_scope.parent = Some(self.cur_scope);
        }

        let mut osh = self.jump_open_scope(new_scope);
        osh.jumped = false;
        osh
    }

    fn jump_open_scope(&mut self, new_scope: &Scope) -> OpenScopeHandle {
        debug_assert!(new_scope.parent.is_some());

        let osh = OpenScopeHandle {
            prev_scope: self.cur_scope,
            jumped: true,
            #[cfg(debug_assertions)]
            debug_scope_level: {
                self.debug_scope_level += 1;
                self.debug_scope_level
            },
        };

        self.cur_scope = Ptr::from_ref(new_scope);
        self.defer_stack.open_scope();

        osh
    }

    fn close_scope(&mut self, osh: OpenScopeHandle) {
        #[cfg(debug_assertions)]
        {
            debug_assert_eq!(
                osh.debug_scope_level, self.debug_scope_level,
                "forgot to close scope"
            );
            self.debug_scope_level -= 1;
        }

        // The parent scope of the prelude file is the root scope. During analysis of other files
        // the correct starting value for cur_scope is the prelude scope, not the root scope.
        // Nevertheless, assigning the root scope to cur_scope (after analysis of the prelude) is
        // fine because open_scope is called immediately for the next file which overwrites cur_scope.
        debug_assert!(
            osh.jumped
                || osh.prev_scope == self.cur_scope.parent.u()
                || (osh.prev_scope.kind == ScopeKind::Root
                    && self.cur_scope.parent.u().kind == ScopeKind::File)
        );

        self.cur_scope = osh.prev_scope;
        self.defer_stack.close_scope();
    }

    #[inline]
    fn alloc<T>(&self, val: T) -> SemaResult<Ptr<T>> {
        Ok(self.cctx.alloc.alloc(val)?)
    }
}

#[must_use]
struct OpenScopeHandle {
    prev_scope: Ptr<Scope>,
    jumped: bool,
    #[cfg(debug_assertions)]
    debug_scope_level: usize,
}

/// `expr == None` means silent
pub fn accumulate_type(
    ty_acc: &mut OPtr<ast::Type>,
    next_ty: Ptr<ast::Type>,
    expr: OPtr<Ast>,
) -> Result<(), HandledErr> {
    if let Some(ty_acc) = ty_acc {
        accumulate_type_inner(ty_acc, next_ty, expr, false)
    } else {
        *ty_acc = Some(next_ty);
        Result::Ok(())
    }
}

pub fn accumulate_type_inner(
    ty_acc: &mut Ptr<ast::Type>,
    next_ty: Ptr<ast::Type>,
    expr: OPtr<Ast>,
    quiet: bool,
) -> Result<(), HandledErr> {
    if let Some(common_ty) = common_type(*ty_acc, next_ty) {
        if !quiet {
            *ty_acc = common_ty;
        }
        Result::Ok(())
    } else {
        if let Some(expr) = expr
            && !quiet
        {
            error_mismatched_types(expr.return_val_span(), *ty_acc, next_ty);
        }
        Result::Err(HandledErr)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MutationKind {
    Assign,
    Initialize,
    AddrOf,
    Slice,
}

fn extern_directive_name(directive: Ptr<ast::Ast>) -> &'static str {
    match directive.kind {
        AstKind::ExternDirective => "extern",
        AstKind::IntrinsicDirective => "intrinsic",
        _ => unreachable_debug(),
    }
}

#[derive(Debug, Clone, Copy)]
pub struct PatternSource {
    pub ty: Ptr<ast::Type>,
    pub kind: PatternSourceKind,
}

#[derive(Debug, Clone, Copy)]
pub enum PatternSourceKind {
    ByVal,
    ByRef { is_mut: bool },
}

impl PatternSource {
    pub fn new(ty: Ptr<ast::Type>) -> PatternSource {
        match ty.matchable2() {
            TypeMatch::PtrTy(p) => PatternSource {
                kind: PatternSourceKind::ByRef { is_mut: p.is_mut },
                ty: p.pointee.downcast_type(),
            },
            _ => PatternSource { kind: PatternSourceKind::ByVal, ty },
        }
    }

    fn ty_in_body(self, narrowed_inner: Ptr<ast::Type>) -> Ptr<ast::Type> {
        match self.kind {
            PatternSourceKind::ByRef { is_mut } => {
                type_new!(PtrTy { is_mut, pointee: narrowed_inner.upcast() }).upcast_to_type()
            },
            PatternSourceKind::ByVal => narrowed_inner,
        }
    }
}

fn type_layout_finished(ty: Ptr<ast::Type>) -> bool {
    match ty.matchable().as_ref() {
        TypeEnum::StructDef { fields, .. } | TypeEnum::UnionDef { fields, .. } => {
            fields.iter().all(|f| f.var_ty.is_some())
        },
        TypeEnum::EnumDef { variants, tag_ty, .. } => {
            tag_ty.is_some_and(|i| i.bits.is_some()) && variants.iter().all(|v| v.var_ty.is_some())
        },
        _ => true,
    }
}

fn get_enum_variant_tag(variant: Ptr<ast::Decl>) -> SemaResult<Ptr<ast::IntVal>> {
    match try_get_enum_variant_tag(variant) {
        Some(tag) => Ok(tag),
        None => NotFinished(UnitDependency::EnumVariantTag(variant)),
    }
}

fn try_get_enum_variant_tag(variant: Ptr<ast::Decl>) -> OPtr<ast::IntVal> {
    variant.init?.try_downcast::<ast::IntVal>()
}

fn get_ty(expr: Ptr<ast::Ast>) -> SemaResult<Ptr<ast::Type>> {
    match expr.ty {
        Some(ty) => Ok(ty),
        None => NotFinished(UnitDependency::ExprType(expr)),
    }
}

fn get_var_ty(decl: Ptr<ast::Decl>) -> SemaResult<Ptr<ast::Type>> {
    match decl.var_ty {
        Some(ty) => Ok(ty),
        None => NotFinished(UnitDependency::VarType(decl)),
    }
}

// TODO: also use for SliceTy
fn find_in_namespace(ty: Ptr<ast::Type>, sym: Symbol) -> OPtr<ast::Decl> {
    ty.get_scope()?
        .find_decl_norec(sym, false)
        .or_else(|| Some(ty.get_associated_external_consts()?.find_field(sym)?.1))
}

/// Maybe merge Ast Nodes
#[derive(Debug)]
enum CallKind {
    Function { ty_hint: OPtr<ast::Type> },
    EnumInit { variant: Ptr<ast::Decl> },
    PositionalInitializer,
    Type,
}

/*
bitflags!(CallFlags: u8 {
    IS_FUNCTION_CALL,
    IS_METHOD_CALL,

    IS_ENUM_INIT,

    IS_POSITIONAL_INITIALIZER,
});
*/
