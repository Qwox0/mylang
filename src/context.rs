use crate::{
    arena_allocator::Arena,
    ast::{self, ast_new},
    compiler::CompileDurations,
    diagnostic_reporter::{self, DiagnosticReporter, DiagnosticSeverity},
    error::SpannedError,
    parser::lexer::{Code, Span},
    ptr::Ptr,
    sema::primitives::Primitives,
    util::UnwrapDebug,
};
use std::ops::{Deref, DerefMut};

#[cfg(not(test))]
pub type CtxDiagnosticReporter = diagnostic_reporter::DiagnosticPrinter;
#[cfg(test)]
pub type CtxDiagnosticReporter = diagnostic_reporter::DiagnosticCollector;

pub struct CompilationContextInner {
    pub alloc: Arena,
    pub code: Ptr<Code>,
    pub diagnostic_reporter: CtxDiagnosticReporter,
    pub compile_time: CompileDurations,

    pub primitives: Primitives,
    pub global_scope: Ptr<ast::Block>,
}

pub struct CompilationContext(pub Ptr<CompilationContextInner>);

impl Deref for CompilationContext {
    type Target = CompilationContextInner;

    fn deref(&self) -> &Self::Target {
        self.0.as_ref()
    }
}

impl DerefMut for CompilationContext {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.0.as_mut()
    }
}

impl Drop for CompilationContext {
    fn drop(&mut self) {
        #[allow(static_mut_refs)]
        unsafe {
            debug_assert!(self.0 == Ptr::from_ref(CTX.as_ref().u()));
            drop(CTX.take().u())
        }
    }
}

impl CompilationContext {
    pub fn new(code: Ptr<Code>) -> CompilationContext {
        let alloc = Arena::new();

        let mut stmts = Vec::new();
        let primitives = Primitives::setup(&mut stmts, &alloc);
        let stmts = alloc.alloc_slice(&stmts).unwrap();
        let stmts = Ptr::<[Ptr<ast::Decl>]>::cast_slice::<Ptr<ast::Ast>>(stmts);

        let global_scope =
            ast_new!(Block { span: Span::ZERO, has_trailing_semicolon: false, stmts });
        let global_scope = alloc.alloc(global_scope).unwrap();

        let ctx = CompilationContextInner {
            alloc,
            code,
            diagnostic_reporter: CtxDiagnosticReporter::default(),
            compile_time: CompileDurations::default(),
            primitives,
            global_scope,
        };
        #[allow(static_mut_refs)]
        let ctx: &'static _ = unsafe {
            assert!(CTX.is_none(), "drop the previous context first");
            CTX = Some(ctx);
            CTX.as_ref().u()
        };
        CompilationContext(Ptr::from_ref(ctx))
    }
}

#[thread_local]
static mut CTX: Option<CompilationContextInner> = None;

macro_rules! impl_diagnostic_reporter_for_ctx {
    ($ty:ty) => {
        impl DiagnosticReporter for $ty {
            fn report(&mut self, severity: DiagnosticSeverity, err: &impl SpannedError) {
                self.diagnostic_reporter.report(severity, err)
            }

            fn max_past_severity(&self) -> Option<DiagnosticSeverity> {
                self.diagnostic_reporter.max_past_severity()
            }
        }
    };
}
impl_diagnostic_reporter_for_ctx! {CompilationContextInner}
impl_diagnostic_reporter_for_ctx! {CompilationContext}

#[inline]
pub fn ctx() -> &'static CompilationContextInner {
    #[allow(static_mut_refs)]
    unsafe {
        CTX.as_ref().unwrap()
    }
}

#[inline]
pub fn ctx_mut() -> &'static mut CompilationContextInner {
    #[allow(static_mut_refs)]
    unsafe {
        CTX.as_mut().unwrap()
    }
}

#[inline]
pub fn code() -> &'static Code {
    &ctx().code
}

#[inline]
pub fn primitives() -> &'static Primitives {
    &ctx().primitives
}
