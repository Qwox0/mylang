use crate::{
    arena_allocator::Arena,
    ast::{self, ast_new},
    compiler::CompileDurations,
    diagnostic_reporter,
    parser::lexer::Span,
    ptr::Ptr,
    sema::primitives::Primitives,
    source_file::SourceFile,
    util::UnwrapDebug,
};
use std::{
    collections::HashMap,
    ops::{Deref, DerefMut},
    path::{Path, PathBuf},
};

#[cfg(not(test))]
pub type CtxDiagnosticReporter = diagnostic_reporter::DiagnosticPrinter;
#[cfg(test)]
pub type CtxDiagnosticReporter = diagnostic_reporter::DiagnosticCollector;

pub struct CompilationContextInner {
    pub alloc: Arena,
    pub diagnostic_reporter: CtxDiagnosticReporter,
    pub compile_time: CompileDurations,

    pub primitives: Primitives,
    pub global_scope: Ptr<ast::Block>,

    /// absolute import file path -> index into `files`
    pub imports: HashMap<PathBuf, FilesIndex>,
    pub files: Vec<SourceFile>,

    #[cfg(debug_assertions)]
    pub debug_types: bool,
}

pub type FilesIndex = usize;

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
    pub fn new() -> CompilationContext {
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
            diagnostic_reporter: CtxDiagnosticReporter::default(),
            compile_time: CompileDurations::default(),

            primitives,
            global_scope,

            imports: HashMap::new(),
            files: Vec::new(),

            #[cfg(debug_assertions)]
            debug_types: false,
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

impl Ptr<CompilationContextInner> {
    pub fn add_import(self, mut path: PathBuf) -> Result<FilesIndex, std::io::Error> {
        if path == Path::new("prelude") {
            // TODO: use `std::env::current_exe().unwrap().canonicalize()`
            path = PathBuf::from(concat!(std::env!("HOME"), "/src/mylang/lib/prelude.mylang"));
        }
        let path = path.canonicalize()?;
        if let Some(idx) = self.imports.get(&path) {
            return Ok(*idx);
        }
        let file = SourceFile::read(Ptr::from_ref(path.as_path()), &self.alloc)?;
        self.as_mut().files.push(file);
        let idx = self.files.len() - 1;
        self.as_mut().imports.insert(path, idx);
        Ok(idx)
    }

    pub fn add_import_from_file(self, file: SourceFile) -> FilesIndex {
        if let Some(idx) = self.imports.get(file.path.as_ref()) {
            return *idx;
        }
        let path = file.path.to_path_buf();
        self.as_mut().files.push(file);
        let idx = self.files.len() - 1;
        self.as_mut().imports.insert(path, idx);
        idx
    }
}

#[thread_local]
static mut CTX: Option<CompilationContextInner> = None;

macro_rules! impl_diagnostic_reporter_for_ctx {
    ($ty:ty) => {
        impl crate::diagnostic_reporter::DiagnosticReporter for $ty {
            fn report<M>(
                &mut self,
                severity: crate::diagnostic_reporter::DiagnosticSeverity,
                span: Span,
                msg: &M,
            ) where
                M: std::fmt::Display + ?Sized,
            {
                self.diagnostic_reporter.report(severity, span, msg)
            }

            fn max_past_severity(&self) -> Option<crate::diagnostic_reporter::DiagnosticSeverity> {
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
pub fn primitives() -> &'static Primitives {
    &ctx().primitives
}
