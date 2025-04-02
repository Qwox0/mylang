use crate::{
    arena_allocator::Arena,
    ast::{self, ast_new},
    compiler::CompileDurations,
    diagnostic_reporter,
    parser::lexer::Span,
    ptr::{OPtr, Ptr},
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
    pub primitives_scope: Ptr<ast::Block>,

    /// absolute import file path -> index into `files`
    pub imports: HashMap<PathBuf, FilesIndex>,
    /// This List must contain [`Ptr`]s because it might reallocate while a mutable reference to
    /// [`SourceFile`] is active.
    pub files: Vec<Ptr<SourceFile>>,
    pub root_file_idx: Option<FilesIndex>,

    pub compiler_libs_path: PathBuf,
    pub project_path: PathBuf,

    /// TODO: implement `#mut_checks(.Enabled)`, `#mut_checks(.Disabled)`
    pub do_mut_checks: bool,

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
        let primitives_scope = alloc.alloc(global_scope).unwrap();

        let compiler_binary_path = std::env::current_exe().unwrap();

        // TODO: copy `lib/` to be next to the binary
        let mut compiler_libs_path = compiler_binary_path.parent().unwrap();
        while compiler_libs_path.file_name().unwrap() != "mylang" {
            compiler_libs_path = compiler_libs_path.parent().unwrap();
        }
        let compiler_libs_path = compiler_libs_path.join("lib");

        let ctx = CompilationContextInner {
            alloc,
            diagnostic_reporter: CtxDiagnosticReporter::default(),
            compile_time: CompileDurations::default(),

            primitives,
            primitives_scope,

            imports: HashMap::new(),
            files: Vec::new(),
            root_file_idx: None,

            compiler_libs_path,
            project_path: PathBuf::new(),

            do_mut_checks: true,

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
    pub fn add_import(
        self,
        path: &str,
        cur_path: OPtr<Path>,
    ) -> Result<FilesIndex, std::io::Error> {
        //  if path == "prelude" {
        //     PathBuf::from(concat!(std::env!("HOME"), "/src/mylang/lib/prelude.mylang"))
        let path = if path == "std" {
            self.compiler_libs_path.join("std/mod.mylang")
        } else if path == "libc" {
            self.compiler_libs_path.join("libc.mylang")
        } else {
            let cur_path = cur_path.u();
            debug_assert!(cur_path.is_file());
            cur_path.parent().u().join(path)
        }
        .canonicalize()?;
        if let Some(idx) = self.imports.get(&path) {
            return Ok(*idx);
        }
        let file = SourceFile::read(Ptr::from_ref(path.as_path()), &self.alloc)?;
        let file = self.alloc.alloc(file).unwrap();
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
        let file = self.alloc.alloc(file).unwrap();
        self.as_mut().files.push(file);
        let idx = self.files.len() - 1;
        self.as_mut().imports.insert(path, idx);
        idx
    }

    pub fn path_in_proj(self, path: &Path) -> &Path {
        path.strip_prefix(&self.project_path).unwrap_or(path)
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

            fn hint<M>(&mut self, span: Span, msg: &M)
            where M: std::fmt::Display + ?Sized {
                self.diagnostic_reporter.hint(span, msg)
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
