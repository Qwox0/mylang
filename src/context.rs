use crate::{
    arena_allocator::Arena,
    ast::{self, ast_new},
    cli::BuildArgs,
    compiler::CompileDurations,
    diagnostics::{self, HandledErr, cerror, cerror2, chint, handle_alloc_err},
    parser::lexer::Span,
    ptr::{HashKeyPtr, OPtr, Ptr},
    sema::primitives::Primitives,
    source_file::SourceFile,
    util::{UnwrapDebug, is_canonical},
};
use std::{
    collections::{HashMap, HashSet},
    ops::{Deref, DerefMut},
    path::{Path, PathBuf},
};

#[cfg(not(test))]
pub type CtxDiagnosticReporter = diagnostics::DiagnosticPrinter;
#[cfg(test)]
pub type CtxDiagnosticReporter = diagnostics::DiagnosticCollector;

pub struct CompilationContextInner {
    pub alloc: Arena,
    pub diagnostic_reporter: CtxDiagnosticReporter,
    pub compile_time: CompileDurations,

    pub primitives: Primitives,
    pub primitives_scope: Ptr<ast::Block>,

    /// absolute import file path -> index into `files`
    pub imports: HashMap<Box<Path>, FilesIndex>,
    /// This List must contain [`Ptr`]s because it might reallocate while a mutable reference to
    /// [`SourceFile`] is active.
    pub files: Vec<Ptr<SourceFile>>,
    pub root_file_idx: Option<FilesIndex>,

    pub libraries: HashSet<HashKeyPtr<str>>,
    pub library_search_paths: HashSet<HashKeyPtr<str>>,

    pub compiler_libs_path: PathBuf,
    pub project_path: OPtr<Path>,

    /// TODO: implement `#mut_checks(.Enabled)`, `#mut_checks(.Disabled)`
    pub do_mut_checks: bool,

    pub args: BuildArgs,
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
    pub fn new(args: BuildArgs) -> CompilationContext {
        let alloc = Arena::new();

        let mut stmts = Vec::new();
        let primitives = Primitives::setup(&mut stmts, &alloc);
        let stmts = alloc.alloc_slice(&stmts).unwrap();
        let stmts = Ptr::<[Ptr<ast::Decl>]>::cast_slice::<Ptr<ast::Ast>>(stmts);

        let global_scope =
            ast_new!(Block { span: Span::ZERO, has_trailing_semicolon: false, stmts });
        let primitives_scope = alloc.alloc(global_scope).unwrap();

        let compiler_binary_path = std::env::current_exe().unwrap();
        let compiler_libs_path =
            compiler_binary_path.parent().unwrap().join("lib").canonicalize().unwrap();
        assert!(
            compiler_libs_path.is_dir(),
            "\"{}\" must be a directory",
            compiler_libs_path.display()
        );

        let ctx = CompilationContextInner {
            alloc,
            diagnostic_reporter: CtxDiagnosticReporter::default(),
            compile_time: CompileDurations::default(),

            primitives,
            primitives_scope,

            imports: HashMap::new(),
            files: Vec::new(),
            root_file_idx: None,

            libraries: HashSet::new(),
            library_search_paths: HashSet::new(),

            compiler_libs_path,
            project_path: None,

            do_mut_checks: true,

            args,
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
    pub fn set_source_root(self, path: PathBuf) -> Result<(), HandledErr> {
        let root_file_idx = self.add_path_import(path, false, Span::ZERO)?;
        self.set_root_file(root_file_idx)
    }

    pub fn set_root_file(mut self, root_file_idx: usize) -> Result<(), HandledErr> {
        if self.root_file_idx.is_some() {
            return cerror2!(Span::ZERO, "Tried to set the root file multiple times");
        }
        self.root_file_idx = Some(root_file_idx);
        let root_file = self.files[root_file_idx];
        self.project_path = Some(Ptr::from_ref(root_file.path.parent().unwrap()));
        Ok(())
    }

    pub fn add_import(
        self,
        import_text: &str,
        cur_path: Option<&Path>,
        err_span: Span,
    ) -> Result<FilesIndex, HandledErr> {
        let lib = |path| self.add_path_import(path, true, err_span);
        match import_text {
            "std" => lib(self.compiler_libs_path.join("std/mod.mylang")),
            "libc" => lib(self.compiler_libs_path.join("libc.mylang")),
            "runtime" => lib(self.compiler_libs_path.join("runtime.mylang")),
            p => {
                let cur_path = cur_path.u();
                debug_assert!(is_canonical(cur_path));
                debug_assert!(cur_path.is_file());
                // This needs to be canonicalized because import_text might look like "../a.mylang"
                self.add_path_import(cur_path.parent().u().join(p), false, err_span)
            },
        }
    }

    fn add_path_import(
        self,
        mut path: PathBuf,
        assume_canonicalized: bool,
        err_span: Span,
    ) -> Result<FilesIndex, HandledErr> {
        let res: std::io::Result<_> = try {
            if assume_canonicalized {
                debug_assert!(is_canonical(&path));
            } else {
                path = path.canonicalize()?;
            }
            if let Some(idx) = self.imports.get(path.as_path()) {
                return Ok(*idx);
            }
            SourceFile::read(Ptr::from_ref(path.as_path()), &self.alloc)?
        };
        let file = res.map_err(|e| {
            let p = path.display();
            cerror!(err_span, "cannot import source file \"{p}\": {e}")
        })?;
        self.add_source_file(file, Some(path))
    }

    #[cfg(test)]
    pub fn set_test_root(self, code: Ptr<crate::parser::lexer::Code>) -> Result<(), HandledErr> {
        let root_file_idx = self.add_test_code_buf(code)?;
        self.set_root_file(root_file_idx)
    }

    #[cfg(test)]
    pub fn add_test_code_buf(
        self,
        code: Ptr<crate::parser::lexer::Code>,
    ) -> Result<FilesIndex, HandledErr> {
        self.add_source_file(SourceFile::new(Ptr::from_ref("test.mylang".as_ref()), code), None)
    }

    fn add_source_file(
        self,
        file: SourceFile,
        path: Option<PathBuf>,
    ) -> Result<FilesIndex, HandledErr> {
        let file = self.alloc.alloc(file).map_err(handle_alloc_err)?;
        self.as_mut().files.push(file);
        let idx = self.files.len() - 1;
        if let Some(path) = path {
            self.as_mut().imports.insert(path.into_boxed_path(), idx);
        }
        Ok(idx)
    }

    pub fn add_library(self, str_lit: Ptr<ast::StrVal>) -> Result<(), HandledErr> {
        let name = str_lit.text;
        // TODO: test
        if let Some(slash_idx) = str_lit.text.rfind(['/', '\\']) {
            cerror!(
                str_lit.span,
                "Currently paths in library names are not supported because they are passed \
                 directly to the `-l` flag of lld."
            );
            chint!(
                str_lit.span,
                "Consider adding `#add_library_search_path \"{}\"` instead",
                &name[..slash_idx]
            );
            return Err(HandledErr);
        }
        self.as_mut().libraries.insert(name.as_hash_key());
        Ok(())
    }

    pub fn add_library_search_path(self, path: Ptr<str>) -> Result<(), HandledErr> {
        self.as_mut().library_search_paths.insert(path.as_hash_key());
        Ok(())
    }

    pub fn path_in_proj(self, path: &Path) -> &Path {
        path.strip_prefix(self.project_path.u().as_ref()).unwrap_or(path)
    }
}

impl CompilationContextInner {
    pub fn debug_llvm_module_on_invalid_fn(&self) -> bool {
        self.args.debug_llvm_ir_optimized || self.args.debug_llvm_ir_unoptimized
    }
}

#[thread_local]
static mut CTX: Option<CompilationContextInner> = None;

macro_rules! impl_diagnostic_reporter_for_ctx {
    ($ty:ty) => {
        impl crate::diagnostics::DiagnosticReporter for $ty {
            fn report<M>(
                &mut self,
                severity: crate::diagnostics::DiagnosticSeverity,
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

            fn max_past_severity(&self) -> Option<crate::diagnostics::DiagnosticSeverity> {
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
