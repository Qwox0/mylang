#[cfg(test)]
use crate::parser::lexer;
use crate::{
    arena_allocator::Arena,
    ast,
    cli::BuildArgs,
    compiler::CompileDurations,
    diagnostics::{self, HandledErr, cerror, cerror_fatal, chint},
    intern_pool::{InternPool, Symbol},
    parser::lexer::Span,
    ptr::{HashKeyPtr, OPtr, Ptr},
    scope::{Scope, ScopeKind},
    sema::primitives::Primitives,
    source_file::SourceFile,
    util::{OptionExt, UnwrapDebug, is_canonical},
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
    pub tmp_alloc: Arena,
    pub diagnostic_reporter: CtxDiagnosticReporter,
    pub compile_time: CompileDurations,

    pub symbols: InternPool,
    pub primitives: Primitives,
    pub primitives_scope: Ptr<Scope>,
    /// Parent scope of files
    pub global_scope: Ptr<Scope>,
    pub import_manager: ImportManager,

    pub libraries: HashSet<HashKeyPtr<str>>,
    pub library_search_paths: HashSet<HashKeyPtr<str>>,

    /// TODO: implement `#mut_checks(.Enabled)`, `#mut_checks(.Disabled)`
    pub do_mut_checks: bool,

    pub args: BuildArgs,
    pub entry_point: Symbol,

    pub ty_names: HashMap<Ptr<ast::Type>, Symbol>,

    #[cfg(test)]
    pub stmts: Option<Box<[Ptr<ast::Ast>]>>,
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
    pub fn empty(args: BuildArgs) -> CompilationContext {
        let alloc = Arena::new();
        let diagnostic_reporter = CtxDiagnosticReporter::default();
        let mut symbols = InternPool::new();

        let mut decls = Vec::new();
        let primitives = Primitives::setup(&mut decls, &mut symbols, &alloc);
        let decls = alloc.alloc_slice(&decls).unwrap();
        let primitives_scope = alloc.alloc(Scope::new(decls, ScopeKind::Root)).unwrap();

        let entry_point_sym = symbols.get_or_intern(Ptr::from_ref(args.entry_point.as_ref()));

        let ctx = CompilationContextInner {
            alloc,
            tmp_alloc: Arena::new_scratch(32 * 1024 - Arena::BUMP_OVERHEAD),
            diagnostic_reporter,
            compile_time: CompileDurations::default(),

            symbols,
            primitives,
            primitives_scope,
            global_scope: primitives_scope, // may be replaced with prelude later
            import_manager: ImportManager::new(),

            libraries: HashSet::new(),
            library_search_paths: HashSet::new(),

            do_mut_checks: true,

            args,
            entry_point: entry_point_sym,

            ty_names: HashMap::new(),

            #[cfg(test)]
            stmts: None,
        };
        #[allow(static_mut_refs)]
        let ctx: &'static mut _ = unsafe {
            assert!(CTX.is_none(), "drop the previous context first");
            CTX = Some(ctx);
            CTX.as_mut().u()
        };
        CompilationContext(Ptr::from_ref(ctx))
    }

    pub fn basic(args: BuildArgs) -> Result<CompilationContext, HandledErr> {
        let mut ctx = CompilationContext::empty(args);
        ctx.0.load_prelude();
        ctx.0.load_start_file()?;
        Ok(ctx)
    }

    #[cfg(test)]
    pub fn for_tests<C: AsRef<lexer::Code> + ?Sized>(
        args: BuildArgs,
        code: Ptr<C>,
        load_prelude: bool,
    ) -> CompilationContext {
        let mut ctx = CompilationContext::empty(args);
        if load_prelude {
            ctx.0.load_prelude();
        }
        ctx.0.set_test_file(Ptr::from_ref((*code).as_ref())).unwrap();
        ctx
    }
}

impl CompilationContextInner {
    pub fn load_prelude(mut self: Ptr<Self>) {
        debug_assert_eq!(self.import_manager.files.len(), 0);
        let Ok(prelude_idx) = self.add_import("prelude", None, Span::ZERO) else {
            cerror_fatal!(Span::ZERO, "cannot load prelude")
        };
        self.import_manager.set_prelude_file(prelude_idx);
    }

    pub fn load_start_file(&mut self) -> Result<(), HandledErr> {
        let path = self.args.path.clone();
        let root_file_idx =
            self.import_manager.add_path_import(path, false, Span::ZERO, &self.alloc)?;
        Ok(self.import_manager.set_start_file(root_file_idx))
    }

    pub fn add_import(
        &mut self,
        import_text: &str,
        cur_path: Option<&Path>,
        err_span: Span,
    ) -> Result<FilesIndex, HandledErr> {
        let std_path = self.import_manager.try_resolve_standard_import_path(import_text);
        let assume_canonicalized = std_path.is_some();
        let path = std_path.unwrap_or_else(|| {
            let cur_path = cur_path.u();
            debug_assert!(is_canonical(cur_path));
            debug_assert!(cur_path.is_file());
            // This needs to be canonicalized because import_text might look like "../a.mylang"
            cur_path.parent().u().join(import_text)
        });
        self.import_manager
            .add_path_import(path, assume_canonicalized, err_span, &self.alloc)
    }

    pub fn files(&self) -> &[Ptr<SourceFile>] {
        &self.import_manager.files
    }

    pub fn start_file(&self) -> Ptr<SourceFile> {
        let start_file = self.import_manager.start_file.u();
        debug_assert!(self.import_manager.files.contains(&start_file));
        start_file
    }

    #[cfg(test)]
    fn set_test_file(&mut self, code: Ptr<lexer::Code>) -> Result<(), HandledErr> {
        let test_file_idx = self.import_manager.add_source_file(
            SourceFile::new(Ptr::from_ref(self.args.path.as_ref()), code),
            None,
            &self.alloc,
        )?;
        Ok(self.import_manager.set_start_file(test_file_idx))
    }

    pub fn add_library(&mut self, str_lit: Ptr<ast::StrVal>) -> Result<(), HandledErr> {
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
        self.libraries.insert(name.as_hash_key());
        Ok(())
    }

    pub fn add_library_search_path(&mut self, path: Ptr<str>) -> Result<(), HandledErr> {
        self.library_search_paths.insert(path.as_hash_key());
        Ok(())
    }

    pub fn path_in_proj<'p>(&self, path: &'p Path) -> &'p Path {
        path.strip_prefix(self.import_manager.project_path.u().as_ref()).unwrap_or(path)
    }
}

impl CompilationContextInner {
    pub fn debug_llvm_module_on_invalid_fn(&self) -> bool {
        self.args.debug_llvm_ir_optimized || self.args.debug_llvm_ir_unoptimized
    }
}

pub struct ImportManager {
    /// absolute import file path -> index into `files`
    //pub imports: HashMap<Box<Path>, FilesIndex>, // If I use this valgrind shows false positives
    imports: HashMap<PathBuf, FilesIndex>,
    /// This List must contain [`Ptr`]s because it might reallocate while a mutable reference to
    /// [`SourceFile`] is active.
    files: Vec<Ptr<SourceFile>>,

    /// also contained in [`ImportManager::files`]
    pub start_file: OPtr<SourceFile>,
    /// also contained in [`ImportManager::files`]
    pub prelude_file: OPtr<SourceFile>,

    project_path: OPtr<Path>,
    compiler_libs_path: PathBuf,
}

impl ImportManager {
    fn new() -> Self {
        let compiler_libs_path = if !cfg!(miri) {
            let compiler_binary_path = std::env::current_exe().unwrap();
            compiler_binary_path.parent().unwrap().join("lib").canonicalize().unwrap()
        } else {
            PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/target/debug/lib"))
        };
        assert!(
            compiler_libs_path.is_dir(),
            "\"{}\" must be a directory",
            compiler_libs_path.display()
        );

        ImportManager {
            imports: HashMap::new(),
            files: Vec::new(),
            start_file: None,
            prelude_file: None,
            project_path: None,
            compiler_libs_path,
        }
    }

    fn set_prelude_file(&mut self, prelude_file_idx: usize) {
        debug_assert_eq!(prelude_file_idx, 0);
        self.prelude_file.set_once(*self.files.get(prelude_file_idx).u());
    }

    fn set_start_file(&mut self, start_file_idx: usize) {
        let start_file = *self.files.get(start_file_idx).u();
        self.start_file.set_once(start_file);
        self.project_path = Some(Ptr::from_ref(start_file.path.parent().unwrap()));
    }

    fn try_resolve_standard_import_path(&self, import_text: &str) -> Option<PathBuf> {
        Some(match import_text {
            "std" => self.compiler_libs_path.join("std/mod.mylang"),
            "libc" => self.compiler_libs_path.join("libc.mylang"),
            "runtime" => self.compiler_libs_path.join("runtime.mylang"),
            "prelude" => self.compiler_libs_path.join("prelude.mylang"),
            _ => return None,
        })
    }

    fn add_path_import(
        &mut self,
        mut path: PathBuf,
        assume_canonicalized: bool,
        err_span: Span,
        alloc: &Arena,
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
            SourceFile::read(Ptr::from_ref(path.as_path()), &alloc)?
        };
        let file = res.map_err(|e| {
            let p = path.display();
            cerror!(err_span, "cannot import source file \"{p}\": {e}")
        })?;
        self.add_source_file(file, Some(path), alloc)
    }

    fn add_source_file(
        &mut self,
        file: SourceFile,
        path: Option<PathBuf>,
        alloc: &Arena,
    ) -> Result<FilesIndex, HandledErr> {
        let file = alloc.alloc(file)?;
        self.files.push(file);
        let idx = self.files.len() - 1;
        if let Some(path) = path {
            //self.as_mut().imports.insert(path.into_boxed_path(), idx);
            let old = self.imports.insert(path, idx);
            debug_assert!(old.is_none());
        }
        Ok(idx)
    }
}

macro_rules! impl_diagnostic_reporter_for_ctx {
    ($ty:ty) => {
        impl crate::diagnostics::DiagnosticReporter for $ty {
            fn report(
                &mut self,
                severity: crate::diagnostics::DiagnosticSeverity,
                span: Span,
                msg: impl std::fmt::Display,
            ) {
                self.diagnostic_reporter.report(severity, span, msg)
            }

            fn hint(&mut self, span: Span, msg: impl std::fmt::Display) {
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

#[thread_local]
static mut CTX: Option<CompilationContextInner> = None;

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

#[inline]
pub fn tmp_alloc() -> &'static mut Arena {
    &mut ctx_mut().tmp_alloc
}
