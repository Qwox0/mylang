use crate::{
    ast::{Ast, debug::DebugAst},
    cli::{BuildArgs, OutKind},
    codegen::llvm::{self, CodegenModuleExt, error::CodegenResult},
    context::{CompilationContext, CompilationContextInner, tmp_alloc},
    diagnostics::{DiagnosticReporter, HandledErr, cerror, cwarn},
    parser::{self, lexer::Span},
    ptr::Ptr,
    sema,
    util::{UnwrapDebug, unreachable_debug},
};
use inkwell::context::Context;
use std::{
    assert_matches::debug_assert_matches,
    convert::Infallible,
    mem::ManuallyDrop,
    ops::FromResidual,
    path::Path,
    time::{Duration, Instant},
};

impl<'ctx> llvm::Codegen<'ctx> {
    pub fn compile_all(&mut self, stmts: &[Ptr<Ast>]) -> CodegenResult<()> {
        self.precompile_decls(&stmts)?;

        for s in stmts.iter().copied() {
            self.compile_top_level(s)?;
            tmp_alloc().reset_scratch(s);
        }

        CodegenResult::Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CompileMode {
    Check,
    Build,
    Run,

    // for tests:
    TestRun,

    // for benchmarks:
    Parse,
    Codegen,
}

pub fn compile(mode: CompileMode, args: BuildArgs) -> CompileResult {
    let ctx = CompilationContext::new(args);
    let args = &ctx.args;
    ctx.0.set_source_root(args.path.clone())?;

    if !args.quiet {
        let proj_path = args.path.parent().unwrap();
        eprint!("Compiling file {:?}", args.path.file_name().unwrap());
        if proj_path != Path::new("") {
            eprint!(" in project {:?}", proj_path);
            std::env::set_current_dir(proj_path).unwrap();
        }
        eprintln!("");
    }

    if !args.is_lib {
        let _: usize = ctx.0.add_import("runtime", None, Span::ZERO)?;
    }

    compile_ctx(ctx.0, mode)
}

pub fn compile_ctx(mut ctx: Ptr<CompilationContextInner>, mode: CompileMode) -> CompileResult {
    let args = &ctx.as_ref().args;

    // ##### Parsing #####

    let parse_start = Instant::now();
    let stmts = parser::parse_files_in_ctx(ctx);
    ctx.compile_time.parser = parse_start.elapsed();

    if ctx.do_abort_compilation() {
        if !args.quiet {
            eprintln!("Parser Error(s) in {:?}", ctx.compile_time.parser);
        }
        return CompileResult::Err;
    }

    debug_assert!(ctx.files.iter().all(|f| f.has_been_parsed()));

    macro_rules! debug_ast {
        () => {
            for file in ctx.files.iter() {
                eprintln!("# File {:?}", file.path);
                for s in stmts[file.stmt_range.u()].iter() {
                    eprintln!("stmt @ {:x?}", s);
                    s.print_tree();
                }
                eprintln!();
            }
        };
    }

    if args.debug_ast {
        eprintln!("### AST Nodes:");
        debug_ast!();
    }

    if mode == CompileMode::Parse {
        if !args.quiet {
            ctx.compile_time.print();
        }
        return CompileResult::Ok;
    }

    // ##### Sema #####

    let sema_start = Instant::now();
    sema::analyze(ctx, stmts);
    ctx.compile_time.sema = sema_start.elapsed();

    if ctx.do_abort_compilation() {
        eprintln!("Sema Error(s) in {:?}", ctx.compile_time.sema);
        return CompileResult::Err;
    }

    if args.debug_typed_ast {
        eprintln!("\n### Typed AST Nodes:");
        debug_ast!();
    }

    if mode == CompileMode::Check {
        if !args.quiet {
            ctx.compile_time.print();
        }
        // println!("{} KiB", alloc.0.allocated_bytes() as f64 / 1024.0);
        return CompileResult::Ok;
    }

    // ##### Codegen #####

    let codegen_start = Instant::now();
    let context = Context::create();
    let mut codegen = llvm::Codegen::new(&context, "dev");
    if let CodegenResult::Err(e) = codegen.compile_all(&stmts) {
        cerror!(Span::ZERO, "Codegen failed: {e:?}");
        return CompileResult::Err;
    };
    let module = codegen.module.take().u();
    drop(codegen);
    ctx.compile_time.codegen = codegen_start.elapsed();

    if args.debug_functions {
        eprint!("functions:");
        for a in module.get_functions() {
            eprint!("{:?},", a.get_name());
        }
        eprintln!("\n");
    }

    if mode == CompileMode::Codegen {
        return CompileResult::Ok;
    }

    // ##### Backend #####

    debug_assert_matches!(mode, CompileMode::Build | CompileMode::Run | CompileMode::TestRun);

    let backend_setup_start = Instant::now();
    let target_machine = llvm::Codegen::init_target_machine(args.target_triple.as_deref());
    ctx.compile_time.backend_setup = backend_setup_start.elapsed();

    if args.debug_llvm_ir_unoptimized {
        eprintln!("### Unoptimized LLVM IR:");
        eprintln!("{}\n", module.print_to_string().to_string());
    }

    let backend_start = Instant::now();
    module.optimize(&target_machine, args.optimization_level).unwrap();
    ctx.compile_time.optimization = backend_start.elapsed();

    if args.debug_llvm_ir_optimized {
        eprintln!(";### Optimized LLVM IR:");
        eprintln!("{}\n", module.print_to_string().to_string());
    }

    let exe_file_path = Path::new("out").join(args.path.file_stem().unwrap());
    let obj_file_path = exe_file_path.with_added_extension("o");

    if args.emit_llvm_ir {
        std::fs::write(
            exe_file_path.with_added_extension("ll"),
            module.print_to_string().to_string(),
        )
        .unwrap();
    }

    // ##### Linking #####

    if mode == CompileMode::TestRun {
        let module = ManuallyDrop::new(module);
        let module = module.as_mut_ptr();
        return CompileResult::ModuleForTesting(BackendModule { context, module });
    }

    if args.out != OutKind::None {
        let write_obj_file_start = Instant::now();
        module.compile_to_obj_file(&target_machine, &obj_file_path).unwrap();
        ctx.compile_time.writing_obj = write_obj_file_start.elapsed();
    }

    if args.is_lib {
        cwarn!(
            Span::ZERO,
            "finished building object file. Building a static or dynamic library is currently not \
             implemented"
        );
        return CompileResult::Ok;
    }

    if args.out == OutKind::Executable {
        let linking_start = std::time::Instant::now();
        let mut cmd = std::process::Command::new("ld.lld");
        cmd.arg("-o")
            .arg(exe_file_path.as_os_str())
            .arg("-pie") // "position independent executable"; Note: pointers also look different: 0xceea530 -> 0x56213d9cd530
            .arg("-dynamic-linker=/lib64/ld-linux-x86-64.so.2");
        for lib in ctx.library_search_paths.iter() {
            cmd.arg("-rpath").arg(lib.0.as_ref());
        }
        cmd.arg("/usr/lib/x86_64-linux-gnu/Scrt1.o")
            .arg(obj_file_path.as_os_str())
            .args(ctx.library_search_paths.iter().map(|s| format!("-L{}", s.0.as_ref())))
            .args(ctx.libraries.iter().map(|s| format!("-l{}", s.0.as_ref())))
            .arg("-L/lib/x86_64-linux-gnu")
            .arg("-lc")
            .arg("-lm");
        if args.debug_linker_args {
            eprintln!("### Linker Cmd");
            eprint!("{}", cmd.get_program().display());
            for args in cmd.get_args() {
                eprint!(" {}", args.display());
            }
            eprint!("\n\n");
        }
        let err = cmd.status().unwrap();
        if !err.success() {
            eprintln!("linking failed: {:?}", err);
            return CompileResult::Err;
        }

        ctx.compile_time.linking = linking_start.elapsed();
    }

    if !args.quiet {
        ctx.compile_time.print();
    }

    // ##### Executing #####

    if mode != CompileMode::Run {
        return CompileResult::Ok;
    }

    if args.out != OutKind::Executable {
        ctx.error_without_code("Cannot run program if `--out` is not `exe`");
        return CompileResult::Err;
    }

    eprintln!("\nRunning `{}`", exe_file_path.display());
    if let Err(e) = std::process::Command::new(exe_file_path).status().unwrap().exit_ok() {
        eprintln!("{e}");
        return CompileResult::RunErr { err_code: e.code().unwrap_or(1) };
    }

    CompileResult::Ok
}

#[derive(Default)]
pub struct CompileDurations {
    // frontend:
    parser: Duration,
    sema: Duration,
    codegen: Duration,

    // backend:
    backend_setup: Duration,
    optimization: Duration,
    writing_obj: Duration,

    linking: Duration,
}

impl CompileDurations {
    pub fn print(&self) {
        let frontend_total = self.parser + self.sema + self.codegen;
        eprintln!("  Frontend:              {:?}", frontend_total);
        eprintln!("    Lexer, Parser:         {:?}", self.parser);
        eprintln!("    Semantic Analysis:     {:?}", self.sema);
        if self.codegen.is_zero() {
            return;
        }
        eprintln!("    LLVM IR Codegen:       {:?}", self.codegen);

        let backend_total = self.backend_setup + self.optimization + self.writing_obj;
        eprintln!("  Backend:               {:?}", backend_total);
        eprintln!("    LLVM Setup:            {:?}", self.backend_setup);
        eprintln!("    LLVM pass pipeline:    {:?}", self.optimization);
        if !self.writing_obj.is_zero() {
            eprintln!("    writing obj file:      {:?}", self.writing_obj);
        }

        if !self.linking.is_zero() {
            eprintln!("  Linking:               {:?}", self.linking);
        }

        let total = frontend_total + backend_total + self.linking;
        eprintln!("  Total:                 {:?}", total);
    }
}

#[derive(Debug)]
pub enum CompileResult {
    Ok,
    Err,
    RunErr { err_code: i32 },

    ModuleForTesting(BackendModule),
}

impl CompileResult {
    pub fn exit_code(self, on_err: i32) -> i32 {
        match self {
            CompileResult::Ok => 0,
            CompileResult::Err => on_err,
            CompileResult::RunErr { err_code } => err_code,
            CompileResult::ModuleForTesting(_) => unreachable_debug(),
        }
    }

    pub fn ok(self) -> bool {
        match self {
            CompileResult::Ok => true,
            CompileResult::Err | CompileResult::RunErr { .. } => false,
            CompileResult::ModuleForTesting(_) => unreachable_debug(),
        }
    }
}

impl FromResidual<Result<Infallible, HandledErr>> for CompileResult {
    fn from_residual(_: Result<Infallible, HandledErr>) -> Self {
        CompileResult::Err
    }
}

#[derive(Debug)]
pub struct BackendModule {
    #[allow(unused)]
    context: inkwell::context::Context,
    module: *const inkwell::llvm_sys::LLVMModule,
}

impl Drop for BackendModule {
    fn drop(&mut self) {
        drop(ManuallyDrop::into_inner(self.codegen_module()));
    }
}

impl BackendModule {
    pub fn codegen_module<'m>(&'m self) -> ManuallyDrop<inkwell::module::Module<'m>> {
        ManuallyDrop::new(unsafe { inkwell::module::Module::new(self.module as *mut _) })
    }
}
