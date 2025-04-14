use crate::{
    ast::{Ast, debug::DebugAst},
    cli::{BuildArgs, OutKind},
    codegen::llvm::{self, CodegenModuleExt},
    context::CompilationContextInner,
    diagnostics::DiagnosticReporter,
    parser::{self},
    ptr::Ptr,
    sema::{self},
    source_file::SourceFile,
    util::{UnwrapDebug, unreachable_debug},
};
use inkwell::context::Context;
use std::{
    assert_matches::debug_assert_matches,
    mem::ManuallyDrop,
    path::Path,
    time::{Duration, Instant},
};

impl<'ctx> llvm::Codegen<'ctx> {
    pub fn compile_all(&mut self, stmts: &[Ptr<Ast>], order: &[usize]) {
        debug_assert_eq!(stmts.len(), order.len());
        for idx in order {
            let s = stmts[*idx];
            self.compile_top_level(s);
        }
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

pub fn compile(
    ctx: Ptr<CompilationContextInner>,
    mode: CompileMode,
    args: &mut BuildArgs,
) -> CompileResult {
    if args.path.is_dir() {
        println!("Compiling project at {:?}", args.path);
        todo!();
        /*
        for file in args
            .path
            .read_dir()
            .unwrap()
            .map(|f| f.unwrap().path())
            .filter(|p| p.is_file())
            .filter(|p| p.extension().is_some_and(|ext| ext == "mylang"))
        {
            util::write_file_to_string(&file, &mut code).unwrap();
        }
        */
    } else if !args.path.is_file() {
        panic!("{:?} is not a dir nor a file", args.path)
    }
    let proj_path = args.path.parent().unwrap();
    let entry_file = args.path.file_name().unwrap();
    print!("Compiling file {:?}", entry_file);
    if proj_path != Path::new("") {
        print!(" in project {:?}", proj_path);
        std::env::set_current_dir(proj_path).unwrap();
    }
    println!("");

    let entry_file = Path::new(entry_file);
    let root_file = SourceFile::read(Ptr::from_ref(entry_file), &ctx.alloc).unwrap();
    compile_file(ctx, root_file, mode, args)
}

pub fn compile_file(
    mut ctx: Ptr<CompilationContextInner>,
    file: SourceFile,
    mode: CompileMode,
    args: &BuildArgs,
) -> CompileResult {
    #[cfg(debug_assertions)]
    {
        ctx.debug_types = args.debug_types;
    }

    // ##### Parsing #####

    let parse_start = Instant::now();
    let stmts = parser::parse(ctx, file);
    ctx.compile_time.parser = parse_start.elapsed();

    if ctx.do_abort_compilation() {
        eprintln!("Parser Error(s) in {:?}", ctx.compile_time.parser);
        return CompileResult::Err;
    }

    debug_assert!(ctx.files.iter().all(|f| f.has_been_parsed()));

    macro_rules! debug_ast {
        () => {
            for file in ctx.files.iter() {
                println!("# File {:?}", file.path);
                for s in stmts[file.stmt_range.u()].iter() {
                    println!("stmt @ {:x?}", s);
                    s.print_tree();
                }
                println!();
            }
        };
    }

    if args.debug_ast {
        println!("### AST Nodes:");
        debug_ast!();
    }

    if mode == CompileMode::Parse {
        if args.print_compile_time {
            ctx.compile_time.print();
        }
        return CompileResult::Ok;
    }

    // ##### Sema #####

    let sema_start = Instant::now();
    let order = sema::analyze(ctx, &stmts);
    sema::validate_main(ctx, &stmts, args);
    ctx.compile_time.sema = sema_start.elapsed();

    if ctx.do_abort_compilation() {
        eprintln!("Sema Error(s) in {:?}", ctx.compile_time.sema);
        return CompileResult::Err;
    }

    if args.debug_typed_ast {
        println!("\n### Typed AST Nodes:");
        debug_ast!();
    }

    if mode == CompileMode::Check {
        if args.print_compile_time {
            ctx.compile_time.print();
        }
        // println!("{} KiB", alloc.0.allocated_bytes() as f64 / 1024.0);
        return CompileResult::Ok;
    }

    // ##### Codegen #####

    let codegen_start = Instant::now();
    let context = Context::create();
    let mut codegen = llvm::Codegen::new(&context, "dev");
    codegen.compile_all(&stmts, &order);
    let module = codegen.module.take().u();
    drop(codegen);
    ctx.compile_time.codegen = codegen_start.elapsed();

    if args.debug_functions {
        print!("functions:");
        for a in module.get_functions() {
            print!("{:?},", a.get_name());
        }
        println!("\n");
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
        println!("### Unoptimized LLVM IR:");
        println!("{}\n", module.print_to_string().to_string());
    }

    let backend_start = Instant::now();
    module.optimize(&target_machine, args.optimization_level).unwrap();
    ctx.compile_time.optimization = backend_start.elapsed();

    if args.debug_llvm_ir_optimized {
        println!("### Optimized LLVM IR:");
        println!("{}\n", module.print_to_string().to_string());
    }

    // ##### Linking #####

    if mode == CompileMode::TestRun {
        let module = ManuallyDrop::new(module);
        let module = module.as_mut_ptr();
        return CompileResult::ModuleForTesting(BackendModule { context, module });
    }

    if mode == CompileMode::Run && args.out != OutKind::Executable {
        ctx.error_without_code("Cannot run program if `--out` is not `exe`");
        return CompileResult::Err;
    }

    if args.out == OutKind::None {
        if args.print_compile_time {
            ctx.compile_time.print();
        }
        return CompileResult::Ok;
    }

    let exe_file_path = Path::new("out").join(args.path.file_stem().unwrap());
    let obj_file_path = exe_file_path.with_added_extension("o");
    let write_obj_file_start = Instant::now();
    module.compile_to_obj_file(&target_machine, &obj_file_path).unwrap();
    ctx.compile_time.writing_obj = write_obj_file_start.elapsed();

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
            println!("### Linker Cmd");
            print!("{}", cmd.get_program().display());
            for args in cmd.get_args() {
                print!(" {}", args.display());
            }
            print!("\n\n");
        }
        let err = cmd.status().unwrap();
        if !err.success() {
            println!("linking failed: {:?}", err);
            return CompileResult::Err;
        }

        ctx.compile_time.linking = linking_start.elapsed();
    }

    if args.print_compile_time {
        ctx.compile_time.print();
    }

    // ##### Executing #####

    if args.out == OutKind::Executable && mode == CompileMode::Run {
        println!("\nRunning `{}`", exe_file_path.display());
        if let Err(e) = std::process::Command::new(exe_file_path).status().unwrap().exit_ok() {
            println!("{e}");
            return CompileResult::RunErr { err_code: e.code().unwrap_or(1) };
        }
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
        println!("  Frontend:              {:?}", frontend_total);
        println!("    Lexer, Parser:         {:?}", self.parser);
        println!("    Semantic Analysis:     {:?}", self.sema);
        if self.codegen.is_zero() {
            return;
        }
        println!("    LLVM IR Codegen:       {:?}", self.codegen);

        let backend_total = self.backend_setup + self.optimization + self.writing_obj;
        println!("  Backend:               {:?}", backend_total);
        println!("    LLVM Setup:            {:?}", self.backend_setup);
        println!("    LLVM pass pipeline:    {:?}", self.optimization);
        if self.writing_obj.is_zero() {
            return;
        }
        println!("    writing obj file:      {:?}", self.writing_obj);

        if self.linking.is_zero() {
            return;
        }
        println!("  Linking:               {:?}", self.linking);

        let total = frontend_total + backend_total + self.linking;
        println!("  Total:                 {:?}", total);
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
    pub fn exit_code(self) -> i32 {
        match self {
            CompileResult::Ok => 0,
            CompileResult::Err => 1,
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
