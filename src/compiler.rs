use crate::{
    ast::{Ast, debug::DebugAst},
    cli::{BuildArgs, OutKind},
    codegen::llvm::{self, CodegenModuleExt},
    context::CompilationContext,
    diagnostic_reporter::DiagnosticReporter,
    parser::{
        Parser,
        lexer::{Code, Lexer},
        parser_helper::ParserInterface,
    },
    ptr::Ptr,
    sema::Sema,
    util::{self, UnwrapDebug},
};
use inkwell::{context::Context, targets::TargetMachine};
use std::{
    assert_matches::debug_assert_matches,
    mem::ManuallyDrop,
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

    // for benchmarks:
    Parse,
    Codegen,
}

pub fn compile2(mode: CompileMode, args: &BuildArgs) -> i32 {
    let mut code = String::with_capacity(4096);
    if !args.no_prelude {
        let prelude_path = concat!(std::env!("HOME"), "/src/mylang/lib/prelude.mylang");
        util::write_file_to_string(prelude_path, &mut code).unwrap();
    }

    if args.path.is_dir() {
        println!("Compiling project at {:?}", args.path);
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
    } else if args.path.is_file() {
        println!("Compiling file at {:?}", args.path);
        util::write_file_to_string(&args.path, &mut code).unwrap();
    } else {
        panic!("{:?} is not a dir nor a file", args.path)
    }
    compile(code.as_ref(), mode, args)
}

pub fn compile(code: &Code, mode: CompileMode, args: &BuildArgs) -> i32 {
    let mut ctx = CompilationContext::new(Ptr::from_ref(code));
    let res = compile_ctx(&mut ctx, mode, args);
    if !res.ok {
        return 1;
    }
    let Some(out) = res.backend_out else {
        return 0;
    };

    debug_assert_matches!(mode, CompileMode::Build | CompileMode::Run);

    if args.path.is_dir() {
        todo!()
    }

    if args.out == OutKind::None {
        #[cfg(not(test))]
        ctx.compile_time.print();
        return 0;
    }

    let mut exe_file_path = args.path.with_file_name("out");
    exe_file_path.push(args.path.file_stem().unwrap());
    let obj_file_path = exe_file_path.with_added_extension("o");
    let write_obj_file_start = Instant::now();
    out.codegen_module()
        .compile_to_obj_file(&out.target_machine, &obj_file_path)
        .unwrap();
    ctx.compile_time.writing_obj = write_obj_file_start.elapsed();

    if args.out == OutKind::Executable {
        let linking_start = std::time::Instant::now();
        let err = std::process::Command::new("gcc")
            .arg(obj_file_path.as_os_str())
            .arg("-o")
            .arg(exe_file_path.as_os_str())
            .arg("-lm")
            .status()
            .unwrap();
        if !err.success() {
            println!("linking with gcc failed: {:?}", err);
        }

        ctx.compile_time.linking = linking_start.elapsed();
    }

    ctx.compile_time.print();

    if args.out == OutKind::Executable && mode == CompileMode::Run {
        println!("\nRunning `{}`", exe_file_path.display());
        if let Err(e) = std::process::Command::new(exe_file_path).status().unwrap().exit_ok() {
            println!("{e}");
            return e.code().unwrap_or(1);
        }
    }
    0
}

pub fn compile_ctx(
    ctx: &mut CompilationContext,
    mode: CompileMode,
    args: &BuildArgs,
) -> CompileResult {
    let code = ctx.code.as_ref();

    if args.debug_tokens {
        println!("### Tokens:");
        let mut lex = Lexer::new(code);
        while let Some(t) = lex.next() {
            println!("{:?}", t)
        }
        println!();
    }

    let frontend_parse_start = Instant::now();
    let parse_res = Parser::parse(code, &ctx.alloc);
    ctx.compile_time.parser = frontend_parse_start.elapsed();

    if !parse_res.errors.is_empty() {
        eprintln!("Parse Error in {:?}", ctx.compile_time.parser);
        for e in &parse_res.errors {
            ctx.error(e);
        }
        return CompileResult::ERR;
    }

    let top_level_scope = parse_res.top_level_scope.u();

    if args.debug_ast {
        println!("### AST Nodes:");
        for s in top_level_scope.stmts.iter() {
            println!("stmt @ {:x?}", s);
            s.print_tree();
        }
        println!();
    }

    if mode == CompileMode::Parse {
        return CompileResult { ok: true, backend_out: None };
    }

    // ##### Sema #####

    let sema_start = Instant::now();
    let (sema, order) = Sema::analyze2(top_level_scope, args.debug_types);
    ctx.compile_time.sema = sema_start.elapsed();

    if !sema.errors.is_empty() {
        eprintln!("Sema Error in {:?}", ctx.compile_time.sema);
        for e in &sema.errors {
            ctx.error(e);
        }
        return CompileResult::ERR;
    }

    if args.debug_typed_ast {
        println!("\n### Typed AST Nodes:");
        for s in top_level_scope.stmts.iter() {
            println!("stmt @ {:x?}", s);
            s.print_tree();
        }
        println!();
    }

    if mode == CompileMode::Check {
        #[cfg(not(test))]
        ctx.compile_time.print();
        // println!("{} KiB", alloc.0.allocated_bytes() as f64 / 1024.0);
        return CompileResult { ok: true, backend_out: None };
    }

    // ##### Codegen #####

    let codegen_start = Instant::now();
    let context = Context::create();
    let mut codegen = llvm::Codegen::new(&context, "dev");
    codegen.compile_all(&top_level_scope.stmts, &order);
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
        return CompileResult { ok: true, backend_out: None };
    }

    // ##### Backend #####

    debug_assert_matches!(mode, CompileMode::Build | CompileMode::Run);

    let backend_setup_start = Instant::now();
    let target_machine = llvm::Codegen::init_target_machine(args.target_triple.as_deref());
    ctx.compile_time.backend_setup = backend_setup_start.elapsed();

    if args.debug_llvm_ir_unoptimized {
        println!("### Unoptimized LLVM IR:");
        println!("{}\n", module.print_to_string());
    }

    let backend_start = Instant::now();
    module.optimize(&target_machine, args.optimization_level).unwrap();
    ctx.compile_time.optimization = backend_start.elapsed();

    if args.debug_llvm_ir_optimized {
        println!("### Optimized LLVM IR:");
        println!("{}\n", module.print_to_string());
    }

    let module = ManuallyDrop::new(module);
    let module = module.as_mut_ptr();
    CompileResult { ok: true, backend_out: Some(BackendOut { context, target_machine, module }) }
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
        println!("  Linking with gcc:      {:?}", self.linking);

        let total = frontend_total + backend_total + self.linking;
        println!("  Total:                 {:?}", total);
    }
}

pub struct BackendOut {
    #[allow(unused)]
    context: inkwell::context::Context,
    target_machine: TargetMachine,
    module: *const inkwell::llvm_sys::LLVMModule,
}

impl Drop for BackendOut {
    fn drop(&mut self) {
        drop(ManuallyDrop::into_inner(self.codegen_module()));
    }
}

impl BackendOut {
    pub fn codegen_module<'m>(&'m self) -> ManuallyDrop<inkwell::module::Module<'m>> {
        ManuallyDrop::new(unsafe { inkwell::module::Module::new(self.module as *mut _) })
    }
}

pub struct CompileResult {
    pub ok: bool,
    pub backend_out: Option<BackendOut>,
}

impl CompileResult {
    pub const ERR: Self = Self { ok: false, backend_out: None };
}
