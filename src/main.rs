#![feature(path_add_extension)]

use clap::Parser;
use inkwell::context::Context;
use mylang::{
    ast::debug::DebugAst,
    cli::{BuildArgs, Cli, Command},
    codegen::llvm,
    compiler::Compiler,
    parser::{
        StmtIter,
        lexer::{Code, Lexer},
        parser_helper::ParserInterface,
    },
    sema,
    type_::Type,
    util::display_spanned_error,
};
use std::time::{Duration, Instant};

fn main() {
    let cli = Cli::parse();
    let err = match &cli.command {
        Command::Repl {} => todo!("repl"),
        Command::Clean(_) => todo!("clean"),
        Command::Build(args) => compile2(CompileMode::Build, args),
        Command::Run(args) => compile2(CompileMode::Run, args),
        Command::Check(args) => compile2(CompileMode::Check, args),
        Command::Dev {} => {
            dev();
            std::process::ExitStatus::default()
        },
    };
    std::process::exit(err.code().unwrap())
}

#[derive(Debug, PartialEq)]
enum CompileMode {
    Check,
    Build,
    Run,
}

fn compile2(mode: CompileMode, args: &BuildArgs) -> std::process::ExitStatus {
    if args.path.is_dir() {
        let code = args
            .path
            .read_dir()
            .unwrap()
            .map(|f| f.unwrap().path())
            .filter(|p| p.is_file())
            .filter(|p| p.extension().is_some_and(|ext| ext == "mylang"))
            .map(|f| std::fs::read_to_string(f).unwrap())
            .collect::<String>();
        compile(code.as_ref(), mode, args)
    } else if args.path.is_file() {
        let code = std::fs::read_to_string(&args.path).unwrap();
        compile(code.as_ref(), mode, args)
    } else {
        panic!("{:?} is not a dir nor a file", args.path)
    }
}

fn compile(code: &Code, mode: CompileMode, args: &BuildArgs) -> std::process::ExitStatus {
    let alloc = bumpalo::Bump::new();

    if args.debug_tokens {
        println!("### Tokens:");
        let mut lex = Lexer::new(code);
        while let Some(t) = lex.next() {
            println!("{:?}", t)
        }
        println!();
    }

    if args.debug_ast {
        println!("### AST Nodes:");
        if let Err(()) = StmtIter::parse_and_debug(code) {
            std::process::exit(1)
        }
        println!();
    }

    println!("### Frontend:");
    let frontend_parse_start = Instant::now();
    let stmts = StmtIter::parse_all_or_fail(code, &alloc);
    let frontend_parse_duration = frontend_parse_start.elapsed();

    let sema = sema::Sema::new(code, &alloc, args.debug_types);
    let context = Context::create();
    let codegen = llvm::Codegen::new_module(&context, "dev");
    let mut compiler = Compiler::new(sema, codegen);

    enum FrontendDurations {
        Detailed { sema: Duration, codegen: Duration },
        Combined(Duration),
    }

    let frontend2_duration = if cfg!(debug_assertions) {
        let (sema, codegen) = compiler.compile_stmts_dev(&stmts, code, args.debug_typed_ast);
        FrontendDurations::Detailed { sema, codegen }
    } else {
        let frontend2_start = Instant::now();
        let _ = compiler.compile_stmts(&stmts);

        if !compiler.sema.errors.is_empty() {
            for e in compiler.sema.errors {
                display_spanned_error(&e, code);
            }
            std::process::exit(1);
        }

        let frontend2_duration = frontend2_start.elapsed();

        if args.debug_typed_ast {
            println!("\n### Typed AST Nodes:");
            for s in stmts.iter().copied() {
                println!("stmt @ {:?}", s);
                s.print_tree();
            }
            println!();
        }

        FrontendDurations::Combined(frontend2_duration)
    };
    let total_frontend_duration = frontend_parse_duration
        + match frontend2_duration {
            FrontendDurations::Detailed { sema, codegen } => sema + codegen,
            FrontendDurations::Combined(d) => d,
        };

    print!("functions:");
    for a in compiler.codegen.module.get_functions() {
        print!("{:?},", a.get_name());
    }
    println!("\n");

    let target_machine = llvm::Codegen::init_target_machine(args.target_triple.as_deref());

    if args.debug_llvm_ir_unoptimized {
        println!("### Unoptimized LLVM IR:");
        compiler.codegen.module.print_to_stderr();
        println!();
    }

    let backend_start = Instant::now();
    compiler.optimize(&target_machine, args.optimization_level).unwrap();
    let backend_duration = backend_start.elapsed();
    let total_duration = total_frontend_duration + backend_duration;

    if args.debug_llvm_ir_optimized {
        println!("### Optimized LLVM IR:");
        compiler.codegen.module.print_to_stderr();
        println!();
    }

    println!("### Compilation time:");
    println!("  Frontend:                             {:?}", total_frontend_duration);
    println!("    Lexer, Parser:                      {:?}", frontend_parse_duration);

    match frontend2_duration {
        FrontendDurations::Detailed { sema, codegen } => {
            println!("    Semantic Analysis:                  {:?}", sema);
            println!("    LLVM IR Codegen:                    {:?}", codegen);
        },
        FrontendDurations::Combined(d) => {
            println!("    Semantic Analysis, LLVM IR Codegen: {:?}", d);
        },
    }

    println!("  LLVM Backend (LLVM pass pipeline):    {:?}", backend_duration);
    println!("  Total:                                {:?}", total_duration);
    println!();

    if args.path.is_dir() {
        todo!()
    }

    let mut exe_file_path = args.path.with_file_name("out");
    exe_file_path.push(args.path.file_stem().unwrap());
    let obj_file_path = exe_file_path.with_added_extension("o");
    compiler.codegen.compile_to_obj_file(&target_machine, &obj_file_path).unwrap();

    std::process::Command::new("gcc")
        .arg(obj_file_path.as_os_str())
        .arg("-o")
        .arg(exe_file_path.as_os_str())
        .status()
        .unwrap();

    if mode == CompileMode::Run {
        std::process::Command::new(exe_file_path).status().unwrap()
    } else {
        std::process::ExitStatus::default()
    }
}

fn dev() {
    const DEBUG_TOKENS: bool = false;
    const DEBUG_AST: bool = true;
    const DEBUG_TYPES: bool = false;
    const DEBUG_TYPED_AST: bool = false;
    const DEBUG_LLVM_IR_UNOPTIMIZED: bool = false;
    const DEBUG_LLVM_IR_OPTIMIZED: bool = true;
    //const LLVM_TARGET_TRIPLE: Option<&str> = Some("arm-linux-gnueabihf");
    const LLVM_TARGET_TRIPLE: Option<&str> = None;
    const LLVM_OPTIMIZATION_LEVEL: u8 = 0;
    type MyMainRetTy = i64;

    let alloc = bumpalo::Bump::new();

    let code = "
A :: struct {
    a: i64,
    b := 2,
}

add :: (a: i64, b: i64) -> a + b;

main :: -> {
    three := 3;
    myarr: [5]f32 = [2.0; 5];
    myarr := [3, 30, three, 5, 5];

    b := true || false;

    a := A.{ a = 1 };

    mut sum := a.a;
    myarr | for x {
        sum += add(x, 1);
    };
    sum
};

// rec factorial :: (x: f64) -> x == 0 | if 1 else x * factorial(x-1);
// mymain :: -> factorial(10) == 3628800;

// TODO: This causes too many loops
// a :: -> b();
// b :: -> c();
// c :: -> d();
// d :: -> 10;
";
    let code = code.as_ref();

    if DEBUG_TOKENS {
        println!("### Tokens:");
        let mut lex = Lexer::new(code);
        while let Some(t) = lex.next() {
            println!("{:?}", t)
        }
        println!();
    }

    if DEBUG_AST {
        println!("### AST Nodes:");
        if let Err(()) = StmtIter::parse_and_debug(code) {
            std::process::exit(1)
        }
        println!();
    }

    println!("### Frontend:");
    let frontend_parse_start = Instant::now();
    let stmts = StmtIter::parse_all_or_fail(code, &alloc);
    let frontend_parse_duration = frontend_parse_start.elapsed();

    let sema = sema::Sema::new(code, &alloc, DEBUG_TYPES);
    let context = Context::create();
    let codegen = llvm::Codegen::new_module(&context, "dev");
    let mut compiler = Compiler::new(sema, codegen);

    enum FrontendDurations {
        Detailed { sema: Duration, codegen: Duration },
        Combined(Duration),
    }

    let frontend2_duration = if cfg!(debug_assertions) {
        let (sema, codegen) = compiler.compile_stmts_dev(&stmts, code, DEBUG_TYPED_AST);
        FrontendDurations::Detailed { sema, codegen }
    } else {
        let frontend2_start = Instant::now();
        let _ = compiler.compile_stmts(&stmts);

        if !compiler.sema.errors.is_empty() {
            for e in compiler.sema.errors {
                display_spanned_error(&e, code);
            }
            std::process::exit(1);
        }

        let frontend2_duration = frontend2_start.elapsed();

        if DEBUG_TYPED_AST {
            println!("\n### Typed AST Nodes:");
            for s in stmts.iter().copied() {
                println!("stmt @ {:?}", s);
                s.print_tree();
            }
            println!();
        }

        FrontendDurations::Combined(frontend2_duration)
    };
    let total_frontend_duration = frontend_parse_duration
        + match frontend2_duration {
            FrontendDurations::Detailed { sema, codegen } => sema + codegen,
            FrontendDurations::Combined(d) => d,
        };

    print!("functions:");
    for a in compiler.codegen.module.get_functions() {
        print!("{:?},", a.get_name());
    }
    println!("\n");

    let target_machine = llvm::Codegen::init_target_machine(LLVM_TARGET_TRIPLE);

    if DEBUG_LLVM_IR_UNOPTIMIZED {
        println!("### Unoptimized LLVM IR:");
        compiler.codegen.module.print_to_stderr();
        println!();
    }

    let backend_start = Instant::now();
    compiler.optimize(&target_machine, LLVM_OPTIMIZATION_LEVEL).unwrap();
    let backend_duration = backend_start.elapsed();
    let total_duration = total_frontend_duration + backend_duration;

    if DEBUG_LLVM_IR_OPTIMIZED {
        println!("### Optimized LLVM IR:");
        compiler.codegen.module.print_to_stderr();
        println!();
    }

    #[allow(unused)]
    enum ExecutionVariant {
        ObjectCode,
        Jit,
    }

    const EXE_VAR: ExecutionVariant = ExecutionVariant::ObjectCode;
    //const EXE_VAR: ExecutionVariant = ExecutionVariant::Jit;
    match EXE_VAR {
        ExecutionVariant::ObjectCode => {
            let Type::Function(f) =
                compiler.sema.symbols.get("mymain").unwrap().get_type().ok().unwrap()
            else {
                panic!()
            };
            let c_file = match f.ret_type {
                Type::Int { .. } => "../../test-int.c",
                Type::Float { .. } => "../../test-double.c",
                _ => "../../test-other.c",
            };
            println!("{:?}", c_file);
            let _ = std::fs::create_dir("target/build_dev");
            let _ = std::fs::remove_file("target/build_dev/test.c");
            std::os::unix::fs::symlink(c_file, "target/build_dev/test.c").unwrap();
            let filename = "target/build_dev/output.o";
            compiler
                .codegen
                .compile_to_obj_file(&target_machine, filename.as_ref())
                .unwrap();
        },
        ExecutionVariant::Jit => {
            let fn_name = "mymain";
            let out = compiler
                .codegen
                .jit_run_fn::<MyMainRetTy>(fn_name, inkwell::OptimizationLevel::None)
                .unwrap();
            println!("{fn_name} returned {}", out);
        },
    }

    println!("### Compilation time:");
    println!("  Frontend:                             {:?}", total_frontend_duration);
    println!("    Lexer, Parser:                      {:?}", frontend_parse_duration);

    match frontend2_duration {
        FrontendDurations::Detailed { sema, codegen } => {
            println!("    Semantic Analysis:                  {:?}", sema);
            println!("    LLVM IR Codegen:                    {:?}", codegen);
        },
        FrontendDurations::Combined(d) => {
            println!("    Semantic Analysis, LLVM IR Codegen: {:?}", d);
        },
    }

    println!("  LLVM Backend (LLVM pass pipeline):    {:?}", backend_duration);
    println!("  Total:                                {:?}", total_duration);
}
