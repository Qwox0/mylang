use inkwell::context::Context;
use mylang::{
    ast::debug::DebugAst,
    cli::Cli,
    codegen::llvm,
    compiler::Compiler,
    parser::{StmtIter, lexer::Lexer, parser_helper::ParserInterface},
    sema,
    type_::Type,
    util::display_spanned_error,
};
use std::{
    io::Read,
    path::Path,
    time::{Duration, Instant},
};

fn main() {
    dev();
    /*
    let cli = Cli::parse();

    println!("{:#?}", cli);

    match cli.command {
        Command::RunScript(RunScriptArgs { script }) => {
            let code = read_code_file(&script);
            let code = code.as_ref();
            let stmts = StmtIter::parse(code);

            if cli.debug == Some(mylang::cli::DebugOptions::Ast) {
                for s in stmts {
                    match s {
                        ResultWithFatal::Ok(s) => s.print_tree(),
                        ResultWithFatal::Err(e) | ResultWithFatal::Fatal(e) => {
                            eprintln!("ERROR: {:?}", e)
                        },
                    }
                }
                return;
            }

            let context = Context::create();
            let mut compiler = codegen::llvm::Compiler::new_module(
                &context,
                script.file_stem().expect("has filename").to_str().expect("is valid utf-8"),
            );

            for pres in stmts {
                let stmt = pres.unwrap_or_else(|e| panic!("ERROR: {:?}", e));
                /*
                    match stmt.kind {
                        StmtKind::Decl { markers, ident, kind } => {
                            let (ty, Some(value)) = kind.into_ty_val() else {
                                eprintln!("top-level item needs initialization");
                                break
                            };
                            compiler
                                .add_item(Item { markers, ident, ty, value, code })
                                .unwrap_or_else(|e| panic!("ERROR: {:?}", e))
                        },
                        StmtKind::Semicolon(expr) | // => todo!(),
                        StmtKind::Expr(expr) => {
                            panic!("top-level expressions are not allowed")

                            /*
                            match compiler.jit_run_expr(&expr, code, cli.debug) {
                                Ok(out) => println!("=> {}", out),
                                Err(err) => {
                                    eprintln!("ERROR: {:?}", err);
                                    break;
                                },
                            }
                            */
                        },
                    }
                */
            }

            compiler.run_passes_debug(cli.debug);

            //let out = compiler.jit_run_fn("main").expect("has main
            // function"); println!("main returned {}", out);
        },
        Command::Build { build_script } => todo!("`build` command"),
        Command::Compile { file } => todo!("`compile` command"),
        Command::Repl {} => {
            //todo!("`repl` command");
            let context = Context::create();
            let mut compiler = codegen::llvm::Compiler::new_module(&context, "repl");
            let mut jit = Jit::default();
            loop {
                print!("\n> ");
                std::io::stdout().flush().expect("Could flush stdout");
                let mut line = String::new();
                std::io::stdin().read_line(&mut line).expect("Could read line from stdin");

                // leaking the String is fine because it is needed for the entire compilation
                // process.
                let code = Code::new(line.leak());

                for pres in StmtIter::parse(code) {
                    let stmt = pres.unwrap_or_else(|e| panic!("ERROR: {:?}", e));
                    /*
                        match stmt.kind {
                            StmtKind::Decl { markers, ident, kind } => {
                                //compiler.compile_var_decl(ident, kind.clone(), code).unwrap();
                                let (ty, Some(value)) = kind.into_ty_val() else {
                                    eprintln!("top-level item needs initialization");
                                    break
                                };
                                compiler
                                    .add_item(Item { markers, ident, ty, value, code })
                                    .unwrap_or_else(|e| eprintln!("ERROR: {:?}", e));
                                compiler.move_module_to(&mut jit);
                            },
                            StmtKind::Semicolon(expr) | // => todo!(),
                            StmtKind::Expr(expr) => {
                                match compiler.compile_repl_expr(&expr, code, cli.debug) {
                                    Ok(()) => (),
                                    Err(err) => {eprintln!("ERROR: {:?}", err); continue;},
                                }
                                //compiler.run_passes_debug(cli.debug);
                                let module = compiler.take_module();
                                match jit.run_repl_expr(module) {
                                    Ok(out) => println!("=> {}", out),
                                    Err(err) => {
                                        eprintln!("ERROR: {:?}", err);
                                        break;
                                    },
                                }
                                /*
                                let out = compiler.compile_repl_expr(&expr).unwrap();
                                out.print_to_stderr();
                                unsafe { out.delete() };
                                */
                            },
                        }
                    */
                }
            }
        },
        Command::Check {} => todo!("`check` command"),
        Command::Clean {} => todo!("`clean` command"),

        Command::Dev {} => dev(),
    }
    */
}

#[allow(unused)]
fn read_code_file(path: &Path) -> String {
    fn inner(path: &Path) -> Result<String, std::io::Error> {
        let mut buf = String::new();
        std::fs::File::open(path)?.read_to_string(&mut buf)?;
        Ok(buf)
    }
    inner(path).unwrap_or_else(|e| {
        Cli::print_help().unwrap();
        eprintln!("\nERROR: {}", e);
        std::process::exit(1);
    })
}

fn dev() {
    const DEBUG_TOKENS: bool = false;
    const DEBUG_AST: bool = true;
    const DEBUG_TYPES: bool = true;
    const DEBUG_TYPED_AST: bool = false;
    const DEBUG_LLVM_IR_UNOPTIMIZED: bool = false;
    const DEBUG_LLVM_IR_OPTIMIZED: bool = false;
    const LLVM_OPTIMIZATION_LEVEL: u8 = 1;
    type MyMainRetTy = i64;

    let alloc = bumpalo::Bump::new();

    let code = "
A :: struct {
    a: i64,
    b := 2,
}

add :: (a: i64, b: i64) -> a + b;

mymain :: -> {
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

    let sema = sema::Sema::<DEBUG_TYPES>::new(code, &alloc);
    let context = Context::create();
    let codegen = llvm::Codegen::new_module(&context, "dev", &alloc);
    let mut compiler = Compiler::new(sema, codegen);

    enum FrontendDurations {
        Detailed { sema: Duration, codegen: Duration },
        Combined(Duration),
    }

    let frontend2_duration = if cfg!(debug_assertions) {
        let (sema, codegen) = compiler.compile_stmts_dev::<DEBUG_TYPED_AST>(&stmts, code);
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

    let target_machine = llvm::Codegen::init_target_machine();

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
            let _ = std::fs::create_dir("target/build_dev");
            let _ = std::fs::remove_file("target/build_dev/test.c");
            std::os::unix::fs::symlink(c_file, "target/build_dev/test.c").unwrap();
            let filename = "target/build_dev/output.o";
            compiler.codegen.compile_to_obj_file(&target_machine, filename).unwrap();
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
