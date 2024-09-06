#![feature(test)]
#![allow(unused)]

use inkwell::{
    context::Context,
    data_layout::DataLayout,
    llvm_sys::target::LLVM_InitializeAllTargetInfos,
    module::Module,
    passes::PassManager,
    targets::{
        CodeModel, InitializationConfig, RelocMode, Target, TargetData, TargetMachine, TargetTriple,
    },
    OptimizationLevel,
};
use mylang::{
    cli::Cli,
    codegen::{self, llvm::Compiler},
    parser::{
        lexer::{Lexer, Span},
        parser_helper::Parser,
        StmtIter,
    },
};
use std::{
    io::{Read, Write},
    path::Path,
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
    const DEBUG_LLVM_IR_UNOPTIMIZED: bool = true;
    const DEBUG_LLVM_IR_OPTIMIZED: bool = true;

    let alloc = bumpalo::Bump::new();

    let code = "
pub test :: x -> 1+2*x;
pub sub :: (a, mut b) -> -b + a;
//main :: -> test(1) + test(2);
//mymain :: -> false | if test(1) else (10 | sub(1)) | sub(3);
// factorial :: x -> x == 0 | if 1 else x * factorial(x-1);
// mymain :: -> factorial(10) == 3628800;
mymain :: -> {
    mut a := test(1);
    // a = 100;
    b := test(2);
    a + b
};
";

    let code = code.as_ref();
    let stmts = StmtIter::parse(code, &alloc);

    if DEBUG_TOKENS {
        println!("\n### Tokens:");
        let mut lex = Lexer::new(code);
        while let Some(t) = lex.next() {
            println!("{:?}", t)
        }
    }

    if DEBUG_AST {
        println!("\n### AST Nodes:");
        for s in stmts.clone() {
            match s {
                Ok(s) => {
                    println!("stmt @ {:?}", s);
                    unsafe { s.as_ref().print_tree(code) };
                },
                Err(e) => {
                    eprintln!("ERROR: {:?}", e);
                    const VIEW_SIZE: usize = 20;
                    let view_start = e.span.start.saturating_sub(VIEW_SIZE);
                    let view = Span::new(view_start, e.span.end.saturating_add(VIEW_SIZE));
                    let newline_count =
                        code[Span::new(view_start, e.span.start)].lines().skip(1).count();
                    eprintln!("  {:?}", &code[view]);
                    eprintln!(
                        "  {}{}",
                        " ".repeat(VIEW_SIZE + newline_count + 1),
                        "^".repeat(e.span.len())
                    );
                    break;
                },
            }
        }
    }

    let context = Context::create();
    let mut compiler = codegen::llvm::Compiler::new_module(&context, "dev", code);

    for pres in stmts {
        let expr = pres.unwrap_or_else(|e| panic!("ERROR: {:?}", e));

        compiler
            .compile_top_level(unsafe { expr.as_ref() }, code)
            .unwrap_or_else(|e| panic!("ERROR: {:?}", e));

        /*
            match stmt.kind {
                    StmtKind::VarDecl { markers, ident, kind } => {
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

    print!("functions:");
    for a in compiler.module.get_functions() {
        print!("{:?},", a.get_name());
    }
    println!("");

    enum ExecutionVariant {
        ObjectCode,
        Jit,
    }

    const EXE_VAR: ExecutionVariant = ExecutionVariant::ObjectCode;

    let target_machine = Compiler::init_target_machine();

    if DEBUG_LLVM_IR_UNOPTIMIZED {
        println!("\n### Unoptimized LLVM IR:");
        compiler.module.print_to_stderr();
    }

    compiler.run_passes(&target_machine);

    if DEBUG_LLVM_IR_OPTIMIZED {
        println!("\n### Optimized LLVM IR:");
        compiler.module.print_to_stderr();
    }

    match EXE_VAR {
        ExecutionVariant::ObjectCode => compile(&compiler.module, &target_machine),
        ExecutionVariant::Jit => run_with_jit(compiler),
    }
}

fn compile(module: &Module, target_machine: &TargetMachine) {
    let filename = "target/output.o";
    target_machine
        .write_to_file(module, inkwell::targets::FileType::Object, Path::new(filename))
        .unwrap();
}

fn run_with_jit(mut compiler: Compiler) {
    let out = compiler.jit_run_fn("main").expect("has main function");
    println!("main returned {}", out);
}

#[cfg(test)]
mod benches {
    extern crate test;

    use super::*;
    use test::*;

    /// old: 11ms  <- so bad
    /// new: 3000ns  (3667x faster)
    #[bench]
    fn bench_parse(b: &mut Bencher) {
        b.iter(|| {
            let alloc = bumpalo::Bump::new();
            let code = "
test :: x -> 1+2+x;
main :: -> test(1) + test(2);
//main :: -> if true test(1) else test(2);
/*
main :: -> {
    a := test(1);
    b := test(2);
    a + b
};
*/
";
            let code = code.as_ref();
            let mut stmts = StmtIter::parse(code, &alloc);
            while let Some(_) = black_box(StmtIter::next(black_box(&mut stmts))) {}
        })
    }

    /// old: 23ms  <- so bad
    /// new: 3900ns  (5897x faster)
    #[bench]
    fn bench_parse2(b: &mut Bencher) {
        b.iter(|| {
            let alloc = bumpalo::Bump::new();
            let code = "
test :: x -> {
    a := (x + 3 * 2) * x + 1;
    b := x * 2 * x;
    a + b
};
main :: -> test(10);
";
            let code = code.as_ref();
            let mut stmts = StmtIter::parse(code, &alloc);
            while let Some(_) = black_box(StmtIter::next(black_box(&mut stmts))) {}
        })
    }

    #[bench]
    fn bench_parse3(b: &mut Bencher) {
        b.iter(|| {
            let alloc = bumpalo::Bump::new();
            let code = "
pub test :: x -> 1+2*x;
pub sub :: (a, b) -> -b + a;
main :: -> false | if test(1) else (10 | sub(3));
";
            let code = code.as_ref();
            let mut stmts = StmtIter::parse(code, &alloc);
            while let Some(_) = black_box(StmtIter::next(black_box(&mut stmts))) {}
        })
    }
}
