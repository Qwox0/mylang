#![feature(test)]

use inkwell::context::Context;
use mylang::{
    cli::{Cli, Command, RunScriptArgs},
    codegen::{self, llvm::jit::Jit},
    parser::{lexer::Code, result_with_fatal::ResultWithFatal, Item, StmtIter},
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
    const DEBUG_AST: bool = true;

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
    let stmts = StmtIter::parse(code);

    if DEBUG_AST {
        for s in stmts.clone() {
            match s {
                ResultWithFatal::Ok(s) => {
                    println!("{:?}", s);
                    s.print_tree()
                },
                ResultWithFatal::Err(e) | ResultWithFatal::Fatal(e) => {
                    eprintln!("ERROR: {:?}", e);
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
            .compile_top_level(&expr, code)
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

    compiler.run_passes_debug(None);

    let out = compiler.jit_run_fn("main").expect("has main function");
    println!("main returned {}", out);
}

#[cfg(test)]
mod benches {
    extern crate test;

    use super::*;
    use test::*;

    /// old: 11ms  <- so bad
    #[bench]
    fn bench_parse(b: &mut Bencher) {
        b.iter(|| {
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
            let mut stmts = StmtIter::parse(code);
            while let Some(_) = black_box(StmtIter::next(black_box(&mut stmts))) {}
        })
    }

    /// old: 23ms  <- so bad
    #[bench]
    fn bench_parse2(b: &mut Bencher) {
        b.iter(|| {
            let code = "
test :: x -> {
    a := (x + 3 * 2) * x + 1;
    b := x * 2 * x;
    a + b
};
main :: -> test(10);
";
            let code = code.as_ref();
            let mut stmts = StmtIter::parse(code);
            while let Some(_) = black_box(StmtIter::next(black_box(&mut stmts))) {}
        })
    }
}
