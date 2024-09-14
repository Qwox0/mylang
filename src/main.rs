#![feature(test)]

use inkwell::{context::Context, module::Module, targets::TargetMachine};
use mylang::{
    cli::Cli,
    codegen::llvm,
    parser::{lexer::Lexer, parser_helper::ParserInterface, DebugAst, StmtIter},
    sema,
    util::display_spanned_error,
};
use std::{io::Read, path::Path, time::Instant};

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
    const DEBUG_AST: bool = false;
    const DEBUG_TYPES: bool = false;
    const DEBUG_TYPED_AST: bool = true;
    const DEBUG_LLVM_IR_UNOPTIMIZED: bool = false;
    const DEBUG_LLVM_IR_OPTIMIZED: bool = false;
    const LLVM_OPTIMIZATION_LEVEL: u8 = 1;

    let alloc = bumpalo::Bump::new();

    let code = "
pub sub :: (a: f64, b: f64, ) -> -b + a;

// Sub :: struct {
//     a: f64,
//     b: f64,
// }
// pub sub2 :: (values: Sub) -> values.a - values.b;

//main :: -> test(1) + test(2);
//mymain :: -> false | if test(1) else (10 | sub(1)) | sub(3);
// factorial :: x -> x == 0 | if 1 else x * factorial(x-1);
// mymain :: -> factorial(10) == 3628800;
mymain :: -> {
    mut a := test(1);
    mut a := 10;
    a = 100;
    b := test();

    //sub2(Sub.{ a = a, b })
    sub(a, b)
};
pub test :: (mut x := 1) -> {
    x += 1;
    1+2*x
};

// pub infer_problem1 :: () -> infer_problem2();
// pub infer_problem2 :: () -> infer_problem1();
";

    let code = code.as_ref();
    let stmts = StmtIter::parse(code, &alloc);

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
        for s in stmts.clone() {
            match s {
                Ok(s) => {
                    println!("stmt @ {:?}", s);
                    s.print_tree();
                },
                Err(e) => {
                    display_spanned_error(e, code);
                },
            }
        }
        println!();
    }

    println!("### Frontend:");
    let frontend_parse_start = Instant::now();
    let context = Context::create();
    let mut compiler = llvm::Codegen::new_module(&context, "dev");

    let mut has_parse_error = false;
    let stmts = stmts
        .filter_map(|s| match s {
            Ok(stmt) => {
                compiler.add_top_level_symbol(stmt);
                Some(stmt)
            },
            Err(err) => {
                display_spanned_error(err, code);
                has_parse_error = true;
                None
            },
        })
        .collect::<Vec<_>>();

    if has_parse_error {
        panic!("Parse ERROR")
    }
    let frontend_parse_duration = frontend_parse_start.elapsed();

    let frontend_sema_start = Instant::now();
    let mut sema = sema::Sema::new(code, DEBUG_TYPES);

    for s in stmts.iter().copied() {
        sema.preload_top_level(s);
    }

    let mut finished = vec![false; stmts.len()];
    let mut remaining_fns = stmts.len();
    while finished.iter().any(std::ops::Not::not) {
        let old_remaining_fns = remaining_fns;
        debug_assert!(stmts.len() == finished.len());
        remaining_fns = stmts
            .iter()
            .zip(finished.iter_mut())
            .filter(|(_, finished)| !**finished)
            .map(|(s, finished)| {
                // TODO: don't reanalyse the entire function if `NotFinished` is returned
                *finished = sema.analyze_top_level(*s);
                *finished
            })
            .filter(|finished| !*finished)
            .count();
        if remaining_fns == old_remaining_fns {
            panic!("cycle detected") // TODO: find location of cycle
        }
    }

    if !sema.errors.is_empty() {
        for e in sema.errors {
            display_spanned_error(e, code);
        }
        panic!("Semantic analysis ERROR")
    }

    let frontend_sema_duration = frontend_sema_start.elapsed();

    if DEBUG_TYPED_AST {
        println!("\n### Typed AST Nodes:");
        for s in stmts.iter().copied() {
            println!("stmt @ {:?}", s);
            s.print_tree();
        }
        println!();
    }

    let frontend_codegen_start = Instant::now();
    for expr in stmts {
        compiler
            .compile_top_level_fn_body(&expr)
            .unwrap_or_else(|e| panic!("ERROR: {:?}", e));
    }
    let module = compiler.into_module();
    let frontend_codegen_duration = frontend_codegen_start.elapsed();
    let frontend_duration =
        frontend_parse_duration + frontend_sema_duration + frontend_codegen_duration;

    print!("functions:");
    for a in module.get_functions() {
        print!("{:?},", a.get_name());
    }
    println!("\n");

    let target_machine = llvm::Codegen::init_target_machine();

    if DEBUG_LLVM_IR_UNOPTIMIZED {
        println!("### Unoptimized LLVM IR:");
        module.print_to_stderr();
        println!();
    }

    let backend_start = Instant::now();
    module.run_passes(&target_machine, LLVM_OPTIMIZATION_LEVEL);
    let backend_duration = backend_start.elapsed();

    if DEBUG_LLVM_IR_OPTIMIZED {
        println!("### Optimized LLVM IR:");
        module.print_to_stderr();
        println!();
    }

    #[allow(unused)]
    enum ExecutionVariant {
        ObjectCode,
        Jit,
    }

    const EXE_VAR: ExecutionVariant = ExecutionVariant::ObjectCode;
    match EXE_VAR {
        ExecutionVariant::ObjectCode => compile(module.get_inner(), &target_machine),
        ExecutionVariant::Jit => run_with_jit(module),
    }

    println!("### Compilation time:");
    println!("  Frontend:            {:?}", frontend_duration);
    println!("    Lexer, Parser:     {:?}", frontend_parse_duration);
    println!("    Semantic Analysis: {:?}", frontend_sema_duration);
    println!("    LLVM IR Codegen:   {:?}", frontend_codegen_duration);
    println!("  LLVM Backend:        {:?} (LLVM pass pipeline)", backend_duration);
    println!("  Total:               {:?}", frontend_duration + backend_duration);
}

fn compile(module: &Module, target_machine: &TargetMachine) {
    let filename = "target/output.o";
    target_machine
        .write_to_file(module, inkwell::targets::FileType::Object, Path::new(filename))
        .unwrap();
}

fn run_with_jit(mut module: llvm::CodegenModule) {
    let out = module.jit_run_fn("main").expect("has main function");
    println!("main returned {}", out);
}

/// old (clone in Lexer::peek)
/// ```
/// test benches::bench_parse  ... bench:       2,898.31 ns/iter (+/- 229.93)
/// test benches::bench_parse2 ... bench:       4,115.37 ns/iter (+/- 397.96)
/// test benches::bench_parse3 ... bench:       4,569.32 ns/iter (+/- 537.69)
/// test benches::bench_parse4 ... bench:       6,576.30 ns/iter (+/- 638.84)
/// ```
///
/// next_tok field in Lexer
/// ```
/// test benches::bench_parse  ... bench:       1,880.97 ns/iter (+/- 386.20)
/// test benches::bench_parse2 ... bench:       2,780.48 ns/iter (+/- 851.97)
/// test benches::bench_parse3 ... bench:       2,908.87 ns/iter (+/- 408.08)
/// test benches::bench_parse4 ... bench:       4,063.92 ns/iter (+/- 567.65)
/// ```
#[cfg(test)]
mod benches {
    extern crate test;

    use super::*;
    use test::*;

    #[inline]
    fn bench(code: &str) {
        let alloc = bumpalo::Bump::new();
        let code = code.as_ref();
        let mut stmts = StmtIter::parse(code, &alloc);
        while let Some(res) = black_box(StmtIter::next(black_box(&mut stmts))) {
            res.unwrap();
        }
    }

    /// old: 11ms  <- so bad
    /// new: 3000ns  (3667x faster)
    #[bench]
    fn bench_parse(b: &mut Bencher) {
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
        b.iter(|| bench(code));
    }

    /// old: 23ms  <- so bad
    /// new: 3900ns  (5897x faster)
    #[bench]
    fn bench_parse2(b: &mut Bencher) {
        let code = "
test :: x -> {
    a := (x + 3 * 2) * x + 1;
    b := x * 2 * x;
    a + b
};
main :: -> test(10);
";
        b.iter(|| bench(code));
    }

    #[bench]
    fn bench_parse3(b: &mut Bencher) {
        let code = "
pub test :: x -> 1+2*x;
pub sub :: (a, b) -> -b + a;
main :: -> false | if test(1) else (10 | sub(3));
";
        b.iter(|| bench(code));
    }

    #[bench]
    fn bench_parse4(b: &mut Bencher) {
        let code = "
pub test :: (x := 2) -> 1+2*x;
pub sub :: (a, mut b) -> -b + a;
mymain :: -> {
    mut a := test(1);
    mut a := 10;
    a = 100;
    b := test(2);
    a + b
};
";
        b.iter(|| bench(code));
    }
}
