use inkwell::context::Context;
use mylang::{
    cli::{Cli, Command, DebugOptions, RunScriptArgs},
    codegen::{self},
    parser::{
        lexer::{Code, Lexer, Token, TokenKind},
        parser::{PErrKind, PError},
        result_with_fatal::ResultWithFatal,
        stmt, ws0, Stmt, StmtKind,
    },
};
use std::{
    io::{Read, Write},
    path::Path,
};

fn main() {
    let cli = Cli::parse();

    println!("{:#?}", cli);

    match cli.command {
        Command::RunScript(RunScriptArgs { script }) => {
            //run_script(Code::new(&read_code_file(script)), cli.debug)
            let code = read_code_file(&script);
            let stmts = parse(code.as_ref(), cli.debug);

            let context = Context::create();
            let mut compiler = codegen::llvm::Compiler::new_module(
                &context,
                script.file_stem().expect("has filename").to_str().expect("is valid utf-8"),
            );

            for stmt in stmts {
                match stmt.kind {
                    StmtKind::VarDecl { markers, ident, kind } => {
                        let _f = compiler.compile_var_decl(ident, kind, code.as_ref()).unwrap();
                    },
                    StmtKind::Semicolon(_) => todo!(),
                    StmtKind::Expr(expr) => panic!("top-level expressions are not allowed"),
                    /*
                    StmtKind::Expr(expr) => {
                        let out = compiler.compile_repl_expr(&expr).unwrap();
                        out.print_to_stderr();
                        unsafe { out.delete() };
                    },
                    */
                }
            }

            if cli.debug == Some(DebugOptions::LlvmIrUnoptimized) {
                compiler.module.print_to_stderr();
                std::process::exit(0);
            }

            compiler.run_passes();

            if cli.debug == Some(DebugOptions::LlvmIrOptimized) {
                compiler.module.print_to_stderr();
                std::process::exit(0);
            }
        },
        Command::Build { build_script } => todo!("`build` command"),
        Command::Compile { file } => todo!("`compile` command"),
        Command::Repl {} => {
            //todo!("`repl` command");
            let context = Context::create();
            let mut compiler = codegen::llvm::Compiler::new_module(&context, "repl");
            loop {
                print!("\n> ");
                std::io::stdout().flush().expect("Could flush stdout");
                let mut line = String::new();
                std::io::stdin().read_line(&mut line).expect("Could read line from stdin");
                let code = line.as_ref();

                for stmt in parse(code, cli.debug) {
                    match stmt.kind {
                        StmtKind::VarDecl { markers, ident, kind } => {
                            let _f = compiler.compile_var_decl(ident, kind, code).unwrap();
                        },
                        StmtKind::Semicolon(_) => todo!(),
                        StmtKind::Expr(expr) => {
                            match compiler.jit_run_expr(&expr, code) {
                                Ok(out) => println!("=> {}", out),
                                Err(err) => panic!("ERROR: {:?}", err),
                            }

                            /*
                            let out = compiler.compile_repl_expr(&expr).unwrap();
                            out.print_to_stderr();
                            unsafe { out.delete() };
                            */
                        },
                    }
                }
            }
        },
        Command::Check {} => todo!("`check` command"),
        Command::Clean {} => todo!("`clean` command"),
    }
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

fn parse(code: &Code, debug: Option<DebugOptions>) -> Vec<Stmt> {
    println!("+++ CODE START\n\"{}\"\n+++ CODE END", code);

    if debug == Some(DebugOptions::Tokens) {
        debug_tokens(code);
        std::process::exit(0);
    }

    let now = std::time::Instant::now();
    let res = parse_into_vec(code);
    let took = now.elapsed();

    let stmts = res.unwrap_or_else(|e| {
        eprintln!("{}", e.display(code));
        panic!("ERROR")
    });

    println!(
        "+++ PARSING took: {}ms {}us {}ns",
        took.as_millis(),
        took.as_micros() % 1000,
        took.as_nanos() % 1000
    );

    if debug == Some(DebugOptions::Ast) {
        for e in stmts.iter() {
            println!("\n{:#?}", e);
            println!("> {}", e.to_text());
            e.print_tree();
        }
        std::process::exit(0);
    }

    stmts
}

fn debug_lex(code: &Code, cer: impl Iterator<Item = Token>) {
    let mut full = String::new();
    let mut prev_was_ident = false;

    for Token { kind, span } in cer {
        let text = code.get(span.bytes()).expect("correctly parsed span");
        let is_ident = kind == TokenKind::Ident;
        if prev_was_ident && is_ident {
            full.push(' ');
        }
        full.push_str(&text);
        if kind == TokenKind::Semicolon {
            full.push('\n');
        }
        prev_was_ident = is_ident;
        let text = format!("{:?}", text);
        println!("{:<20} -> {:?} {}", text, kind, span);
    }

    println!("+++ full code:\n{}", full);
}

pub fn debug_tokens(code: &Code) {
    let lex = Lexer::new(code).filter(|t| {
        !matches!(
            t.kind,
            TokenKind::Whitespace | TokenKind::LineComment(_) | TokenKind::BlockComment(_)
        )
    });

    debug_lex(code, lex);
}

pub fn parse_into_vec(code: &Code) -> ResultWithFatal<Vec<Stmt>, PError> {
    //debug_tokens(code);

    let mut lex = Lexer::new(code);

    if let ResultWithFatal::Ok((_, new_lex)) = ws0().run(lex) {
        lex = new_lex;
    }

    let mut stmts = Vec::new();

    use ResultWithFatal::*;

    loop {
        match stmt().run(lex) {
            Ok((stmt, l)) => {
                stmts.push(stmt);
                lex = l;
            },
            Err(PError { kind: PErrKind::NoInput, .. }) => break,
            Err(e) => return Err(e),
            Fatal(e) => return Fatal(e),
        }
    }
    assert!(lex.is_empty(), "The lexer must parse the entire input!");

    Ok(stmts)
}
