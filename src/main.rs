use inkwell::context::Context;
use mylang::{
    cli::{Cli, Command, DebugOptions, RunScriptArgs},
    codegen,
    parser::{
        lexer::{Code, Lexer, Token, TokenKind},
        parser::{PErrKind, PError},
        result_with_fatal::ResultWithFatal,
        stmt, ws0, ws1, Stmt, StmtKind,
    },
};
use std::{io::Read, path::PathBuf};

fn main() {
    let cli = Cli::parse();

    println!("{:#?}", cli);

    match cli.command {
        Command::RunScript(RunScriptArgs { script }) => {
            run_script(Code::new(&read_code_file(script)), cli.debug)
        },
        Command::Build { build_script } => todo!("`build` command"),
        Command::Compile { file } => todo!("`compile` command"),
        Command::Repl {} => todo!("`repl` command"),
        Command::Check {} => todo!("`check` command"),
        Command::Clean {} => todo!("`clean` command"),
    }
}

fn read_code_file(path: PathBuf) -> String {
    fn inner(path: PathBuf) -> Result<String, std::io::Error> {
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

fn run_script(code: &Code, debug: Option<DebugOptions>) {
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
        for e in stmts {
            println!("\n{:#?}", e);
            println!("> {}", e.to_text());
            e.print_tree();
        }
        std::process::exit(0);
    }

    println!("+++ CODEGEN LLVM START");

    let context = Context::create();
    let mut compiler = codegen::Compiler::new_module(&context, code, "debug");

    for stmt in stmts {
        match stmt.kind {
            StmtKind::Let { markers, ident, ty, kind } => {
                let f = compiler.compile_let(ident, kind).unwrap();
                f.print_to_stderr();
            },
            StmtKind::Semicolon(_) => todo!(),
            StmtKind::Expr(expr) => {
                let out = compiler.compile_repl_expr(&expr).unwrap();
                out.print_to_stderr();
                unsafe { out.delete() };
            },
        }
    }

    println!("END");
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
