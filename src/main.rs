use mylang::parser::{
    lexer::{Lexer, Token, TokenKind},
    parser::{PErrKind, PError},
    result_with_fatal::ResultWithFatal,
    stmt, Stmt,
};
use ResultWithFatal::*;

fn main() {
    #[allow(unused)]
    let code = r#"
// This is a normal comment
/* This is also a normal comment */
//!This is a inner doc comment
/*! This is also a inner doc comment */

/*
    /*
        /*
            Comment in a comment in a comment
        */
    */
*/

// type: `i32 -> i32`
let square1 = (x: i32) -> i32 {
    x * x
};
let square4 = <T: Mul<T>>(x: T) -> x * x;
let square9 = x -> x * x; // automatic infer generic

let MyStruct = <T> -> struct {
    x: T,
};

/// this is the main function
let main = -> {
    let mut a = -3.1415;
    a = a * -1.as<f32>;
    a += 0.5;
    let my_char = 'a';
    let my_str = "hello world";

    let a = Some(1);
    let a = a.map(a -> a + 1).unwrap_or_else(-> 1);
};
"#;

    let code = r#"
let curry = a -> b -> a + b;

let add = (a, b) -> a + b;

    let mut rec a = a.add(1):add(1):String.from_int().len();
//  let mut rec - = ---------------------------------------;
//              a   -------------------------------------()
//                  ---------------------------------.---
//                  -------------------------------() len
//                  ---------------:---------------
//                  ------------(1) ------.--------
//                  --------:---    String from_int
//                  -----(1) add
//                  -.---
//                  a add

/// this is the main function
let main = -> {
    let rec a: int = 1;
    let mut a = a;
    let a: u8 = a.add(1);
};"#;
    println!("+++ CODE START\n{}\n+++ CODE END", code);

    let now = std::time::Instant::now();
    let res = parse(code);
    let took = now.elapsed();

    let exprs = res.unwrap_or_else(|e| {
        eprintln!("{}", e.display(code));
        panic!("ERROR")
    });

    for e in exprs {
        println!("{:#?}", e);
        println!("> {}", e.to_text());
        e.print_tree();
        println!("\n");
    }

    println!(
        "took: {}ms {}us {}ns",
        took.as_millis(),
        took.as_micros() % 1000,
        took.as_nanos() % 1000
    );
    println!("END");
}

fn debug_lex(code: &str, cer: impl Iterator<Item = Token>) {
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

pub fn debug_tokens(code: &str) {
    let lex = Lexer::new(code).filter(|t| {
        !matches!(
            t.kind,
            TokenKind::Whitespace | TokenKind::LineComment(_) | TokenKind::BlockComment(_)
        )
    });

    debug_lex(code, lex);
}

pub fn parse(code: &str) -> ResultWithFatal<Vec<Stmt>, PError> {
    //debug_tokens(code);

    let mut lex = Lexer::new(code);

    let mut stmts = Vec::new();

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
    Ok(stmts)
}
