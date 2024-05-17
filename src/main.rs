#![feature(control_flow_enum)]

use mylang::parser::{
    lexer::{Lexer, Span, Token, TokenKind},
    stmt, ParseError, Stmt,
};
use std::{iter, ops::ControlFlow};

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
let rec a: int = 1;
let mut a = a;
let a: u8 = a.add(1);
let rec mut a = a.add(1):add(1):String.from_int().len();
/*
let add = (a, b) -> a + b;

/// this is the main function
let main = -> {
    let a = 1;
    let a = a.add(1):add(1):String.from_int().len();
//  -----------------------------------------------;
//  let - = ---------------------------------------
//      a   -------------------------------------()
//          ---------------------------------.---
//          -------------------------------() len
//          ---------------:---------------
//          ------------(1) ------.--------
//          --------:---    String from_int
//          -----(1) add
//          -.---
//          a add
};
*/
"#;
    println!("+++ CODE START\n{}\n+++ CODE END", code);

    let show_span_error = |span: Span, err: ParseError| {
        let (line_start, line) = code
            .lines()
            .try_fold(0, |idx, line| {
                let end = idx + line.len() + 1;
                if (idx..end).contains(&span.start) {
                    ControlFlow::Break((idx, line))
                } else {
                    ControlFlow::Continue(end)
                }
            })
            .break_value()
            .unwrap();
        let err_start = span.start - line_start;
        let err_len = span.len();
        println!("\nERROR:",);
        println!("| {}", line);
        println!("| {}", " ".repeat(err_start) + &"^".repeat(err_len));
        panic!("ERROR: {:?}", err);
    };

    let exprs = parse(code).unwrap_or_else(|e| match e {
        ParseError::NoInput => todo!(),
        ParseError::UnexpectedToken(Token { span, .. })
        | ParseError::DoubleLetMarker(Token { span, .. })
        | ParseError::TooManyLetIdents(span)
        | ParseError::Tmp(_, span) => show_span_error(span, e),
        e => panic!("\nERROR: {:?}", e),
    });

    for e in exprs {
        //println!("{:#?}", e);
        println!("> {}", e.to_text());
        e.print_tree();
        println!("\n");
    }

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

pub fn parse(code: &str) -> Result<Vec<Stmt>, ParseError> {
    debug_tokens(code);

    let lex = Lexer::new(code);

    iter::repeat(())
        .scan(lex, |lex, ()| match stmt(*lex) {
            Ok((stmt, l)) => {
                *lex = l;
                Some(Ok(stmt))
            },
            Err(ParseError::NoInput) => None,
            Err(e) => Some(Err(e)),
        })
        .collect()

    /*
    let mut stmts = Vec::new();

    loop {
        match stmt(lex) {
            Ok((stmt, l)) => {
                stmts.push(stmt);
                lex = l;
            },
            Err(ParseError::NoInput) => break,
            Err(e) => return Err(e),
        }
    }
    Ok(stmts)
    */
}
