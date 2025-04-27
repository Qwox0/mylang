use super::{TestSpan, test_compile_err};
use crate::{
    parser::{
        lexer::{Code, Lexer, Span, Token, TokenKind},
        parser_helper::ParserInterface,
    },
    ptr::Ptr,
    source_file::SourceFile,
    tests::jit_run_test,
};
use std::path::Path;

fn parse_as_tokens(code: &str) -> Vec<Token> {
    static TEST_FILE_PATH: &'static str = "test_file.mylang";

    let test_file =
        SourceFile::new(Ptr::from_ref(Path::new(TEST_FILE_PATH)), Ptr::from_ref(Code::new(code)));
    let mut lex = Lexer::new(Ptr::from_ref(&test_file));
    let mut tokens = Vec::new();
    while let Some(t) = lex.next() {
        tokens.push(t);
    }
    tokens
}

macro_rules! t {
    ($token_kind:ident $token_range:expr) => {
        Token { kind: TokenKind::$token_kind, span: Span::new($token_range, None) }
    };
}

fn parsed_tokens_eq(code: &str, expected_tokens: impl AsRef<[Token]>) -> bool {
    println!("check tokens of `{code}`");
    let mut parsed_tokens = parse_as_tokens(code).into_iter();
    let expected_tokens = expected_tokens.as_ref();
    let mut expected_tokens_iter = expected_tokens.into_iter();
    for idx in 1.. {
        match (parsed_tokens.next(), expected_tokens_iter.next()) {
            (Some(t1), Some(t2)) => {
                if t1.kind != t2.kind {
                    println!("TokenKind mismatch: {:?} != {:?}", t1.kind, t2.kind);
                    return false;
                }
                if t1.span.bytes() != t2.span.bytes() {
                    println!("Span mismatch: {:?} != {:?}", t1.span, t2.span);
                    return false;
                }
                println!("correct token: {t1:?}");
            },
            (None, None) => return true,
            _ => {
                println!(
                    "length mismatch: expected {} tokens, got {} tokens",
                    expected_tokens.len(),
                    idx + parsed_tokens.count()
                );
                return false;
            },
        }
    }
    unreachable!()
}

#[test]
fn space_after_dot() {
    assert!(parsed_tokens_eq("1.&", [t!(IntLit 0..1), t!(DotAmpersand 1..3)]));
    assert!(parsed_tokens_eq("1. &", [t!(FloatLit 0..2), t!(Whitespace 2..3), t!(Ampersand 3..4)]));

    assert!(parsed_tokens_eq("1.0", [t!(FloatLit 0..3)]));
    assert!(parsed_tokens_eq("1. 0", [t!(FloatLit 0..2), t!(Whitespace 2..3), t!(IntLit 3..4)]));

    assert!(parsed_tokens_eq("1.hello()", [
        t!(IntLit 0..1),
        t!(Dot 1..2),
        t!(Ident 2..7),
        t!(OpenParenthesis 7..8),
        t!(CloseParenthesis 8..9),
    ]));
    assert!(parsed_tokens_eq("1. hello()", [
        t!(IntLit 0..1),
        t!(Dot 1..2),
        t!(Whitespace 2..3),
        t!(Ident 3..8),
        t!(OpenParenthesis 8..9),
        t!(CloseParenthesis 9..10),
    ]));

    // Although '_' is valid in a number literal, it is parsed as an ident start if it is the first
    // "digit" after the float '.'
    assert!(parsed_tokens_eq("1._2", [t!(IntLit 0..1), t!(Dot 1..2), t!(Ident 2..4)]));
}

#[test]
fn different_int_lit_base() {
    assert!(parsed_tokens_eq("0b1", [t!(IntLit 0..3)]));
    assert_eq!(*jit_run_test::<i64>("0b1").ok(), 0b1);
    assert_eq!(*jit_run_test::<i64>("0o10").ok(), 8);
    assert_eq!(*jit_run_test::<i64>("0xAbc").ok(), 0xabc);
}

#[test]
#[ignore = "not yet implemented"]
fn different_float_lit_base() {
    assert!(parsed_tokens_eq("0b0101.0111", [t!(FloatLit 0..11)]));
    assert_eq!(*jit_run_test::<i64>("0b0101.0111").ok(), 0b1);
    assert_eq!(*jit_run_test::<i64>("0xAbc").ok(), 0xabc);
}

#[test]
#[ignore = "not yet implemented"]
fn invalid_literal_digit() {
    test_compile_err("0b12", "TODO", |code| TestSpan::of_substr(code, "2"));
}

#[test]
fn base_prefix_without_number() {
    assert!(parsed_tokens_eq("0b", [t!(IntLit 0..1), t!(Ident 1..2)]));
    // TODO: implement and test compiler error
}

#[test]
#[ignore = "not yet implemented"]
fn todo_float_lit_starting_with_dot() {
    assert!(parsed_tokens_eq(".5", [t!(FloatLit 0..2)]));
}
