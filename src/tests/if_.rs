use super::{jit_run_test, jit_run_test_raw};
use crate::{
    tests::{TestSpan, test_compile_err, test_parse},
    util::IteratorExt,
};

#[test]
fn if_variants() {
    for my_bool in [true, false] {
        for variant_body in ["x = 1", "do x = 1", "{ x = 1 }"] {
            let code = format!(
                "
my_bool := {my_bool};
mut x := 0;
if my_bool {variant_body};
x"
            );
            let out = *jit_run_test::<i32>(&code).ok();
            let expected = if my_bool { 1 } else { 0 };
            assert!(out == expected, "```{code}\n``` -> expected: {expected}; got: {out}");
        }
    }
}

#[test]
fn if_expr() {
    for my_bool in [true, false] {
        for variant_then_body in ["5", "do 5", "then 5", "{ 5 }"] {
            let code = format!("my_bool := {my_bool}; if my_bool {variant_then_body} else 10");
            let out = *jit_run_test::<i32>(&code).ok();
            let expected = if my_bool { 5 } else { 10 };
            assert!(out == expected, "```{code}\n``` -> expected: {expected}; got: {out}");
        }
    }
}

#[test]
#[ignore = "not yet implemented"]
fn todo_fix_parser() {
    assert_eq!(*jit_run_test_raw::<i32>("test :: -> i32 { if true 10 else 20 }").ok(), 10);

    let code = "test :: -> struct { ok: bool } { if true .(true) else .(false) }";
    assert_eq!(*jit_run_test_raw::<bool>(code).ok(), true);
}

#[test]
fn parse_err_missing_if_body() {
    let res = test_parse("if a .A");
    let err = res.errors().expect_one();
    assert!(err.msg.starts_with("unexpected token: EOF"));
    assert_eq!(err.span.range(), 7..8);
}

#[test]
fn single_branch_yields_value() {
    test_compile_err(
        "{ if true 1; }",
        "Cannot yield a value from this `if` because it doesn't have an `else` branch.",
        |code| TestSpan::of_substr(code, "1"),
    );

    // function calls don't trigger the error.
    let code = r#"{
print :: -> i32 #import "libc".printf("hello world".ptr);
if true print();
}"#;
    jit_run_test::<()>(code).ok();
}
