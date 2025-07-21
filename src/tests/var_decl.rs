use super::jit_run_test_raw;
use crate::tests::{TestSpan, test_compile_err, test_compile_err_raw};

#[test]
fn error_invalid_lhs() {
    let err_msg = "expected a variable name, got an expression";

    test_compile_err("{ 1 :: 2; }", err_msg, |code| TestSpan::of_substr(code, "1"));

    test_compile_err("{ a+b := 1; }", err_msg, |code| TestSpan::of_substr(code, "a+b"));

    // The `mut` marker shouln't change the error.
    test_compile_err("{ mut a+b := 1; }", err_msg, |code| TestSpan::of_substr(code, "a+b"));
}

#[test]
fn static_var() {
    let code = "
get_num :: -> {
    static mut counter := 0;
    counter += 1;
    counter
}
test :: -> { get_num(); get_num(); get_num() }";
    let res = jit_run_test_raw::<i64>(code);
    assert_eq!(*res.ok(), 3);
    assert!(res.llvm_ir().contains("@get_num.counter = internal global i64 0, align 8"))
}

#[test]
fn good_invalid_token_error() {
    // here we have to guess that `num` might be a decl (is this a good idea?)
    test_compile_err("{ num i32 = 1; }", "expected `:`, `:=`, `::`, or `;`", |code| {
        TestSpan::of_substr(code, "num").after()
    });

    // with the `mut` we now know that this is meant to be a decl
    test_compile_err(
        "{ mut num i32 = 1; }",
        "expected `:`, `:=`, or `::`, got an identifier",
        |code| TestSpan::of_substr(code, "i32"),
    );

    let err_msg = "expected `:`, `:=`, `,`, or `)`, got `#`";

    // an ident alone is a valid function parameter -> different error
    test_compile_err_raw("f :: (a # i32 = 1) -> {}", err_msg, |code| {
        TestSpan::of_substr(code, "#")
    });

    // `mut` marker shouln't change error message
    test_compile_err_raw(
        "f :: (mut a # i32 = 1) -> {}",
        "expected `:`, `:=`, or `::`, got `#`", // curently a worse error
        |code| TestSpan::of_substr(code, "#"),
    );

    test_compile_err_raw("f :: (a: i32 = 1, b # i32 = 2) -> {}", err_msg, |code| {
        TestSpan::of_substr(code, "#")
    });

    // `mut` marker shouln't change error message
    test_compile_err_raw("f :: (a: i32 = 1, mut b # i32 = 2) -> {}", err_msg, |code| {
        TestSpan::of_substr(code, "#")
    });
}
