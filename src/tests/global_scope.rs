use crate::tests::{TestSpan, has_duplicate_symbol, jit_run_test_raw, test_compile_err_raw};

#[test]
fn error_duplicate() {
    let code = "
    test :: -> 1;
    test :: -> 2;";
    test_compile_err_raw(code, "duplicate definition in file scope", |code| {
        TestSpan::of_nth_substr(code, 1, "test")
    });
}

#[test]
fn error_global_variable() {
    let code = "
    var := 0;
    test :: -> var;";
    test_compile_err_raw(
        code,
        "Global variables must be marked as const (`var :: ...`), static (`static var := ...`) or \
         extern (`extern var: ...)",
        |code| TestSpan::of_substr(code, "var"),
    );
}

#[test]
fn unexpected_toplevel_expr() {
    let code = "1 + 1;";
    test_compile_err_raw(code, "unexpected top level expression", |code| {
        TestSpan::of_substr(code, "1 + 1")
    });
}

#[test]
fn order_independent() {
    let code = "
test :: -> A;
A :: 3;";
    let res = jit_run_test_raw::<i64>(code);
    assert_eq!(*res.ok(), 3);
    assert!(!res.llvm_ir().contains("@A"));
    drop(res);

    let code = "
test :: -> A;
static A := 3;";
    let res = jit_run_test_raw::<i64>(code);
    assert_eq!(*res.ok(), 3);
    assert!(!has_duplicate_symbol(res.llvm_ir(), "@A"));
}
