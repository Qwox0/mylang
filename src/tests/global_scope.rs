use crate::tests::{TestSpan, test_compile_err_raw};

#[test]
fn error_duplicate() {
    let code = "
    test :: -> 1;
    test :: -> 2;";
    test_compile_err_raw(code, "Duplicate definition in module scope", |code| {
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
