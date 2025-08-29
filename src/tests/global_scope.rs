use crate::tests::{has_duplicate_symbol, substr, test};

#[test]
fn error_duplicate() {
    let code = "
    test :: -> 1;
    test :: -> 2;";
    test(code).error("duplicate definition in file scope", substr!("test";skip=1));
}

#[test]
fn error_global_variable() {
    let code = "
    var := 0;
    test :: -> var;";
    test(code).error(
        "Global variables must be marked as const (`var :: ...`) or static (`static var := ...`)",
        substr!("var"),
    );
}

#[test]
fn unexpected_toplevel_expr() {
    let code = "1 + 1;";
    test(code).error("unexpected top level expression", substr!("1 + 1"));
}

#[test]
fn order_independent() {
    let code = "
test :: -> A;
A :: 3;";
    let res = test(code).ok(3i64);
    assert!(!res.llvm_ir().contains("@A"));
    drop(res);

    let code = "
test :: -> A;
static A := 3;";
    let res = test(code).ok(3i64);
    assert!(!has_duplicate_symbol(res.llvm_ir(), "@A"));
}
