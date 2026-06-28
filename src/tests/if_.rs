use crate::tests::*;

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
            test_body(code).ok(if my_bool { 1i32 } else { 0 });
        }
    }
}

#[test]
fn if_expr() {
    for my_bool in [true, false] {
        for variant_then_body in ["5", "do 5", "then 5", "{ 5 }"] {
            test_body(format!("my_bool := {my_bool}; if my_bool {variant_then_body} else 10"))
                .ok(if my_bool { 5i32 } else { 10 });
        }
    }
}

#[test]
#[ignore = "not yet implemented"]
fn todo_fix_parser() {
    test("test :: -> i32 { if true 10 else 20 }").ok(10i32);

    test("test :: -> struct { ok: bool } { if true .(true) else .(false) }").ok(true);
}

#[test]
fn parse_err_missing_if_body() {
    test("if a .A").parse().error("unexpected token: EOF", substr!("A";.after()));
}

#[test]
fn single_branch_yields_value() {
    test_body("{ if true 1; }").error(
        "Cannot yield a value from this `if` because it doesn't have an `else` branch.",
        substr!("1"),
    );

    // function calls don't trigger the error.
    let code = r#"{
print :: -> i32 #import "libc".printf("hello world".ptr);
if true print();
}"#;
    test_body(code).with_prelude().ok(());
}

#[test]
fn merge_bb_codegen() {
    // void
    test_body("if true {} else {}").compile_no_err();

    // zst struct
    test_body("S :: struct {}; if true S.() else S.()").compile_no_err();

    // never
    test_body("if true { return 1; } else { return 2; }").compile_no_err();

    // WriteTarget::Phi
    let res = test_body("if true then 1 else return 2").compile_no_err();
    assert!(res.llvm_ir().contains("phi i64 [ 1, %then ]"));
    drop(res);

    // WriteTarget::Ptr
    let res = test_body("a := if true then 1 else return 2; a").compile_no_err();
    assert!(res.llvm_ir().contains("store i64 1, ptr %a, align 8"));
    assert!(!res.llvm_ir().contains("phi"));
    drop(res);
}
