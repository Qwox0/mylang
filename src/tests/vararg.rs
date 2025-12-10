use crate::tests::test;

#[test]
fn no_other_params() {
    let code = "
f1 :: #varargs -> {};
f2 :: #varargs () -> {};";
    let res = test(code).compile_no_err();
    assert!(res.llvm_ir().contains("void @f1(...)"));
    assert!(res.llvm_ir().contains("void @f2(...)"));
}

#[test]
fn printf_codegen() {
    let code = include_str!("../../tests/printf_vararg.mylang");
    let res = test(code).compile_no_err();
    assert!(res.llvm_ir().contains("declare noundef i32 @printf(ptr noundef, ...)"));
    assert!(res.llvm_ir().contains(
        "call noundef i32 (ptr, ...) @printf(ptr noundef @0, i64 10, double 1.234560e+02, ptr @1)"
    ));
}
