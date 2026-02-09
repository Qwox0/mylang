use crate::tests::test_body;

#[test]
fn zst_variable_codegen() {
    let res = test_body("mut a: void; a = {}; a").ok(());
    assert!(res.llvm_ir().contains("%a = alloca [0 x i8], align 1"));
    drop(res);

    let res = test_body("mut a: never; a = return; a").ok(());
    assert!(res.llvm_ir().contains("%a = alloca [0 x i8], align 1"));
}
