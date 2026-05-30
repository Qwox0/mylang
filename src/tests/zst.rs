use crate::tests::test_body;

#[test]
fn zst_variable_codegen() {
    let res = test_body("mut a: void; a = {}; a").ok(());
    assert!(!res.llvm_ir().contains("%a = alloca [0 x i8], align 1"), "no stack alloc needed");
    drop(res);

    let res = test_body("mut a: never; a = return; a").ok(());
    assert!(!res.llvm_ir().contains("%a = alloca [0 x i8], align 1"), "no stack alloc needed");
    drop(res);

    let res = test_body("take_void_ptr :: (p: *void) -> {}; a: void; take_void_ptr(&a);").ok(());
    assert!(res.llvm_ir().contains("%0 = alloca [0 x i8], align 1"), "stack alloc needed");
    //                               ^ unnamed because AddrOf doesn't know the variable name
    drop(res);
}
