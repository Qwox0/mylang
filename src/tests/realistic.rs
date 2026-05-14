use crate::tests::test;

#[test]
fn compile_std_lib() {
    test("_ :: #import \"std\";")
        .with_prelude()
        .print_llvm_module(false)
        .compile_no_err();
}
