use crate::tests::test;

#[test]
fn compile_std_lib() {
    test("_ :: #import \"std\";").with_prelude().compile_no_err();
}
