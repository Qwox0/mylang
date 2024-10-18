use crate::tests::jit_run_test;

#[test]
fn gt_for_bool() {
    let code = "true > false";
    let ok = jit_run_test!(code => bool).unwrap();
    assert!(ok, "`{code}` -> expected `true`");

    let code = "true < false";
    let ok = jit_run_test!(code => bool).unwrap();
    assert!(!ok, "`{code}` -> expected `false`");
}
