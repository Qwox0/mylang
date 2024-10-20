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

#[test]
fn infer_literal_type() {
    let code = "a: i8 = 3 + 8; a";
    let out = jit_run_test!(code => i8);
    assert_eq!(out.unwrap(), 3 + 8);
}
