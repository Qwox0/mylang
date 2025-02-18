use crate::tests::jit_run_test;

#[test]
fn gt_for_bool() {
    let code = "true > false";
    let res = jit_run_test::<bool>(code);
    assert!(*res.ok(), "`{code}` -> expected `true`");
    drop(res);

    let code = "true < false";
    let res = jit_run_test::<bool>(code);
    assert!(!*res.ok(), "`{code}` -> expected `false`");
}

#[test]
fn infer_literal_type() {
    let code = "a: i8 = 3 + 8; a";
    assert_eq!(*jit_run_test::<i8>(code).ok(), 3 + 8);
}
