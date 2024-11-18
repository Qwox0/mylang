use crate::tests::jit_run_test;

#[test]
fn basic_call() {
    let out = jit_run_test!(raw "
my_fn :: (a: i64, b: i64) -> a + b;
test :: -> my_fn(1, 2);" => i64);
    assert_eq!(out.unwrap(), 3);
}

#[test]
fn nested_fns() {
    let out = jit_run_test!(raw "
test :: -> {
    my_fn :: (a: i64, b: i64) -> a + b;
    my_fn(1, 2)
};" => i64);
    assert_eq!(out.unwrap(), 3);
}

#[allow(unused)]
//#[test] // TODO
fn recursive_fn() {
    let out = jit_run_test!("
rec factorial :: (x: i64) -> if x == 0 do 1 else x * factorial(x-1);
factorial(10)" => i64);
    assert_eq!(out.unwrap(), 3628800);
    todo!()
}

#[test]
fn infer_number_based_on_ret_type() {
    let out = jit_run_test!(raw "test :: -> i8 { 1 };" => i8);
    assert_eq!(out.unwrap(), 1);
    todo!()
}
