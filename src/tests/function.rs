use crate::tests::jit_run_test;

#[allow(unused)]
//#[test] // TODO
fn recursive_fn() {
    let out = jit_run_test!("
rec factorial :: (x: i64) -> if x == 0 do 1 else x * factorial(x-1);
factorial(10)" => i64);
    assert_eq!(out.unwrap(), 3628800);
    todo!()
}
