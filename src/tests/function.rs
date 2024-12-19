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
    let out = jit_run_test!(raw "
test :: -> i8 {
    if true return 1;
    2
};" => i8);
    assert_eq!(out.unwrap(), 1);
}

#[test]
fn infer_struct_pos_initalizer_based_on_ret_type() {
    let out = jit_run_test!(raw "
test :: -> struct { ok: bool } {
    if false return .(false);
    .(true)
};" => bool);
    assert_eq!(out.unwrap(), true);
}

#[test]
fn infer_struct_named_initalizer_based_on_ret_type() {
    let out = jit_run_test!(raw "
test :: -> struct { ok: bool } {
    if false return .{ ok = false };
    .{ ok = true }
};" => bool);
    assert_eq!(out.unwrap(), true);
}

#[test]
fn call_like_method() {
    let out = jit_run_test!(raw "
add :: (l: i64, r: i64) -> l + r;
test :: -> 123.add(456);" => i64);
    assert_eq!(out.unwrap(), 123 + 456);
}
