use crate::tests::{jit_run_test, jit_run_test_raw};

#[test]
fn basic_call() {
    let code = "
my_fn :: (a: i64, b: i64) -> a + b;
test :: -> my_fn(1, 2);";
    assert_eq!(*jit_run_test_raw::<i64>(code).ok(), 3);
}

#[test]
fn nested_fns() {
    let code = "
test :: -> {
    my_fn :: (a: i64, b: i64) -> a + b;
    my_fn(1, 2)
};";
    assert_eq!(*jit_run_test_raw::<i64>(code).ok(), 3);
}

#[test]
#[ignore = "unfinished test"]
fn recursive_fn() {
    let code = "
rec factorial :: (x: i64) -> if x == 0 do 1 else x * factorial(x-1);
factorial(10)";
    assert_eq!(*jit_run_test::<i64>(code).ok(), 3628800);
    todo!()
}

#[test]
fn infer_number_based_on_ret_type() {
    let code = "
test :: -> i8 {
    if true return 1;
    2
};";
    assert_eq!(*jit_run_test_raw::<i8>(code).ok(), 1);
}

#[test]
fn infer_struct_pos_initalizer_based_on_ret_type() {
    let code = "
test :: -> struct { ok: bool } {
    if false return .(false);
    .(true)
};";
    assert_eq!(*jit_run_test_raw::<bool>(code).ok(), true);
}

#[test]
fn infer_struct_named_initalizer_based_on_ret_type() {
    let code = "
test :: -> struct { ok: bool } {
    if false return .{ ok = false };
    .{ ok = true }
};";
    assert_eq!(*jit_run_test_raw::<bool>(code).ok(), true);
}

#[test]
fn call_like_method() {
    let code = "
add :: (l: i64, r: i64) -> l + r;
test :: -> 123.add(456);";
    assert_eq!(*jit_run_test_raw::<i64>(code).ok(), 123 + 456);
}
