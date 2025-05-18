use super::{TestSpan, test_compile_err};
use crate::{
    ast,
    tests::{jit_run_test_raw, test_parse},
    util::IteratorExt,
};

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
    let out = jit_run_test_raw::<i8>(code);
    assert_eq!(*out.ok(), 1);
    let module_text = out.module_text().unwrap();
    assert!(module_text.contains("ret i8 1"));
    assert!(module_text.contains("ret i8 2"));
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

#[test]
fn use_correct_return_type() {
    let code = "test :: -> {
        if false return 1.0;
        5
    };";
    assert_eq!(*jit_run_test_raw::<f64>(code).ok(), 5.0);

    let code = "test :: -> {
        if false return MyStruct.(1);
        .(5)
    };
    MyStruct :: struct { x: i32 };";
    assert_eq!(*jit_run_test_raw::<i32>(code).ok(), 5);
}

#[test]
fn specialize_return_type() {
    let code = "test :: -> {
        if false return 1;
        5.0
    };";
    assert_eq!(*jit_run_test_raw::<f64>(code).ok(), 5.0);
}

#[test]
#[ignore = "not yet implemented"]
fn lambda_type_mismatch() {
    let code = "
take_lambda :: (f: (x: i32) -> i32) -> f(5);
take_lambda(() -> 10)";
    test_compile_err(code, "TODO", |code| TestSpan::of_substr(code, "() -> 10"));

    let code = "
take_lambda :: (f: (x: i32) -> i32) -> f(5);
take_lambda((x: f32) -> 10)";
    test_compile_err(code, "TODO", |code| TestSpan::of_substr(code, "() -> 10"));
}

// TODO: duplicate function parameter names

#[test]
fn parse_params_without_types() {
    let res = test_parse("(a, b, c) -> {};").no_error();
    let f = res.stmts.into_iter().expect_one().downcast::<ast::Fn>();
    assert_eq!(f.params.len(), 3);
}

#[test]
fn parse_params_without_types2() {
    let res = test_parse("(a, b) -> {};").no_error();
    let f = res.stmts.into_iter().expect_one().downcast::<ast::Fn>();
    assert_eq!(f.params.len(), 2);
}

#[test]
#[ignore = "not yet implemented"]
fn lambda_infer_arg_types() {
    let code = "
f :: (lambda: (x: i8, y: f32) -> i32) -> lambda(1, 2.3);
test :: -> f((x, y) -> x + y.as(i8));";
    let out = jit_run_test_raw::<i8>(code);
    assert_eq!(*out.ok(), 1 + 2);
}
