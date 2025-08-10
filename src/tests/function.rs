use super::{TestSpan, jit_run_test, test_compile_err, test_compile_err_raw};
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
fn recursive_fn() {
    let factorial_code = "
factorial :: (x: i64) -> i64 {
    if x == 0 then 1 else x * factorial(x-1)
}";

    // global with `rec`
    let code = format!(
        "rec {factorial_code}
test :: -> factorial(10);"
    );
    assert_eq!(*jit_run_test_raw::<i64>(code).ok(), 3628800);

    // global without `rec`
    let code = format!(
        "{factorial_code}
test :: -> factorial(10);"
    );
    assert_eq!(*jit_run_test_raw::<i64>(code).ok(), 3628800);

    /* TODO
    // local with `rec`
    let code = format!(
        "test :: -> {{
    rec {factorial_code}
    factorial(10)
}}"
    );
    assert_eq!(*jit_run_test_raw::<i64>(code).ok(), 3628800);
    */
}

#[test]
fn recursive_fn_difficult_ret_infer() {
    let code = "
rec factorial :: (x: i64) -> if x == 0 then 1 else x * factorial(x-1);
test :: -> factorial(10);";
    assert_eq!(*jit_run_test_raw::<i64>(code).ok(), 3628800);
}

#[test]
fn recursive_fn_cannot_infer_ret_ty() {
    test_compile_err_raw(
        "test :: -> test();",
        "cannot infer the return type of this recursive function",
        |code| TestSpan::of_substr(code, "test"),
    );
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
    assert!(out.llvm_ir().contains("ret i8 1"));
    assert!(out.llvm_ir().contains("ret i8 2"));
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
#[ignore = "not implemented"]
fn specialize_return_type_to_optional() {
    let code = "test :: -> { // return type: `unknown`
        if false return 1; // return type: `{int_literal}`
        None // return type: `?{int_literal}`
    };"; // return type: `?i64`
    assert_eq!(*jit_run_test_raw::<f64>(code).ok(), 5.0);

    let code = "test :: -> {
        if false return 1;
        Some(5.0)
    };";
    assert_eq!(*jit_run_test_raw::<f64>(code).ok(), 5.0);
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
fn function_currying() {
    assert_eq!(*jit_run_test_raw::<i64>("f :: -> -> 1; test :: -> f()();").ok(), 1);

    let res = jit_run_test_raw::<i32>("f :: -> -> i32 { 1 }; test :: -> f()();");
    assert_eq!(*res.ok(), 1);
    assert!(res.llvm_ir().contains("ret ptr @lambda"));
    assert!(res.llvm_ir().contains(
        "%call = call noundef ptr @f()
  %call1 = call noundef i32 %call()"
    ));
}

#[test]
fn duplicate_parameter() {
    test_compile_err("f :: (a: i32, a: f64) -> {}", "duplicate parameter 'a'", |code| {
        TestSpan::of_nth_substr(code, 1, "a")
    });
}

#[test]
fn parse_params_without_types() {
    {
        let res = test_parse("(a, b, c) -> {};").no_error();
        let f = res.stmts.into_iter().expect_one().downcast::<ast::Fn>();
        assert_eq!(f.params().len(), 3);
    }

    {
        let res = test_parse("(a, b) -> {};").no_error();
        let f = res.stmts.into_iter().expect_one().downcast::<ast::Fn>();
        assert_eq!(f.params().len(), 2);
    }
}

#[test]
fn lambda() {
    let code = "
take_lambda :: (f: () -> i64) -> f();
take_lambda(() -> 10)";
    assert_eq!(*jit_run_test::<i32>(code).ok(), 10);

    let code = "
take_lambda :: (f: (x: i32) -> i32) -> f(5);
take_lambda((x: i32) -> x)";
    assert_eq!(*jit_run_test::<i32>(code).ok(), 5);
}

#[test]
fn lambda_infer_arg_types() {
    let code = "
f :: (lambda: (x: i8, y: f32) -> i8) -> lambda(1, 2.3);
test :: -> f((x, y) -> x + y.as(i8));";
    let out = jit_run_test_raw::<i8>(code);
    assert_eq!(*out.ok(), 1 + 2);
}

#[test]
fn lambda_infer_ret_type() {
    let code = "
take_lambda :: (f: () -> i8) -> f();
take_lambda(() -> 10)";
    let out = jit_run_test::<i8>(code);
    assert_eq!(*out.ok(), 10);
    assert!(out.llvm_ir().contains("ret i8 10"));
}

/// TODO: show more details about the mismatch, like param counts or individual type mismatches
#[test]
fn lambda_type_mismatch() {
    let code = "
take_lambda :: (f: (x: i32) -> i16) -> f(5);
take_lambda(() -> 10)";
    test_compile_err(code, "mismatched types: expected (x:i32)->i16; got ()->i16", |code| {
        TestSpan::of_substr(code, "() -> 10")
    });

    let code = "
take_lambda :: (f: (x: i32) -> i32) -> f(5);
take_lambda((x: f32) -> 10)";
    test_compile_err(code, "mismatched types: expected (x:i32)->i32; got (x:f32)->i32", |code| {
        TestSpan::of_substr(code, "(x: f32) -> 10")
    });
}

#[test]
fn error_missing_semicolon_after_fn() {
    let code = "
a :: -> 1
ok :: -> 1; // decls are not allowed as the second expression after `->`
ok2 :: -> _ := ok(); // but decls are allowed directly after `->`
";
    test_compile_err_raw(code, "expected `;`", |code| {
        TestSpan::of_substr(code, "a :: -> 1").after()
    });
}

#[test]
fn unorderable_indirectly_recursive_functions() {
    let code = |b_ret_ty| {
        format!(
            "
a :: (x: bool) -> {{
    if x return b();
    val: u16 = 10;
    return val;
}};
b :: -> {b_ret_ty} a(false) + 5;
test :: -> a(true);
_ :: 1; // TODO: remove this
        "
        )
    };
    test_compile_err_raw(code(""), "cycle(s) detected:", |_| TestSpan::ZERO);
    // TODO: "The compiler currently doesn't try to infer return types across multiple indirectly
    // recursive functions."

    // works with an explicit return type:
    assert_eq!(*jit_run_test_raw::<u16>(code("u16")).ok(), 15);
}

#[test]
fn dont_use_partially_infered_return_type() {
    let code = "
a :: -> {
    if false return 1;
    pause();
    a: u8 = 2;
    return a;
}
b :: -> {
    x: i32 = 1;
    a() + x
};
test :: -> b();
pause :: -> {};
        ";
    test_compile_err_raw(code, "MismatchedTypesBinOp (lhs: u8, rhs: i32)", |code| {
        TestSpan::of_substr(code, "+")
    });
}
