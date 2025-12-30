use crate::{
    ast,
    tests::{TestSpan, substr, test, test_body, test_parse},
    util::IteratorExt,
};

#[test]
fn basic_call() {
    let code = "
my_fn :: (a: i64, b: i64) -> a + b;
test :: -> my_fn(1, 2);";
    test(code).ok(3i64);
}

#[test]
fn nested_fns() {
    let code = "
test :: -> {
    my_fn :: (a: i64, b: i64) -> a + b;
    my_fn(1, 2)
};";
    test(code).ok(3i64);
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
    test(code).ok(3628800i64);

    // global without `rec`
    let code = format!(
        "{factorial_code}
test :: -> factorial(10);"
    );
    test(code).ok(3628800i64);

    /* TODO
    // local with `rec`
    let code = format!(
        "test :: -> {{
    rec {factorial_code}
    factorial(10)
}}"
    );
    test_raw(code).expect(3628800i64);
    */
}

#[test]
fn recursive_fn_difficult_ret_infer() {
    let code = "
rec factorial :: (x: i64) -> if x == 0 then 1 else x * factorial(x-1);
test :: -> factorial(10);";
    test(code).ok(3628800i64);
}

#[test]
fn recursive_fn_cannot_infer_ret_ty() {
    test("test :: -> test();")
        .error("cannot infer the return type of this recursive function", substr!("test"));
}

#[test]
fn infer_number_based_on_ret_type() {
    let code = "
test :: -> i8 {
    if true return 1;
    2
};";
    let out = test(code).ok(1i8);
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
    test(code).ok(true);
}

#[test]
fn infer_struct_named_initalizer_based_on_ret_type() {
    let code = "
test :: -> struct { ok: bool } {
    if false return .{ ok = false };
    .{ ok = true }
};";
    test(code).ok(true);
}

#[test]
fn call_like_method() {
    let code = "
add :: (l: i64, r: i64) -> l + r;
test :: -> 123.add(456);";
    test(code).ok(123 + 456i64);
}

#[test]
fn use_correct_return_type() {
    let code = "test :: -> {
        if false return 1.0;
        5
    };";
    test(code).ok(5.0f64);

    let code = "test :: -> {
        if false return MyStruct.(1);
        .(5)
    };
    MyStruct :: struct { x: i32 };";
    test(code).ok(5i32);
}

#[test]
#[ignore = "not implemented"]
fn specialize_return_type_to_optional() {
    let code = "test :: -> { // return type: `unknown`
        if false return 1; // return type: `{int_literal}`
        None // return type: `?{int_literal}`
    };"; // return type: `?i64`
    test(code).ok(5.0f64);

    let code = "test :: -> {
        if false return 1;
        Some(5.0)
    };";
    test(code).ok(5.0f64);
}

#[test]
fn specialize_return_type() {
    let code = "test :: -> {
        if false return 1;
        5.0
    };";
    test(code).ok(5.0f64);
}

#[test]
fn function_currying() {
    test("f :: -> -> 1; test :: -> f()();").ok(1i64);

    let res = test("f :: -> -> i32 { 1 }; test :: -> f()();").ok(1i32);
    assert!(res.llvm_ir().contains("ret ptr @lambda"));
    assert!(res.llvm_ir().contains(
        "%call = call noundef ptr @f()
  %call1 = call noundef i32 %call()"
    ));
}

#[test]
fn duplicate_parameter() {
    test_body("f :: (a: i32, a: f64) -> {}").error("duplicate parameter 'a'", substr!("a";skip=1));
}

#[test]
fn parse_params_without_types() {
    {
        let res = test_parse("(a, b, c) -> {};");
        let f = res.stmts().iter().expect_one().downcast::<ast::Fn>();
        assert_eq!(f.params().len(), 3);
    }

    {
        let res = test_parse("(a, b) -> {};");
        let f = res.stmts().iter().expect_one().downcast::<ast::Fn>();
        assert_eq!(f.params().len(), 2);
    }
}

#[test]
fn parse_invalid_comma_in_params() {
    let code = "
f :: (,,a:i32) -> {};
g :: (a:i32,,b:i32) -> {};
f(,1);
g(1,,2);";
    test_parse(code)
        .error("expected parameter, got `,`", substr!(",,";.start()))
        .error("expected parameter, got `,`", substr!("g :: (a:i32,,";.end()))
        .error("expected expression, got `,`", substr!("f(,";.end()))
        .error("expected expression, got `,`", substr!("g(1,,";.end()));
}

#[test]
fn parse_empty_params() {
    let code = "
(); // err
() -> {}; // ok";
    test_parse(code).error("expected expression, got `)`", substr!(");";.start()));
}

#[test]
fn lambda() {
    let code = "
take_lambda :: (f: () -> i64) -> f();
take_lambda(() -> 10)";
    test_body(code).ok(10i32);

    let code = "
take_lambda :: (f: (x: i32) -> i32) -> f(5);
take_lambda((x: i32) -> x)";
    test_body(code).ok(5i32);
}

#[test]
fn lambda_infer_arg_types() {
    let code = "
f :: (lambda: (x: i8, y: f32) -> i8) -> lambda(1, 2.3);
test :: -> f((x, y) -> x + y.as(i8));";
    test(code).ok(1 + 2i8);
}

#[test]
fn lambda_infer_ret_type() {
    let code = "
take_lambda :: (f: () -> i8) -> f();
take_lambda(() -> 10)";
    let out = test_body(code).ok(10i8);
    assert!(out.llvm_ir().contains("ret i8 10"));
}

/// TODO: show more details about the mismatch, like param counts or individual type mismatches
#[test]
fn lambda_type_mismatch() {
    let code = "
take_lambda :: (f: (x: i32) -> i16) -> f(5);
take_lambda(() -> 10)";
    test_body(code)
        .error("mismatched types: expected `(x:i32)->i16`; got `()->i16`", substr!("() -> 10"));

    let code = "
take_lambda :: (f: (x: i32) -> i32) -> f(5);
take_lambda((x: f32) -> 10)";
    test_body(code).error(
        "mismatched types: expected `(x:i32)->i32`; got `(x:f32)->i32`",
        substr!("(x: f32) -> 10"),
    );
}

#[test]
fn error_missing_semicolon_after_fn() {
    let code = "
a :: -> 1
ok :: -> 1; // decls are not allowed as the second expression after `->`
ok2 :: -> _ := ok(); // but decls are allowed directly after `->`
";
    test(code).error("expected `;`", substr!("a :: -> 1";.after()));
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
        "
        )
    };
    test(code("")).error("cycle(s) detected:", |_| TestSpan::ZERO);
    // TODO: "The compiler currently doesn't try to infer return types across multiple indirectly
    // recursive functions."

    // works with an explicit return type:
    test(code("u16")).ok(15u16);
}

#[test]
fn dont_use_partially_inferred_return_type() {
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
    test(code).error("mismatched types (left: `u8`, right: `i32`)", substr!("+"));
}

#[test]
fn function_ptr() {
    let code = "
f :: (x: int) -> x + 1;
F :: *(_: int) -> int;

get_f :: -> F { ptr := &f; ptr };

test :: -> {
    fn: F = get_f();
    out: int = fn.*(10);
    return out;
}";
    test(code).ok(11);

    let code = "
f :: (x: int) -> x + 1;
test :: -> f.&.*(10);
";
    test(code).ok(11);
}
