use crate::tests::{TestSpan, substr, test, test_body, test_parse};

#[test]
#[ignore = "unfinished test"]
fn good_error_message1() {
    let code = "
pub test :: (mut x := 1) { // TODO: test this (better error)
    x += 1;
    420;
    1+2*x
};";
    todo!("better error message for \"{code}\"");
}

#[test]
fn good_error_message2() {
    test("test :: -> A; A :: 1 ").error("expected `;`", |code| TestSpan::pos(code.len() - 1));

    test("test :: -> { 1 ").error("expected `}`, got EOF", |code| TestSpan::pos(code.len()));

    let code = "
test :: -> {
    MyStruct :: struct { x: i64 };
    MyStruct.{ x = 5 }
";
    test(code).error("expected `}`, got EOF", |code| TestSpan::pos(code.len()));
}

/*
#[test]
fn good_error_message2_2() {
    let msg = res.diagnostics();
    println!("{:?}", msg );
    /*
    jit_run_test_raw("
test :: -> {
    MyStruct :: struct { x: i64 };
    MyStruct.{ x = 5 }
" )
    .ok();
    */
    todo!("better error message");
    panic!("OK")
}
*/

#[test]
#[ignore = "unfinished test"]
fn sret_no_memcpy() {
    #[derive(Debug, PartialEq)]
    struct MyStruct {
        a: i64,
        b: i64,
        c: i64,
    }
    let code = "
        MyStruct :: struct { a: i64, b: i64, c: i64 };
        test :: -> MyStruct.{ a = 5, b = 10, c = 15 };";
    let res = test(code).ok(MyStruct { a: 5, b: 10, c: 15 });
    assert!(!res.llvm_ir().contains("memcpy")); // TODO: implement this
}

#[test]
#[ignore = "unfinished test"]
#[allow(unused)]
fn fix_precedence_range() {
    test_parse("for x in 1.. do print(x);");

    let a = 1..{
        let x = 1;
        x
    };
    for _ in {
        let x = 1;
        0..x
    } {}
    for _ in 1.. {
        let x = 1;
        //x // -> Error
        break;
    }

    //jit_run_test!("for x in 0.. do break;" ).ok();
    //jit_run_test!("if 1 == 0.. do {};" ).ok();
    test_body("if true do 0.. else 1..;").ok(());
}

#[test]
#[ignore = "unfinished test"]
fn test_display_span1() {
    test("test :: ->").error("todo", |_| todo!());
}

#[test]
#[ignore = "unfinished test"]
fn test_display_span2() {
    let code = "test :: ->
";
    test(code).error("todo", |_| todo!());
}

#[test]
fn call_fn_with_sret() {
    #[derive(Debug, PartialEq, Eq)]
    #[repr(C)]
    struct MyStruct {
        a: i8,
        b: u32,
        c: u64,
        d: u64,
    }

    let code = "
MyStruct :: struct {
    a: i8,
    b: u32,
    c: u64,
    d: u64 = 4,
}
new :: -> MyStruct.(-5, 10, 123);
test :: -> {
    a := new();
    return a;
}";
    test(code).ok(MyStruct { a: -5, b: 10, c: 123, d: 4 });
}

#[test]
#[ignore = "not yet implemented"]
fn todo_fix_pos_initializer_codegen() {
    test_body("if false { struct { ok: bool }.(true); };").ok(());
}

#[test]
#[ignore = "not yet implemented"]
fn fix_panic() {
    test("f :: (x: i32) -> {}; test :: -> f(xx i32);").error("todo", |_| todo!());
    // TODO: check if err is type mismatch
    panic!("OK")
}

#[test]
#[ignore = "not yet implemented"]
fn fix_multi_fn_compile() {
    test_body("std :: #import \"std\"; println :: std.println;").ok(());
    panic!("OK")
}

/// solution 1: keep this error
/// solution 2: add special case for statements ([`crate::ast::Ast::block_expects_trailing_semicolon`] == false)
///     parse `for ... {}.{ val = 3 }` as `for ... {};.{ val = 3 }`
#[test]
#[ignore = "not yet implemented"]
fn parse_for_block() {
    let code = "
MyStruct :: struct { val: i32 };
for _ in 0..10 {} // currently a ';' is required
.{ val = 3 }";
    test_body(code).ok(3i32);
}

#[test]
fn validate_lvalue() {
    test_body("1 = 2")
        .error("Cannot assign a value to an expression of kind 'IntVal'", substr!("1"));
    test_body("f :: -> struct {x:i32}.(5); f().x = 2").ok(());
}

#[test]
#[ignore = "not yet implemented"]
fn prevent_errors_caused_by_errors() {
    let code = "
MyStruct :: struct { a: MissingType }
extern foreign_fn: (_: MyStruct) -> i32;";
    test_body(code).error("unknown ident `MissingType`", substr!("MissingType"));

    let code = "
MyStruct :: struct { a: MissingType }
extern foreign_fn: (_: i32) -> MyStruct;";
    test_body(code).error("unknown ident `MissingType`", substr!("MissingType"));
}

#[test]
fn invalid_array_lit_with_hint() {
    test_body("[1, 2]").error("expected `]`, got `,`", substr!(",")).info(
        "if you want to create an array value, consider using an array initializer `.[...]` \
         instead",
        substr!("[1,"),
    );
}

#[test]
fn incorrect_signedness_of_int_lit() {
    test("test :: -> u8 { -1 }")
        .error("Cannot apply unary operator `-` to type `u8`", substr!("-1"));

    test_body("a: u8 = if true { -1 } else 1;")
        .error("Cannot apply unary operator `-` to type `u8`", substr!("-1"));

    let code = "
f :: (p: *u32) -> p.*;
test :: -> f((-1).&);";
    test(code)
        .error("mismatched types: expected *u32; got *{signed integer literal}", substr!("(-1).&"));
}

#[test]
#[ignore = "not implemented"]
fn fix_enum_parse_error() {
    let code = "MyEnum :: enum { x: i32, x :: (s: *MyEnum) -> s.*.x; };";
    test(code).error("duplicate symbol `x` in struct scope", substr!("x";skip=1));
}

#[test]
fn use_intrinsics() {
    let code = "
cos : f64 -> f64 : #intrinsic \"llvm.cos.f64\";
test :: -> cos(10);";
    test(code).ok(10.0f64.cos());

    let code = "
ctz : (u32, is_zero_poison: bool) -> u32 : #intrinsic \"llvm.ctlz.i32\";
test :: -> ctz(10, true);";
    test(code).ok(10u32.leading_zeros());
}

#[test]
#[ignore = "not implemented"]
fn intrinsics_fix_immarg() {
    let code = "
ctz : (u32, is_zero_poison: bool) -> u32 : #intrinsic \"llvm.ctlz.i32\";
a :: (is_zero_poison: bool) -> ctz(10, is_zero_poison); // not allowed!
test :: -> a(true);";
    test(code).ok(10u32.leading_zeros());
}
