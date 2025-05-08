use crate::{
    context::CompilationContext,
    parser,
    tests::{
        TestSpan, jit_run_test, jit_run_test_raw, test_compile_err, test_compile_err_raw,
        test_file_mock,
    },
    util::IteratorExt,
};

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
    test_compile_err_raw("test :: -> A; A :: 1 ", "expected ';'", |code| {
        TestSpan::pos(code.len() - 1)
    });

    test_compile_err_raw("test :: -> { 1 ", "expected '}'", |code| TestSpan::pos(code.len()));

    let code = "
test :: -> {
    MyStruct :: struct { x: i64 };
    MyStruct.{ x = 5 }
";
    test_compile_err_raw(code, "expected '}'", |code| TestSpan::pos(code.len()));
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
fn return_struct_u2_i64_correctly() {
    let out = *jit_run_test::<u128>("struct { tag: u2 = 1, val: i64 = -1 }.{}").ok();
    let out_hex_string = format!("{out:032x}");
    println!("{out_hex_string}");
    // expect "ffffffffffffffff______________01"
    assert!(out_hex_string.starts_with(&"f".repeat(16)));
    assert!(out_hex_string.ends_with("01"));
}

#[test]
#[ignore = "unfinished test"]
fn return_struct_f64() {
    let big_float_bits: u64 = 0xEFFFFFFFFFFFFFF3;
    let big_float = f64::from_bits(big_float_bits);

    #[derive(Debug, Clone, Copy)]
    struct Out {
        tag: u8,
        val: f64,
    }
    let code = format!(
        "
MySumType :: struct {{
    tag: u2,
    val: f64,
}};
test :: -> {{
    retval: MySumType;
    retval.tag=1;
    retval.val={big_float};
    retval
}};"
    );
    let out = *jit_run_test_raw::<Out>(&code).ok();
    assert_eq!(out.tag, 1);
    assert_eq!(format!("{:x}", out.val.to_bits()), format!("{:x}", big_float_bits));
}

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
    let res = jit_run_test_raw::<MyStruct>(code);
    assert_eq!(*res.ok(), MyStruct { a: 5, b: 10, c: 15 });
    let llvm_module_text = res.module_text().unwrap();
    assert!(!llvm_module_text.contains("memcpy")); // TODO: implement this
}

#[test]
fn parse_err_missing_if_body() {
    let ctx = CompilationContext::new();
    let test_file = test_file_mock("if a .A".as_ref());
    parser::parse(ctx.0, test_file);
    assert_eq!(ctx.diagnostic_reporter.diagnostics.len(), 1);
    let err = &ctx.diagnostic_reporter.diagnostics[0];
    assert!(err.msg.starts_with("NoInput"));
    assert_eq!(err.span.range(), 7..8);
}

#[test]
#[ignore = "unfinished test"]
#[allow(unused)]
fn fix_precedence_range() {
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
    jit_run_test::<()>("if true do 0.. else 1..;").ok();
}

#[test]
fn good_error_message3() {
    let res = jit_run_test::<()>("A :: enum { B }; x := 1; A.B.(1);");
    let errors = res.diagnostics();
    assert_eq!(errors.len(), 1);
    assert_eq!(errors[0].span.range(), 38..41);
}

#[test]
fn good_error_message4() {
    let code = "A :: enum { B }; test :: -> A { .B.(1) };";
    let start = code.find(".B.(1)").unwrap();
    let range = start..start + 2;
    let res = jit_run_test_raw::<()>(code);
    let errors = res.diagnostics();
    assert_eq!(errors.len(), 1);
    assert_eq!(errors[0].span.range(), range);
}

#[test]
#[ignore = "TODO: better error"]
fn good_error_cannot_apply_initializer_to_type() {
    let code = "A :: enum { B }; test :: -> A.(1);";
    let start = code.find("A.(1)").unwrap();
    let range = start..start + 1;
    let res = jit_run_test_raw::<()>(code);
    let errors = res.diagnostics();
    assert_eq!(errors.len(), 1);
    assert_eq!(errors[0].span.range(), range);
    assert!(!errors[0].msg.contains("of type `type`"))
}

#[test]
#[ignore = "unfinished test"]
fn test_display_span1() {
    jit_run_test_raw::<()>("test :: ->").errors().expect_one();
}

#[test]
#[ignore = "unfinished test"]
fn test_display_span2() {
    let code = "test :: ->
";
    jit_run_test_raw::<()>(code).errors().expect_one();
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
    assert_eq!(*jit_run_test_raw::<MyStruct>(code).ok(), MyStruct { a: -5, b: 10, c: 123, d: 4 });
}

extern crate test;

#[bench]
fn bench_sema_error(b: &mut test::Bencher) {
    b.iter(|| {
        let res = jit_run_test::<()>(test::black_box("A :: enum { B }; x := 1; A.B.(1);"));
        let errors = res.diagnostics();
        assert_eq!(errors.len(), 1);
        assert_eq!(errors[0].span.range(), 38..41);
    });
}

#[test]
#[ignore = "not yet implemented"]
fn todo_fix_pos_initializer_codegen() {
    jit_run_test::<()>("if false { struct { ok: bool }.(true); };").ok();
}

#[test]
#[ignore = "not yet implemented"]
fn fix_panic() {
    let _err = jit_run_test_raw::<()>("f :: (x: i32) -> {}; test :: -> f(xx i32);")
        .errors()
        .expect_one();
    // TODO: check if err is type mismatch
    panic!("OK")
}

#[test]
#[ignore = "not yet implemented"]
fn fix_multi_fn_compile() {
    jit_run_test::<()>("std :: #import \"std\"; println :: std.println;").ok();
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
    let out = *jit_run_test::<i32>(code).ok();
    assert_eq!(out, 3);
}

#[test]
fn validate_lvalue() {
    test_compile_err("1 = 2", "Cannot assign a value to an expression of kind 'IntVal'", |code| {
        TestSpan::of_substr(code, "1")
    });
    jit_run_test::<()>("f :: -> struct {x:i32}.(5); f().x = 2").ok();
}

#[test]
#[ignore = "not yet implemented"]
fn lambda_type_mismatch() {
    let code = "
take_lambda :: (f: (x: i32) -> i32) -> f(5);
take_lambda(() -> 10)";
    test_compile_err(code, "TODO", |code| TestSpan::of_substr(code, "() -> 10"));
}
