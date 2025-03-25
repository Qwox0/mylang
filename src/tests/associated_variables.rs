use super::jit_run_test;
use crate::{
    diagnostic_reporter::DiagnosticSeverity,
    tests::{TestSpan, jit_run_test_raw},
};

#[test]
fn struct_method() {
    let code = "
MyStruct :: struct { val: i64 };
MyStruct.new :: -> MyStruct.(0);
MyStruct.inc :: (self: MyStruct) -> self.val += 1;
test :: -> {
    mut a := MyStruct.new();
    a.inc();
    a.val
}";
    assert_eq!(*jit_run_test_raw::<i64>(code).ok(), 1);
}

#[test]
fn use_before_definition() {
    let code = "
MyStruct :: struct { val: i64 };
test :: -> MyStruct.new().val;
MyStruct.new :: -> MyStruct.(MyStruct.DEFAULT_VAL);
MyStruct.DEFAULT_VAL :: 10;";
    assert_eq!(*jit_run_test_raw::<i64>(code).ok(), 10)
}

#[test]
fn define_const_in_struct() {
    let code = "
MyStruct :: struct {
    x: i64;
    VAL :: 10;
};
test :: -> MyStruct.VAL;";
    assert_eq!(*jit_run_test_raw::<i64>(code).ok(), 10)
}

#[test]
#[ignore = "not implemented"]
fn define_and_use_const_in_struct() {
    let code = "
MyStruct :: struct {
    val: i64 = MyStruct.DEFAULT_VAL;
    DEFAULT_VAL :: 10;
};
test :: -> MyStruct.().val;";
    assert_eq!(*jit_run_test_raw::<i64>(code).ok(), 10)
}

#[test]
fn error_missing_type_name() {
    let code = ".method :: -> {};";
    let res = jit_run_test::<()>(code);
    let err = res.one_err();
    assert_eq!(err.severity, DiagnosticSeverity::Error);
    assert_eq!(err.span, TestSpan::pos(res.full_code.find(".").unwrap()));
    assert_eq!(err.msg.as_ref(), "A member declaration requires an associated type name");
}

#[test]
#[ignore = "unfinished test"]
fn error_access_static_through_value() {
    let code = "
MyStruct :: struct { val: i64 };
MyStruct.NUM :: 10;
test :: -> {
    a := MyStruct.(0);
    a.NUM
}";
    let res = jit_run_test::<()>(code);
    let err = res.err();
    assert_eq!(err[0].severity, DiagnosticSeverity::Error);
    let expected_err_span = TestSpan::of_substr(&res.full_code, "a.NUM");
    assert_eq!(err[0].span, expected_err_span);
    assert_eq!(err[0].msg.as_ref(), "cannot access a static constant through a value");

    assert_eq!(err[1].severity, DiagnosticSeverity::Info);
    assert_eq!(err[1].span, expected_err_span.start());
    assert_eq!(err[1].msg.as_ref(), "consider replacing the value with its type 'MyStruct'"); // not implemented
}

#[test]
#[ignore = "todo: better error instead of cycle detection"]
fn error_access_missing_const() {
    let code = "
MyStruct :: struct {};
test :: -> MyStruct.SOME_MISSING_CONST;";
    assert_eq!(*jit_run_test_raw::<i64>(code).ok(), 10)
}

