use super::{jit_run_test, jit_run_test_raw};
use crate::{diagnostics::DiagnosticSeverity, tests::TestSpan, util::IteratorExt};

#[test]
fn error_invalid_lhs() {
    let code = "1 :: 2;";
    let res = jit_run_test::<()>(code);
    let err = res.errors().expect_one();
    assert_eq!(err.severity, DiagnosticSeverity::Error);
    assert_eq!(err.span, TestSpan::pos(res.full_code.find("1").unwrap()));
    assert_eq!(err.msg.as_ref(), "expected variable name");
}

#[test]
fn static_var() {
    let code = "
get_num :: -> {
    static mut counter := 0;
    counter += 1;
    counter
}
test :: -> { get_num(); get_num(); get_num() }";
    let res = jit_run_test_raw::<i64>(code);
    assert_eq!(*res.ok(), 3);
    assert!(
        res.module_text()
            .unwrap()
            .contains("@get_num.counter = internal global i64 0, align 8")
    )
}
