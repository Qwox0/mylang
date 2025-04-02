use super::jit_run_test;
use crate::{diagnostic_reporter::DiagnosticSeverity, tests::TestSpan, util::IteratorExt};

#[test]
fn error_invalid_lhs() {
    let code = "1 :: 2;";
    let res = jit_run_test::<()>(code);
    let err = res.errors().expect_one();
    assert_eq!(err.severity, DiagnosticSeverity::Error);
    assert_eq!(err.span, TestSpan::pos(res.full_code.find("1").unwrap()));
    assert_eq!(err.msg.as_ref(), "expected variable name");
}
