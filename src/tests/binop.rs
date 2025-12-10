use crate::tests::{substr, test_body};

#[test]
fn gt_for_bool() {
    test_body("true > false").ok(true);
    test_body("true < false").ok(false);
}

#[test]
fn infer_literal_type() {
    test_body("a: i8 = 3 + 8; a").ok(3i8 + 8);
}

#[test]
fn correct_error_span_with_parens() {
    test_body("_: []u8 = (-1).&").error(
        "mismatched types: expected `[]u8`; got `*{signed integer literal}`",
        substr!("(-1).&"),
    );

    test_body("1 + \"\"")
        .error("mismatched types (left: `{integer literal}`, right: `[]u8`)", substr!("+"));

    test_body("(1 + \"\")")
        .error("mismatched types (left: `{integer literal}`, right: `[]u8`)", substr!("+"));
}
