use crate::tests::{TestSpan, substr, test};

fn test_compile_err_for_call_and_pos_initializer(
    params: &str,
    args: &str,
    expected_msg_start: &str,
    expected_span: impl Fn(&str) -> TestSpan,
) {
    test(format!("f :: ({params}) -> {{}};\ntest :: -> f({args});"))
        .error(expected_msg_start, &expected_span);

    test(format!("T :: struct {{ {params} }};\ntest :: -> T.({args});"))
        .error(expected_msg_start, &expected_span);
}

#[test]
fn too_many_args() {
    test_compile_err_for_call_and_pos_initializer(
        "a: i32, b: i32",
        "1, 2, 3, 4",
        "Got 4 positional arguments, but expected at most 2 arguments",
        substr!("3, 4";.start()),
    );
}

#[test]
fn named_arg_conflicts_with_pos_arg() {
    test_compile_err_for_call_and_pos_initializer(
        "a: i32, b: i32",
        "1, 2, a=3",
        "Parameter 'a' specified multiple times",
        substr!("a=3";.start()),
    );
}

#[test]
fn duplicate_named_arg() {
    test_compile_err_for_call_and_pos_initializer(
        "a: i32, b: i32",
        "1, b=2, b=3",
        "Parameter 'b' specified multiple times",
        substr!("b=3";.start()),
    );
}

#[test]
fn missing_args() {
    test_compile_err_for_call_and_pos_initializer(
        "a: i32, b: i32, c: i32",
        "b=5",
        "Missing arguments for parameters `a: i32`, `c: i32`",
        substr!("(b=5)";.end()),
    );
}

#[test]
fn unknown_named_param() {
    test_compile_err_for_call_and_pos_initializer(
        "a: i32, b: i32",
        "unknown=5",
        "Unknown parameter",
        substr!("unknown"),
    );
}
