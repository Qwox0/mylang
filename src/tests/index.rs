use super::{TestSpan, test_compile_err};

#[test]
fn invalid_index_type() {
    test_compile_err(".[1,2,3][1.0]", "Cannot index into array with `f64`", |code| {
        TestSpan::of_substr(code, "1.0")
    });
}

#[test]
fn invalid_slice_mut_marker() {
    test_compile_err(
        ".[1,2,3][1]mut",
        "The `mut` marker can only be used when slicing, not when indexing",
        |code| TestSpan::of_substr(code, "mut"),
    );
}
