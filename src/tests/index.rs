use crate::tests::{substr, test_body};

#[test]
fn invalid_index_type() {
    test_body(".[1,2,3][1.0]").error("Cannot index into array with `f64`", substr!("1.0"));
}

#[test]
fn invalid_slice_mut_marker() {
    test_body(".[1,2,3][1]mut")
        .error("The `mut` marker can only be used when slicing, not when indexing", substr!("mut"));
}
