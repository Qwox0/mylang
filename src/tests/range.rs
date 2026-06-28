use crate::tests::*;

#[test]
fn inclusive_range_without_end_bound() {
    for code in ["..=", "123..="] {
        test_body(code).error("an inclusive range must have an end bound", substr!("..="));
    }
}
