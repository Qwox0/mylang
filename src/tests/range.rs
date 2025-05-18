use super::{TestSpan, test_compile_err};

#[test]
fn inclusive_range_without_end_bound() {
    for code in ["..=", "123..="] {
        test_compile_err(code, "an inclusive range must have an end bound", |code| {
            TestSpan::of_substr(code, "..=")
        });
    }
}
