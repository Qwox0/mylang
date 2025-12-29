use crate::tests::{substr, test, test_body};

#[test]
fn incorrect_signedness_of_int_lit() {
    test("test :: -> u8 { -1 }")
        .error("Cannot apply unary operator `-` to type `u8`", substr!("-1"));

    test_body("a: u8 = if true { -1 } else 1;")
        .error("Cannot apply unary operator `-` to type `u8`", substr!("-1"));

    // type inference through AddrOf
    let code = "
f :: (p: *u32) -> p.*;
test :: -> f(&-1);";
    test(code).error("Cannot apply unary operator `-` to type `u32`", substr!("-1"));
}
