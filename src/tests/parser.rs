use crate::tests::{substr, test};

#[test]
fn error_unterminated_block_comment() {
    test("f :: -> { /* */ }").compile_no_err();

    test("f :: -> { /* }")
        .error("unterminated block comment: missing trailing `*/`", substr!("/*"));

    let code = "
    /*
    //*/
";
    test(code).error("unterminated block comment: missing trailing `*/`", substr!("/*"));

    test("f :: -> { a := /* }")
        .error("unterminated block comment: missing trailing `*/`", substr!("/*"));

    test("{ (struct {} /* ) } ")
        .error("unterminated block comment: missing trailing `*/`", substr!("/*"));

    test("f :: (a: i32, b: i32, /* ")
        .error("unterminated block comment: missing trailing `*/`", substr!("/*"));

    test("f :: MyStruct.{/* } ")
        .error("unterminated block comment: missing trailing `*/`", substr!("/*"));

    test("test :: my_func(1, 2, /* ) ")
        .error("unterminated block comment: missing trailing `*/`", substr!("/*"));
}
