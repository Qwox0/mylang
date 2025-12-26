use crate::{
    ast,
    tests::{TestSpan, substr, test},
    util::IteratorExt,
};

/// common_type([]T, []mut T) == []T (allowed)
/// ty_match(got=[]T, expected=[]mut T) => Error
#[test]
fn cannot_assign_more_general_type() {
    test("test :: (str: []u8) -> a: []mut u8 = str;")
        .error("mismatched types: expected `[]mut u8`; got `[]u8`", substr!("str";skip=1));
}

#[test]
fn can_infer_more_general_return_type() {
    fn t(code: &'static str, expected_ty: &str) {
        let res = test(code).load_prelude(false).compile_no_err();
        let test_decl = res.stmts().iter().expect_one().downcast::<ast::Decl>();
        let test_fn = test_decl.init.unwrap().downcast::<ast::Fn>();
        assert_eq!(test_fn.ret_ty.unwrap().to_string(), expected_ty)
    }

    t("test :: (str: []u8, mut_str: []mut u8) -> { return mut_str; str }", "[]u8");
    t("test :: (str: []u8, mut_str: []mut u8) -> { return str; mut_str }", "[]u8");
    t("test :: (str: [][]u8, mut_str: []mut []mut u8) -> { return mut_str; str }", "[][]u8");
    t("test :: (str: [][]u8, mut_str: []mut []mut u8) -> { return str; mut_str }", "[][]u8");
}

/// `common_type([][]mut T, []mut []T)` should return `[][]T`, but I don't want to allocate those
/// return types.
#[test]
fn error_no_allocation_in_common_type() {
    test("test :: (s1: [][]mut u8, s2: []mut []u8) -> { return s1; s2 }")
        .error(
            "The compiler currently cannot use `[][]u8` as the combined type of `[][]mut u8` and \
             `[]mut []u8`. Consider specifying `[][]u8` explicitly.",
            |_| TestSpan::ZERO,
        )
        .error("mismatched types: expected `[][]mut u8`; got `[]mut []u8`", substr!("s2";skip=1)); // always emit another error message?
    test("test :: (s1: [][]mut u8, s2: []mut []u8) -> { return s2; s1 }")
        .error(
            "The compiler currently cannot use `[][]u8` as the combined type of `[]mut []u8` and \
             `[][]mut u8`. Consider specifying `[][]u8` explicitly.",
            |_| TestSpan::ZERO,
        )
        .error("mismatched types: expected `[]mut []u8`; got `[][]mut u8`", substr!("s1";skip=1)); // always emit another error message?
}
