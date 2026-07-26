use crate::{ast, tests::*};

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
        let res = test(code).compile_no_err();
        let test_decl = res.one_stmt::<ast::Decl>();
        let test_fn = test_decl.init.unwrap().downcast::<ast::Fn>();
        assert_eq!(test_fn.ret_ty.unwrap().to_string(), expected_ty)
    }

    t("test :: (str: []u8, mut_str: []mut u8) -> { return mut_str; str }", "[]u8");
    t("test :: (str: []u8, mut_str: []mut u8) -> { return str; mut_str }", "[]u8");
    t("test :: (str: [][]u8, mut_str: []mut []mut u8) -> { return mut_str; str }", "[][]u8");
    t("test :: (str: [][]u8, mut_str: []mut []mut u8) -> { return str; mut_str }", "[][]u8");
}

#[test]
fn dont_break_return_type_on_finalize() {
    let code = "
dont_break_my_return_type :: -> ?*mut i64 { mut a := 1; &mut a };
_ :: -> []mut u8.(dont_break_my_return_type() orelse return null, 0); // this previously changed \
                the return type of `dont_break_my_return_type` to `*any`
test :: -> { dont_break_my_return_type() orelse return; };
";
    test(code).compile_no_err();
}
