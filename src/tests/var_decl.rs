use crate::{
    ast::{self, AstKind},
    tests::{substr, test, test_body, test_parse},
    util::IteratorExt,
};

#[test]
fn error_invalid_lhs() {
    let err_msg = "expected a variable name, got an expression";

    test_body("{ 1 :: 2; }").error(err_msg, substr!("1"));

    test_body("{ a+b := 1; }").error(err_msg, substr!("a+b"));

    // The `mut` marker shouln't change the error.
    test_body("{ mut a+b := 1; }").error(err_msg, substr!("a+b"));

    test_body("{ mut a+b; }").error(err_msg, substr!("a+b"));
}

#[test]
fn static_var() {
    let code = "
get_num :: -> {
    static mut counter := 0;
    counter += 1;
    counter
}
test :: -> { get_num(); get_num(); get_num() }";
    let res = test(code).ok(3i64);
    assert!(res.llvm_ir().contains("@get_num.counter = internal global i64 0, align 8"));
}

#[test]
fn good_invalid_token_error() {
    // here we have to guess that `num` might be a decl (is this a good idea?)
    test_body("{ num i32 = 1; }")
        .error("expected `:`, `:=`, `::`, or `;`", substr!("num";.after()));

    // with the `mut` we now know that this is meant to be a decl
    test_body("{ mut num i32 = 1; }")
        .error("expected `:`, `:=`, or `::`, got an identifier", substr!("i32"));

    let err_msg = "expected `:`, `:=`, `,`, or `)`, got `#`";

    // an ident alone is a valid function parameter -> different error
    test("f :: (a # i32 = 1) -> {}").error(err_msg, substr!("#"));

    // `mut` marker shouln't change error message
    test("f :: (mut a # i32 = 1) -> {}")
        .error("expected `:`, `:=`, or `::`, got `#`", substr!("#")); // curently a worse error

    test("f :: (a: i32 = 1, b # i32 = 2) -> {}").error(err_msg, substr!("#"));

    // `mut` marker shouln't change error message
    test("f :: (a: i32 = 1, mut b # i32 = 2) -> {}").error(err_msg, substr!("#"));

    test_body("mut a = 1").error("expected `:`, `:=`, or `::`, got `=`", substr!("="));

    test_body("mut 1").error(
        "expected an identifier, `mut`, `rec`, `pub`, or `static`, got an integer literal",
        substr!("1"),
    );
}

#[test]
fn parse_colon() {
    fn test_fn_in_var_ty(code: &str) {
        let res = test_parse(code);
        let cos = res.stmts().iter().expect_one().downcast::<ast::Decl>();
        assert_eq!(cos.ident.sym.text(), "cos");
        let var_ty = cos.var_ty_expr.unwrap().downcast::<ast::Fn>();
        let var_ty_ret = var_ty.body.unwrap().downcast::<ast::Ident>();
        assert_eq!(var_ty_ret.sym.text(), "f64");
        assert_eq!(cos.init.unwrap().kind, AstKind::IntrinsicDirective);
    }
    test_fn_in_var_ty("cos : f64 -> f64 : #intrinsic \"llvm.cos.f64\";");
    test_fn_in_var_ty("cos : (f64) -> f64 : #intrinsic \"llvm.cos.f64\";");

    {
        let res = test_parse("v :: x : i32 : 3;");
        let v = res.stmts().iter().expect_one().downcast::<ast::Decl>();
        assert_eq!(v.ident.sym.text(), "v");
        assert!(v.var_ty_expr.is_none());
        let x = v.init.unwrap().downcast::<ast::Decl>();
        assert_eq!(x.ident.sym.text(), "x");
        assert_eq!(x.var_ty_expr.unwrap().downcast::<ast::Ident>().sym.text(), "i32");
        assert_eq!(x.init.unwrap().downcast::<ast::IntVal>().val, 3);
    }
}

#[test]
#[ignore = "not fixed yet"]
fn decl_in_weird_places_panic() {
    let code = r#"
        .[x := 1];
        N // <- this symbol lookup panics because `x := 1` increase cur_decl_pos
    "#;
    test_body(code).ok(());

    let code = r#"
        N :: 1;
        f :: (a: void) -> {}
        f(x := N);
        N // <- this symbol lookup panics because `x := N` increase cur_decl_pos
    "#;
    test_body(code).ok(());
}
