use crate::tests::{TestSpan, arr, substr, test_body};

#[test]
fn array_initializer() {
    #[rustfmt::skip]
    let code = |inner_ret_val| format!("
mut a := 0;
return .[
    {{ a += 1; 1 }},
    {{ a += 1; 2 }},
    return {inner_ret_val},
    {{ a += 1; 4 }},
]");

    test_body(code(".[a, a]")).error(
        "mismatched types: expected `[2]i64`; got `[4]{integer literal}`",
        |code| {
            let start = code.find(".[").unwrap();
            let end = code.rfind("]").unwrap() + 1;
            TestSpan::new(start, end)
        },
    );

    test_body(code(".[a, a, a, a]")).ok(arr([2i64; 4]));

    let code = "
mut a := 0;
return .[
    return .[a, a, a],
    { a += 1; 2 },
    { a += 1; 3 },
]";
    test_body(code).ok([0i64; 3]);
}

#[test]
fn array_initializer_short() {
    test_body(".[return .[1, 2, 3]; 3]").ok(arr([1_i64, 2, 3]));
    test_body(".[return .[1, 2, 3]; 10]").error(
        "mismatched types: expected `[3]{integer literal}`; got `[10]never`",
        substr!(".[return .[1, 2, 3]; 10]"),
    );

    test_body(".[3; return .[1, 2, 3]]")
        .error("Cannot evaluate array length at compile time", substr!("return .[1, 2, 3]"));
}

#[test]
fn dot() {
    test_body("(return 99).some_field").ok(99_i64);
}

#[test]
fn cast() {
    test_body("(return 99).as(u16)").ok(99_u16);

    test_body("(return 99).as(bool)").error(
        "mismatched types: expected `{integer literal}`; got `bool`",
        substr!("(return 99).as(bool)"),
    );
}

#[test]
fn index() {
    let code = "
myarr := .[1, 2, 3, 4, 5];
myarr[return 99]";
    test_body(code).error("Cannot index into array with `never`", substr!("return 99"));

    test_body("(return 99)[0]").ok(99_i64);
}

#[test]
fn binop() {
    test_body("(return 3) + 7").ok(3_i64);
}

#[test]
fn assign() {
    test_body("a := return 3; a = 7; a")
        .error("mismatched types: expected `never`; got `{integer literal}`", substr!("7"));
    test_body("a := return 3; a += 7; a")
        .error("mismatched types: expected `never`; got `{integer literal}`", substr!("7"));

    test_body("a := return 3; a = return 7;")
        .error("Cannot assign to `a`, as it is not declared as mutable", substr!("a = return 7"));
    test_body("a := return 3; a += return 7;")
        .error("Cannot assign to `a`, as it is not declared as mutable", substr!("a += return 7"));

    test_body("mut a := return 3; a = return 7;").ok(3_i64);
    test_body("mut a := return 3; a += return 7;").ok(3_i64);
}
