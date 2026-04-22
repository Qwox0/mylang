use crate::tests::{NewTest, substr, test, test_body};
use std::fmt;

fn test_all_coercion_sites(
    expr: impl fmt::Display,
    source_ty: impl fmt::Display,
    coerced_ty: impl fmt::Display,
) -> NewTest {
    let code = format!(
        "
coerce :: (coerced: {coerced_ty}) -> {{}};
Coerce :: struct {{ coerced: {coerced_ty} }}
MyEnum :: enum {{ Coerce({coerced_ty}) }}

test :: -> {{
    a: {source_ty};

    coerce({expr});
    mut ptr: {coerced_ty} = {expr};
    ptr = {expr};
    {coerced_ty}.[{expr}];
    {coerced_ty}.[{expr}; 5];
    Coerce.{{ coerced={expr} }};
    Coerce.({expr});
    MyEnum.Coerce({expr});
    // _: ?{coerced_ty} = Some({expr}); // TODO: allow optional coercion in Some?
    ((a: {source_ty}) -> {coerced_ty} {{ {expr} }})(a);
    ((a: {source_ty}) -> {coerced_ty} {{ return {expr}; }})(a);
}}"
    );
    test(code)
}

#[test]
fn addr_of_to_optional() {
    test_all_coercion_sites("&a", "i32", "?*i32").ok(());
}

#[test]
fn deref_ptr_to_optional() {
    test_all_coercion_sites("a.*", "**i32", "?*i32").ok(());

    // In this case the optional is not produces by type coercion
    test_all_coercion_sites("a.*", "*?i32", "?i32").ok(());
}

#[test]
fn cast() {
    let code = "
test :: () -> ?*mut u8 {
    1.as(?*mut u8) orelse return null
}";
    test(code).ok(1_u64);
}

// TODO: is this coercion a good idea?
#[test]
fn coerce_orelse_rhs() {
    let code = "
    NonNullEnum :: enum { A = 123, B };
    a := NonNullEnum.A;
    opt: ??NonNullEnum = null;
    opt orelse a
";
    test_body(code).ok(123_u8);
}

#[test]
fn cannot_coerce_to_multiple_optionals() {
    // `NonNull` -> `?NonNull` is allowed; `?NonNull` -> `??NonNull` is not allowed

    // ty_match
    test_body("NonNullEnum :: enum { A = 123, B }; a: ??NonNullEnum = Some(NonNullEnum.A);").error(
        "mismatched types: expected `??NonNullEnum`; got `?NonNullEnum`",
        substr!("Some(NonNullEnum.A)"),
    );

    test_body("NonNullEnum :: enum { A = 123, B }; a: ?NonNullEnum = NonNullEnum.A; a").ok(123_u8);

    // common_type
    test_body(
        "NonNullEnum :: enum { A = 123, B, C, D };
        a := .[Some(Some(NonNullEnum.A)), Some(NonNullEnum.B)];
        a := .[Some(NonNullEnum.C), Some(Some(NonNullEnum.D))];",
    )
    .error(
        "mismatched types: expected `??NonNullEnum`; got `?NonNullEnum`",
        substr!("Some(NonNullEnum.B)"),
    )
    .error(
        "mismatched types: expected `?NonNullEnum`; got `??NonNullEnum`",
        substr!("Some(Some(NonNullEnum.D))"),
    );
}
