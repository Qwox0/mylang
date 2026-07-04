use crate::tests::*;

#[test]
fn expr_type() {
    let code = "
MyStruct :: struct { x := DEFINED_LATER };
test :: -> MyStruct.{};
DEFINED_LATER :: EVEN_LATER;
EVEN_LATER :: 123;
";
    test(code).ok(123);
}

#[test]
fn associated_const() {
    let code = "
MyStruct :: struct {};
test :: -> MyStruct.MISSING;
";
    test(code).error("no associated constant `MISSING` on type `MyStruct`", substr!("MISSING"));

    let code = "
MyStruct :: struct {};
test :: -> MyStruct.DEFINED_LATER;
MyStruct.DEFINED_LATER :: 123;
";
    test(code).ok(123);
}

#[test]
fn dot() {
    let code = "
MyStruct :: struct {};
test :: (m: MyStruct) -> m.missing;
";
    test(code).error("no field `missing` on type `MyStruct`", substr!("missing"));

    // fields cannot be defined later, but methods can

    let code = "
MyStruct :: struct {};
test :: (m: MyStruct) -> m.later_defined();
MyStruct.later_defined :: (self: MyStruct) -> 123;
";
    test(code).ok(123);
}
