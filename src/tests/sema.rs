use crate::tests::{optional::some, *};

#[test]
fn cycle_detection() {
    let code = "
a :: -> b();
b :: -> a();
a2 :: -> b2();
b2 :: -> a2();
not_part_of_cycle :: -> struct {}.MISSING;
";
    test(code)
        .error("no associated constant `MISSING` on type `struct{}`", substr!("MISSING"))
        .error("cycle(s) detected:", |_| TestSpan::ZERO);
}

#[test]
fn error_cycle_in_struct() {
    let code = "
MyStruct :: struct {
    a: I;
    arr: [NEW.a]I = .[0; NEW.a];
    NEW :: MyStruct.(7);
    not_part_of_cycle :: -> struct {}.MISSING;
};
I :: u64;
test :: -> MyStruct.NEW.a;";
    test(code)
        .error("no associated constant `MISSING` on type `struct{}`", substr!("MISSING"))
        .error("cycle(s) detected:", |_| TestSpan::ZERO); // TODO: test full cycle report
}

#[test]
fn correctly_handle_error_in_later_cycles() {
    let code = "
MyStruct :: struct { arr := .[7; LEN]; };
CONST :: MyStruct.();
LEN :: \"\";
test :: -> CONST.arr[4];";
    test(code).error("mismatched types: expected `u64`; got `[]u8`", substr!("LEN"));
}

#[test]
fn recursive_type() {
    let code = "
A :: struct { a: *A };
";
    test(code).compile_no_err();
}

#[test]
fn indirectly_recursive_types() {
    let code = "
A :: struct { b: *B };
B :: struct { a: *A };
";
    test(code).compile_no_err();
}

#[test]
fn indirectly_recursive_nested_types() {
    let code = "
A :: struct {
    B :: struct { a: *A };
    b: *B;
};";
    test(code).compile_no_err();

    let code = "
A :: struct {
    b: ?*B;
    X : A : .(null); // must pause on `A` because type_check would fail otherwise
    Y :: A.(null); // must pause on `A`
    B :: struct { a: *A };
};";
    test(code).compile_no_err();
}

#[test]
fn guessing_type_inference_on_mismatch() {
    let code = "
take_f :: (f: *(x: int, y: int) -> int) -> f.*(1, 2);
test :: -> take_f(/* missing '&' */ (x, y) -> x + y);
";
    test(code).with_prelude().error(
        "mismatched types: expected `*(x:int,y:int)->int`; got `(x:i64,y:i64)->i64`",
        substr!("(x, y) -> x + y"),
    );
}

#[test]
#[ignore = "todo"]
fn optional_inner_type_variance() {
    // covariant
    let code = "
    insert_10 :: (opt: *mut ?i32) -> opt.* = Some(10);

    opt_never: ?never = null;
    mut opt_coerced: ?i32 = opt_never; // copy => coercion (/covariance) ok
    insert_10(&mut opt_coerced);
    opt_coerced
";
    test_body(code).ok(some(10_i32));

    // invariant
    let code = "
    insert_10 :: (opt: *mut ?i32) -> opt.* = Some(10);

    mut opt_never: ?never = null;
    opt_coerced: *mut ?i32 = &mut opt_never; // ptr cast => coercion (/covariance) NOT ok
    insert_10(opt_coerced);
    opt_coerced
";
    test_body(code).ok(some(10_i32)); // TODO: error
}

#[test]
fn finalize_doesnt_change_finalized_types() {
    let code = r#"
get_u8 :: (slice: []u8, idx: u64) -> ?u8 {
    if idx < slice.len then Some(slice[idx]) else null
}
min :: (a: u64, b: u64) -> if a < b then a else b;

trim_partial_prefix :: (name: *mut []mut u8) -> {
    name_char := get_u8(name.*, 0); // finalize_ty changes `name.*` to `[]u8`
    max_idx := min(0, name.*.len); // `name.*` expects `[]mut u8`
}
    "#;
    test(code).compile_no_err();

    // Is it possible to have `*mut {integer}` which is then finalized to `*i64`, instead of
    // `*mut i64`?
}

#[test]
#[ignore = "todo"]
fn analyze_after_unknown_associated_const() {
    let code = "
MyStruct :: struct {};
test :: -> {
    a := MyStruct.UNKNOWN;
    b: []u8 = 1;
}
";
    test(code)
        .error("no associated constant `UNKNOWN` on type `MyStruct`", substr!("UNKNOWN"))
        .error("mismatched types: expected `[]u8`; got `{integer}`", substr!("1"));
}
