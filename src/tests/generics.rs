use crate::tests::{CompileTest, arr, assert_contains, substr, test, test_body};

#[test]
fn generic_function_call() {
    #[derive(Debug, PartialEq)]
    #[repr(C)]
    struct Sums(i8, i16, i32, i64, i128);

    let code = "
add :: (l: $Num, r: Num) -> Num l + r;
test :: -> {
    sum8 := add(10, r=20, Num=i8);         // Explicit type annotation
    sum16 := add(1000.as(i16), 2000);      // Inferred, based on input 1
    sum32 := add(100000, 200000.as(i32));  // Inferred, based on input 2
    sum64 := add(r=-1000, l=2000.as(i64)); // Inferred, based on named arg
    sum128: i128 = add(0, -1);             // Inferred, based on output
    struct { sum8: i8, sum16: i16, sum32: i32, sum64: i64, sum128: i128 }
        .{ sum8, sum16, sum32, sum64, sum128 }
};
";
    let res = test(code).ok(Sums(30_i8, 3000_i16, 300_000_i32, 1000_i64, -1_i128));
    for ty in ["i8", "i16", "i32", "i64", "i128"] {
        assert_contains!(res.llvm_ir(), "{ty} @add.{ty}(");
        #[cfg(debug_assertions)]
        assert_contains!(res.llvm_ir(), "%add = add {ty} %l, %r");
        #[cfg(not(debug_assertions))]
        assert_contains!(res.llvm_ir(), "%add = add {ty} %0, %1");
        assert_contains!(res.llvm_ir(), "call noundef {ty} @add.{ty}(");
    }
    drop(res);

    // finalized type == generic default
    let code = "
add :: (l: $Num, r: Num) -> l + r;
test :: -> add(10, 20);
";
    let res = test(code).ok(30_i64);
    assert_contains!(res.llvm_ir(), "@add.i64");
    drop(res);

    // error: not a normal parameter (see const_parameter)
    test_body("id :: (x: $T) -> x; id(5, u32)")
        .error("Got 2 positional arguments, but expected at most 1 arguments", substr!("u32"));
}

#[test]
fn generic_inner_types() {
    let code = r#"
MyStruct :: struct { a: i8, b: []u8 };
func :: (ptr: *$A, arr: [3]$B, opt: ?$C) -> {}
test :: -> {
    int: i32 = 10;
    s: MyStruct = .(-1, "Hello World");

    func(&int, .[1, 2, 3], Some(s));
    func(&s, u16.[1, 2, 3], Some(s));
}
"#;
    let res = test(code).compile_no_err();
    assert_contains!(res.llvm_ir(), "@func.i32.i64.MyStruct");
    assert_contains!(res.llvm_ir(), "@func.MyStruct.u16.MyStruct");
    drop(res);

    // cannot infer type inside optional
    let code = r#"
take_opt :: (opt: ?$A) -> {};
test :: -> take_opt(null);
"#;
    test(code).error("Cannot infer value of generic argument `A`", substr!("take_opt(null)"));
}

#[test]
#[ignore = "TODO"]
fn infer_generic_based_on_inferred_return_type() {
    let code = "
id :: (x: $T) -> /* inferred: T */ x;
test :: -> a: u16 = id(1);
";
    // Currently `1` is inferred as `i64`, which causes a mismatch with `u16`
    test(code).compile_no_err();
}

#[test]
fn generic_type_error() {
    let code = r#"
add_pi :: (val: $Num) -> val + 3.1415;
test :: -> {
    add_pi(1.0);
    add_pi(1);
    add_pi("Hello World");
    add_pi(struct { val := 1 }.{});
}
"#;
    // TODO: also print call which generated instantiation
    test(code)
        .error("mismatched types (left: `i64`, right: `{float}`)", substr!("+"))
        .error("mismatched types (left: `[]u8`, right: `{float}`)", substr!("+"))
        .error("mismatched types (left: `struct{val:i64}`, right: `{float}`)", substr!("+"));
}

#[test]
fn generic_mismatches() {
    let code = r#"
f :: (a: $T, b: T) -> {};
test :: -> {
    f(1, "Hello");
}"#;
    test(code).error(
        "mismatched types: expected `$T` (inferred as `{integer}`); got `[]u8`",
        substr!("\"Hello\""),
    );

    /*// TODO
    let code = "
f :: (a: $T) -> {};
test :: -> {
    empty: ?*i32 = null;
    f(empty, T=*i32);
}";
    test(code).error(
        "mismatched types: expected `$T` (inferred as `*i32`); got `?*i32`",
        substr!("empty"; skip=1),
    );
    */

    // allow coercion if the generic is not specified explicitly
    let code = "
f :: (a: $T, b: T) -> {};
test :: -> {
    val: i32 = 1;
    empty: ?*i32 = null;
    f(&val, empty);
}
";
    test(code).compile_no_err();
}

#[test]
fn duplicate_generic_def() {
    // TODO: better error?
    test("f :: (a: $T, b: $T) -> {}")
        .error("duplicate parameter 'T'", substr!("T"; skip=1))
        .info("first definition of 'T'", substr!("T"));
}

#[test]
#[ignore = "is this a good idea?"]
fn parameter_generic_name_collision() {
    // TODO: better error?
    test("f :: (some_name: $some_name) -> {}")
        .error("duplicate parameter 'some_name'", substr!("some_name"; skip=1))
        .info("first definition of 'some_name'", substr!("some_name"));
}

#[test]
#[ignore = "TODO"]
fn infer_through_generic() {
    let code = "
MyStruct :: struct { val: i32 };
func :: (a: $T, b: T) -> {}
test :: -> {
    func(MyStruct.(1), .(2));
    func(.(1), MyStruct.(2));
        // a: T is unknown => a is unknown
        // b: b is known => T is set
        // a: T is known => a is known
}";
    test(code).compile_no_err();
}

#[test]
#[ignore = "TODO"]
fn const_positional_parameter() {
    // type
    let code = "
sizeof :: ($ty) -> #sizeof(ty);
test :: -> sizeof(i32);
";
    test(code).ok(4);

    // number
    let code = "
arr_splat :: (val: $T, len: $N) -> [N]T {
    mut arr: [N]T;
    for idx in 0..N do arr[idx] = default;
    arr
}
test :: -> arr_splat(123, 5);
";
    let res = test(code).ok(arr([123; 5]));
    drop(res);
}

#[test]
fn non_type_const_parameter() {
    /* TODO
    init_arr :: (default: $T) -> [$N]T {
        mut arr: [N]T;
        for idx in 0..N do arr[idx] = default;
        arr
    }
    */
    let code = r#"
arr_len :: (arr: [$N]$T) -> u64 N;
test :: ->
    arr_len(.[10, 20, 30, 40, 50])
    + arr_len(u16.[])
    + arr_len(.["Hello", "World", "!"])
;"#;
    let res = test(code).ok(8_usize);
    assert_contains!(res.llvm_ir(), "@arr_len.i64.5");
    assert_contains!(res.llvm_ir(), "@arr_len.u16.0");
    assert_contains!(res.llvm_ir(), "@\"arr_len.[]u8.3\"");
    drop(res);
}

// TODO: test :: -> MyStruct.(1, "Hello World", &test);

#[test]
fn generic_data_struct() {
    // Positional initializer
    let code = r#"
MyStruct :: struct($A, $B, $C) {
    a: A,
    b: []B,
    c: ?*C,
}
static arr := u16.[1, 2, 3];
test :: -> MyStruct.(1, "Hello World", &arr);
"#;
    let res = test(code).compile_no_err();
    assert_contains!(res.llvm_ir(), "[3 x i16] [i16 1, i16 2, i16 3]");
    assert_contains!(res.llvm_ir(), "sret({{ i64, {{ ptr, i64 }}, ptr }})");
    drop(res);

    /* TODO
    // Named Initializer
    let code = r#"
MyStruct :: struct($A, $B, $C) {
    a: A,
    b: []B,
    c: ?*C,
}
static arr := u16.[1, 2, 3];
test :: -> MyStruct.{
    a = 1,
    b = "Hello World",
    c = &arr,
};
"#;
    test(code).compile_no_err();
    */
}
