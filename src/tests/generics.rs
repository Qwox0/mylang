use crate::tests::{CompileTest, assert_contains, substr, test, test_body};

#[test]
fn generic_function_call() {
    #[derive(Debug, PartialEq)]
    #[repr(C)]
    struct Sums(i8, i16, i32, i64, i128);

    let code = "
add :: (l: $Num, r: Num) -> l + r;
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
    assert!(res.llvm_ir().contains("@add.i64"));
    drop(res);

    // error: not a normal parameter (see const_parameter)
    test_body("id :: (x: $T) -> x; id(5, u32)")
        .error("Got 2 positional arguments, but expected at most 1 arguments", substr!("u32"));
}

#[test]
fn generic_inner_types() {
    let code = r#"
func :: (ptr: *$A, arr: [3]$B, opt: ?$C) -> {}
test :: -> {
    int: i32 = 10;
    s: struct { a: i8, b: []u8 } = .(-1, "Hello World");

    func(&int, .[1, 2, 3], Some(s));
    func(&s, .[1, 2, 3], Some(s));
}
"#;
    test(code).compile_no_err();

    // cannot infer type inside optional
    let code = r#"
take_opt :: (opt: ?$A) -> {};
test :: -> take_opt(null);
"#;
    test(code).error("Cannot infer value of generic argument `A`", substr!("take_opt(null)"));
}

#[test]
fn generic_mismatches() {
    /*
    let code = "
f :: (a: $T) -> {};
test :: -> {
    empty: ?*i32 = null;
    f(empty, T=*i32);
}";
    test(code).error("todo", substr!("todo"));
    */

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
    let code = "
sizeof :: ($ty) -> #sizeof(ty);
test :: -> sizeof(i32);
";
    test(code).ok(4);
}

#[test]
#[ignore = "TODO"]
fn generic_number_val() {
    let code = "
init_arr :: (default: $T) -> [$N]T {
    mut arr: [N]T;
    for idx in 0..N do arr[idx] = default;
    arr
}";
    test(code).compile_no_err();
}

// TODO: generic structs
// TODO: error: param name == generic name
