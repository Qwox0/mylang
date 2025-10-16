use crate::tests::{arr, substr, test, test_body};

/// more tests about associated constants: [`crate::tests::associated_consts`]
#[test]
fn const_on_struct() {
    let code = "
MyStruct :: struct { field: u8, MY_CONST : u64 : 3 };
test :: -> {
    arr: [MyStruct.MY_CONST]u8 = .[1, 2, 3];
    return arr;
}";
    test(code).ok(arr([1u8, 2, 3]));

    let code = "
MyStruct :: struct { field: u8, MY_CONST : u64 : 10 };
test :: -> {
    arr: [MyStruct.MY_CONST]u8 = .[1, 2, 3];
    return arr;
}";
    test(code).error("mismatched types: expected [10]u8; got [3]u8", substr!(".[1, 2, 3]"));
}

#[test]
fn const_struct() {
    let code = "
MyStruct :: struct { text: []u8, number: u64 };
CONST_STRUCT :: MyStruct.(\"Hello World\", 3);
test :: -> CONST_STRUCT;
";
    let res = test(code).ok(("Hello World", 3u64));
    assert!(res.llvm_ir().contains("{ { ptr, i64 } { ptr @0, i64 11 }, i64 3 }"));
    assert!(!res.llvm_ir().contains("alloca"));
    drop(res);

    let code = "
MyStruct :: struct { text: []u8, inner: struct { number: u64 }};
CONST_STRUCT :: MyStruct.(\"Hello World\", .(3));
test :: -> CONST_STRUCT;
";
    let res = test(code).ok(("Hello World", 3u64));
    assert!(res.llvm_ir().contains("{ { ptr, i64 } { ptr @0, i64 11 }, { i64 } { i64 3 } }"));
    assert!(!res.llvm_ir().contains("alloca"));
    drop(res);

    let code = "
MyStruct :: struct { text: []u8, inner: struct { number: u64 }};
CONST_STRUCT :: MyStruct.{ text = \"Hello World\", inner = .{ number = 3 }};
test :: -> {
    arr: [CONST_STRUCT.inner.number]u8 = .[1, 2, 3];
    return CONST_STRUCT;
}";
    let res = test(code).ok(("Hello World", 3u64));
    drop(res);

    let code = "
MyStruct :: struct { text: []u8, inner: struct { number: u64 }};
CONST_STRUCT :: MyStruct.(\"Hello World\", .(10));
test :: -> {
    arr: [CONST_STRUCT.inner.number]u8 = .[1, 2, 3];
}";
    test(code).error("mismatched types: expected [10]u8; got [3]u8", substr!(".[1, 2, 3]"));
}

#[test]
fn const_array() {
    let code = "
CONST_ARR :: u64.[1, 2, 3, 4];
CONST_ARR2 : [CONST_ARR[3]]u64 : CONST_ARR;
test :: -> CONST_ARR;
";
    let res = test(code).ok([1u64, 2, 3, 4]);
    assert!(res.llvm_ir().contains("[4 x i64] [i64 1, i64 2, i64 3, i64 4]"));
    assert!(!res.llvm_ir().contains("alloca"));
    drop(res);

    let code = "
MyStruct :: struct { text: []u8, number: u64 };
CONST_ARR :: MyStruct.[
    .(\"Hello\", 1),
    .(\"World\", 2),
];
test :: -> {
    CONST_ARR
}";
    let res = test(code).ok([("Hello", 1u64), ("World", 2)]);
    assert!(res.llvm_ir().contains(
        "[2 x { { ptr, i64 }, i64 }] [{ { ptr, i64 }, i64 } { { ptr, i64 } { ptr @0, i64 5 }, i64 \
         1 }, { { ptr, i64 }, i64 } { { ptr, i64 } { ptr @1, i64 5 }, i64 2 }]"
    ));
    assert!(!res.llvm_ir().contains("alloca"));
    drop(res);
}

#[test]
fn codegen_use_constant_aggregate() {
    // manual stack allocation
    let code = "
CONST :: .[7; 10];
test :: -> {
    arr := CONST;
    arr[1]
}";
    test(code).ok(7i32);

    // currently uses an automatic stack allocation
    let code = "
CONST :: .[7; 10];
test :: -> CONST[1];";
    let res = test(code).ok(7i32);
    //assert!(!res.llvm_ir().contains("alloca")); // TODO
    drop(res);

    // correct codegen for structs (see build_struct_access)
    let code = "
MyStruct :: struct { a: i32 };
CONST :: MyStruct.{ a=7 };
test :: -> CONST.a;";
    test(code).ok(7i32);
}

#[test]
fn prefer_type_error_over_non_const_error() {
    test("test :: (len: []u8) -> { .[1; len] }")
        .error("mismatched types: expected u64; got []u8", substr!("len";skip=1));
    test("test :: (len: u64) -> { .[1; len] }")
        .error("Array length must be known at compile time", substr!("len";skip=1));
}

#[test]
fn unfinalized_consts() {
    let code = "
MY_INT :: 3;
a: i64 = MY_INT;
b: u16 = MY_INT;
c := MY_INT;
a + b.as(i64) + c";
    test_body(code).ok(9i64);

    let code = "
f :: (x: u16) -> x * 2;
A :: 5;
test :: -> f((A + 1).as(u16)); // `finalize_arg_type` during codegen is still valid, as it only
                               // changes the type of the `A` ident, not the `A` decl.
test2 :: -> i32 A + 1;
";
    let res = test(code).ok(12u16);
    assert!(res.llvm_ir().contains("@f(i16 noundef 6)"));
    assert!(res.llvm_ir().contains("ret i32 6"));
}

#[test]
fn allow_cast_on_unfinalized_consts() {
    let code = "
MY_INT :: 3;
take_ptr :: (ptr: *any) -> {};
take_ptr(xx MY_INT.as(i8));";
    let res = test_body(code).ok(());
    // `int_lit` -> cast to `i8` -> autocast to `*any`
    assert!(res.llvm_ir().contains("take_ptr(ptr noundef inttoptr (i8 3 to ptr))"));
    drop(res);

    let code = "
MY_INT :: 3;
take_ptr :: (ptr: *any) -> {};
take_ptr(xx MY_INT);";
    let res = test_body(code).ok(());
    // `int_lit` -> finalize to `i64` -> autocast to `*any`
    assert!(res.llvm_ir().contains("take_ptr(ptr noundef inttoptr (i64 3 to ptr))"));
}

#[test]
fn fix_llvm_ty_mismatch_for_const_struct_arg() {
    let code = "
MyStruct :: struct { val: u8 };
CONST :: MyStruct.(1);
take_struct_val :: (val: MyStruct) -> {};
test :: -> take_struct_val(CONST);";
    test(code).ok(());

    let code = "
MyStruct :: struct { val: u8, val2: u8, val3: i64 };
CONST :: MyStruct.(1, 2, 3);
take_struct_val :: (val: MyStruct) -> {};
test :: -> take_struct_val(CONST);";
    test(code).ok(());
}

#[test]
fn error_use_runtime_var_in_const() {
    let code = "
test :: -> {
    a := 1;
    A :: a;
}";
    test(code).error("Cannot access a non-constant symbol at compile time", substr!("a";skip=1));
}
