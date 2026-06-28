use crate::tests::*;

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
    test(code).error("mismatched types: expected `[10]u8`; got `[3]u8`", substr!(".[1, 2, 3]"));
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
    test(code).error("mismatched types: expected `[10]u8`; got `[3]u8`", substr!(".[1, 2, 3]"));
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
        .error("mismatched types: expected `u64`; got `[]u8`", substr!("len";skip=1));
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

#[test]
fn const_cast_num_to_ptr() {
    test("NULL :: 0.as(*never); test :: -> ?*u8 NULL;").ok(std::ptr::null::<()>());
}

#[test]
fn const_addr_of() {
    let res = test_body("a :: 1 + 1; const_addr :: &a; const_addr.*").ok(2_i64);
    assert!(res.llvm_ir().contains("@test.a = private unnamed_addr constant i64 2, align 8"));
    assert!(res.llvm_ir().contains("load i64, ptr @test.a, align 8"));
    drop(res);

    let res = test_body("static a := 1 + 1; const_addr :: &a; const_addr.*").ok(2_i64);
    assert!(res.llvm_ir().contains("@test.a = internal constant i64 2, align 8"));
    assert!(res.llvm_ir().contains("load i64, ptr @test.a, align 8"));
    drop(res);

    let res = test_body("mut static a := 1 + 1; const_addr :: &mut a; const_addr.*").ok(2_i64);
    assert!(res.llvm_ir().contains("@test.a = internal global i64 2, align 8"));
    assert!(res.llvm_ir().contains("load i64, ptr @test.a, align 8"));
    drop(res);

    test_body("a := 1 + 1; const_addr :: &a; const_addr.*").error(
        "Can only take the address of a static or constant value at compile time",
        substr!("&a"),
    );
    test_body("const_addr :: &1; const_addr.*").error(
        "Can only take the address of a static or constant value at compile time",
        substr!("&1"),
    );
}

#[test]
fn codegen_duplicate_const_alloc() {
    let code = "
a :: 1 + 1;
const_addr_f32 : *f32 : &a;
const_addr_i32 : *i32 : &a;
b := const_addr_f32.*;          // creates first global const
c := const_addr_i32.*;          // creates second global const
c
    ";
    let res = test_body(code).ok(2_i32);
    assert!(
        res.llvm_ir()
            .contains("@test.a = private unnamed_addr constant float 2.000000e+00, align 4")
    );
    assert!(
        res.llvm_ir()
            .contains("@test.a.1 = private unnamed_addr constant i32 2, align 4")
    );
    drop(res);

    let code = "
a :: 1 + 1;
const_addr_1 : *f32 : &a;
const_addr_2 : *f32 : &a;
b := const_addr_1.*;            // creates global const
c := const_addr_2.*;            // uses global const
c
    ";
    let res = test_body(code).ok(2_f32);
    assert!(
        res.llvm_ir()
            .contains("@test.a = private unnamed_addr constant float 2.000000e+00, align 4")
    );
    assert!(!res.llvm_ir().contains("@test.a.1"));
    drop(res);

    // theoretical duplicate const alloc problem. Doesn't exist in reality because constant values
    // must not contain cycles
    let code = "
MyStruct :: struct { val: *MyStruct };
val :: MyStruct.{ val=&val };   // 2: create global const for `val`
const_addr :: &val;
test :: -> const_addr.*;        // 1: doesn't find global for `val` > compiles `val` init > 2 >
                                //    global for `val` exists now > don't create second global!
    ";
    let res = test(code).error("cycle(s) detected:", |_| TestSpan::ZERO);
    drop(res);
}

/// ```llvm
/// @alloc_ddadadacd92a4149d036d1533e90cb9e = private unnamed_addr constant [8 x i8] c"\0A\00\00\00\00\00\00\00", align 8
///
/// ; [...]
///
/// define void @example[d797943fde5fa4b6]::test2() unnamed_addr {
/// start:
///   call void @example[d797943fde5fa4b6]::f2(i64 10)
///   call void @example[d797943fde5fa4b6]::f(ptr align 8 @alloc_ddadadacd92a4149d036d1533e90cb9e)
///   call void @example[d797943fde5fa4b6]::f(ptr align 8 @alloc_ddadadacd92a4149d036d1533e90cb9e)
///   ret void
/// }
/// ```
#[test]
#[ignore = "todo"]
fn addr_of_const() {
    let code = "
C: i64 : 10;

f :: (x: *i64) -> {}
f2 :: (x: i64) -> {}

test :: -> {
    f2(C);
    f(&C);
    f(&C);
}
        ";
    let res = test(code).compile_no_err();
    assert!(!res.llvm_ir().contains("alloca"));
    drop(res);
}
