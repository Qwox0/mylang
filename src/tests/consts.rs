use crate::tests::{TestSpan, array::CRetArr, jit_run_test_raw, test_compile_err_raw};

#[test]
fn const_on_struct() {
    let code = "
MyStruct :: struct { field: u8, MY_CONST : u64 : 3 };
test :: -> {
    arr: [MyStruct.MY_CONST]u8 = .[1, 2, 3];
    return arr;
}";
    debug_assert_eq!(jit_run_test_raw::<CRetArr<u8, 3>>(code).ok().val, [1, 2, 3]);

    let code = "
MyStruct :: struct { field: u8, MY_CONST : u64 : 10 };
test :: -> {
    arr: [MyStruct.MY_CONST]u8 = .[1, 2, 3];
    return arr;
}";
    test_compile_err_raw(code, "mismatched types: expected [10]u8; got [3]u8", |code| {
        TestSpan::of_substr(code, ".[1, 2, 3]")
    });
}

#[test]
fn const_struct() {
    let code = "
MyStruct :: struct { text: []u8, number: u64 };
CONST_STRUCT :: MyStruct.(\"Hello World\", 3);
test :: -> CONST_STRUCT;
";
    let res = jit_run_test_raw::<(&'static str, u64)>(code);
    assert_eq!(*res.ok(), ("Hello World", 3));
    assert!(res.llvm_ir().contains("{ { ptr, i64 } { ptr @0, i64 11 }, i64 3 }"));
    assert!(!res.llvm_ir().contains("alloca"));
    drop(res);

    let code = "
MyStruct :: struct { text: []u8, inner: struct { number: u64 }};
CONST_STRUCT :: MyStruct.(\"Hello World\", .(3));
test :: -> CONST_STRUCT;
";
    let res = jit_run_test_raw::<(&'static str, u64)>(code);
    assert_eq!(*res.ok(), ("Hello World", 3));
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
    let res = jit_run_test_raw::<(&'static str, u64)>(code);
    assert_eq!(*res.ok(), ("Hello World", 3));
    drop(res);

    let code = "
MyStruct :: struct { text: []u8, inner: struct { number: u64 }};
CONST_STRUCT :: MyStruct.(\"Hello World\", .(10));
test :: -> {
    arr: [CONST_STRUCT.inner.number]u8 = .[1, 2, 3];
}";
    test_compile_err_raw(code, "mismatched types: expected [10]u8; got [3]u8", |code| {
        TestSpan::of_substr(code, ".[1, 2, 3]")
    });
}

#[test]
fn const_array() {
    let code = "
CONST_ARR :: u64.[1, 2, 3, 4];
CONST_ARR2 : [CONST_ARR[3]]u64 : CONST_ARR;
test :: -> CONST_ARR;
";
    let res = jit_run_test_raw::<[u64; 4]>(code);
    assert_eq!(res.ok(), &[1, 2, 3, 4]);
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
    let res = jit_run_test_raw::<[(&'static str, u64); 2]>(code);
    assert_eq!(res.ok(), &[("Hello", 1), ("World", 2)]);
    assert!(res.llvm_ir().contains(
        "[2 x { { ptr, i64 }, i64 }] [{ { ptr, i64 }, i64 } { { ptr, i64 } { ptr @0, i64 5 }, i64 \
         1 }, { { ptr, i64 }, i64 } { { ptr, i64 } { ptr @1, i64 5 }, i64 2 }]"
    ));
    assert!(!res.llvm_ir().contains("alloca"));
    drop(res);
}
