//! # About ABI
//!
//! * floats are are returned in SIMD registers. => don't convert them to integers.
//! * <https://discourse.llvm.org/t/another-struct-return-question/35099/2>
//! * <https://learn.microsoft.com/en-us/cpp/build/x64-software-conventions?view=msvc-170>
//!
//! Special cases:
//! llvm struct  -> llvm return type
//! { f32, f32 } -> <2 x f32>
//! { f32, f32, f32 } -> { <2 x f32>, f32 }
//! { i32, i32 } -> i64
//! { i32, f32 } -> i64
//! { f32, i32 } -> i64

use crate::tests::jit_run_test;

macro_rules! test_struct_return {
    ($test_name:ident : { $($field:ident : $ty:ty = $val:expr),* $(,)? }) => {
        #[test]
        fn $test_name() {
            #[derive(Debug, PartialEq)]
            #[repr(C)]
            struct Struct { $($field: $ty,)* }
            let out = jit_run_test!(concat!("struct{", stringify!($( $field: $ty = $val, )* ), "}.{}") => Struct).unwrap();
            assert_eq!(out, Struct { $( $field : $val ),* });
        }
    };
}

test_struct_return!(return_struct_i32_i32 : {
    a: i32 = -17,
    b: i32 = 5,
});
test_struct_return!(return_struct_i64_i64 : {
    a: i64 = -17,
    b: i64 = 5,
});
test_struct_return!(return_struct_i32_i64 : {
    a: i32 = -17,
    b: i64 = 5,
});
test_struct_return!(return_struct_i64_i32 : {
    a: i64 = -17,
    b: i32 = 5,
});

test_struct_return!(return_struct_f32_i32 : {
    a: f32 = 10.123,
    b: i32 = -17,
});
test_struct_return!(return_struct_f64_i64 : {
    a: f64 = 10.123,
    b: i64 = -17,
});
test_struct_return!(return_struct_f32_i64 : {
    a: f32 = 10.123,
    b: i64 = -17,
});
test_struct_return!(return_struct_f64_i32 : {
    a: f64 = 10.123,
    b: i32 = -17,
});

test_struct_return!(return_struct_i32_f32 : {
    a: i32 = -17,
    b: f32 = 10.123,
});
test_struct_return!(return_struct_i64_f64 : {
    a: i64 = -17,
    b: f64 = 10.123,
});
test_struct_return!(return_struct_i32_f64 : {
    a: i32 = -17,
    b: f64 = 10.123,
});
test_struct_return!(return_struct_i64_f32 : {
    a: i64 = -17,
    b: f32 = 10.123,
});

test_struct_return!(return_struct_f32_f32 : {
    a: f32 = -17.5,
    b: f32 = 10.123,
});
test_struct_return!(return_struct_f64_f64 : {
    a: f64 = -17.5,
    b: f64 = 10.123,
});
test_struct_return!(return_struct_f32_f64 : {
    a: f32 = -17.5,
    b: f64 = 10.123,
});
test_struct_return!(return_struct_f64_f32 : {
    a: f64 = -17.5,
    b: f32 = 10.123,
});

test_struct_return!(return_struct_f32_f32_f32 : {
    a: f32 = -17.5,
    b: f32 = 10.123,
    c: f32 = 1.6180339887,
});

test_struct_return!(return_struct_i64_i64_i64 : {
    a: i64 = 100,
    b: i64 = 200,
    c: i64 = 300,
});

#[test]
fn return_nested_struct() {
    #[derive(Debug, PartialEq)]
    #[repr(C)]
    struct Inner {
        a: i8,
        b: i16,
    }

    #[derive(Debug, PartialEq)]
    #[repr(C)]
    struct Struct {
        a: i8,
        b: Inner,
        c: i32,
    }

    let out =
        jit_run_test!("struct { a: i8 = 1, b: struct { a: i8 = 2, b: i16 = 3 } = .{}, c: i32 = 4 }.{}" => Struct)
            .unwrap();

    assert_eq!(out, Struct { a: 1, b: Inner { a: 2, b: 3 }, c: 4 });
}
