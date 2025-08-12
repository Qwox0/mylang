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

use crate::tests::{jit_run_test, jit_run_test_raw};

macro_rules! test_struct_return {
    ($test_name:ident : { $($field:ident : $ty:ty = $val:expr),* $(,)? }) => {
        #[test]
        fn $test_name() {
            #[derive(Debug, PartialEq)]
            #[repr(C)]
            struct Struct { $($field: $ty,)* }
            let res = jit_run_test::<Struct>(concat!("struct{", stringify!($( $field: $ty = $val, )* ), "}.{}"));
            assert_eq!(*res.ok(), Struct { $( $field : $val ),* });
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
fn return_struct_u2_i64() {
    let out = *jit_run_test::<u128>("struct { tag: u2 = 1, val: i64 = -1 }.{}").ok();
    let out_hex_string = format!("{out:032x}");
    println!("{out_hex_string}");
    // expect "ffffffffffffffff______________01"
    assert!(out_hex_string.starts_with(&"f".repeat(16)));
    assert!(out_hex_string.ends_with("01"));
}

#[test]
#[ignore = "unfinished test"]
fn return_struct_f64() {
    let big_float_bits: u64 = 0xEFFFFFFFFFFFFFF3;
    let big_float = f64::from_bits(big_float_bits);

    #[derive(Debug, Clone, Copy)]
    struct Out {
        tag: u8,
        val: f64,
    }
    let code = format!(
        "
MySumType :: struct {{
    tag: u2,
    val: f64,
}};
test :: -> {{
    retval: MySumType;
    retval.tag=1;
    retval.val={big_float};
    retval
}};"
    );
    let out = *jit_run_test_raw::<Out>(&code).ok();
    assert_eq!(out.tag, 1);
    assert_eq!(format!("{:x}", out.val.to_bits()), format!("{:x}", big_float_bits));
}

#[test]
fn return_struct_ptr_i64() {
    #[derive(Debug, Clone, Copy, PartialEq)]
    #[repr(C)]
    struct Struct {
        ptr: *const i64,
        b: i64,
    }
    let out = *jit_run_test::<Struct>("x := 100; struct{ptr: *i64, b: i64 = 200 }.(&x)").ok();
    let a: i64 = 100;
    let ptr = &a as *const i64;
    assert!(
        (ptr as u64 - out.ptr as u64) < 2000,
        "The stack pointers should be close to each other"
    );
    assert_eq!(out.b, 200);
}

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

    let code = "struct { a: i8 = 1, b: struct { a: i8 = 2, b: i16 = 3 } = .{}, c: i32 = 4 }.{}";
    assert_eq!(*jit_run_test::<Struct>(code).ok(), Struct { a: 1, b: Inner { a: 2, b: 3 }, c: 4 });
}

#[test]
fn struct_with_array_param_pass_through_to_another_call() {
    let code = "
MyNum :: struct { val: [2]i32 }
add :: (n: MyNum, other: i64) -> MyNum.(.[n.val[0] + xx other, n.val[1]]);
pass_through :: (n: MyNum, other: i64) -> add(n, other);
test :: -> pass_through(.(.[1,2]), 2);";
    let out = *jit_run_test_raw::<[i32; 2]>(code).ok();
    assert_eq!(out, [3, 2]);
}

#[test]
fn call_by_value_struct_with_many_small_fields() {
    let code = "
MyStruct :: struct { a: i8, b: i8, c: [6]i8, x: u32 }
take_struct :: (s: MyStruct) -> s.x;
test :: -> take_struct(.(0, 10, .[1; 6], 99));";
    let out = *jit_run_test_raw::<u32>(code).ok();
    assert_eq!(out, 99);
}

#[test]
#[ignore = "unimplemented"]
fn call_by_value_i128() {
    let code = "
take_i128 :: (i: i128) -> i.as(i32);
test :: -> take_i128(-10);";
    let res = jit_run_test_raw::<i32>(code);
    assert_eq!(*res.ok(), -10);
    let param_name_prefix = if cfg!(debug_assertions) { "i." } else { "" };
    assert!(res.llvm_ir().contains(&format!(
        "define i32 @take_i128(i64 %{param_name_prefix}0, i64 %{param_name_prefix}1)"
    )));
}

#[test]
fn call_by_value_array() {
    // arrays are a special case for some reason.
    let code = "
take_array :: (arr: [3]i32) -> arr[1];
test :: -> take_array(.[1,-2, 0]);";
    let res = jit_run_test_raw::<i32>(code);
    assert_eq!(*res.ok(), -2);
    let param_name = if cfg!(debug_assertions) { "arr" } else { "0" };
    assert!(
        res.llvm_ir()
            .contains(&format!("define noundef i32 @take_array(ptr noundef %{param_name})"))
    );
    drop(res);

    let code = "
Arr :: struct { arr: [3]i32 }
take_array :: (arr: Arr) -> arr.arr[1];
test :: -> take_array(.(.[1,-2, 0]));";
    let res = jit_run_test_raw::<i32>(code);
    assert_eq!(*res.ok(), -2);
    let param_name_prefix = if cfg!(debug_assertions) { "arr." } else { "" };
    assert!(res.llvm_ir().contains(&format!(
        "define noundef i32 @take_array(i64 %{param_name_prefix}0, i32 %{param_name_prefix}1)",
    )));
}

#[test]
fn enum_size() {
    let res = jit_run_test::<i32>("enum { A, B, C }.B");
    assert_eq!(*res.ok(), 1);
    assert!(res.llvm_ir().contains(&format!("define noundef i32 @test()")));
}
