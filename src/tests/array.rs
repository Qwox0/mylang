use super::{TestSpan, test_compile_err};
use crate::tests::jit_run_test;

/// In the C calling convention arrays are always passed as a pointer and can't be returned.
/// In mylang arrays are also always returned as a (sret) pointer. This helper is needed because
/// Rust does something else.
#[derive(Clone, Copy)]
#[repr(C)]
struct Arr<T, const LEN: usize> {
    val: [T; LEN],
    __force_sret: [u64; 10],
}

#[test]
fn arr_initializer_with_lhs() {
    assert_eq!(jit_run_test::<Arr<u8, 4>>("u8.[1, 2, 3, 4]").ok().val, [1, 2, 3, 4]);
}

/// also tests that the `elem_ty` is still inferred correctly.
#[test]
fn arr_initializer_len_mismatch() {
    test_compile_err(
        "{ arr: [3]u8 = .[1, 2, 3, 4]; }",
        "mismatched types: expected [3]u8; got [4]u8",
        |code| TestSpan::of_substr(code, ".[1, 2, 3, 4]"),
    );
}

#[test]
fn arr_initializer_on_ptr() {
    let code = "{ mut arr: [4]i64; ptr := &mut arr; ptr.[5, 6, 7, 8]; arr }";
    assert_eq!(jit_run_test::<Arr<i64, 4>>(code).ok().val, [5, 6, 7, 8]);
}

#[test]
fn arr_initializer_on_ptr_len_mismatch() {
    test_compile_err(
        "{ mut arr := .[1, 2, 3, 4]; ptr := &mut arr; ptr.[5, 6, 7, 8, 9]; arr }",
        "Cannot initialize the array behind the pointer `*mut [4]i64` with 5 items",
        |code| TestSpan::of_substr(code, "ptr.[5, 6, 7, 8, 9]"),
    );
}

#[test]
#[ignore = "not yet implemented"]
fn arr_initializer_on_slice() {
    let out = *jit_run_test::<Arr<i64, 4>>(
        "{ arr := .[1, 2, 3, 4]; slice := arr[1..]; slice.[5, 6, 7]; arr }",
    )
    .ok();
    assert_eq!(out.val, [1, 5, 6, 7]);
}

#[test]
#[ignore = "not yet implemented"]
fn arr_initializer_on_slice_len_mismatch() {
    // "{ mut arr := .[1, 2, 3, 4]; slice := arr[1..]mut; slice.[5, 6]; arr }"
    // => Runtime error?
}

#[test]
fn arr_initializer_on_ref_mut_check() {
    test_compile_err(
        "{ arr := .[1, 2, 3, 4]; ptr := &arr; ptr.[5, 6, 7, 8]; arr }",
        "Cannot mutate the value behind an immutable pointer",
        |code| TestSpan::of_substr(code, "ptr.[5, 6, 7, 8]"),
    );

    /*
    test_compile_err(
        "{ arr := .[1, 2, 3, 4]; slice := arr[1..]; slice.[5, 6, 7]; arr }",
        "TODO",
        |code| TestSpan::of_substr(code,  "slice.[5, 6, 7]"),
    );
    */
}

#[test]
#[ignore = "not yet implemented"]
fn array_constant() {
    let out = *jit_run_test::<[i64; 5]>("MY_NUMBERS :: .[1,2,3,4,5]; MY_NUMBERS").ok();
    debug_assert_eq!(out, [1, 2, 3, 4, 5]);
}

#[test]
fn never_val_in_array() {
    #[rustfmt::skip]
    let code = |inner_ret_val| format!("
mut a := 0;
return .[
    {{ a += 1; 1 }},
    {{ a += 1; 2 }},
    return {inner_ret_val},
    {{ a += 1; 4 }},
]");

    test_compile_err(
        code(".[a, a]"),
        "mismatched types: expected [2]i64; got [4]{integer literal}",
        |code| {
            let start = code.find(".[").unwrap();
            let end = code.rfind("]").unwrap() + 1;
            TestSpan::new(start, end)
        },
    );

    debug_assert_eq!(*jit_run_test::<[i64; 4]>(code(".[a, a, a, a]"),).ok(), [2, 2, 2, 2]);
}
