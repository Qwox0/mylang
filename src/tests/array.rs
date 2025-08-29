use super::TestSpan;
use crate::tests::{substr, test_body};

/// In the C calling convention arrays are always passed as a pointer and can't be returned.
/// In mylang arrays are also always returned as a (sret) pointer. This helper is needed because
/// Rust does something else.
#[derive(Clone, Copy)]
#[repr(C)]
pub struct CRetArr<T, const LEN: usize> {
    pub val: [T; LEN],
    __force_sret: [u64; 10],
}

#[test]
fn arr_initializer_with_lhs() {
    assert_eq!(test_body("u8.[1, 2, 3, 4]").get_out::<CRetArr<u8, 4>>().val, [1, 2, 3, 4]);
}

/// also tests that the `elem_ty` is still inferred correctly.
#[test]
fn arr_initializer_len_mismatch() {
    test_body("{ arr: [3]u8 = .[1, 2, 3, 4]; }")
        .error("mismatched types: expected [3]u8; got [4]u8", substr!(".[1, 2, 3, 4]"));
}

#[test]
fn arr_initializer_on_ptr() {
    let code = "{ mut arr: [4]i64; ptr := &mut arr; ptr.[5, 6, 7, 8]; arr }";
    assert_eq!(test_body(code).get_out::<CRetArr<i64, 4>>().val, [5, 6, 7, 8]);
}

#[test]
fn arr_initializer_on_ptr_len_mismatch() {
    test_body("{ mut arr := .[1, 2, 3, 4]; ptr := &mut arr; ptr.[5, 6, 7, 8, 9]; arr }").error(
        "Cannot initialize the array behind the pointer `*mut [4]i64` with 5 items",
        substr!("ptr.[5, 6, 7, 8, 9]"),
    );
}

#[test]
#[ignore = "not yet implemented"]
fn arr_initializer_on_slice() {
    let code = "{ arr := .[1, 2, 3, 4]; slice := arr[1..]; slice.[5, 6, 7]; arr }";
    assert_eq!(test_body(code).get_out::<CRetArr<i64, 4>>().val, [1, 5, 6, 7]);
}

#[test]
#[ignore = "not yet implemented"]
fn arr_initializer_on_slice_len_mismatch() {
    // "{ mut arr := .[1, 2, 3, 4]; slice := arr[1..]mut; slice.[5, 6]; arr }"
    // => Runtime error?
}

#[test]
fn arr_initializer_on_ref_mut_check() {
    test_body("{ arr := .[1, 2, 3, 4]; ptr := &arr; ptr.[5, 6, 7, 8]; arr }")
        .error("Cannot mutate the value behind an immutable pointer", substr!("ptr.[5, 6, 7, 8]"));

    /*
    test_compile_err(
        "{ arr := .[1, 2, 3, 4]; slice := arr[1..]; slice.[5, 6, 7]; arr }",
        "TODO",
        substr!("slice.[5, 6, 7]"),
    );
    */
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

    test_body(code(".[a, a]")).error(
        "mismatched types: expected [2]i64; got [4]{integer literal}",
        |code| {
            let start = code.find(".[").unwrap();
            let end = code.rfind("]").unwrap() + 1;
            TestSpan::new(start, end)
        },
    );

    test_body(code(".[a, a, a, a]")).ok([2i64; 4]);
}
