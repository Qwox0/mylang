use super::{TestSpan, jit_run_test, test_compile_err};
use crate::util::IteratorExt;

#[test]
fn create_mut_ptr() {
    let get_code = |mut_| format!("{mut_} my_var := 5; &mut my_var");
    let out = *jit_run_test::<*mut i64>(get_code("mut")).ok();
    assert!(!out.is_null());

    test_compile_err(get_code(""), "Cannot mutably reference `my_var`", |code| {
        TestSpan::of_nth_substr(code, 1, "my_var")
    });
}

#[test]
fn create_mut_slice() {
    let get_code = |mut_| format!("{mut_} my_arr := .[1,2,3]; my_arr[..]mut");
    let out = *jit_run_test::<(*mut i64, u64)>(get_code("mut")).ok();
    assert!(!out.0.is_null());
    assert_eq!(out.1, 3);

    test_compile_err(get_code(""), "Cannot mutably reference `my_arr`", |code| {
        TestSpan::of_nth_substr(code, 1, "my_arr")
    });
}

#[test]
fn nested_ptr_slice() {
    let get_code = |var_mut: &str, slice_mut: &str, ptr_mut: &str| {
        format!(
            "{var_mut} arr := .[1,2,3]; ptr := arr[1..]{slice_mut}[0].&{ptr_mut}; ptr.* = 5; arr",
        )
    };
    assert_eq!(*jit_run_test::<[i64; 3]>(get_code("mut", "mut", "mut")).ok(), [1, 5, 3]);
    test_compile_err(
        get_code("", "", ""),
        "Cannot modify the value behind an immutable pointer",
        |code| TestSpan::of_substr(code, "ptr.* = 5"),
    );
    test_compile_err(
        get_code("", "", "mut"),
        "Cannot modify the elements of the immutable slice",
        |code| TestSpan::of_substr(code, "arr[1..][0].&mut"),
    );
    test_compile_err(get_code("", "mut", "mut"), "Cannot mutably reference `arr`", |code| {
        TestSpan::of_substr(code, "arr[1..]mut[0].&mut").start_with_len(3)
    });
}

#[test]
fn mut_ptr_to_const() {
    let res = jit_run_test::<i64>("MY_CONST :: 5; p := &mut MY_CONST; p.* += 100; MY_CONST");
    assert_eq!(*res.ok(), 5);
    let warning = res.warnings().expect_one();
    assert!(warning.msg.starts_with(
        "The mutable pointer will reference a local copy of `MY_CONST`, not the constant itself"
    ));
    assert_eq!(warning.span, TestSpan::of_substr(&res.full_code, "&mut MY_CONST"));
}

#[test]
fn receive_mut_ptr() {
    let get_code = |mut_marker: &str| {
        format!(
            "
increment_by5 :: (x: *mut i64) -> x.* += 5;
mut num := 0;
increment_by5(num.&{mut_marker});
num"
        )
    };
    let out = *jit_run_test::<i64>(get_code("mut")).ok();
    assert_eq!(out, 5);

    test_compile_err(get_code(""), "expected *mut i64; got *i64", |code| {
        TestSpan::of_substr(code, "num.&")
    });
}

#[test]
fn deref_ptr_hint() {
    test_compile_err(
        "my_fn :: (ptr: *mut i64) -> ptr += 1;",
        "Cannot apply binary operatator `+=` to pointer type `*mut i64`",
        |code| TestSpan::of_nth_substr(code, 1, "ptr"),
    );
}

#[test]
fn check_lvalue_mutability() {
    let get_code = |mut_marker| format!("{mut_marker} x := 5; x *= 2; x");
    assert_eq!(*jit_run_test::<i64>(get_code("mut")).ok(), 10);
    test_compile_err(get_code(""), "Cannot assign to immutable variable 'x'", |code| {
        TestSpan::of_substr(code, "x *= 2")
    });

    let get_code = |mut_marker| format!("{mut_marker} arr := .[1,2,3]; arr[1] = 5; arr");
    assert_eq!(*jit_run_test::<[i64; 3]>(get_code("mut")).ok(), [1, 5, 3]);
    test_compile_err(get_code(""), "Cannot assign to immutable variable 'arr'", |code| {
        TestSpan::of_substr(code, "arr[1] = 5")
    });

    let get_code = |mut_marker| format!("mut x := 5; ptr := &{mut_marker} x; ptr.* *= 2; x");
    assert_eq!(*jit_run_test::<i64>(get_code("mut")).ok(), 10);
    test_compile_err(get_code(""), "Cannot modify the value behind an immutable pointer", |code| {
        TestSpan::of_substr(code, "ptr.* *= 2")
    });

    // TODO: is `arr[1..]mut` good syntax?
    let get_code = |mut_marker| {
        format!("mut arr := .[1,2,3]; slice := arr[1..]{mut_marker}; slice[0] = 5; arr")
    };
    assert_eq!(*jit_run_test::<[i64; 3]>(get_code("mut")).ok(), [1, 5, 3]);
    test_compile_err(get_code(""), "Cannot modify the elements of an immutable slice", |code| {
        TestSpan::of_substr(code, "slice[0] = 5")
    });
}
