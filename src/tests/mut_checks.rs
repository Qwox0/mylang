use crate::tests::{substr, test_body};

#[test]
fn create_mut_ptr() {
    let get_code = |mut_| format!("{mut_} my_var := 5; &mut my_var");
    let out = test_body(get_code("mut")).get_out::<*mut i64>();
    assert!(!out.is_null());

    test_body(get_code("")).error("Cannot mutably reference `my_var`", substr!("my_var";skip=1));
}

#[test]
fn create_mut_slice() {
    let get_code = |mut_| format!("{mut_} my_arr := .[1,2,3]; my_arr[..]mut");
    let out = test_body(get_code("mut")).get_out::<(*mut i64, u64)>();
    assert!(!out.0.is_null());
    assert_eq!(out.1, 3);

    test_body(get_code("")).error("Cannot mutably reference `my_arr`", substr!("my_arr";skip=1));
}

#[test]
fn nested_ptr_slice() {
    let get_code = |var_mut: &str, slice_mut: &str, ptr_mut: &str| {
        format!(
            "{var_mut} arr := .[1,2,3]; ptr := arr[1..]{slice_mut}[0].&{ptr_mut}; ptr.* = 5; arr",
        )
    };
    test_body(get_code("mut", "mut", "mut")).ok([1i64, 5, 3]);
    test_body(get_code("", "", ""))
        .error("Cannot mutate the value behind an immutable pointer", substr!("ptr.* = 5"));
    test_body(get_code("", "", "mut"))
        .error("Cannot mutate the elements of an immutable slice", substr!("arr[1..][0].&mut"));
    test_body(get_code("", "mut", "mut"))
        .error("Cannot mutably reference `arr`", substr!("arr[1..]mut[0].&mut";.start_with_len(3)));
}

#[test]
fn mut_ptr_to_const() {
    test_body("MY_CONST :: 5; p := &mut MY_CONST; p.* += 100; MY_CONST")
        .ok(5i64)
        .warn(
            "The mutable pointer will reference a local copy of `MY_CONST`, not the constant \
             itself",
            substr!("&mut MY_CONST"),
        );
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
    test_body(get_code("mut")).ok(5i64);
    test_body(get_code(""))
        .error("mismatched types: expected *mut i64; got *i64", substr!("num.&"));
}

#[test]
fn deref_ptr_hint() {
    test_body("my_fn :: (ptr: *mut i64) -> ptr += 1;").error(
        "Cannot apply binary operatator `+=` to pointer type `*mut i64`",
        substr!("ptr";skip=1),
    );
}

#[test]
fn check_lvalue_mutability() {
    let get_code = |mut_marker| format!("{mut_marker} x := 5; x *= 2; x");
    test_body(get_code("mut")).ok(10i64);
    test_body(get_code("")).error("Cannot assign to immutable variable 'x'", substr!("x *= 2"));

    let get_code = |mut_marker| format!("{mut_marker} arr := .[1,2,3]; arr[1] = 5; arr");
    test_body(get_code("mut")).ok([1i64, 5, 3]);
    test_body(get_code(""))
        .error("Cannot assign to immutable variable 'arr'", substr!("arr[1] = 5"));

    let get_code = |mut_marker| format!("mut x := 5; ptr := &{mut_marker} x; ptr.* *= 2; x");
    test_body(get_code("mut")).ok(10i64);
    test_body(get_code(""))
        .error("Cannot mutate the value behind an immutable pointer", substr!("ptr.* *= 2"));

    // TODO: is `arr[1..]mut` good syntax?
    let get_code = |mut_marker| {
        format!("mut arr := .[1,2,3]; slice := arr[1..]{mut_marker}; slice[0] = 5; arr")
    };
    test_body(get_code("mut")).ok([1i64, 5, 3]);
    test_body(get_code(""))
        .error("Cannot mutate the elements of an immutable slice", substr!("slice[0] = 5"));
}
