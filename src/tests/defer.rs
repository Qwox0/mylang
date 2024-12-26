use crate::tests::jit_run_test;

#[test]
fn defer_reverse_order() {
    let out = jit_run_test!(raw "
test :: -> {
    mut var: i64 = 0;
    {
        defer var += 1;
        defer var *= 1000000;
    }
    var
}" => i64);
    assert_eq!(out.unwrap(), 1);
}

/// Semantics of return:
///
/// ```mylang
/// defer /* ... */;
/// return x;
///
/// // ->
///
/// return_value := x;
/// // compile defer exprs; -> mutating x results in ...
/// // -> a compiler error (moved value)
/// // -> nothing (x was copied into `return_value`)
/// return return_value;
/// ```
#[test]
#[ignore = "unfinished test (TODO: implement move/copy)"]
fn defer_doesnt_mutate_return() {
    let out = jit_run_test!(raw "
test :: -> {
    mut var: i64 = 0;
    defer var += 1;
    return var;
}" => i64);
    assert_eq!(out.unwrap(), 0);
}

#[test]
fn defer_multiple_blocks() {
    let out = jit_run_test!(raw "
inner :: (val: *mut i64) -> {
    {
        defer val.* += 1;
        {
            defer val.* += 2;
            defer val.* += 3;
            {
                defer val.* += 1;
                return;
            }
            defer val.* += 10;
        }
        defer val.* += 10;
    }
    defer val.* += 10;
};
test :: -> { mut x: i64 = 0; inner(x.&mut); x }" => i64);
    assert_eq!(out.unwrap(), 7);
}

#[test]
fn defer_in_nested_functions() {
    let out = jit_run_test!(raw "
test :: -> {
    inner_x := 0;
    inner_x := &mut inner_x;
    defer inner_x.* += 10;
    mut x := 0;
    inner :: (inner_x: *mut i64) -> {
        defer inner_x.* += 1;
        {
            defer inner_x.* += 2;
        }
    };
    inner(&mut x);
    x
}" => i64);
    assert_eq!(out.unwrap(), 3);

    let out = jit_run_test!(raw "
test :: -> {
    inner_x := 0;
    inner_x := &mut inner_x;
    defer inner_x.* += 10;
    mut x := 0;
    inner :: (inner_x: *mut i64) -> {
        defer inner_x.* += 1;
        {
            defer inner_x.* += 2;
            return;
        }
    };
    inner(&mut x);
    x
}" => i64);
    assert_eq!(out.unwrap(), 3);
}

#[test]
fn defer_in_loop() {
    let out = jit_run_test!(raw "
test :: -> {
    mut x := 100;
    defer x += 1;
    for _ in 0..2 {
        defer x += 100;
        {
            defer x *= 2;
        }
    }
    x
}" => i64);
    assert_eq!(out.unwrap(), 701);

    let out = jit_run_test!(raw "
test :: -> {
    mut x := 100;
    defer x += 1;
    for _ in 0..2 {
        defer x += 100;
        {
            defer x *= 2;
            continue;
        }
    }
    x
}" => i64);
    assert_eq!(out.unwrap(), 701);

    let out = jit_run_test!(raw "
test :: -> {
    mut x := 100;
    defer x += 1;
    0.. |> for _ {
        defer x += 100;
        {
            defer x *= 2;
            break;
        }
    }
    x
}" => i64);
    assert_eq!(out.unwrap(), 301);
}

#[test]
fn correctly_close_scope_on_return() {
    let out = jit_run_test!(raw "
test :: -> {
    a := 0;
    if false {
        a := 1;
        return -1;
    }
    return a;
}" => i64);
    assert_eq!(out.unwrap(), 0);
}
