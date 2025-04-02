use crate::tests::jit_run_test_raw;

#[test]
fn defer_reverse_order() {
    let code = "
test :: -> {
    mut var: i64 = 0;
    {
        defer var += 1;
        defer var *= 1000000;
    }
    var
}";
    assert_eq!(*jit_run_test_raw::<i64>(code).ok(), 1);
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
    let code = "
test :: -> {
    mut var: i64 = 0;
    defer var += 1;
    return var;
}";
    assert_eq!(*jit_run_test_raw::<i64>(code).ok(), 0);
}

#[test]
fn defer_multiple_blocks() {
    let code = "
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
test:: -> { mut x: i64 = 0; inner(x.&mut); x }";
    assert_eq!(*jit_run_test_raw::<i64>(code).ok(), 7);
}

#[test]
fn defer_in_nested_functions() {
    let code = "
test :: -> {
    mut inner_x := 0;
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
}";
    assert_eq!(*jit_run_test_raw::<i64>(code).ok(), 3);

    let code = "
test :: -> {
    mut inner_x := 0;
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
}";
    assert_eq!(*jit_run_test_raw::<i64>(code).ok(), 3);
}

#[test]
fn defer_in_loop() {
    let code = "
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
}";
    assert_eq!(*jit_run_test_raw::<i64>(code).ok(), 701);

    let code = "
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
}";
    assert_eq!(*jit_run_test_raw::<i64>(code).ok(), 701);

    let code = "
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
}";
    assert_eq!(*jit_run_test_raw::<i64>(code).ok(), 301);
}

#[test]
fn correctly_close_scope_on_return() {
    let code = "
test :: -> {
    a := 0;
    if false {
        a := 1;
        return -1;
    }
    return a;
}";
    assert_eq!(*jit_run_test_raw::<i64>(code).ok(), 0);
}
