use crate::tests::jit_run_test;

#[test]
fn for_loop_sum_array() {
    let array = [1, 2, 3, 4, 10, 123];
    let code = format!(
        "
mut sum := 0;
.{array:?} |> for x {{
    sum += x;
}};
sum"
    );
    assert_eq!(*jit_run_test::<i64>(code).ok(), array.iter().sum::<i64>());
}

#[test]
fn for_break_in_if() {
    let code = "
arr := .[1, 2, 3, 4, 5];
mut sum := 0;
for x in arr {
    if x > 3 break;
    sum += x;
};
sum";
    assert_eq!(*jit_run_test::<i64>(code).ok(), 1 + 2 + 3);
}

#[test]
fn for_continue_at_end() {
    let code = "
arr := .[1, 2, 3, 4, 5];
mut x := 0;
for elem in arr {
    x = elem;
    continue;
};
x";
    assert_eq!(*jit_run_test::<i64>(code).ok(), 5);
}

#[test]
fn return_in_for() {
    assert_eq!(*jit_run_test::<f64>(".[5, 10.0, 15] |> for x { return x; }; 0.0").ok(), 5.0);
}

#[test]
fn for_with_do() {
    let arr = [2, 3, 7, 10];
    let code = format!(
        "
arr := .{arr:?};
mut product := 1;
arr |> for x do product *= x;
product"
    );
    assert_eq!(*jit_run_test::<i64>(code).ok(), arr.iter().product());
}

#[test]
#[ignore = "unfinished test"]
fn invalid_source_type() {
    *jit_run_test::<()>("true |> for _x { break; };").ok()
}
