use crate::tests::jit_run_test;

#[test]
fn while_loop_sum_array() {
    let array = [1, 2, 3, 4, 10, 123];
    let arr_len = array.len();
    let code = format!(
        "
arr := .{array:?};
arr_len := {arr_len};
mut sum := 0;
mut idx := 0;
while idx < arr_len {{
    sum += arr[idx];
    idx += 1;
}};
sum"
    );
    assert_eq!(*jit_run_test::<i64>(code).ok(), array.iter().sum::<i64>());
}

#[test]
fn while_pipe_condition() {
    let code = format!(
        "
mut idx := 0;
idx < 3 |> while idx += 1;
idx"
    );
    assert_eq!(*jit_run_test::<i64>(code).ok(), 3);
}

#[test]
fn while_break_in_if() {
    let code = "
mut idx := 0;
true |> while {
    if idx == 3 break;
    idx += 1;
};
idx";
    assert_eq!(*jit_run_test::<i64>(code).ok(), 3);
}

#[test]
fn while_continue_at_end() {
    let code = "
mut idx := 0;
idx < 5 |> while {
    idx += 1;
    continue;
};
idx";
    assert_eq!(*jit_run_test::<i64>(code).ok(), 5);
}

#[test]
fn while_continue_and_break() {
    let code = "
mut idx := 0;
while true {
    idx += 1;
    if idx < 10 continue;
    break;
};
idx";
    assert_eq!(*jit_run_test::<i64>(code).ok(), 10);
}

#[test]
fn return_in_while() {
    assert_eq!(*jit_run_test::<f64>("while true { return 5.0; }; 0.0").ok(), 5.0);
}

#[test]
fn while_with_do() {
    let code = format!(
        "
mut product := 1;
while product < 100 do product <<= 1;
product"
    );
    assert_eq!(*jit_run_test::<i64>(code).ok(), 128);
}
