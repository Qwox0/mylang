use crate::tests::jit_run_test;

#[test]
fn while_loop_sum_array() {
    let array = [1, 2, 3, 4, 10, 123];
    let arr_len = array.len();
    let out = jit_run_test!(&format!("
arr := {array:?};
arr_len := {arr_len};
mut sum := 0;
mut idx := 0;
while idx < arr_len {{
    sum += arr[idx];
    idx += 1;
}};
sum") => i64)
    .unwrap();
    assert_eq!(out, array.iter().sum::<i64>());
}

#[test]
fn while_pipe_condition() {
    let out = jit_run_test!(&format!("
mut idx := 0;
idx < 3 | while idx += 1;
idx") => i64)
    .unwrap();
    assert_eq!(out, 3);
}

#[test]
fn while_break_in_if() {
    let out = jit_run_test!(&"
mut idx := 0;
true | while {
    if idx == 3 break;
    idx += 1;
};
idx" => i64)
    .unwrap();
    assert_eq!(out, 3);
}

#[test]
fn while_continue_at_end() {
    let out = jit_run_test!(&"
mut idx := 0;
idx < 5 | while {
    idx += 1;
    continue;
};
idx" => i64)
    .unwrap();
    assert_eq!(out, 5);
}

#[test]
fn while_continue_and_break() {
    let out = jit_run_test!(&"
mut idx := 0;
true | while {
    idx += 1;
    if idx < 10 continue;
    break;
};
idx" => i64)
    .unwrap();
    assert_eq!(out, 10);
}
