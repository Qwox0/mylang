use crate::tests::test_body;

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
    test_body(code).ok(array.iter().sum::<i64>());
}

#[test]
fn while_pipe_condition() {
    let code = format!(
        "
mut idx := 0;
idx < 3 |> while idx += 1;
idx"
    );
    test_body(code).ok(3i64);
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
    test_body(code).ok(3i64);
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
    test_body(code).ok(5i64);
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
    test_body(code).ok(10i64);
}

#[test]
fn return_in_while() {
    test_body("while true { return 5.0; }; 0.0").ok(5.0f64);
}

#[test]
fn while_with_do() {
    let code = format!(
        "
mut product := 1;
while product < 100 do product <<= 1;
product"
    );
    test_body(code).ok(128i64);
}
