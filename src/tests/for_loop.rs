use crate::tests::jit_run_test;

#[test]
fn for_loop_sum_array() {
    let array = [1, 2, 3, 4, 10, 123];
    let out = jit_run_test!(&format!("
mut sum := 0;
{array:?} | for x {{
    sum += x;
}};
sum") => i64)
    .unwrap();
    assert_eq!(out, array.iter().sum::<i64>());
}

#[test]
fn for_break_in_if() {
    let out = jit_run_test!(&"
arr := [1, 2, 3, 4, 5];
mut sum := 0;
arr | for x {
    if x > 3 break;
    sum += x;
};
sum" => i64)
    .unwrap();
    assert_eq!(out, 1 + 2 + 3);
}

#[test]
fn for_continue_at_end() {
    let out = jit_run_test!(&"
arr := [1, 2, 3, 4, 5];
mut x := 0;
arr | for elem {
    x = elem;
    continue;
};
x" => i64)
    .unwrap();
    assert_eq!(out, 5);
}

/// TODO
#[test]
fn parse_panic() {
    let out = jit_run_test!(&"true | for { break; };" => i64).unwrap();
    assert_eq!(out, 10);
}
