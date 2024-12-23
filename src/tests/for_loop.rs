use crate::tests::jit_run_test;

#[test]
fn for_loop_sum_array() {
    let array = [1, 2, 3, 4, 10, 123];
    let out = jit_run_test!(format!("
mut sum := 0;
.{array:?} |> for x {{
    sum += x;
}};
sum") => i64);
    assert_eq!(out.unwrap(), array.iter().sum::<i64>());
}

#[test]
fn for_break_in_if() {
    let out = jit_run_test!("
arr := .[1, 2, 3, 4, 5];
mut sum := 0;
for x in arr {
    if x > 3 break;
    sum += x;
};
sum" => i64);
    assert_eq!(out.unwrap(), 1 + 2 + 3);
}

#[test]
fn for_continue_at_end() {
    let out = jit_run_test!("
arr := .[1, 2, 3, 4, 5];
mut x := 0;
for elem in arr {
    x = elem;
    continue;
};
x" => i64);
    assert_eq!(out.unwrap(), 5);
}

#[test]
fn return_in_for() {
    assert_eq!(5.0, jit_run_test!(".[5, 10.0, 15] |> for x { return x; }; 0.0" => f64).unwrap());
}

#[test]
fn for_with_do() {
    let arr = [2, 3, 7, 10];
    let out = jit_run_test!(format!("
arr := .{arr:?};
mut product := 1;
arr |> for x do product *= x;
product") => i64);
    assert_eq!(out.unwrap(), arr.iter().product());
}

/*
/// TODO
#[test]
fn parse_panic() {
    let out = jit_run_test!("true | for { break; };" => i64).unwrap();
    assert_eq!(out, 10);
}
*/
