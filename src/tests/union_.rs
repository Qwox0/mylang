use super::jit_run_test;

#[test]
fn transmute_with_union_same_size() {
    let int = 1200000000;
    let float = f32::from_bits(int);
    let code = format!(
        "
mut ok := true;
mut union_val: union {{ int: i32, float: f32 }};
union_val.int = {int};
ok &&= union_val.int == {int};
ok &&= union_val.float == {float};
ok"
    );
    assert!(*jit_run_test::<bool>(&code).ok(), "```{code}\n``` -> expected `true`");
}
