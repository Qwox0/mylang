use crate::tests::jit_run_test;

#[test]
fn initialize_ptr_to_struct() {
    let code = "{
MyStruct :: struct {
    a: i64,
    b := false,
    c: [3]f64,
};

a: MyStruct;
ptr := &a;
ptr.{
    a = 5,
    c = [5.2, 2.0, 3.3],
};
mut sum := 0.0;
ptr.*.c | for x {
    sum += x;
};
sum
}";
    let out = jit_run_test!(&code => f64).unwrap();
    assert_eq!(out, 10.5);
}
