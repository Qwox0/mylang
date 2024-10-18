use super::jit_run_test;

#[test]
fn set_field_value() {
    let code = "
MyStruct :: struct { a: i64 };
a: MyStruct;
a.a = 123;
a.a == 123";
    let ok = jit_run_test!(code => bool).unwrap();
    assert!(ok, "```{code}\n``` -> expected `true`");
}

#[test]
fn anon_struct_no_init() {
    jit_run_test!("a: struct { a: i64 }" => ()).unwrap();
}
