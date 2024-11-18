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
fn anon_struct_on_decl() {
    jit_run_test!("a: struct { a: i64 };" => ()).unwrap();
}

#[test]
fn tuple() {
    let code = "
MyTuple :: struct { i64, i64 };
a := MyTuple.(3, 7);
a.0 == 3 && a.1 == 7";
    let ok = jit_run_test!(code => bool).unwrap();
    assert!(ok, "```{code}\n``` -> expected `true`");
}
