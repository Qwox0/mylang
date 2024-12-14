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
#[ignore = "unfinished test"]
fn tuple() {
    let code = "
MyTuple :: struct { i64, i64 };
a := MyTuple.(3, 7);
a.0 == 3 && a.1 == 7";
    let ok = jit_run_test!(code => bool).unwrap();
    assert!(ok, "```{code}\n``` -> expected `true`");
}

#[test]
fn return_local_vs_global_type() {
    // global
    let (out, llvm_module_text_global) = jit_run_test!(raw "
MyStruct :: struct { x: i64 };
test :: -> MyStruct.{ x = 5 };
" => i64, llvm_module)
    .unwrap();
    assert_eq!(out, 5);

    // local
    let (out, llvm_module_text_local) = jit_run_test!(raw "
test :: -> {
    MyStruct :: struct { x: i64 };
    MyStruct.{ x = 5 }
}" => i64, llvm_module)
    .unwrap();
    assert_eq!(out, 5);

    assert_eq!(llvm_module_text_local, llvm_module_text_global);
}
