use super::jit_run_test;
use crate::tests::{TestSpan, jit_run_test_raw, test_compile_err_raw};

#[test]
fn set_field_value() {
    let code = "
MyStruct :: struct { a: i64 };
mut a: MyStruct;
a.a = 123;
a.a == 123";
    assert!(jit_run_test::<bool>(code).ok(), "```{code}\n``` -> expected `true`");
}

#[test]
fn anon_struct_on_decl() {
    jit_run_test::<()>("a: struct { a: i64 };").ok();
}

#[test]
#[ignore = "unfinished test"]
fn tuple() {
    let code = "
MyTuple :: struct { i64, i64 };
a := MyTuple.(3, 7);
a.0 == 3 && a.1 == 7";
    assert!(jit_run_test::<bool>(code).ok(), "```{code}\n``` -> expected `true`");
}

#[test]
fn return_local_vs_global_type() {
    // global
    let code = "
MyStruct :: struct { x: i64 };
test :: -> MyStruct.{ x = 5 };
";
    let mut res = jit_run_test_raw::<i64>(code);
    assert_eq!(*res.ok(), 5);
    let llvm_module_text_global = res.take_llvm_ir();
    drop(res);

    // local
    let code = "
        test :: -> {
            MyStruct :: struct { x: i64 };
            MyStruct.{ x = 5 }
        }";
    let res = jit_run_test_raw::<i64>(code);
    assert_eq!(*res.ok(), 5);
    let llvm_module_text_local = res.llvm_ir();

    assert_eq!(llvm_module_text_local, llvm_module_text_global);
}

#[test]
fn duplicate_field() {
    let code = "MyStruct :: struct { x: i32, x: i64 };";
    test_compile_err_raw(code, "duplicate struct field `x`", |code| {
        TestSpan::of_nth_substr(code, 1, "x")
    });
}

#[test]
#[ignore = "not implemented"]
fn duplicate_field2() {
    // TODO: allow field and method with the same name?
    let code = "MyStruct :: struct { x: i32, x :: (s: *MyStruct) -> s.x; };";
    jit_run_test_raw::<()>(code).ok();

    let code = "MyStruct :: struct { x: i32 }; MyStruct.x :: (s: *MyStruct) -> s.x;";
    jit_run_test_raw::<()>(code).ok();
}
