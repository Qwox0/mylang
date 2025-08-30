use crate::tests::{substr, test, test_body};

#[test]
fn set_field_value() {
    let code = "
MyStruct :: struct { a: i64 };
mut a: MyStruct;
a.a = 123;
a.a == 123";
    test_body(code).ok(true);
}

#[test]
fn anon_struct_on_decl() {
    test_body("a: struct { a: i64 };").ok(());
}

#[test]
#[ignore = "unfinished test"]
fn tuple() {
    let code = "
MyTuple :: struct { i64, i64 };
a := MyTuple.(3, 7);
a.0 == 3 && a.1 == 7";
    test_body(code).ok(true);
}

#[test]
fn return_local_vs_global_type() {
    // global
    let code = "
MyStruct :: struct { x: i64 };
test :: -> MyStruct.{ x = 5 };
";
    let mut res = test(code).ok(5i64);
    let llvm_module_text_global = res.take_llvm_ir();
    drop(res);

    // local
    let code = "
        test :: -> {
            MyStruct :: struct { x: i64 };
            MyStruct.{ x = 5 }
        }";
    let res = test(code).ok(5i64);
    let llvm_module_text_local = res.llvm_ir();

    assert_eq!(llvm_module_text_local, llvm_module_text_global);
}

#[test]
fn duplicate_field() {
    let code = "MyStruct :: struct { x: i32, x: i64 };";
    test(code).error("duplicate struct field `x`", substr!("x";skip=1));

    // TODO: allow methods with the same name as a field?
    let code = "MyStruct :: struct { x: i32, x :: (s: *MyStruct) -> s.*.x; };";
    test(code).error("duplicate symbol `x` in struct scope", substr!("x";skip=1));

    let code = "MyStruct :: struct { x: i32 }; MyStruct.x :: (s: *MyStruct) -> s.*.x;";
    // TODO: better error message
    test(code).error("duplicate definition of `struct{x:i32}.x`", substr!("MyStruct.x"));

    // union
    {
        let code = "MyUnion :: union { x: i32, x: i64 };";
        test(code).error("duplicate union field `x`", substr!("x";skip=1));

        let code = "MyUnion :: union { x: i32, x :: (s: *MyUnion) -> s.*.x; };";
        test(code).error("duplicate symbol `x` in union scope", substr!("x";skip=1));

        let code = "MyUnion :: union { x: i32 }; MyUnion.x :: (s: *MyUnion) -> s.*.x;";
        // TODO: better error message
        test(code).error("duplicate definition of `union{x:i32}.x`", substr!("MyUnion.x"));
    }

    // enum
    {
        let code = "MyEnum :: enum { X, X(i64) };";
        test(code).error("duplicate enum variant `X`", substr!("X";skip=1));

        /* TODO: allow consts in enums
        let code = "MyEnum :: enum { x: i32, x :: (s: *MyEnum) -> s.*.x; };";
        test_raw(code).one_err("duplicate symbol `x` in struct scope", substr!("x";skip=1));
        */

        let code = "MyEnum :: enum { X }; MyEnum.X :: (s: *MyEnum) -> s.*.x;";
        // TODO: better error message
        test(code).error("duplicate definition of `enum{X}.X`", substr!("MyEnum.X"));
    }
}

#[test]
fn use_param_default_in_const() {
    let code = "
MyStruct :: struct { arr := .[DEFAULT; 10]; };
CONST :: MyStruct.();
DEFAULT :: 7;
test :: -> CONST.arr[1];";
    test(code).ok(7i32);
}
