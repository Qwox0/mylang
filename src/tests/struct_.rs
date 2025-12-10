use crate::tests::{substr, test, test_analyzed_struct, test_body};
use std::mem::offset_of;

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
    test(code).error("duplicate definition of `MyStruct.x`", substr!("MyStruct.x"));

    // union
    {
        let code = "MyUnion :: union { x: i32, x: i64 };";
        test(code).error("duplicate union field `x`", substr!("x";skip=1));

        let code = "MyUnion :: union { x: i32, x :: (s: *MyUnion) -> s.*.x; };";
        test(code).error("duplicate symbol `x` in union scope", substr!("x";skip=1));

        let code = "MyUnion :: union { x: i32 }; MyUnion.x :: (s: *MyUnion) -> s.*.x;";
        // TODO: better error message
        test(code).error("duplicate definition of `MyUnion.x`", substr!("MyUnion.x"));
    }

    // enum
    {
        let code = "MyEnum :: enum { X, X(i64) };";
        test(code).error("duplicate enum variant `X`", substr!("X";skip=1));

        /* TODO: allow consts in enums
        let code = "MyEnum :: enum { x: i32, x :: (s: *MyEnum) -> s.*.x; };";
        test_raw(code).one_err("duplicate symbol `x` in struct scope", substr!("x";skip=1));
        */

        let code = "MyEnum :: enum { X }; MyEnum.X :: (s: *MyEnum) -> s.*.as(u8);";
        // TODO: better error message
        test(code).error("duplicate definition of `MyEnum.X`", substr!("MyEnum.X"));
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

#[test]
fn struct_layout() {
    #[repr(C)]
    #[rustfmt::skip]
    struct A { a: i64, b: u8, c: u16 }
    // layout: `aaaaaaaab_cc____`
    let res = test_analyzed_struct("struct { a: i64, b: u8, c: u16 }");

    assert_eq!(crate::type_::struct_size(&res.data.fields), std::mem::size_of::<A>());
    assert_eq!(crate::type_::struct_alignment(&res.data.fields), std::mem::align_of::<A>());

    assert_eq!(crate::type_::struct_offset(&res.data.fields, 0), offset_of!(A, a));
    assert_eq!(crate::type_::struct_offset(&res.data.fields, 1), offset_of!(A, b));
    assert_eq!(crate::type_::struct_offset(&res.data.fields, 2), offset_of!(A, c));
}
