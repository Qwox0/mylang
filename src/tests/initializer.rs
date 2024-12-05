use crate::tests::jit_run_test;

#[test]
fn initializer_on_struct_type() {
    let code = "
MyStruct :: struct {
    a: i64,
    b := true,
    c: [3]f64,
};

a := MyStruct.{
    a = 5,
    c = .[1.0, 2.0, 3.0],
};

if a.a != 5 return false;
if !a.b return false;
if a.c[0] != 1.0 return false;
if a.c[1] != 2.0 return false;
if a.c[2] != 3.0 return false;
true";
    let ok = jit_run_test!(code => bool).unwrap();
    assert!(ok);
}

#[test]
fn infer_initialize_ty() {
    let code = "
MyStruct :: struct {
    a: i64,
    b := true,
    c: [3]f64,
};

a := .{
    a = 5,
    c = .[1.0, 2.0, 3.0],
};

if a.a != 5 return false;
if !a.b return false;
if a.c[0] != 1.0 return false;
if a.c[1] != 2.0 return false;
if a.c[2] != 3.0 return false;
true";
    let ok = jit_run_test!(code => bool).unwrap();
    assert!(ok);
}

#[test]
fn initializer_on_anonymous_struct_type() {
    let code = "
a := struct { a: i64, b := true, c: [3]f64 }.{
    a = 5,
    c = .[1.0, 2.0, 3.0],
};

if a.a != 5 return false;
if !a.b return false;
if a.c[0] != 1.0 return false;
if a.c[1] != 2.0 return false;
if a.c[2] != 3.0 return false;
true";
    let ok = jit_run_test!(code => bool).unwrap();
    assert!(ok);
}

#[test]
fn initializer_on_anonymous_struct_type_with_defaults() {
    let code = "
a := struct { a := 5, b := true, c := .[1.0, 2.0, 3.0] }.{};

if a.a != 5 return false;
if !a.b return false;
if a.c[0] != 1.0 return false;
if a.c[1] != 2.0 return false;
if a.c[2] != 3.0 return false;
true";
    let ok = jit_run_test!(code => bool).unwrap();
    assert!(ok);
}

#[test]
fn initialize_ptr_to_struct() {
    let code = "
MyStruct :: struct {
    a: i64,
    b := false,
    c: [3]f64,
};

a: MyStruct;
ptr := &a;
ptr.{
    a = 5,
    c = .[5.2, 2.0, 3.3],
};
mut sum := 0.0;
ptr.*.c | for x {
    sum += x;
};
sum";
    let out = jit_run_test!(code => f64).unwrap();
    assert_eq!(out, 10.5);
}

#[test]
fn nested_initializers() {
    let code = "
MyStruct :: struct {
    a: i64,
    inner: struct { b: [4]i8, c: f32 = 12.34 },
};

a := MyStruct.{
    a = 5,
    inner = .{ b = .[2, 3, 5, 7] },
};

if a.a != 5 return false;
if a.inner.b[0] != 2 return false;
if a.inner.b[1] != 3 return false;
if a.inner.b[2] != 5 return false;
if a.inner.b[3] != 7 return false;
if a.inner.c != 12.34 return false;
true";
    let (ok, llvm_module_text) = jit_run_test!(code => bool,llvm_module).unwrap();
    assert!(ok);
    let stack_allocations = llvm_module_text.lines().filter(|l| l.contains("alloca")).count();
    assert_eq!(stack_allocations, 1, "this code should only do one stack allocation");
}

#[test]
fn positional_initializer() {
    #[derive(Debug, PartialEq)]
    #[repr(C)]
    struct Vec3 {
        x: f32,
        y: f32,
        z: f32,
    }
    let out = jit_run_test!("
Vec3 :: struct { x: f32, y: f32, z: f32 };
Vec3.(1.0, 0.0, 2.5)" => Vec3)
    .unwrap();
    assert_eq!(out, Vec3 { x: 1.0, y: 0.0, z: 2.5 });
}
