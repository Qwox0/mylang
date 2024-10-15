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
    c = [1.0, 2.0, 3.0],
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
    c = [1.0, 2.0, 3.0],
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
    c = [1.0, 2.0, 3.0],
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
a := struct { a := 5, b := true, c := [1.0, 2.0, 3.0] }.{};

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
    c = [5.2, 2.0, 3.3],
};
mut sum := 0.0;
ptr.*.c | for x {
    sum += x;
};
sum";
    let out = jit_run_test!(code => f64).unwrap();
    assert_eq!(out, 10.5);
}

// TODO: test errors for invalid cases
