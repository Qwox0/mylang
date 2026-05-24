use crate::tests::{TestSpan, substr, test};

#[test]
fn recursive_value_cannot_infer_type() {
    let code = "
MyStruct :: struct {
    val: i32,
    prev: *MyStruct,
}
static STATIC := MyStruct.{ val = 123, prev = &STATIC };
";
    test(code).error("cycle(s) detected:", |_| TestSpan::ZERO);
    // TODO: better error like "cannot infer type of recursive value. Consider explicitly annotating the type."
}

#[test]
#[ignore = "todo?"]
fn use_static_in_const() {
    let code = "
static STATIC := 123;
CONST :: STATIC;
";
    test(code).compile_no_err();

    let code = "
mut static STATIC := 123;
CONST :: STATIC;
";
    test(code).error("", substr!("STATIC";skip=1));
}

#[test]
fn static_vs_const_codegen() {
    let code = "
static MY_STATIC := 10;
f :: (x: *i64) -> {};
test :: -> f(&MY_STATIC);
";
    let res = test(code).compile_no_err();
    assert!(res.llvm_ir().contains("@MY_STATIC = internal constant i64 10, align 8")); // symbol for MY_STATIC
    assert!(!res.llvm_ir().contains("alloca i64")); // no stack allocation
    assert!(res.llvm_ir().contains("call void @f(ptr noundef @MY_STATIC)")); // use ptr to global
    drop(res);

    // TODO: move to `tests/consts`
    let code = "
MY_CONST :: 10;
f :: (x: *i64) -> {};
test :: -> f(&MY_CONST);
";
    let res = test(code).compile_no_err();
    assert!(!res.llvm_ir().contains("@MY_CONST")); // no symbol for MY_CONST
    assert!(res.llvm_ir().contains("%0 = alloca i64, align 8")); // stack allocation for value
    assert!(res.llvm_ir().contains("call void @f(ptr noundef %0)")); // use ptr to stack allocation
}

#[test]
fn recursive_static() {
    let code = "
MyStruct :: struct { val: i32, prev: *MyStruct };
static STATIC: MyStruct = .{ val = 123, prev = &STATIC };
static mut MUT_STATIC: MyStruct = .{ val = 123, prev = &MUT_STATIC };
";
    let res = test(code).compile_no_err();
    assert!(
        res.llvm_ir()
            .contains("@STATIC = internal constant { i32, ptr } { i32 123, ptr @STATIC }, align 8")
    );
    assert!(res.llvm_ir().contains(
        "@MUT_STATIC = internal global { i32, ptr } { i32 123, ptr @MUT_STATIC }, align 8"
    ));
    drop(res);
}

#[test]
fn indirectly_recursive_statics() {
    let code = "
A :: struct { b: *B };
B :: struct { a: *A };
static a: A = .{ b = &b };
static b: B = .{ a = &a };
";
    test(code).compile_no_err();
}

#[test]
fn fn_ptr_in_static() {
    let code = "
f :: -> 1;
static f_ptr := &f;
static lambda_ptr := & -> 2;

test :: -> f_ptr.*() + lambda_ptr.*();
";
    test(code).ok(3_i64);
}
