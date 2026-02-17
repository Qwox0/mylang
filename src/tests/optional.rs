use crate::{
    ast::debug::DebugAst,
    ptr::OPtr,
    tests::{substr, test, test_body},
};
use std::fmt;

#[derive(Debug, Clone, Copy, Eq)]
#[repr(C)]
struct Optional<Data> {
    tag: u8,
    data: Data,
}

impl<Data: PartialEq> PartialEq for Optional<Data> {
    fn eq(&self, other: &Self) -> bool {
        self.tag == other.tag && (self.tag == 0 || self.data == other.data)
    }
}

impl<Data: fmt::Display> fmt::Display for Optional<Data> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.tag == 0 { "null".fmt(f) } else { write!(f, "Some({})", self.data) }
    }
}

fn some<Data>(data: Data) -> Optional<Data> {
    Optional { tag: 1, data }
}

impl<Data> Optional<Data> {
    const NULL: Optional<Data> =
        Optional { tag: 0, data: unsafe { std::mem::MaybeUninit::zeroed().assume_init() } };

    fn is_none(&self) -> bool {
        self.tag == 0
    }

    fn into_option(self) -> Option<Data> {
        if self.tag == 0 { None } else { Some(self.data) }
    }
}

#[test]
fn create_some() {
    let code = "test :: -> ?i16 return Some(123);";
    let res = test(code).ok(some(123_i16));
    assert!(res.llvm_ir().contains("define i32 @test()"));
    drop(res);

    let code = "test :: -> ?i64 return Some(123);";
    let res = test(code).ok(some(123_i64));
    assert!(res.llvm_ir().contains("define { i64, i64 } @test()"));
}

#[test]
fn dynamic_const_type_behind_optional() {
    let code = "
VAL :: Some(123);
static a: ?i8 = VAL; // VAL can be any ?<int>
static b: ?i64 = VAL;
test :: -> ?i32 VAL;";
    let res = test(code).ok(some(123_i32));
    assert!(res.llvm_ir().contains("@a = constant { i8, i8 } { i8 1, i8 123 }, align 1"));
    assert!(res.llvm_ir().contains("@b = constant { i8, i64 } { i8 1, i64 123 }, align 8"));
    assert!(res.llvm_ir().contains("define i64 @test()"));
}

#[test]
fn create_none() {
    let code = "test :: -> ?i32 return null;";
    let res = test(code).ok(Optional::<i32>::NULL);
    assert!(res.llvm_ir().contains("define i64 @test()"));
    drop(res);

    let code = "test :: -> ?i64 return null;";
    let res = test(code).ok(<Optional<i64>>::NULL);
    assert!(res.llvm_ir().contains("define { i64, i64 } @test()"));
}

#[test]
fn coerce_non_null_to_optional() {
    let res = test("static num: u8 = 123; test :: -> ?*u8 { ptr: *u8 = &num; ptr };")
        .get_out::<*const u8>();
    assert!(!res.is_null());
    assert!(format!("{res:p}").starts_with("0x7f"));
    assert_eq!(unsafe { *res }, 123);

    let res = test("test :: -> ?*u8 { &1 };").get_out::<*const u8>();
    assert!(!res.is_null());
    assert!(format!("{res:p}").starts_with("0x7f"));
}

#[test]
fn orelse_unwrapping() {
    let default = 10;
    for lhs in [Optional::NULL, some(123)] {
        let res = test(format!("test :: -> i32 {lhs} orelse {default};")).get_out::<i32>();
        assert_eq!(res, lhs.into_option().unwrap_or(default));
    }

    test("test :: (opt: ?i32) -> i32 { opt orelse 10 }").compile_no_err();

    test("test :: -> ?i32 Some(Some(123)) orelse null;").ok(some(123_i32));
}

#[test]
fn orelse_never() {
    for lhs in [Optional::NULL, some(123)] {
        let res = test_body(format!("a: i32 = {lhs} orelse return 10; a + 1")).get_out::<i32>();
        assert_eq!(res, lhs.into_option().map(|x| x + 1).unwrap_or(10));
    }

    test_body(format!("a: i32 = (null orelse return 10) + 1; a")).ok(10_i32);
}

#[test]
fn orelse_infer() {
    test("MyStruct :: struct { val: i32 }; test :: -> MyStruct { Some(.(3)) orelse .(5) } ")
        .ok(3_i32);
    test("MyStruct :: struct { val: i32 }; test :: -> MyStruct { null orelse .(5) } ").ok(5_i32);
    test("MyStruct :: struct { val: i32 }; test :: -> { Some(MyStruct.(3)) orelse .(5) } ")
        .ok(3_i32);
    test("MyStruct :: struct { val: i32 }; test :: -> { a: ?MyStruct = null; a orelse .(5) } ")
        .ok(5_i32);
}

#[test]
fn orelse_combine_types() {
    use crate::ast::*;
    let res = test("test :: (a: *mut i32, b: *i32) -> { Some(a) orelse b } ").compile_no_err();
    let func = res.one_stmt::<Decl>().init.unwrap().downcast::<Fn>();
    let ret_ptr_ty = func.ret_ty.unwrap().downcast::<PtrTy>();
    debug_assert!(!ret_ptr_ty.is_mut);
    drop(res);

    test("MyStruct :: struct { val: i32 }; test :: -> { null orelse MyStruct.(5) } ").ok(5_i32);
}

#[test]
fn error_optional_orelse_optional() {
    test("Wrong :: struct{}; test :: -> Wrong Some(12) orelse null;")
        .error("mismatched types: expected `{integer}`; got `?never`", substr!("null"))
        .info("Consider using `or` operator instead", substr!("orelse"))
        .error(
            "mismatched types: expected `Wrong`; got `{integer}`",
            substr!("Some(12) orelse null"),
        );
    /* TODO: decide if these errors are better:
        .error("mismatched types: expected `Wrong`; got `{integer}`", substr!("12"))
        .error("mismatched types: expected `Wrong`; got `?never`", substr!("None"))
        .info("Consider using `or` operator instead", substr!("orelse"));
    */

    test("test :: -> ??i32 Some(Some(Some(123))) orelse Some(Some(null));")
        .error("mismatched types: expected `??i32`; got `???never`", substr!("Some(Some(null))"))
        .info("Consider using `or` operator instead", substr!("orelse"));

    // inference still works
    test("A :: struct {val:i32}; test :: -> A Some(.{val=1}) orelse Some(.{val=2});")
        .error("mismatched types: expected `A`; got `?A`", substr!("Some(.{val=2})"))
        .info("Consider using `or` operator instead", substr!("orelse"));

    test("test :: -> i32 null orelse null;")
        //.error("mismatched types: expected `i32`; got `?never`", substr!("null";skip=1)); // TODO: is this hint better?
        .error("mismatched types: expected `i32`; got `?never`", substr!("null orelse null"));
    // `null orelse null` can be valid:
    let res = test("test :: -> ?i32 null orelse null;").get_out::<Optional<i32>>();
    assert!(res.is_none());
}

#[test]
fn error_optional_orelse_invalid() {
    test("test :: -> Some(123) orelse \"\";")
        .error("mismatched types: expected `{integer}`; got `[]u8`", substr!("\"\""));
}

#[test]
#[ignore = "not implemented"]
fn optional_or_optional() {
    for lhs in [None, Some(123)] {
        for default in [None, Some(456)] {
            let res = test(format!("test :: -> ?*i32 {lhs:?} or {default:?};"))
                .get_out::<Optional<i32>>();
            let expected = lhs.or(default);
            assert_eq!(res.tag, expected.is_some() as u8);
            if let Some(expected) = expected {
                assert_eq!(res.data, expected);
            }
        }
    }
}

#[test]
fn orelse_on_complex_non_null_struct() {
    let code = "
MyNonNullStruct :: struct { a: f64, b: [4]i32, c: [2]*f64, d: bool };
test :: -> {
    a := 3.14;
    mut val := Some(MyNonNullStruct.(a, .[1, 2, 3, 4], .[&a, &a], true));
    val orelse return false;

    val = null;
    val orelse return true;
    return false;
}";
    test(code).ok(true);

    let code = "
MyNonNullStruct :: struct { a: f64, b: [4]i32, c: [2]*f64, d: bool };
VAL :: Some(MyNonNullStruct.(3.14, .[1, 2, 3, 4], .[0x7f123.as(*f64); 2], true));
test :: -> {
    VAL orelse return false;
    return true;
}";
    test(code).ok(true);
}

#[test]
fn correctly_combine_return_types() {
    use crate::ast::*;

    let code = "
test :: (opt_ptr: ?*u8) -> {
    ptr := opt_ptr orelse return null;
    ptr
};";
    let res = test(code).compile_no_err();
    let func = res.one_stmt::<Decl>().init.unwrap().downcast::<Fn>();
    assert_eq!(func.ret_ty.unwrap().to_string(), "?*u8");
    drop(res);

    let res =
        test("test :: (x: ?*f64) -> []f64.{ ptr=x orelse return null, len=1 };").compile_no_err();
    assert_eq!(res.one_decl_init::<Fn>().ret_ty.unwrap().to_text(true), "?[]f64");
    drop(res);

    let res = test("test :: (x: ?*f64) -> { s: []f64 = .{ ptr=x orelse return null, len=1 }; s };")
        .compile_no_err();
    assert_eq!(res.one_decl_init::<Fn>().ret_ty.unwrap().to_text(true), "?[]f64");
}

#[test]
fn error_cannot_coerce_zeroable_to_optional() {
    test_body("if true return null; 5")
        .error("mismatched types: expected `?never`; got `{integer}`", substr!("5"))
        // TODO: .hint("Consider explicitly wrapping the value with `Some`", |_| todo!())
    ;
}

#[test]
fn orelse_with_out_coercion() {
    // ?*mut u8 -> orelse -> *mut u8 -> coerce -> ?*mut u8
    let code = "
test :: () -> ?*mut u8 {
    1.as(?*mut u8) orelse return null
}";
    test(code).ok(1_u64);

    let code = "
test :: () -> {
    1.as(?*mut u8) orelse return null
}";
    test(code).ok(1_u64);
}

#[test]
fn return_nested_optional_null() {
    test("test :: -> ??*u8        null;").ok(Optional::<OPtr<u8>>::NULL);
    test("test :: -> ??*u8 return null;").ok(Optional::<OPtr<u8>>::NULL);
}
