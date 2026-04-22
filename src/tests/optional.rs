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

    fn is_null(&self) -> bool {
        self.tag == 0
    }

    fn into_option(self) -> Option<Data> {
        if self.tag == 0 { None } else { Some(self.data) }
    }
}

#[test]
fn optional_type_check() {
    #[track_caller]
    fn t(ty: &str, expected_ty: &str, res: Result<(), &str>) {
        #[rustfmt::skip]
        let t = test(format!("// {ty} -> {expected_ty}
T :: struct {{ x: i32 }};
NonNull :: struct {{ x: enum {{ A = 1, B }} }};
test :: -> {expected_ty} {{ val: {ty}; val }}"
        ));
        if let Err(msg) = res {
            t.error(msg, substr!("val";skip=1));
        } else {
            t.compile_no_err();
        }
    }

    t("???T", "???T", Ok(()));

    t("?i32", "?never", Err("mismatched types: expected `?never`; got `?i32`")); // int is not assignable to never

    t("?T", "T", Err("mismatched types: expected `T`; got `?T`"));
    t("?NonNull", "NonNull", Err("mismatched types: expected `NonNull`; got `?NonNull`"));

    t("T", "?T", Err("mismatched types: expected `?T`; got `T`")); // T is not non-null
    t("NonNull", "?NonNull", Ok(()));
    t("*NonNull", "*?NonNull", Ok(()));

    t("?NonNull", "??NonNull", Err("mismatched types: expected `??NonNull`; got `?NonNull`")); // Optional is not non-null

    t("?never", "??T", Ok(()));
    t("never", "?T", Ok(()));

    t("??never", "?T", Err("mismatched types: expected `?T`; got `??never`"));

    t("??T", "?any", Ok(()));
    t("?T", "any", Ok(()));

    t("T", "?any", Err("mismatched types: expected `?any`; got `T`"));
    t("NonNull", "?any", Ok(()));
}

#[test]
fn optional_common_type() {
    #[track_caller]
    fn t(lhs_ty: &str, rhs_ty: &str, res: Result<&str, &str>) {
        #[rustfmt::skip]
        let t = test(format!("// {lhs_ty} & {rhs_ty}
T :: struct {{ x: i32 }};
NonNull :: struct {{ x: enum {{ A = 1, B }} }};
test :: -> {{ lhs: {lhs_ty}; rhs: {rhs_ty}; .[lhs, rhs] }}"
        ));
        match res {
            Ok(expected_ret_ty) => {
                t.compile_no_err().ret_ty(&format!("[2]{expected_ret_ty}"));
            },
            Err(msg) => {
                t.error(msg, substr!("rhs";skip=1));
            },
        }
    }

    t("???T", "???T", Ok("???T"));

    t("T", "?T", Err("mismatched types: expected `T`; got `?T`"));
    t("?T", "T", Err("mismatched types: expected `?T`; got `T`"));

    t("NonNull", "?NonNull", Ok("?NonNull"));
    t("?NonNull", "NonNull", Ok("?NonNull"));

    t("?NonNull", "??NonNull", Err("mismatched types: expected `?NonNull`; got `??NonNull`"));
    t("??NonNull", "?NonNull", Err("mismatched types: expected `??NonNull`; got `?NonNull`"));

    t("??T", "?never", Ok("??T"));
    t("?never", "??T", Ok("??T"));

    t("?T", "??never", Err("mismatched types: expected `?T`; got `??never`"));
    t("??never", "?T", Err("mismatched types: expected `??never`; got `?T`"));

    t("??T", "?any", Ok("?any"));
    t("?any", "??T", Ok("?any"));
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
    for lhs in [Optional::NULL, some(123_i32)] {
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
    assert!(res.is_null());
}

#[test]
fn error_optional_orelse_invalid() {
    test("test :: -> Some(123) orelse \"\";")
        .error("mismatched types: expected `{integer}`; got `[]u8`", substr!("\"\""));
}

#[test]
fn error_orelse_invalid_lhs() {
    let code = "
f :: (i: i32) -> i + 1;
get_opt_i64 :: -> ?i64 Some(10);
test :: -> get_opt_i64() orelse 20 |> f();
//                              ^^ no second error here!
";
    test(code)
        .error("mismatched types: expected `i32`; got `i64`", substr!("get_opt_i64() orelse 20"));

    let code = "
f :: (i: i32) -> i + 1;
test :: -> Some(\"abc\") orelse 20 |> f();
";
    test(code)
        .error("mismatched types: expected `[]u8`; got `{integer}`", substr!("20"))
        .error("mismatched types: expected `i32`; got `[]u8`", substr!("Some(\"abc\") orelse 20"));
    // 2 errors are fine here
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
pub(super) fn optional_repr() {
    #[track_caller]
    fn test_size(ty: &str, expected_size: usize) {
        #[rustfmt::skip]
        let code = format!("
Ty :: {ty};
a: struct {{ start: Ty, end: u8 }};
llvm_size := a.end.&.as(usize) - a.start.&.as(usize);
struct {{ mylang_size: usize, llvm_size: usize }}.(#sizeof(Ty), llvm_size)
            ");
        #[repr(C)]
        #[derive(Debug, PartialEq, Eq)]
        struct TypeSizes {
            mylang_size: usize,
            llvm_size: usize,
        }
        test_body(code)
            .with_prelude()
            .ok(TypeSizes { mylang_size: expected_size, llvm_size: expected_size });
    }

    test_size(" struct { a: i32 }", 4);
    test_size("?struct { a: i32 }", 8);
    test_size(" struct { n: never, a: i32 }", 4);
    test_size("?struct { n: never, a: i32 }", 8);
    test_size(" struct { a: *i32 }", 8);
    test_size("?struct { a: *i32 }", 8);

    //test_size(" never", 0); // TODO: Addrof never causes codegen problems
    test_body("#sizeof( never)").ok(0_usize);
    test_size("?never", 0);

    test_size(" enum {         }", 0); // []
    test_size("?enum {         }", 0); // [null]
    test_size(" enum { A       }", 0); // [0]
    test_size("?enum { A       }", 1); // [null, 0]
    test_size(" enum { A = 123 }", 0); // [123]
    test_size("?enum { A = 123 }", 1); // [null, 123]
    test_size(" enum { A(i32)  }", 4); // [i32]
    test_size("?enum { A(i32)  }", 8); // [null, i32]
    test_size(" enum { A(*i32) }", 8); // [*i32]
    test_size("?enum { A(*i32) }", 8); // [?*i32]

    test_size("?any", 1);
}

#[test]
fn optional_c_ffi_type() {
    let res = test("test :: -> ?never  null;").compile_no_err();
    assert!(res.llvm_ir().contains("ret void"));
}

#[test]
fn is_some_codegen() {
    let code = "
SmallNonNull :: enum { A = 123, B };
Some(SmallNonNull.A) orelse .B
";
    test_body(code).with_prelude().ok(123_u8);

    let code = "
SmallNonNull :: enum { A = 123, B };
mut val: union { a: SmallNonNull, x: *i32 };
val.x = 0x100.as(*i32);
Some(val) orelse return 1;
val.x.as(usize)
";
    test_body(code).with_prelude().ok(0x100_i32);
}

#[test]
fn coerce_orelse_lhs() {
    // Cannot increase number of wrapped optionals on lhs with coercion
    let code = "
NonNullEnum :: enum { A = 123, B };
a := NonNullEnum.A;
Some(a) orelse Some(.B)
//   ^ cannot coerce `T` to `?T`
";
    test_body(code)
        .error("mismatched types: expected `NonNullEnum`; got `?NonNullEnum`", substr!("Some(.B)"))
        .info("Consider using `or` operator instead", substr!("orelse"));

    let code = "
NonNullEnum :: enum { A = 123, B };
a: ?NonNullEnum = Some(NonNullEnum.A);
a orelse Some(.B)
";
    test_body(code)
        .error("mismatched types: expected `NonNullEnum`; got `?NonNullEnum`", substr!("Some(.B)"))
        .info("Consider using `or` operator instead", substr!("orelse"));

    let code = "
NonNullEnum :: enum { A = 123, B };
Some(NonNullEnum.A) orelse Some(.B)
";
    test_body(code)
        .error("mismatched types: expected `NonNullEnum`; got `?NonNullEnum`", substr!("Some(.B)"))
        .info("Consider using `or` operator instead", substr!("orelse"));

    // Can do non-optional coercion
    let code = "
mut x := 1;
mut_ptr := &mut x;
const_ptr := &x;
Some(mut_ptr) orelse const_ptr
//   ^ can coerce `*mut T` to `*T`
";
    test_body(code).ok_stack_ptr().ret_ty("*i64");
}

#[test]
fn infer_in_some() {
    let code = "
MyStruct :: struct { a: i32 };
test :: -> ?MyStruct { Some(.{ a=-123 }) }
";
    test(code).ok(some(-123_i32));

    test("test :: -> ??i32 { Some(-123) }").error(
        "mismatched types: expected `??i32`; got `?{signed integer}`",
        substr!("Some(-123)"),
    );
    /* TODO: decide if these errors are better:
        .error("mismatched types: expected `?i32`; got `{signed integer}`", substr!("-123"))
    */
}

#[test]
fn no_coercion_to_optional_in_some() {
    // Can coerce `*mut i64` -> `*i64`
    let code = "
test :: -> ?*i64 {
    mut a := 123;
    Some(&mut a)
}";
    test(code).compile_no_err();

    // Cannot coerce to optional
    let code = "
test :: -> ??*mut i64 {
    mut a := 123;
    Some(&mut a)
}";
    test(code)
        .error("mismatched types: expected `??*mut i64`; got `?*mut i64`", substr!("Some(&mut a)"));
}

#[test]
fn return_nested_optional_null() {
    test("test :: -> ??*u8        null;").ok(Optional::<OPtr<u8>>::NULL);
    test("test :: -> ??*u8 return null;").ok(Optional::<OPtr<u8>>::NULL);
}

/// Problem: `opt.* orelse ...` creates a stack allocation. Thus the function returns an invalid
/// pointer.
/// TODO: `<*?T> orelse <*T>` -> `<*T>`
/// Idea: disallow accidental stack allocations (only allow `&` for l-values). add something like
/// `#alloca(r_value)` or `&tmp r_value`
#[test]
#[ignore = "todo"]
fn optional_as_mut() {
    let code = "
T :: struct { val: i32 };
as_mut :: (opt: *mut ?T) -> ?*mut T
    Some(&mut (opt.* orelse return null));
test :: -> {
    mut opt: ?T = Some(.(1));
    ref := opt.&mut.as_mut();
    ref := ref orelse return null;
    ref.*.val = 10;
    opt
}";
    test(code).ok(some(10_i32));
}

#[test]
#[ignore = "todo"]
fn fix_mutated_var_ty_of_const_decl() {
    let code = "
A :: Some(1);
f :: -> Some(&mut A); // TODO: this finalizes A during sema, which is incorrect
test :: -> { a: ?u32 = A; }";
    test(code).ok(123_i64);

    let code = "
A :: Some(1);
test :: -> {
    a: i64 = A orelse return 0; // TODO: this finalizes A during codegen, which is incorrect
    b: ?u32 = A;
    123
}";
    test(code).ok(123_i64);
}

/// ```llvm
/// %opt_int = alloca { i8, i32 }, align 4
/// %opt_never = alloca [0 x i8], align 1
/// store [0 x i8] zeroinitializer, ptr %opt_never, align 1
/// call void @llvm.memcpy.p0.p0.i64(ptr align 4 %opt_int, ptr align 4 %opt_never, i64 8, i1 false)
/// %ret = load i64, ptr %opt_int, align 4
/// ret i64 %ret
/// ```
#[test]
#[ignore = "todo"]
fn codegen_bug_tag_uninitialized() {
    let code = "
opt_never: ?never = null;
opt_int: ?i32 = opt_never;
opt_int
";
    test_body(code).ok(Optional::<i32>::NULL);
}
