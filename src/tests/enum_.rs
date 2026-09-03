use crate::{tests::*, util::transmute_unchecked};

#[test]
fn basic_enum() {
    let code = "
MyBasicEnum :: enum { A, B, C };
test :: -> {
    mut val := MyBasicEnum.A;
    val = MyBasicEnum.C;
    val
};";
    test(code).ok(2u8);
}

#[test]
fn sum_type() {
    // TODO: finalize enum syntax

    /*
    #[derive(Debug)]
    #[repr(C)]
    enum MySumType {
        Void,
        Int(i64),
        Float(f64),
    }
    */
    #[derive(Debug, Clone, Copy)]
    #[repr(C)]
    struct MySumType {
        tag: u8,
        val: i64, // don't change this to f64
    }

    let typedef = "
MySumType :: enum {
    Void,
    Int(i64),
    Float(f64),
}";

    let code = format!("{typedef}; test :: -> MySumType.Void;");
    let out = test(code).get_out::<MySumType>();
    println!("got bits  : {:016x?}", transmute_unchecked::<_, [u64; 2]>(out));
    println!("got       : {:?}", out);
    assert_eq!(out.tag, 0);

    let code = format!("{typedef}; test :: -> MySumType.Int(-13);");
    let out = test(code).get_out::<MySumType>();
    println!("got bits  : {:016x?}", transmute_unchecked::<_, [u64; 2]>(out));
    println!("got       : {:?}", out);
    assert_eq!(out.tag, 1);
    assert_eq!(transmute_unchecked::<_, i64>(out.val), -13);

    let code = format!("{typedef}; test :: -> MySumType.Float(-331.5);");
    let out = test(code).get_out::<MySumType>();
    println!("got bits  : {:016x?}", transmute_unchecked::<_, [u64; 2]>(out));
    println!("got       : {:?}", out);
    let val = transmute_unchecked::<_, f64>(out.val);
    assert_eq!(out.tag, 2);
    assert_eq!(val, -331.5);

    let code = &format!(
        "{typedef};
test :: -> {{
    mut val := MySumType.Void;
    val = MySumType.Int(-13);
    return val;
}}"
    );
    let out = test(code).get_out::<MySumType>();
    println!("got bits  : {:016x?}", transmute_unchecked::<_, [u64; 2]>(out));
    println!("got       : {:?}", out);
    assert_eq!(out.tag, 1);
    assert_eq!(transmute_unchecked::<_, i64>(out.val), -13);
}

#[test]
fn anon_struct_in_sum_type() {
    #[derive(Clone, Copy)]
    #[repr(C)]
    struct MySumType {
        tag: u8,
        inner: Inner,
    }
    #[derive(Clone, Copy)]
    #[allow(non_snake_case)]
    #[repr(C)]
    union Inner {
        /// # Important
        /// This field increases the aligment of [`Inner`] to 8. Therefore this field is also
        /// needed in Rust to prevent ABI problems.
        A: i64,
        B: InnerA,
    }
    #[derive(Clone, Copy)]
    #[repr(C)]
    struct InnerA {
        a: i32,
        b: i32,
    }

    let code = "
MySumType :: enum {
    A(i64),
    B(struct { a: i32, b: f32 }),
    C,
};
test :: -> MySumType.B(.{ a = -17, b = 10.123 });";
    let out = test(code).get_out::<MySumType>();
    assert_eq!(out.tag, 1);
    let InnerA { a, b } = unsafe { out.inner.B };
    assert_eq!(a, -17);
    assert_eq!(transmute_unchecked::<_, f32>(b), 10.123);
}

#[test]
fn infer_enum_ty() {
    #[derive(Clone, Copy)]
    #[repr(C)]
    struct MyEnum {
        tag: u8,
        a: i64,
        b: f64,
    }

    let code = "
MyEnum :: enum {
    A,
    B(struct { a: i64, b: f64 }),
};
mut val: MyEnum = .A;
val = .B(.{ a = 5, b = 10.123 });
val";
    let out = test_body(code).get_out::<MyEnum>();
    assert_eq!(out.tag, 1);
    assert_eq!(out.a, 5);
    assert_eq!(out.b, 10.123);
}

#[test]
fn return_inner_ty() {
    for (variant, expected_tag) in [("A", 0u8), ("B", 1), ("C", 2)] {
        test(&format!(
            "test :: -> {{
                MyBasicEnum :: enum {{ A, B, C }};
                MyBasicEnum.{variant}
            }};"
        ))
        .ok(expected_tag);
    }
}

#[test]
#[ignore = "unfinished test"]
fn enum_eq() {
    let code = "
MyBasicEnum :: enum { A, B, C };
a := MyBasicEnum.A;
if a == MyBasicEnum.B return false;
if a != MyBasicEnum.A return false;
true";
    test_body(code).ok(true);
}

#[test]
#[ignore = "unfinished test"]
fn variant_eq() {
    let code = "
MyBasicEnum :: enum { A, B(i32), C(struct { a: u8, b: i32 }) };
val := MyBasicEnum.B(5);
if val == MyBasicEnum.B {
    if val != 5 return false;
};
true";
    test_body(code).ok(true);
}

#[test]
fn custom_enum_tag_value() {
    test_body("enum { A = 10, B, C }.A").ok(10i32);
    test_body("enum { A = 10, B, C }.B").ok(11i32);
}

#[test]
fn no_noundef_with_sum_type() {
    let res = test_body("enum { A(u8), B, C }.B").ok((1_u8, any::<u8>()));
    assert!(!res.llvm_ir().contains("noundef"));
}

#[test]
fn cast_negative_tag() {
    let code = "
MyEnum :: enum { A = -1, B = -2, C = -3 };
test :: -> MyEnum.B.as(i64);";
    test(code).ok(-2i64);
}

#[test]
fn invalid_tag_ty() {
    test_body("enum { A = \"hello\" }")
        .error("mismatched types: expected `{integer}`; got `[]u8`", substr!("\"hello\""));
}

#[test]
fn error_expected_ident() {
    // we want `expected ident` instead of `expected '}'`
    test("E :: enum { += };")
        .parse()
        .error("expected an identifier, got `+=`", substr!("+="));
}

#[test]
fn one_variant() {
    // in local
    let res = test_body("a := enum { OneVariant }.OneVariant; a").ok(());
    assert!(res.llvm_ir().contains("define void @test()"));
    assert!(!res.llvm_ir().contains("alloca"), "no stack allocation needed");
    assert!(res.llvm_ir().contains("ret void"));
    drop(res);

    // in constant
    let code = "
MyEnum :: enum { OneVariant };
CONST : MyEnum : .OneVariant;
test :: -> CONST;";
    let res = test(code).ok(());
    assert!(res.llvm_ir().contains("define void @test() {\nentry:\n  ret void")); // empty function
    assert!(!res.llvm_ir().contains("alloca {}"));
}

#[test]
fn one_variant_with_data() {
    // in local
    let code = "
MyEnum :: enum { OneVariant(struct { a := 123, b := 456 }) };
test :: -> {
    a: MyEnum = .OneVariant(.{ a=7 });
    a
}";
    let res = test(code).ok(fields([7i64, 456]));
    assert!(res.llvm_ir().contains("alloca { {}, { { i64, i64 }, [0 x i8] } }, align 8"));
    assert!(res.llvm_ir().contains("ret { i64, i64 } %ret"));
    drop(res);

    // in constant
    let code = "
MyEnum :: enum { OneVariant(struct { a := 123, b := 456 }) };
CONST :: MyEnum.OneVariant(.{ b=7 });
test :: -> CONST;";
    let res = test(code).ok(fields([123i64, 7]));
    let const_val = "{ {} zeroinitializer, { i64, i64 } { i64 123, i64 7 } }";
    assert!(res.llvm_ir().contains(const_val));
    assert!(res.llvm_ir().contains("ret { i64, i64 } %ret"));
}

#[test]
fn good_error_message3() {
    test_body("A :: enum { B }; x := 1; A.B.(1);")
        .error("Cannot apply a positional initializer to value of type `A`", substr!("A.B"));
}

#[test]
fn good_error_message4() {
    test("A :: enum { B }; test :: -> A { .B.(1) };")
        .error("Cannot infer enum variant or type of associated constant", substr!(".B"));

    test("A :: enum { B }; test :: -> A { .B(1) };").error(
        "Cannot call value of type 'A'; expected function",
        substr!(".B(1)";.start_with_len(2)),
    );
}

#[test]
fn good_error_cannot_apply_initializer_to_type() {
    // TODO: better error
    test("A :: enum { B }; test :: -> A.(1);").error(
        "Cannot initialize a value of type `A` using a positional initializer",
        substr!("A.(1)";.start_with_len(1)),
    );
}

#[test]
fn enum_repr_type() {
    let res = test_body("MyEnum :: enum { A, B = 1000000.as(u32), C }; MyEnum.C").ok(1000001_u32);
    assert!(res.llvm_ir().contains("ret i32 1000001"));
    drop(res);

    test("MyEnum :: enum { A, B = 1000000.as(u32), C = -1 }")
        .error("Cannot apply unary operator `-` to type `u32`", substr!("-1"));
}

/// see also [`super::optional::optional_repr`]
#[test]
fn enum_size() {
    test_body("#sizeof(enum { A, B, C })").ok(1_usize);
    test_body("#sizeof(enum { })").ok(0_usize);
    test_body("#sizeof(enum { A })").ok(0_usize);
}

#[test]
fn non_null_enum() {
    test_body("E :: enum { A = 123, B }; opt: ?E = E.A").ok(());

    // A == 0
    test_body("E :: enum { A, B = 123 }; opt: ?E = E.A")
        .error("mismatched types: expected `?E`; got `E`", substr!("E.A"));

    // size(enum) == 0 => 0 is invalid => not non-null
    test_body("E :: enum { A = 123 }; opt: ?E = E.A")
        .error("mismatched types: expected `?E`; got `E`", substr!("E.A"));
}

/// During EnumDef sema I incorrectly used analyze_scope with a local used_tags array. On following
/// sema passes only the new used_tags were set, meaning the order and values of the variant tags
/// were incorrect.
#[test]
fn correct_variant_tags_on_sema_pause() {
    let code = "
E :: enum {
    A,
    B,
    C = C_TAG, // Sema pauses here
    D,
    E = 11,
};
C_TAG : i32 : 10;
test :: -> E.[.A, .B, .C, .D, .E];
";
    test(code).ok(arr([0_i32, 1, 10, 11, 11]));
}

#[test]
fn const_on_enum() {
    // use variant in const
    let code = "
E :: enum {
    CONST :: E.A;
    A,
    B = 10,
    C(i64),
    D(f64) = 20,
}";
    test(code).compile_no_err();

    // use const in variant
    let code = "
E :: enum {
    START_TAG :: 5;
    A = START_TAG,
    B = 10,
    C(i64),
    D(f64) = 20,
}
test :: -> E.A;
";
    test(code).ok(5_u8);

    // cycle (UnitDependency::VarTy)
    let code = "
E :: enum {
    CONST :: E.A.as(i8);
    A = CONST,
}";
    test(code).error("cycle(s) detected:", |_| TestSpan::ZERO);

    // cycle (UnitDependency::ConstVal)
    let code = "
E :: enum {
    CONST : i8 : E.A.as(i8);
    A = CONST,
}";
    test(code).error("cycle(s) detected:", |_| TestSpan::ZERO);
}

#[test]
fn enum_c_ffi_codegen() {
    let test_input = "Hello World\n";
    let test_fd = test_fd(test_input);

    #[rustfmt::skip]
    let code = format!(r#"
printf : #varargs (fmt: *u8) -> i32 : #extern;
read : (fd: i32, buf: ?*mut any, buf_size: usize) -> isize : #extern;
abort : () -> never : #extern;
__errno_location : -> *i32 : #extern;

Result :: enum {{
    Ok(usize),
    Err(i32),

    unwrap :: (self: Result) -> switch self {{
        .Ok :> self,
        .Err :> {{
            printf("errno = %d\n".ptr, self);
            abort()
        }}
    }};
}}
result :: (result: isize) -> Result {{
    if result == -1 return .Err(__errno_location().*);
    if not (result >= 0) abort();
    .Ok(result.as(usize))
}}

test :: -> {{
    mut buf: [1024]u8;
    buf := buf[..]mut;

    result(read({test_fd}, buf.ptr, buf.len)).unwrap()
}}
"#);

    let res = test(&code)
        .with_prelude()
        //.update_options(|o| o.llvm_optimization_level = 1)
        .ok(test_input.len());
    assert!(!res.llvm_ir().contains("define { i128 } @result"));
    assert!(res.llvm_ir().contains("define { i8, i64 } @result"));
    drop(res);

    reset_test_fd(test_fd);
    let res = test(code)
        .with_prelude()
        .update_options(|o| o.llvm_optimization_level = 1)
        .ok(test_input.len());
    assert!(!res.llvm_ir().contains("define { i128 } @result"));
    assert!(res.llvm_ir().contains("define { i8, i64 } @result"));
    drop(res);
}

#[test]
fn error_missing_sum_type_data() {
    // TODO: better error message?
    test_body("a := enum { A(i32) }.A(); a")
        .error("Missing argument for parameter `A: i32`", substr!(".A()";.end()));
}
