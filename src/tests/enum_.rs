use crate::{
    tests::{substr, test, test_body, test_parse},
    util::transmute_unchecked,
};

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
    let res = test_body("enum { A(u8), B, C }.B").ok(1u16);
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
        .error("mismatched types: expected `{integer literal}`; got `[]u8`", substr!("\"hello\""));
}

#[test]
fn error_expected_ident() {
    // we want `expected ident` instead of `expected '}'`
    test_parse("E :: enum { += };").error("expected an identifier, got `+=`", substr!("+="));
}

#[test]
fn one_variant() {
    // in local
    let res = test_body("a := enum { OneVariant }.OneVariant; a").ok(());
    assert!(res.llvm_ir().contains("define void @test()"));
    assert!(res.llvm_ir().contains("alloca {}, align 1")); // TODO: allow this?
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
    let res = test(code).ok([7i64, 456]);
    assert!(res.llvm_ir().contains("alloca { {}, { { i64, i64 }, [0 x i8] } }, align 8"));
    assert!(res.llvm_ir().contains("ret { i128 } %ret"));
    drop(res);

    // in constant
    let code = "
MyEnum :: enum { OneVariant(struct { a := 123, b := 456 }) };
CONST :: MyEnum.OneVariant(.{ b=7 });
test :: -> CONST;";
    let res = test(code).ok([123i64, 7]);
    let const_val = "{ {} zeroinitializer, { i64, i64 } { i64 123, i64 7 } }";
    assert!(res.llvm_ir().contains(const_val));
    assert!(res.llvm_ir().contains("ret { i128 } %ret"));
}

#[test]
fn good_error_message3() {
    test_body("A :: enum { B }; x := 1; A.B.(1);")
        .error("Cannot apply a positional initializer to a value of type `A`", substr!("A.B"));
}

#[test]
fn good_error_message4() {
    test("A :: enum { B }; test :: -> A { .B.(1) };")
        .error("Cannot infer enum type", substr!(".B.(1)";.start_with_len(2)));

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
    let res = test_body("MyEnum :: enum { A, B = 1000000.as(u32), C }; MyEnum.C").ok(1000001u32);
    assert!(res.llvm_ir().contains("ret i32 1000001"));
    drop(res);

    test("MyEnum :: enum { A, B = 1000000.as(u32), C = -1 }")
        .error("Cannot apply unary operator `-` to type `u32`", substr!("-1"));
}
