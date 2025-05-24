use super::{jit_run_test, test_compile_err, TestSpan};
use crate::{tests::jit_run_test_raw, util::transmute_unchecked};

#[test]
fn basic_enum() {
    let code = "
MyBasicEnum :: enum { A, B, C };
test :: -> {
    mut val := MyBasicEnum.A;
    val = MyBasicEnum.C;
    val
};";
    assert_eq!(*jit_run_test_raw::<u8>(code).ok(), 2);
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
    let out = *jit_run_test_raw::<MySumType>(code).ok();
    println!("got bits  : {:016x?}", transmute_unchecked::<_, [u64; 2]>(&out));
    println!("got       : {:?}", out);
    assert_eq!(out.tag, 0);

    let code = format!("{typedef}; test :: -> MySumType.Int(-13);");
    let out = *jit_run_test_raw::<MySumType>(code).ok();
    println!("got bits  : {:016x?}", transmute_unchecked::<_, [u64; 2]>(&out));
    println!("got       : {:?}", out);
    assert_eq!(out.tag, 1);
    assert_eq!(transmute_unchecked::<_, i64>(&out.val), -13);

    let code = format!("{typedef}; test :: -> MySumType.Float(-331.5);");
    let out = *jit_run_test_raw::<MySumType>(code).ok();
    println!("got bits  : {:016x?}", transmute_unchecked::<_, [u64; 2]>(&out));
    println!("got       : {:?}", out);
    let val = transmute_unchecked::<_, f64>(&out.val);
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
    let out = *jit_run_test_raw::<MySumType>(code).ok();
    println!("got bits  : {:016x?}", transmute_unchecked::<_, [u64; 2]>(&out));
    println!("got       : {:?}", out);
    assert_eq!(out.tag, 1);
    assert_eq!(transmute_unchecked::<_, i64>(&out.val), -13);
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
    let out = *jit_run_test_raw::<MySumType>(code).ok();
    assert_eq!(out.tag, 1);
    let InnerA { a, b } = unsafe { out.inner.B };
    assert_eq!(a, -17);
    assert_eq!(transmute_unchecked::<_, f32>(&b), 10.123);
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
    let out = *jit_run_test::<MyEnum>(code).ok();
    assert_eq!(out.tag, 1);
    assert_eq!(out.a, 5);
    assert_eq!(out.b, 10.123);
}

#[test]
fn return_inner_ty() {
    for (variant, expected_tag) in [("A", 0), ("B", 1), ("C", 2)] {
        let res = jit_run_test_raw::<u8>(&format!(
            "
            test :: -> {{
                MyBasicEnum :: enum {{ A, B, C }};
                MyBasicEnum.{variant}
            }};"
        ));
        assert_eq!(*res.ok(), expected_tag);
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
    assert!(*jit_run_test::<bool>(code).ok())
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
    assert!(*jit_run_test::<bool>(code).ok())
}

#[test]
fn custom_enum_tag_value() {
    assert_eq!(*jit_run_test::<i32>("enum { A = 10, B, C }.A").ok(), 10);
    assert_eq!(*jit_run_test::<i32>("enum { A = 10, B, C }.B").ok(), 11);
}

#[test]
fn no_noundef_with_sum_type() {
    let res = jit_run_test::<u16>("enum { A(u8), B, C }.B");
    assert_eq!(*res.ok(), 1);
    assert!(!res.module_text().unwrap().contains("noundef"));
}

#[test]
fn cast_negative_tag() {
    let code = "
MyEnum :: enum { A = -1, B = -2, C = -3 };
test :: -> MyEnum.B.as(i64);";
    assert_eq!(*jit_run_test_raw::<i64>(code).ok(), -2)
}

#[test]
fn invalid_tag_ty() {
    test_compile_err(
        "enum { A = \"hello\" }",
        "mismatched types: expected {integer literal}; got []u8",
        |code| TestSpan::of_substr(code, "\"hello\"")
    );
}
