use super::jit_run_test;
use crate::util::transmute_unchecked;

#[test]
fn basic_enum() {
    let out = jit_run_test!(raw "
MyBasicEnum :: enum { A, B, C };
test :: -> {
    mut val := MyBasicEnum.A;
    val = MyBasicEnum.C;
    val
};" => u8);
    assert_eq!(out.unwrap(), 2);
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
    #[derive(Debug)]
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

    let out =
        jit_run_test!(raw &format!("{typedef}; test :: -> MySumType.Void;") => MySumType).unwrap();
    println!("got bits  : {:032x}", transmute_unchecked::<_, u128>(&out));
    println!("got       : {:?}", out);
    assert_eq!(out.tag, 0);

    let out = jit_run_test!(raw &format!("{typedef}; test :: -> MySumType.Int(-13);") => MySumType)
        .unwrap();
    println!("got bits  : {:032x}", transmute_unchecked::<_, u128>(&out));
    println!("got       : {:?}", out);
    assert_eq!(out.tag, 1);
    assert_eq!(out.val, -13);

    let out =
        jit_run_test!(raw &format!("{typedef}; test :: -> MySumType.Float(-331.5);") => MySumType)
            .unwrap();
    println!("got bits  : {:032x}", transmute_unchecked::<_, u128>(&out));
    println!("got       : {:?}", out);
    let val = transmute_unchecked::<_, f64>(&out.val);
    assert_eq!(out.tag, 2);
    assert_eq!(val, -331.5);

    let out = jit_run_test!(raw &format!("
{typedef};
test :: -> {{
    mut val := MySumType.Void;
    val = MySumType.Int(-13);
    return val;
}}") => i128)
    .unwrap();
    println!("got bits  : {:032x}", out);
    let out = transmute_unchecked::<_, MySumType>(&out);
    println!("got       : {:?}", out);
    assert_eq!(out.tag, 1);
    assert_eq!(out.val, -13);
}

#[test]
fn anon_struct_in_sum_type() {
    #[derive(Debug)]
    #[repr(C)]
    struct MySumType {
        tag: u8,
        a: i64,
        b: f64,
    }

    let out = jit_run_test!(raw "
MySumType :: enum {
    A(struct { a: i64, b: f64 }),
    B(i64),
};
test :: -> MySumType.A(.{ a = 5, b = 10.123 });" => MySumType)
    .unwrap();
    println!("out:  {:?}", out);
    print!("bits:");
    let bits = transmute_unchecked::<_, [u64; 3]>(&out);
    for b in bits {
        print!(" {:016x}", b);
    }
    println!();
    assert_eq!(out.tag, 0);
    assert_eq!(out.a, 5);
    assert_eq!(out.b, 10.123);
}

#[test]
#[allow(unused)]
fn anon_struct_in_sum_type2() {
    #[derive(Debug)]
    #[repr(C)]
    struct MySumType {
        tag: u8,
        a: i32,
        b: f64,
    }

    #[derive(Debug)]
    #[repr(C)]
    struct Inner {
        a: i64,
        b: i64,
    }

    #[derive(Debug)]
    #[repr(C)]
    struct Enum_ {
        tag: u8,
        val: Inner,
    }

    let out = jit_run_test!(raw "
MySumType :: enum {
    A(struct { a: i32, b: f64 }),
    B(i64),
};
test :: -> MySumType.A(.{ a = 5, b = 10.123 });" => Enum_)
    .unwrap();
    println!("{:?}", out );
    /*
    println!("out:  {:?}", out);
    for b in transmute_unchecked::<_, [u32; 4]>(&out) {
        println!("{:08x}", b);
    }
    println!("bits:           {:032x}", transmute_unchecked::<_, u128>(&out));
    println!("10.123 -> bits: {:x}", 10.123f64.to_bits());
    assert_eq!(out.tag, 0);
    assert_eq!(out.a, 5);
    assert_eq!(out.b, 10.123);
    */

    todo!("{}", transmute_unchecked::<u64, f64>(&0x40243ef9db22d0e5))
}

#[test]
fn infer_enum_ty() {
    jit_run_test!("
MyEnum :: enum {
    A,
    B(struct { a: i64, b: f64 }),
};
a := .A;
b := .B(.{ a = 5, b = 10.123 });
" => ())
    .unwrap();
    todo!()
}

#[test]
fn return_inner_ty() {
    let out = jit_run_test!(raw "
test :: -> {
    MyBasicEnum :: enum { A, B, C };
    MyBasicEnum.A
};" => u8)
    .unwrap();
    panic!("{:?}", out)
}
