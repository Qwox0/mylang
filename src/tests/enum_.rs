use super::jit_run_test;
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
    println!("got bits  : {:032x}", transmute_unchecked::<_, u128>(&out));
    println!("got       : {:?}", out);
    assert_eq!(out.tag, 0);

    let code = format!("{typedef}; test :: -> MySumType.Int(-13);");
    let out = *jit_run_test_raw::<MySumType>(code).ok();
    println!("got bits  : {:032x}", transmute_unchecked::<_, u128>(&out));
    println!("got       : {:?}", out);
    assert_eq!(out.tag, 1);
    assert_eq!(transmute_unchecked::<_, i64>(&out.val), -13);

    let code = format!("{typedef}; test :: -> MySumType.Float(-331.5);");
    let out = *jit_run_test_raw::<MySumType>(code).ok();
    println!("got bits  : {:032x}", transmute_unchecked::<_, u128>(&out));
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

    let out = *jit_run_test_raw::<i128>(code).ok();
    println!("got bits  : {:032x}", out);
    let out = transmute_unchecked::<_, MySumType>(&out);
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

/*
#[test]
fn dev() {
    use inkwell::{context::Context, llvm_sys::core::LLVMBuildStore, module::Linkage};

    let context = Context::create();
    let builder = context.create_builder();
    let module = context.create_module("dev");

    let i64_ty = context.i64_type();
    let main_fn = i64_ty.fn_type(&[], false);
    let main_fn = module.add_function("main", main_fn, Some(Linkage::External));
    type Ret = i64;
    let entry = context.append_basic_block(main_fn, "entry");
    builder.position_at_end(entry);

    let a_ptr = builder.build_alloca(i64_ty, "a").unwrap();

    let ten = i64_ty.const_int(10, true);
    let b = transmute_unchecked::<_, LLVMBuilderRef>(&builder);
    let value = unsafe { LLVMBuildStore(b, ten.as_value_ref(), a_ptr.as_value_ref()) };

    let v = builder.build_store(a_ptr, ten).unwrap();
    v.set_alignment(8);

    let ret = builder.build_load(i64_ty, a_ptr, "").unwrap();
    builder.build_return(Some(&ret)).unwrap();

    module.print_to_stderr();

    let jit = module.create_jit_execution_engine(inkwell::OptimizationLevel::None).unwrap();
    let out = unsafe { jit.get_function::<unsafe extern "C" fn() -> Ret>("main").unwrap().call() };
    panic!("{}", out)
}
*/
