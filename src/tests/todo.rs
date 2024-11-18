use crate::tests::jit_run_test;

#[test]
fn todo1() {
    let code = "
pub test :: (mut x := 1) { // TODO: test this (better error)
    x += 1;
    420;
    1+2*x
};";
    todo!("{:?}", code);
}

#[test]
fn todo2() {
    jit_run_test!("mut a := 1;" => ()).unwrap();
    panic!("alignment of store instruction is 4");
}

#[test]
fn return_struct_u2_i64_correctly() {
    let out = jit_run_test!("struct { tag: u2 = 1, val: i64 = -1 }.{}" => u128).unwrap();
    let out_hex_string = format!("{out:032x}");
    println!("{out_hex_string}");
    // expect "ffffffffffffffff______________01"
    assert!(out_hex_string.starts_with(&"f".repeat(16)));
    assert!(out_hex_string.ends_with("01"));
}

#[test]
fn return_struct_f64() {
    let big_float_bits: u64 = 0xEFFFFFFFFFFFFFF3;
    let big_float = f64::from_bits(big_float_bits);

    #[derive(Debug)]
    struct Out {
        tag: u8,
        val: f64,
    }
    let code = format!(
        "
MySumType :: struct {{
    tag: u2,
    val: f64,
}};
test :: -> {{
    retval: MySumType;
    retval.tag=1;
    retval.val={big_float};
    retval
}};"
    );
    let out = jit_run_test!(raw &code => Out).unwrap();
    assert_eq!(out.tag, 1);
    assert_eq!(format!("{:x}", out.val.to_bits()), format!("{:x}", big_float_bits));
}

#[test]
fn return_local_vs_global_type() {
    // global
    // ```
    // ; ModuleID = 'test'
    // source_filename = "test"
    //
    // %MyStruct = type { i64 }
    //
    // define %MyStruct @test() {
    // entry:
    //   %struct = alloca %MyStruct, align 8
    //   %x = getelementptr inbounds %MyStruct, ptr %struct, i32 0, i32 0
    //   store i64 5, ptr %x, align 4
    //   %0 = load %MyStruct, ptr %struct, align 4
    //   ret %MyStruct %0
    // }
    // ```
    let out = jit_run_test!(raw "
MyStruct :: struct { x: i64 };
test :: -> MyStruct.{ x = 5 };
" => i64)
    .unwrap();
    assert_eq!(out, 5);

    // local
    // ```
    // ; ModuleID = 'test'
    // source_filename = "test"
    //
    // define { i64 } @test() {
    // entry:
    //   %struct = alloca { i64 }, align 8
    //   %x = getelementptr inbounds { i64 }, ptr %struct, i32 0, i32 0
    //   store i64 5, ptr %x, align 4
    //   %0 = load { i64 }, ptr %struct, align 4
    //   ret { i64 } %0
    // }
    // ```
    let out = jit_run_test!(raw "
test :: -> {
    MyStruct :: struct { x: i64 };
    MyStruct.{ x = 5 }
};" => i64)
    .unwrap();
    assert_eq!(out, 5);

    todo!();
}
