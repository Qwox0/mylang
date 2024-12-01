use crate::tests::jit_run_test;

#[test]
fn better_error_message1() {
    let code = "
pub test :: (mut x := 1) { // TODO: test this (better error)
    x += 1;
    420;
    1+2*x
};";
    todo!("better error message for \"{code}\"");
}

#[test]
fn better_error_message2() {
    jit_run_test!(raw "
test :: -> { 1 " => ())
    .unwrap();
    /*
    jit_run_test!(raw "
test :: -> {
    MyStruct :: struct { x: i64 };
    MyStruct.{ x = 5 }
" => ())
    .unwrap();
    */
    todo!("better error message");
    panic!("OK")
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
fn sret() {
    #[derive(Debug, PartialEq)]
    struct MyStruct {
        a: i64,
        b: i64,
        c: i64,
    }
    let (out, llvm_module_text) = jit_run_test!(raw "
        MyStruct :: struct { a: i64, b: i64, c: i64 };
        test :: -> MyStruct.{ a = 5, b = 10, c = 15 };" => MyStruct,llvm_module)
    .unwrap();
    assert_eq!(out, MyStruct { a: 5, b: 10, c: 15 });
    assert!(!llvm_module_text.contains("memcpy")); // TODO: implement this
}
