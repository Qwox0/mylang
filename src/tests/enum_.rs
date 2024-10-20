use super::jit_run_test;

#[test]
fn basic_enum() {
    jit_run_test!("
MyBasicEnum :: enum { A, B, C };
mut val = MyBasicEnum.A;
val = MyBasicEnum.B;" => ())
    .unwrap();
}

#[test]
fn sum_type() {
    jit_run_test!("
MySumType :: enum {
    Void,
    Int: i64,
    Float: f64,
};
mut val = MyBasicEnum.Void;
val = MyBasicEnum.Int(5);
" => ())
    .unwrap();
    todo!()
}

#[test]
fn anon_struct_in_sum_type() {
    jit_run_test!("
MySumType :: enum {
    A: struct { a: i64, b: f64 },
    B: i64,
};
a := MySumType.A(.{ a = 5, b = 10.123 });
TODO
" => ())
    .unwrap();
    todo!()
}
