use crate::tests::jit_run_test;

#[test]
fn string_dev() {
    jit_run_test!("my_string := \"Hello World\"" => ()).unwrap();
    panic!("OK")
}
