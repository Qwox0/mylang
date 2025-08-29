use crate::tests::{TestSpan, substr, test};

#[test]
fn error_cycle_in_struct() {
    let code = "
MyStruct :: struct {
    a: I;
    arr: [NEW.a]I = .[0; NEW.a];
    NEW :: MyStruct.(7);
};
I :: u64;
test :: -> MyStruct.NEW.a;";
    test(code).error("cycle(s) detected:", |_| TestSpan::ZERO);
}

#[test]
fn correctly_handle_error_in_later_cycles() {
    let code = "
MyStruct :: struct { arr := .[7; LEN]; };
CONST :: MyStruct.();
LEN :: \"\";
test :: -> CONST.arr[4];";
    test(code).error("mismatched types: expected u64; got []u8", substr!("LEN"));
}
