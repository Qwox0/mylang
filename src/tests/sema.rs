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
    test(code).error("cycle(s) detected:", |_| TestSpan::ZERO); // TODO: test full cycle report
}

#[test]
fn correctly_handle_error_in_later_cycles() {
    let code = "
MyStruct :: struct { arr := .[7; LEN]; };
CONST :: MyStruct.();
LEN :: \"\";
test :: -> CONST.arr[4];";
    test(code).error("mismatched types: expected `u64`; got `[]u8`", substr!("LEN"));
}

#[test]
fn recursive_type() {
    let code = "
A :: struct { a: *A };
";
    test(code).compile_no_err();
}

#[test]
fn indirectly_recursive_types() {
    let code = "
A :: struct { b: *B };
B :: struct { a: *A };
";
    test(code).compile_no_err();
}

#[test]
fn indirectly_recursive_nested_types() {
    let code = "
A :: struct {
    B :: struct { a: *A };
    b: *B;
};";
    test(code).compile_no_err();

    let code = "
A :: struct {
    b: *B;
    X : A : .(nil); // must pause on `A` because type_check would fail otherwise
    Y :: A.(nil); // must pause on `A`
    B :: struct { a: *A };
};";
    test(code).compile_no_err();
}

#[test]
fn guessing_type_inference_on_mismatch() {
    let code = "
take_f :: (f: *(x: int, y: int) -> int) -> f.*(1, 2);
test :: -> take_f(/* missing '&' */ (x, y) -> x + y);
";
    test(code).error(
        "mismatched types: expected `*(x:int,y:int)->int`; got `(x:i64,y:i64)->i64`",
        substr!("(x, y) -> x + y"),
    );
}
