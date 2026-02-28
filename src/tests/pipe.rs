use crate::tests::{test, test_body};

#[test]
fn orelse_pipe_precedence() {
    let code = "
f :: (i: i32) -> i + 1;
test :: -> Some(10) orelse 20 |> f();
";
    test(code).ok(11i32);
}

#[test]
#[ignore = "todo"]
fn orelse_return_pipe_precedence() {
    let code = "
MyStruct :: struct { val :: (self: MyStruct) -> i32 456 };
try_new :: -> ?MyStruct null;
test :: -> try_new() orelse return 123 |>.val();
//        (                           )          or
//                         (                   ) ?
";
    test(code).ok(123i32);
}

#[test]
#[ignore = "todo"]
fn pipe_dot_field() {
    test_body("struct { x: i32 }.(123) |>.x").ok(123i32);
}
