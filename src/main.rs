/*! This is a inner doc comment */

fn main() {
    let code = r#"
// This is a normal comment
/* This is also a normal comment */
//!This is a inner doc comment
/*! This is also a inner doc comment */

/*
    /*
        /*
            Comment in a comment in a comment
        */
    */
*/

// type: `i32 -> i32`
let square1 = (x: i32) -> i32 {
    x * x
};
let square4 = <T: Mul<T>>(x: T) -> x * x;
let square9 = x -> x * x; // automatic infer generic

let MyStruct = <T> -> struct {
    x: T,
};

/// this is the main function
let main = -> {
    let mut a = -3.1415;
    a = a * -1.as<f32>;
    a += 0.5;
    let my_char = 'a';
    let my_str = "hello world";

    let a = Some(1);
    let a = a.map(a -> a + 1).unwrap_or_else(-> 1);
};
"#;

    let code = r#"
/// this is the main function
let main = -> {
    let a = 1;
    let b = a + 1;
};
"#;
    println!("+++ CODE START\n{}\n+++ CODE END", code);

    mylang::parser::parse(code)
}
