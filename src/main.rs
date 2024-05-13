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
let a = a.add(1):add(1):String.from_int().len();
/*
let add = (a, b) -> a + b;

/// this is the main function
let main = -> {
    let a = 1;
    let a = a.add(1):add(1):String.from_int().len();
*/
//  -----------------------------------------------;
//  let - = ---------------------------------------
//      a   -------------------------------------()
//          ---------------------------------.---
//          -------------------------------() len
//          ---------------:---------------
//          ------------(1) ------.--------
//          --------:---    String from_int
//          -----(1) add
//          -.---
//          a add

    let a = a*add(1)+add(1)+String*from_int()*len();
};
"#;
    println!("+++ CODE START\n{}\n+++ CODE END", code);

    let exprs = mylang::parser::parse(code);

    for e in exprs {
        println!("{:#?}", e);
    }
}
