use super::jit_run_test;

#[test]
fn if_variants() {
    for my_bool in [true, false] {
        for variant_body in ["x = 1", "do x = 1", "{ x = 1 }"] {
            let code = format!(
                "
my_bool := {my_bool};
mut x := 0;
if my_bool {variant_body};
x"
            );
            let out = jit_run_test!(&code => i32).unwrap();
            let expected = if my_bool { 1 } else { 0 };
            assert!(out == expected, "```{code}\n``` -> expected: {expected}; got: {out}");
        }
    }
}

#[test]
fn if_expr() {
    for my_bool in [true, false] {
        for variant_then_body in ["5", "do 5", "{ 5 }"] {
            let code = format!(
                "
my_bool := {my_bool};
if my_bool {variant_then_body} else 10"
            );
            let out = jit_run_test!(&code => i32).unwrap();
            let expected = if my_bool { 5 } else { 10 };
            assert!(out == expected, "```{code}\n``` -> expected: {expected}; got: {out}");
        }
    }
}
