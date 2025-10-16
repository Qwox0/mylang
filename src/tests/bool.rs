use crate::tests::test_body;

#[test]
fn cast_bool_to_int() {
    for int in ["u8", "i8", "i32", "u32", "i64", "u64"] {
        test_body(format!(
            "cast :: (b: bool) -> b.as({int});
if cast(false) != 0 return 1;
if cast(true) != 1 return 2;
0"
        ))
        .ok(0i64);
    }
}
