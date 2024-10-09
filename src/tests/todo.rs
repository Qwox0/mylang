#[allow(unused)]
//#[test]
fn todo1() {
    let code = "
pub test :: (mut x := 1) { // TODO: test this
    x += 1;
    420;
    1+2*x
};";
    todo!("{:?}", code);
}
