use crate::tests::jit_run_test;

#[test]
fn string_lit() {
    let code = "
ERR_BYTE : u8 : 0;
my_string := \"Hello World\";
if my_string[ 0] !=  72 return ERR_BYTE; // 'H'
if my_string[ 1] != 101 return ERR_BYTE; // 'e'
if my_string[ 2] != 108 return ERR_BYTE; // 'l'
if my_string[ 3] != 108 return ERR_BYTE; // 'l'
if my_string[ 4] != 111 return ERR_BYTE; // 'o'
if my_string[ 5] !=  32 return ERR_BYTE; // ' '
if my_string[ 6] !=  87 return ERR_BYTE; // 'W'
if my_string[ 7] != 111 return ERR_BYTE; // 'o'
if my_string[ 8] != 114 return ERR_BYTE; // 'r'
if my_string[ 9] != 108 return ERR_BYTE; // 'l'
if my_string[10] != 100 return ERR_BYTE; // 'd'
my_string[6]";
    assert_eq!(*jit_run_test::<u8>(code).ok() as char, 'W');
}
