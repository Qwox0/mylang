use crate::tests::{substr, test, test_body};

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
    test_body(code).ok(b'W');
}

/// ```c
/// static char *a = "Hello World";
/// static char b[] = "Hello World";
///
/// void change_str(char **str) {
///     (*str)[0] = 'X';
/// }
///
/// int main() {
///     char *ptr = b;
///     change_str(&ptr);
///     printf("%s\n", b);
///     // change_str(&a); // => SIGSEGV
///     printf("%s\n", a);
/// }
/// ```
///
/// ```llvm
/// @b = internal global [12 x i8] c"Hello World\00", align 1
/// @a = internal global ptr @.str.1, align 8
/// @.str.1 = private unnamed_addr constant [12 x i8] c"Hello World\00", align 1
/// ```
#[test]
fn static_string() {
    test(r#"static text := "Hello World"; test :: -> {}"#).ok(());
    test(r#"static text: []mut u8 = "Hello World"; test :: -> {}"#)
        .error("mismatched types: expected `[]mut u8`; got `[]u8`", substr!("\"Hello World\""));
    // TODO: allow mutation of the string?
}
