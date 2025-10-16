use crate::tests::{arr, substr, test_body};

#[test]
fn mut_checks() {
    let code = "
    a :=   1 ; a = 2;
mut b :=   1 ; b = 2;
    c :=   1 ; c.&mut;
mut d :=   1 ; d.&mut;
    e := .[1]; e[0] = 2;
mut f := .[1]; f[0] = 2;
    g := .[1]; g[..]mut;
mut h := .[1]; h[..]mut;";
    test_body(code)
        .error("Cannot assign to `a`, as it is not declared as mutable", substr!("a = 2"))
        .info("consider changing `a` to be mutable", substr!("a :=";.start()))
        .error("Cannot mutably reference `c`, as it is not declared as mutable", substr!("c.&mut"))
        .info("consider changing `c` to be mutable", substr!("c := ";.start()))
        .error("Cannot assign to `e[0]`, as `e` is not declared as mutable", substr!("e[0] = 2"))
        .info("consider changing `e` to be mutable", substr!("e :=";.start()))
        .error(
            "Cannot mutably reference `g`, as it is not declared as mutable",
            substr!("g[..]mut"),
        )
        .info("consider changing `g` to be mutable", substr!("g := ";.start()));

    let code = "
a :=   1   .&   ; a.* = 2;
b :=   1   .&mut; b.* = 2;
c := .[1][..]   ; c[0] = 2;
d := .[1][..]mut; d[0] = 2;
e := .[1][..]   ; e[..]mut;
f := .[1][..]mut; f[..]mut;";
    test_body(code)
        .error(
            "Cannot assign to `a.*`, which is behind the immutable pointer `a`",
            substr!("a.* = 2"),
        )
        .info("The pointer type `*i64` is not `mut`", substr!("a.*";.start()))
        .error(
            "Cannot assign to `c[0]`, which is behind the immutable slice `c`",
            substr!("c[0] = 2"),
        )
        .info("The slice type `[]i64` is not `mut`", substr!("c[0]";.start()))
        .error(
            "Cannot mutably reference the elements of `e`, because it is an immutable slice",
            substr!("e[..]mut"),
        )
        .info("The slice type `[]i64` is not `mut`", substr!("e[..]mut";.start()));
}

#[test]
fn nested_slice_addrof_op() {
    let get_code = |var_mut: &str, slice_mut: &str, ptr_mut: &str| {
        format!(
            "{var_mut} arr := .[1,2,3]; ptr := arr[1..]{slice_mut}[0].&{ptr_mut}; ptr.* = 5; arr",
        )
    };
    test_body(get_code("mut", "mut", "mut")).ok(arr([1i64, 5, 3]));
    test_body(get_code("", "", "")).error(
        "Cannot assign to `ptr.*`, which is behind the immutable pointer `ptr`",
        substr!("ptr.* = 5"),
    );
    test_body(get_code("", "", "mut")).error(
        "Cannot mutably reference `arr[1..][0]`, which is behind the immutable slice `arr[1..]`",
        substr!("arr[1..][0].&mut"),
    );
    test_body(get_code("", "mut", "mut")).error(
        "Cannot mutably reference `arr`, as it is not declared as mutable",
        substr!("arr[1..]mut"),
    );
}

#[test]
fn mut_ptr_to_const() {
    test_body("MY_CONST :: 5; p := &mut MY_CONST; p.* += 100; MY_CONST")
        .ok(5i64)
        .warn(
            "The mutable pointer will reference a local copy of `MY_CONST`, not the constant \
             itself",
            substr!("&mut MY_CONST"),
        );
}

#[test]
fn receive_mut_ptr() {
    let get_code = |mut_marker: &str| {
        format!(
            "
increment_by5 :: (x: *mut i64) -> x.* += 5;
mut num := 0;
increment_by5(num.&{mut_marker});
num"
        )
    };
    test_body(get_code("mut")).ok(5i64);
    test_body(get_code(""))
        .error("mismatched types: expected *mut i64; got *i64", substr!("num.&"));
}

#[test]
fn deref_ptr_hint() {
    test_body("my_fn :: (ptr: *mut i64) -> ptr += 1;").error(
        "Cannot apply binary operatator `+=` to pointer type `*mut i64`",
        substr!("ptr";skip=1),
    );
}

#[test]
fn ptr_variable() {
    test_body("mut data := 1; ptr := data.&; ptr.* = 2;").error(
        "Cannot assign to `ptr.*`, which is behind the immutable pointer `ptr`",
        substr!("ptr.* = 2"),
    );
    test_body("mut data := 1; ptr := data.&; ptr = data.&;")
        .error("Cannot assign to `ptr`, as it is not declared as mutable", substr!("ptr = data.&"));

    test_body("mut data := 1; ptr := data.&mut; ptr.* = 2;").ok(());
    test_body("mut data := 1; ptr := data.&mut; ptr = data.&mut;").error(
        "Cannot assign to `ptr`, as it is not declared as mutable",
        substr!("ptr = data.&mut"),
    );

    test_body("mut data := 1; mut ptr := data.&; ptr.* = 2;").error(
        "Cannot assign to `ptr.*`, which is behind the immutable pointer `ptr`",
        substr!("ptr.* = 2"),
    );
    test_body("mut data := 1; mut ptr := data.&; ptr = data.&;").ok(());

    test_body("mut data := 1; mut ptr := data.&mut; ptr.* = 2;").ok(());
    test_body("mut data := 1; mut ptr := data.&mut; ptr = data.&mut;").ok(());
}

#[test]
fn field_mutation() {
    test_body("MyStruct :: struct { val: i32 }; data := MyStruct.(1); data.val = 2;").error(
        "Cannot assign to `data.val`, as `data` is not declared as mutable",
        substr!("data.val = 2"),
    );

    test_body("MyStruct :: struct { val: i32 }; mut data := MyStruct.(1); data.val = 2;").ok(());
}

#[test]
fn deep_check() {
    test_body("mut data := 1; ptr := data.&mut.&mut.&.&mut.&mut; ptr.*.*.*.*.* = 2;").error(
        "Cannot assign to `ptr.*.*.*.*.*`, which is behind the immutable pointer `ptr.*.*`",
        substr!("ptr.*.*.*.*.* = 2"),
    );

    test_body("mut data := 1; ptr := data.&mut.&mut.&mut.&mut.&mut; ptr.*.*.*.*.* = 2;").ok(());

    let code = "
mut data := 1;
MyStruct :: struct { ptr: *mut *mut *mut *mut *mut i64 };
a := MyStruct.(data.&mut.&mut.&mut.&mut.&mut);
a.ptr.*.*.*.*.* = 2;
        ";
    test_body(code).ok(());

    let code = "
mut data := .[1; 10];
slice := data[..]mut[..]mut[..][..]mut;
data[..]mut[..]mut[..][..]mut[1] = 2;
";
    let err = "Cannot mutably reference the elements of `data[..]mut[..]mut[..]`, because it is \
               an immutable slice";
    test_body(code)
        .error(err, substr!("data[..]mut[..]mut[..][..]mut"))
        .info("The slice type `[]i64` is not `mut`", substr!("data[..]mut[..]mut[..]"))
        .error(err, substr!("data[..]mut[..]mut[..][..]mut";skip=1))
        .info("The slice type `[]i64` is not `mut`", substr!("data[..]mut[..]mut[..]";skip=1));
}

#[test]
fn initializer_deep_check() {
    let code = "
MyStruct :: struct { a: i32 };
mut x: MyStruct;
ptr := &mut x;
ptr.{ a = 3 };
x";
    test_body(code).ok(3i32);

    let code = "
MyStruct :: struct { a: i32 };
mut x: MyStruct;
ptr := &x;
ptr.{ a = 3 };
x";
    test_body(code)
        .error(
            "Cannot initialize the value behind `ptr`, because it is an immutable pointer",
            substr!("ptr.{ a = 3 }"),
        )
        .info("The pointer type `*struct{a:i32}` is not `mut`", substr!("ptr";skip=1));

    let code = "
MyStruct :: struct { a: i32 };
mut x: MyStruct;
ptr := x.&mut.&mut.&.&mut.&mut;
ptr.*.*.*.*.{ a = 3 };
x";
    test_body(code)
        .error(
            "Cannot initialize the value behind `ptr.*.*.*.*`, because `ptr.*.*` is an immutable \
             pointer",
            substr!("ptr.*.*.*.*.{ a = 3 }"),
        )
        .info("The pointer type `**mut *mut struct{a:i32}` is not `mut`", substr!("ptr.*.*"));
}

#[test]
fn mut_addr_of_dereference() {
    test_body("mut data := 1; ptr := &data; ptr2 := ptr.*.&mut;")
        .error(
            "Cannot mutably reference `ptr.*`, which is behind the immutable pointer `ptr`",
            substr!("ptr.*.&mut"),
        )
        .info("The pointer type `*i64` is not `mut`", substr!("ptr.*";.start_with_len(3)));

    test_body("mut data := 1; ptr := &mut data; ptr2 := ptr.*.&mut; ptr == ptr2").ok(true);

    test_body("mut data := .[1; 10]; slice := data[..][..]mut;").error(
        "Cannot mutably reference the elements of `data[..]`, because it is an immutable slice",
        substr!("data[..][..]mut"),
    );
}

#[test]
fn mut_ptr_in_array_or_slice() {
    let code = |m| {
        format!(
            "
mut a := 1;
mut b := 1;
c := .[&mut a, &mut b];
c[1].* = 2;
mut d := c;
e := d[..]{m};
e[0].* = 2;

struct {{a:i64, b:i64}}.{{ a, b }}"
        )
    };
    test_body(code("")).error(
        "Cannot assign to `e[0].*`, which is behind the immutable slice `e`",
        substr!("e[0].* = 2"),
    );
    test_body(code("mut")).ok((2i64, 2i64));
}
