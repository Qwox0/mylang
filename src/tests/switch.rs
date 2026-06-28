use crate::tests::*;

#[rustfmt::skip]
fn test_enum_switch(switch_: &str) -> NewTest {
    test(format!("
MyEnum :: enum {{
    Void,
    Int(i32),
    Struct(struct {{ a: i32, b: i32 }}),
    String([]u8),
}}
get_val :: (e: MyEnum) -> i32 {switch_};
test :: -> i32.[
    get_val(.Void),
    get_val(.Int(20)),
    get_val(.Struct(.(20, 30))),
    get_val(.String(\"Hello World\")),
];"))
}

#[test]
fn switch_on_enum() {
    // basic usage
    let code = "
switch e {
    .Void :> 0;
    .Int :> if e > 5 then e else e * 2;
    .Struct :> e.a + e.b;
    .String :> e.len.as(i32);
}";
    test_enum_switch(code).ok(arr([0_i32, 20, 50, 11]));

    // exhaustiveness check
    test_enum_switch("switch e { .Int :> 0; }")
        .error(
            "missing cases `.Void`, `.Struct` and `.String` in exhaustive switch on enum `MyEnum`",
            substr!("switch e"),
        )
        .info("Consider adding an `else` case", substr!("};";.start_with_len(1)));

    // else
    test_enum_switch("switch e { .Int :> 10; } else -1").ok(arr([-1_i32, 10, -1, -1]));
}

#[test]
fn switch_case_with_associated_data() {
    let code = "
switch e {
    .Int(1) :> if e > 5 then e else e * 2;
    .Void :> 0;
    .String(\"Hello World\") :> e.len.as(i32);
}";
    test_enum_switch(code)
        .error("Enum cases with associated data are currently not implemented", substr!(".Int(1)"))
        .error(
            "Enum cases with associated data are currently not implemented",
            substr!(".String(\"Hello World\")"),
        )
        // continues with outer checks:
        .error(
            "missing cases  and `.Struct` in exhaustive switch on enum `MyEnum`",
            substr!("switch e"),
        );

    let code = "
switch e {
    .Int(1) :> if e > 5 then e else e * 2;
    .Void :> 0;
    .String(\"Hello World\") :> e.len.as(i32);
} else 0";
    test_enum_switch(code)
        .error("Enum cases with associated data are currently not implemented", substr!(".Int(1)"))
        .error(
            "Enum cases with associated data are currently not implemented",
            substr!(".String(\"Hello World\")"),
        );

    test_body("switch Some(5) { Some(1) :> 1; null :> 0 }")
        .error("Cases with associated data are currently not implemented", substr!("Some(1)"));
}

#[test]
fn switch_on_optional() {
    let code = "
get_val :: (opt: ?i64) -> i64 switch opt {
    null :> 5;
    Some :> opt * -1;
};
test :: -> .[get_val(null), get_val(Some(123))];
";
    test(code).ok(arr([5_i64, -123_i64]));

    // exhaustiveness check
    test_body("switch Some(1) {}")
        .error(
            "missing cases `null` and `Some` in exhaustive switch on enum `?i64`",
            substr!("switch Some(1)"),
        )
        .info("Consider adding an `else` case", substr!("}"));
}

#[test]
fn infer_out_ty() {
    // no else
    let code = "
MyEnum :: enum { A, B };
MyStruct :: struct { val: i32 };
test :: -> MyStruct {
    switch MyEnum.A { .A :> .(1); .B :> .{ val=2 } }
}";
    test(code).ok(1_i32);

    // with else
    let code = "
MyEnum :: enum { A, B };
MyStruct :: struct { val: i32 };
test :: -> MyStruct {
    switch MyEnum.A { .B :> .{ val=2 } } else .(1)
}";
    test(code).ok(1_i32);
}

#[test]
fn merge_bb_codegen() {
    // void
    let code = "
switch enum { A, B }.A {
    .A :> {};
    .B :> {};
}";
    test_body(code).compile_no_err();

    // zst struct
    let code = "
S :: struct {};
switch enum { A, B }.A {
    .A :> S.();
    .B :> S.();
}";
    test_body(code).compile_no_err();

    // never
    let code = "
switch enum { A, B }.A {
    .A :> return 1;
    .B :> return 2;
}";
    test_body(code).compile_no_err();

    // WriteTarget::Phi
    let code = "
switch enum { A, B }.A {
    .A :> 1;
    .B :> return 2;
}";
    let res = test_body(code).compile_no_err();
    assert!(res.llvm_ir().contains("phi i64"));
    drop(res);

    // WriteTarget::Ptr
    let code = "
a := switch enum { A, B }.A {
    .A :> 1;
    .B :> return 2;
}; a";
    let res = test_body(code).compile_no_err();
    assert!(res.llvm_ir().contains("store i64 1, ptr %a, align 8"));
    assert!(!res.llvm_ir().contains("phi"));
    drop(res);
}

#[test]
fn switch_on_ptr_to_enum() {
    let code = "
MyEnum :: enum {
    Void,
    Int(i32),
    Struct(struct { a: i32, b: i32 }),
}
add_one :: (e: *mut MyEnum) -> switch e {
    .Void :> {};
    .Int :> e.* += 1;
    .Struct :> e.*.a += 1;
};
test :: -> {
    mut v := MyEnum.Void;
    add_one(&mut v);
    mut i := MyEnum.Int(1);
    add_one(&mut i);
    mut s := MyEnum.Struct(.(10, 20));
    add_one(&mut s);
    .[v, i, s]
}";
    #[derive(Debug, Clone, Copy)]
    #[repr(C)]
    struct MyEnum {
        tag: u8,
        data: [i32; 2],
    }
    let res = test(code).get_out::<[MyEnum; 3]>();
    debug_assert_eq!(res[0].tag, 0);
    debug_assert_eq!(res[1].tag, 1);
    debug_assert_eq!(res[1].data[0], 2);
    debug_assert_eq!(res[2].tag, 2);
    debug_assert_eq!(res[2].data[0], 11);
    debug_assert_eq!(res[2].data[1], 20);

    let code = "
MyEnum :: enum { A, B, C }
test :: -> switch &MyEnum.B {
    .A :> 1;
    .B :> 2;
    .C :> 3;
};";
    test(code).ok(2);
}

#[test]
fn switch_on_ptr_to_opt() {
    let code = "
MyEnum :: enum {
    Void,
    Int(i32),
    Struct(struct { a: i32, b: i32 }),
}
add_one :: (e: *mut ?i32) -> switch e {
    Some :> e.* += 1;
    null :> {};
};
test :: -> {
    mut some: ?i32 = Some(5);
    add_one(&mut some);
    mut n: ?i32 = null;
    add_one(&mut n);
    .[some, n]
}";
    #[derive(Debug, Clone, Copy)]
    #[repr(C)]
    struct MyEnum {
        tag: u8,
        data: i32,
    }
    let res = test(code).get_out::<CFfiArray<[MyEnum; 2]>>();
    debug_assert_eq!(res.val[0].tag, 1);
    debug_assert_eq!(res.val[0].data, 6);
    debug_assert_eq!(res.val[1].tag, 0);
    debug_assert_eq!(res.val[1].data, 0);

    let code = "
set_ptr :: (ref: *mut ?*any) -> switch ref {
    Some :> ref.* = 0xabc.as(*any);
    null :> {};
};
test :: -> {
    mut some: ?*any = Some(&{});
    set_ptr(&mut some);
    mut n: ?*any = null;
    set_ptr(&mut n);
    .[some, n]
}";
    test(code).ok(arr([0xabc_usize, 0]));
}
