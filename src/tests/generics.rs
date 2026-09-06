use crate::tests::{CompileTest, arr, assert_contains, fields, substr, test, test_body};

#[test]
fn generic_function_call() {
    #[derive(Debug, PartialEq)]
    #[repr(C)]
    struct Sums(i8, i16, i32, i64, i128);

    let code = "
add :: (l: $Num, r: Num) -> Num l + r;
test :: -> {
    sum8 := add(10, r=20, Num=i8);         // Explicit type annotation
    sum16 := add(1000.as(i16), 2000);      // Inferred, based on input 1
    sum32 := add(100000, 200000.as(i32));  // Inferred, based on input 2
    sum64 := add(r=-1000, l=2000.as(i64)); // Inferred, based on named arg
    sum128: i128 = add(0, -1);             // Inferred, based on output
    struct { sum8: i8, sum16: i16, sum32: i32, sum64: i64, sum128: i128 }
        .{ sum8, sum16, sum32, sum64, sum128 }
};
";
    let res = test(code).ok(Sums(30_i8, 3000_i16, 300_000_i32, 1000_i64, -1_i128));
    for ty in ["i8", "i16", "i32", "i64", "i128"] {
        assert_contains!(res.llvm_ir(), "{ty} @add.{ty}(");
        #[cfg(debug_assertions)]
        assert_contains!(res.llvm_ir(), "%add = add {ty} %l, %r");
        #[cfg(not(debug_assertions))]
        assert_contains!(res.llvm_ir(), "%add = add {ty} %0, %1");
        assert_contains!(res.llvm_ir(), "call noundef {ty} @add.{ty}(");
    }
    drop(res);

    // finalized type == generic default
    let code = "
add :: (l: $Num, r: Num) -> l + r;
test :: -> add(10, 20);
";
    let res = test(code).ok(30_i64);
    assert_contains!(res.llvm_ir(), "@add.i64");
    drop(res);

    // error: not a normal parameter (see const_parameter)
    test_body("id :: (x: $T) -> x; id(5, u32)")
        .error("Got 2 positional arguments, but expected at most 1 arguments", substr!("u32"));
}

#[test]
fn generic_inner_types() {
    let code = r#"
MyStruct :: struct { a: i8, b: []u8 };
func :: (ptr: *$A, arr: [3]$B, opt: ?$C) -> {}
test :: -> {
    int: i32 = 10;
    s: MyStruct = .(-1, "Hello World");

    func(&int, .[1, 2, 3], Some(s));
    func(&s, u16.[1, 2, 3], Some(s));
}
"#;
    let res = test(code).compile_no_err();
    assert_contains!(res.llvm_ir(), "@func.i32.i64.MyStruct");
    assert_contains!(res.llvm_ir(), "@func.MyStruct.u16.MyStruct");
    drop(res);

    // cannot infer type inside optional
    let code = r#"
take_opt :: (opt: ?$A) -> {};
test :: -> take_opt(null);
"#;
    test(code).error("Cannot infer value of generic argument `A`", substr!("take_opt(null)"));
}

#[test]
#[ignore = "TODO"]
fn infer_generic_based_on_inferred_return_type() {
    let code = "
id :: (x: $T) -> /* inferred: T */ x;
test :: -> a: u16 = id(1);
";
    // Currently `1` is inferred as `i64`, which causes a mismatch with `u16`
    test(code).compile_no_err();
}

#[test]
fn generic_type_error() {
    let code = r#"
add_pi :: (val: $Num) -> val + 3.1415;
test :: -> {
    add_pi(1);
    add_pi("Hello World");
    add_pi(struct { val := 1 }.{});
    add_pi(1.0);
}
"#;
    // TODO: also show the call expression which generated instantiation error
    test(code)
        .error("mismatched types (left: `i64`, right: `{float}`)", substr!("+"))
        .error("mismatched types (left: `[]u8`, right: `{float}`)", substr!("+"))
        .error("mismatched types (left: `struct{val:i64}`, right: `{float}`)", substr!("+"));
}

#[test]
fn generic_type_error_after_pause() {
    // error in block returns Ok(err_ty)
    let code = r#"
add_pi :: (val: $Num) -> {
    PAUSE1;
    val + 3.1415
}
test :: -> {
    add_pi(1);
    add_pi("Hello World");
    add_pi(struct { val := 1 }.{});
    add_pi(1.0);
}
PAUSE1 :: PAUSE2; PAUSE2 :: PAUSE3; PAUSE3 :: PAUSE4; PAUSE4 :: 1;
"#;
    test(code)
        .error("mismatched types (left: `i64`, right: `{float}`)", substr!("+"))
        .error("mismatched types (left: `[]u8`, right: `{float}`)", substr!("+"))
        .error("mismatched types (left: `struct{val:i64}`, right: `{float}`)", substr!("+"));

    // error in expression returns Err(HandledErr)
    let code = r#"
add_pi :: (val: $Num) -> PAUSE1 * (val + 3.1415);
test :: -> {
    add_pi(1);
    add_pi("Hello World");
    add_pi(struct { val := 1 }.{});
    add_pi(1.0);
}
PAUSE1 :: PAUSE2; PAUSE2 :: PAUSE3; PAUSE3 :: PAUSE4; PAUSE4 :: 1;
"#;
    test(code)
        .error("mismatched types (left: `i64`, right: `{float}`)", substr!("+"))
        .error("mismatched types (left: `[]u8`, right: `{float}`)", substr!("+"))
        .error("mismatched types (left: `struct{val:i64}`, right: `{float}`)", substr!("+"));
}

#[test]
fn generic_mismatches() {
    let code = r#"
f :: (a: $T, b: T) -> {};
test :: -> {
    f(1, "Hello");
}"#;
    test(code).error(
        "mismatched types: expected `{integer}` (value of `$T`); got `[]u8`",
        substr!("\"Hello\""),
    );

    /*// TODO
    let code = "
f :: (a: $T) -> {};
test :: -> {
    empty: ?*i32 = null;
    f(empty, T=*i32);
}";
    test(code).error(
        "mismatched types: expected `$T` (inferred as `*i32`); got `?*i32`",
        substr!("empty"; skip=1),
    );
    */

    // allow coercion if the generic is not specified explicitly
    let code = "
f :: (a: $T, b: T) -> {};
test :: -> {
    val: i32 = 1;
    empty: ?*i32 = null;
    f(&val, empty);
}
";
    test(code).compile_no_err();
}

#[test]
fn duplicate_generic_def() {
    // TODO: better error?
    test("f :: (a: $T, b: $T) -> {}")
        .error("duplicate parameter 'T'", substr!("T"; skip=1))
        .info("first definition of 'T'", substr!("T"));
}

#[test]
#[ignore = "is this a good idea?"]
fn parameter_generic_name_collision() {
    // TODO: better error?
    test("f :: (some_name: $some_name) -> {}")
        .error("duplicate parameter 'some_name'", substr!("some_name"; skip=1))
        .info("first definition of 'some_name'", substr!("some_name"));
}

#[test]
#[ignore = "TODO"]
fn infer_through_generic() {
    let code = "
MyStruct :: struct { val: i32 };
func :: (a: $T, b: T) -> {}
test :: -> {
    func(MyStruct.(1), .(2));
    func(.(1), MyStruct.(2));
        // a: T is unknown => a is unknown
        // b: b is known => T is set
        // a: T is known => a is known
}";
    test(code).compile_no_err();
}

#[test]
fn const_positional_parameter() {
    // type
    let code = "
sizeof :: ($ty: type) -> #sizeof(ty);
test :: -> sizeof(i32);
";
    let res = test(code).ok(4);
    assert_contains!(res.llvm_ir(), "i64 @sizeof.i32() {{");
    assert_contains!(res.llvm_ir(), "ret i64 4");
    assert_contains!(res.llvm_ir(), "call noundef i64 @sizeof.i32()");
    drop(res);

    // number
    let code = "
arr_splat :: (val: $T, $LEN: u64) -> [LEN]T {
    mut arr: [LEN]T;
    for idx in 0..LEN do arr[idx] = val;
    arr
}
test :: -> arr_splat(123, 5);
";
    let res = test(code).ok(arr([123_i64; 5]));
    assert_contains!(res.llvm_ir(), "@arr_splat.i64.5(");
    assert_contains!(res.llvm_ir(), "call void @arr_splat.i64.5(");
    drop(res);
}

#[test]
fn non_type_const_parameter() {
    /* TODO
    init_arr :: (default: $T) -> [$N]T {
        mut arr: [N]T;
        for idx in 0..N do arr[idx] = default;
        arr
    }
    */
    let code = r#"
arr_len :: (arr: [$N]$T) -> u64 N;
test :: ->
    arr_len(.[10, 20, 30, 40, 50])
    + arr_len(u16.[])
    + arr_len(.["Hello", "World", "!"])
;"#;
    let res = test(code).ok(8_usize);
    assert_contains!(res.llvm_ir(), "@arr_len.i64.5");
    assert_contains!(res.llvm_ir(), "@arr_len.u16.0");
    assert_contains!(res.llvm_ir(), "@\"arr_len.[]u8.3\"");
    drop(res);
}

// TODO: test :: -> MyStruct.(1, "Hello World", &test);

#[test]
fn generic_data_struct() {
    // infer generics
    for init in [r#".(1, "Hello World", &arr)"#, r#".{ a=1, b="Hello World", c=&arr }"#] {
        #[rustfmt::skip]
        let code = format!("
MyStruct :: struct($A, $B, $C) {{ a: A, b: []B, c: ?*C }}
static arr := u16.[1, 2, 3];
test :: -> MyStruct{init};
");
        let res = test(code).compile_no_err();
        assert_contains!(res.llvm_ir(), "[3 x i16] [i16 1, i16 2, i16 3]");
        assert_contains!(res.llvm_ir(), "sret({{ i64, {{ ptr, i64 }}, ptr }})");
    }

    // explicit generics
    for init in [r#".(1, "Hello World", &arr)"#, r#".{ a=1, b="Hello World", c=&arr }"#] {
        #[rustfmt::skip]
        let code = format!("
MyStruct :: struct($A, $B, $C) {{ a: A, b: []B, c: ?*C }}
static arr := u16.[1, 2, 3];
test :: -> MyStruct(i64, u8, [3]u16){init};
");
        let res = test(code).compile_no_err();
        assert_contains!(res.llvm_ir(), "[3 x i16] [i16 1, i16 2, i16 3]");
        assert_contains!(res.llvm_ir(), "sret({{ i64, {{ ptr, i64 }}, ptr }})");
    }

    // explicit generics -> error
    for init in [r#".(10, "Hello World", &arr)"#, r#".{ a=10, b="Hello World", c=&arr }"#] {
        #[rustfmt::skip]
        let code = format!("
MyStruct :: struct($A, $B, $C) {{ a: A, b: []B, c: ?*C }}
static arr := u16.[1, 2, 3];
test :: -> MyStruct(never, void, any){init};
");
        test(code)
            .error("mismatched types: expected `never`; got `{integer}`", substr!("10"))
            .error("mismatched types: expected `[]void`; got `[]u8`", substr!("\"Hello World\""));
    }
}

#[test]
fn generics_without_instantiations() {
    let code = "
my_generic_function :: (a: $A) -> a;
MyGenericStruct :: struct($A) {
    some_method :: -> {};
    some_generic_method :: (b: $B) -> b;
}
test :: -> 1 + 1;
";
    let res = test(code).ok(2);
    assert!(!res.llvm_ir().contains("my_generic_function"));
    assert!(!res.llvm_ir().contains("MyGenericStruct"));
    assert!(!res.llvm_ir().contains("some_method"));
    assert!(!res.llvm_ir().contains("some_generic_method"));
}

#[test]
fn generic_struct_detect_uninferred_generics() {
    let code = "
MyStruct :: struct($T) {
    val: T;
    debug :: (self: *MyStruct) -> { } // No error here because MyStruct in never instantiated
};
take_val :: (x: MyStruct) -> {
    x.&.debug();
    MyStruct.debug(&x);
}
";
    // before fix this emitted a cycle error
    test(code)
        .error(
            "Cannot infer generic parameters of type `MyStruct`",
            substr!("x: MyStruct";.until_end(8)),
        )
        .error(
            "Cannot infer generic parameters of type `MyStruct`",
            substr!("MyStruct.debug";.start_with_len(8)),
        );
}

#[test]
fn generic_def_in_struct_body() {
    test("MyStruct :: struct($A) { a: A; val: $B; }")
        .error("Cannot define generics inside a struct body", substr!("$B"))
        .info("Consider adding the struct generic here. $B", substr!("struct($A)";.end()));

    test("MyStruct :: struct { val: $B; }")
        .error("Cannot define generics inside a struct body", substr!("$B"))
        .info("Consider adding the struct generic here. ($B)", substr!("struct ";.end()));
}

#[test]
fn generic_struct_explicit_instantiation() {
    let code = "
MyStruct :: struct($T) { a: [2]T }
test :: -> MyStruct(i16).(.[-5, 10]);
";
    test(code).ok(fields([-5_i16, 10]));
}

#[test]
fn error_instantiate_non_generic_type() {
    let code = "
MyStruct :: struct { val: i32 };
static a: MyStruct(i32);
";
    test(code).error("Cannot call non-generic type `MyStruct`", substr!("MyStruct(i32)"));
}

#[test]
fn recursive_generic_instantiation() {
    let code = "
MyStruct :: struct($T) {
    val: T;
    DEFAULT :: MyStruct(i64).{ val=0 };  // 2. During analysis of MyStruct(i64), this requires
                                         // MyStruct(i64) => Dont instantiate it again!
    get_val :: (self: MyStruct(T)) -> self.val;
}
test :: -> MyStruct(i64).(5);       // 1. this instantiates MyStruct with T=i64
";
    let res = test(code).ok(5_i64);
    assert_eq!(res.llvm_ir().matches("get_val").count(), 1);
    drop(res);

    let code = "
MyStruct :: struct($T) {
    val: T;
    DEFAULT :: MyStruct.(0);
    get_val :: (self: MyStruct(T)) -> self.val;
}
test :: -> MyStruct(i64).(5);
";
    let res = test(code).ok(5_i64);
    assert_eq!(res.llvm_ir().matches("get_val").count(), 1);
    drop(res);
}

#[test]
fn pass_generic_to_generic() {
    let code = r#"
MyStruct :: struct($A) { val: A, get_val :: (self: MyStruct(A)) -> self.val; }
f :: (s: MyStruct($B)) -> {}
test :: -> {
    f(.{ val = 1 });
    // find f/typeof(f)
    // get typeof(s): `MyStruct($B)`
    // analyze and type check `.{ val = 1 }` with hint `MyStruct($B)`
    // infer $B := typeof(1)
    // instantiate MyStruct(A=i64)
    // instantiate f(B=i64)
    // finalize `.{ val = 1 }` with `MyStruct(i64)`

    f(.{ val = "" });
    f(.{ val = .[&1, &2, &3] });

    f(.(1.as(u16)));
    f(MyStruct.(1.as(i16)));
    s := MyStruct(MyStruct(u8)).{ val=.(5) };
    f(s);
}
"#;
    let res = test(code).compile_no_err();
    assert_contains!(res.llvm_ir(), "@f.i64(i64 %s)");
    assert_contains!(res.llvm_ir(), "@\"f.[]u8\"(");
    assert_contains!(res.llvm_ir(), "@\"f.[3]*i64\"(");

    assert_contains!(res.llvm_ir(), "call void @f.i64(");
    assert_contains!(res.llvm_ir(), "call void @\"f.[]u8\"(");
    assert_contains!(res.llvm_ir(), "call void @\"f.[3]*i64\"(");
    drop(res);

    let code = r#"
MyStruct :: struct($A) { val: ?A = null, get_val :: (self: MyStruct(A)) -> self.val; }
f :: (s: MyStruct($B)) -> {}
test :: -> f(.{ val = Some(1) });
"#;
    let res = test(code).compile_no_err();
    assert_contains!(res.llvm_ir(), "@f.i64(");
    assert_contains!(res.llvm_ir(), "call void @f.i64(");
}

#[test]
#[ignore = "todo"]
fn type_as_generic_value() {
    // currently panics
    let code = "
MyStruct :: struct($A) { val: A, get_val :: (self: MyStruct(A)) -> self.val; }
f :: (s: MyStruct($B)) -> {}
test :: -> f(MyStruct.(MyStruct(u8)).{ val=.(5) });
";
    test(code).error("todo", substr!("todo"));
}

#[test]
fn invalid_position_for_generic() {
    let code = "
test :: -> { a: $T = 1; }
";
    test(code).error("Generics cannot be defined inside a block", substr!("$T"));
}

#[test]
#[ignore = "todo"]
fn error_non_generic_type_parameter() {
    // any non const decl with var_ty `type` causes a panic when trying to access the type, because
    // it doesn't have a const_val
    let code = "
get_size :: (T: type) -> #sizeof(T);
test :: -> get_size(i32);
";
    test(code).compile_no_err();
}

#[test]
fn generic_param_with_init() {
    let code = "
f :: (
    a: $Num = 1, // this init causes a call to finalize, which must not fail for `$Num`
    b := 2,      // init must be analyzed to infer param type (i64)
) -> {}
";
    test(code).compile_no_err();

    let code = "
MyStruct :: struct($T) { val: T; }
f :: (
    a: MyStruct($Num) = .{ val=1 },  // this init causes a call to finalize, which must not fail \
                for `MyStruct($Num)`
    b := MyStruct.{ val=2 },
) -> {}
";
    test(code).compile_no_err();
}

#[test]
fn all_generic_param_cases() {
    let code = "
f :: (
    // currently function parameters are analyzed in order -> Generics are not available
    //use_a: A, use_b: B, use_c: C, use_d: D, use_e: E,

    normal: i32,

    a: $A,          // inferred (pos arg)
    $B: type,       // explicitly set (pos arg)

    c: $C,          // inferred (named arg)
    $D: type,       // explicitly set (named arg)
    e: $E,          // explicitly set (named arg)

    use_a2: A, use_b2: B, use_c2: C, use_d2: D, use_e2: E,
) -> {}

test :: -> f(
    //10      , 20      , 30      , 40      , 50      ,

    0,

    1.as(u64),      // infer A
    i128,           // set B

    c=2.as(u64),    // infer C
    D=i16,          // set D
    e=3, E=u8,      // set E

    use_a2=10, use_b2=20, use_c2=30, use_d2=40, use_e2=50,
);";
    let res = test(code).compile_no_err();
    assert_contains!(res.llvm_ir(), "void @f.u64.i128.u64.i16.u8");

    // same cases, but more compilcated data
    // TODO
}

// generic inference and explicit generic values are diffent and have to be handled seperately.
// e.g. explicit annotations first to determine if a strict type check is needed
#[test]
fn generic_param_mismatch() {
    let code = "
f :: (val: $T) -> {}
test :: -> f(1, T=[]u8);
";
    test(code)
        .error("mismatched types: expected `[]u8` (value of `$T`); got `{integer}`", substr!("1"));
}

#[test]
fn dont_duplicate_dependency_errors_in_generic_items() {
    // previously finalize_instantiation returned NotFinished analyze result of the newly created
    // instantiation. This meant that units for both instantiation expr and call expr stored the
    // dependency. Thus errors, like UNKNOWN associated constant, where emitted twice.
    // Even worse, other identical instantiation expressions assumed that sema of the instantiation
    // was finished. This also proves that the instantiation expression cannot be used wait for
    // sema of the instantiation, meaning that sema.unfinished_instantiations is required.
    let code = "
MyStruct :: struct {
    val: i32;
}
f :: ($T: type) -> MyStruct.UNKNOWN;
test :: -> f(i32);
test2 :: -> f(i32);
";
    test(code).error("no associated constant `UNKNOWN` on type `MyStruct`", substr!("UNKNOWN"))
    // no cycle error!
    ;
}
