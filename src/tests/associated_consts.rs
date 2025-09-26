use crate::tests::{has_duplicate_symbol, substr, test, test_body};

#[test]
fn struct_method() {
    let code = "
MyStruct :: struct { val: i64 };
pub MyStruct.new :: -> MyStruct.(0);
MyStruct.inc :: (self: *mut MyStruct) -> self.*.val += 1;
test :: -> {
    mut a := MyStruct.new();
    a.&mut.inc();
    a.val
}";
    test(code).ok(1i64);
}

#[test]
fn use_before_definition() {
    let code = "
MyStruct :: struct { val: i64 };
test :: -> MyStruct.new().val;
MyStruct.new :: -> MyStruct.(MyStruct.DEFAULT_VAL);
MyStruct.DEFAULT_VAL :: 10;";
    test(code).ok(10i64);
}

#[test]
fn define_const_in_struct() {
    let code = "
MyStruct :: struct {
    x: i64;
    VAL :: 10;
};
test :: -> MyStruct.VAL;";
    test(code).ok(10i64);
}

#[test]
fn define_and_use_const_in_struct() {
    let code = "
MyStruct :: struct {
    val: i64 = DEFAULT_VAL;
    DEFAULT_VAL :: 10;
};
test :: -> MyStruct.().val;";
    test(code).ok(10i64);

    let code = "
Stack :: struct {
    buf: [Stack.SIZE]i64;
    len: u64 = 0,
};
Stack.SIZE : u64 : 128;
test :: -> Stack.(.[10; Stack.SIZE]).buf[127];";
    test(code).ok(10i64);
}

#[test]
fn error_missing_type_name() {
    test_body(".method :: -> {};")
        .error("A member declaration requires an associated type name", substr!("."));
}

#[test]
fn error_access_static_through_value() {
    let code = "
MyStruct :: struct { val: i64 };
MyStruct.NUM :: 10;
test :: -> {
    a := MyStruct.(0);
    a.NUM
}";
    test_body(code)
        .error("cannot access a static constant through a value", substr!("a.NUM"))
        .info(None, substr!("a.NUM";.start())); // not implemented: "consider replacing the value with its type 'MyStruct'"
}

#[test]
fn error_access_field_without_value() {
    let code = "
MyStruct :: struct {
    val: i32;
    f :: -> val;
};
test :: -> MyStruct.f();
";
    test(code).error("unknown identifier `val`", substr!("val";skip=1));
    // TODO: add hint?
}

#[test]
#[ignore = "todo: better error instead of cycle detection"]
fn error_access_missing_const() {
    let code = "
MyStruct :: struct {};
test :: -> MyStruct.SOME_MISSING_CONST;";
    test(code).ok(10i64);
}

#[test]
#[ignore = "not implemented"]
fn use_associated_const_in_struct_def() {
    let code = "
    MyArr :: struct { arr: [MyArr.LEN]i64 };
    MyArr.LEN :: 10;
    test :: -> {}";
    test(code).ok(());
}

#[test]
fn allow_associated_const_on_failed_struct() {
    let code = "
    MyArr :: struct { x: error };
    MyArr.NUM :: 10;
    test :: -> {}";
    test(code).error("unknown identifier `error`", substr!("error"));
}

#[test]
fn nested() {
    let code = "
MyStruct :: struct {};
MyStruct.Inner :: struct {};
MyStruct.Inner.Inner2 :: struct {};
MyStruct.Inner.Inner2.Inner3 :: struct {};
MyStruct.Inner.Inner2.NUM :: 10;
test :: -> MyStruct.Inner.Inner2.NUM;";
    test(code).ok(10i64);
}

#[test]
fn same_codegen_internal_or_external() {
    let code = "
MyStruct :: struct {
    val: i32;
    new :: (val: i32) -> MyStruct.{ val };
};
MyStruct.map :: (self: MyStruct, mapper: i32 -> i32) -> MyStruct.new(mapper(self.val));
test :: -> MyStruct.new(5).map(x -> x * 2);";
    let res = test(code).ok(10i64);

    // Both methods are mangled
    assert!(res.llvm_ir().contains("@\"struct{val:i32}.new\""));
    assert!(!res.llvm_ir().contains("@new"));
    assert!(res.llvm_ir().contains("@\"struct{val:i32}.map\""));
    assert!(!res.llvm_ir().contains("@map"));

    // Both mathods are only generated once
    assert!(!has_duplicate_symbol(res.llvm_ir(), "@\"struct{val:i32}.new\""));
    assert!(!has_duplicate_symbol(res.llvm_ir(), "@\"struct{val:i32}.map\""));
}

#[test]
fn use_static_method_name_in_struct_scope() {
    let code = "
MyStruct :: struct {
    val: i32;

    new :: (val: i32) -> MyStruct.{ val };
    map :: (self: MyStruct, mapper: i32 -> i32) -> new( // `MyStruct.new` not needed
        mapper(self.val)
    );
};
test :: -> MyStruct.new(5).map(x -> x * 2);";
    test(code).ok(10i64);

    let code = "
MyStruct :: struct {
    val: i32;
    new :: (val: i32) -> MyStruct.{ val };
};
MyStruct.map :: (self: MyStruct, mapper: i32 -> i32) -> new( // <- Error
    mapper(self.val)
);
test :: -> MyStruct.new(5).map(x -> x * 2);";
    test(code).error("unknown identifier `new`", substr!("new( // <- Error";.start_with_len(3)));
}

#[test]
fn compilation_order() {
    let code = "
MyStruct :: struct {
    val: i32;

    map :: (self: MyStruct, mapper: i32 -> i32) -> new(mapper(self.val));
    new :: (val: i32) -> MyStruct.{ val };
};
test :: -> MyStruct.new(5).map(x -> x * 2);
_ :: 1; // TODO: remove this
";
    test(code).ok(10i64);
}

#[test]
fn mangle_function_in_anon_struct() {
    let code = "
        struct {
            f :: -> i64 3;
        }.f()";
    test_body(code).ok(3i64);
}
