//! random bugs

use crate::tests::*;

#[test]
fn bug_001() {
    // ERROR: mismatched types: expected `(T:type:$T,allocator:i64=1)->{unknown}`; got `{integer}` (reported @ 'src/sema/mod.rs:2635:21')
    //  --> {test @ src/tests/mod.rs:150:28}:7:37
    //   │
    // 7 │ test :: -> alloc_slice(u8, 1, alloc=1);
    //   │
    //
    // Problem: in `_validate_call`: recursive ty.generics_scope().find_decl was used.
    // Fix: use ty.generics_scope().find_decl_norec instead.
    let code = r#"
other :: #import "other.mylang";

alloc :: ($T: type, allocator := other.DEFAULT) -> {}
alloc_slice :: ($T: type, len: usize, allocator := other.DEFAULT) -> {}

test :: -> alloc_slice(u8, 1, alloc=1);
    "#;
    test(code)
        .with_prelude()
        .add_file("other.mylang", "DEFAULT :: 1;")
        .error("Unknown parameter", substr!("alloc=1";.start_with_len(5)));

    let code = r#"
alloc :: ($T: type, allocator := other.DEFAULT) -> {}
alloc_slice :: ($T: type, len: usize, allocator := other.DEFAULT) -> {}

test :: -> alloc_slice(u8, 1, alloc=1);

other :: struct { DEFAULT :: 1; }
    "#;
    test(code)
        .with_prelude()
        .error("Unknown parameter", substr!("alloc=1";.start_with_len(5)));
}

#[test]
fn bug_002() {
    // Panic during codegen because directives where not replaced.
    // never as a type value shouldn't return early.
    test_body("#sizeof(never)").ok(0_usize);
    test_body("#alignof(never)").ok(1_usize);

    // #sizeof_val as a directive causes too many problems -> create std.mem.sizeof_val instead.
    {
        let code = "
sizeof_val :: (val: $T) -> usize #sizeof(T); // copied from std.mem
test :: -> { sizeof_val(return) };
";
        test(code)
            .with_prelude()
            .error("mismatched types: expected `void`; got `u64`", substr!("sizeof_val(return)"));

        let code = "
sizeof_val :: (val: $T) -> usize #sizeof(T); // copied from std.mem
test :: -> { sizeof_val(return 123) };
";
        test(code).with_prelude().ok(123_usize);
    }
}
